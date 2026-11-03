/**
 * 召回进程的客户端（主线程侧）。
 *
 * 设计要点：
 * - **独立子进程**，不是 worker_threads：onnxruntime 在同一 V8 进程里同时被主线程和
 *   worker 持有会崩整个宿主（HandleScope FATAL / "requested API version [30]"）。
 *   子进程是独立 V8，崩也只崩它自己，主进程照常跑，下一次调用自动重启。
 * - 懒启动：插件加载不付任何代价，第一次召回才 fork。
 * - 复用：一个子进程里按 dataDir+libId 复用同一个 RagLibrary（含缓存好的嵌入器与精排器）。
 * - 带超时：子进程卡死 → kill 掉，下一次自动重启，绝不把主线程吊死。
 * - unref：DSH 退出时不因为这个子进程而拖着不关。
 */
import { fork } from "node:child_process";
import { statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const SERVICE = new URL("./recall-worker.js", import.meta.url);

/**
 * 子进程里跑的这几个文件的"版本指纹"（size + mtime 求和）。
 * 子进程是跨插件热重载复用的单例，代码改了它自己不会知道 ——
 * 所以重载插件时比一下指纹，对不上就把旧子进程干掉重建，
 * 否则改 library.js 之后召回还在跑老逻辑，只有重启 DSH 才生效。
 */
const STAMP_FILES = ["./library.js", "./recall-worker.js", "./store.js", "./embed.js", "./rerank.js", "./condense.js"];

function stampOf() {
	let sum = 0;
	try {
		for (const rel of STAMP_FILES) {
			const st = statSync(fileURLToPath(new URL(rel, import.meta.url)));
			sum += st.size + st.mtimeMs;
		}
		return sum;
	} catch {
		return 0;
	}
}

export class RecallWorker {
	constructor({ timeoutMs = 20_000, logger = null } = {}) {
		this.kind = "fork";
		this.timeoutMs = timeoutMs;
		this.logger = logger;
		this.child = null;
		this.pending = new Map();
		this.seq = 0;
		this.failures = 0;
		this.disposed = false;
		this.stamp = stampOf();
	}

	_spawn() {
		if (this.child?.connected) return this.child;
		const child = fork(SERVICE, [], {
			stdio: ["ignore", "ignore", "ignore", "ipc"],
			// Electron 里 process.execPath 是 electron.exe，得让它以纯 Node 身份跑
			execPath: process.execPath,
			env: { ...process.env, ELECTRON_RUN_AS_NODE: "1" },
			serialization: "json",
		});
		child.unref?.();
		child.channel?.unref?.();
		child.on("message", (message) => {
			const entry = this.pending.get(message?.id);
			if (!entry) return;
			this.pending.delete(message.id);
			clearTimeout(entry.timer);
			this.failures = 0;
			if (message.ok) entry.resolve(message.result);
			else entry.reject(new Error(message.error ?? "recall service failed"));
			this._idle();
		});
		child.on("error", (error) => this._abort(error));
		child.on("exit", (code, signal) => {
			if (this.child === child) this.child = null;
			if (code !== 0 && !this.disposed) {
				this._abort(new Error(`recall service exited (code=${code} signal=${signal ?? "-"})`));
			}
		});
		this.child = child;
		return child;
	}

	_abort(error) {
		for (const [, entry] of this.pending) {
			clearTimeout(entry.timer);
			entry.reject(error instanceof Error ? error : new Error(String(error)));
		}
		this.pending.clear();
		const child = this.child;
		this.child = null;
		this.failures += 1;
		try {
			child?.kill();
		} catch {
			/* ignore */
		}
	}

	/**
	 * 空闲时把子进程从事件循环里摘掉（DSH 退出不被它拖住），
	 * 一有请求就重新 ref 回来，否则 Node 会因为「没有活着的句柄」直接
	 * 把带 pending promise 的进程退掉（unsettled top-level await, exit 13）。
	 */
	_idle() {
		if (this.pending.size > 0) return;
		try {
			this.child?.channel?.unref?.();
			this.child?.unref?.();
		} catch {
			/* ignore */
		}
	}

	_busy() {
		try {
			this.child?.ref?.();
			this.child?.channel?.ref?.();
		} catch {
			/* ignore */
		}
	}

	_call(type, payload) {
		let child;
		try {
			child = this._spawn();
		} catch (error) {
			return Promise.reject(error);
		}
		const id = ++this.seq;
		return new Promise((resolve, reject) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				reject(new Error(`recall service timeout after ${this.timeoutMs}ms`));
				this._abort(new Error("recall service timeout"));
			}, this.timeoutMs);
			this.pending.set(id, { resolve, reject, timer });
			try {
				this._busy();
				child.send({ id, type, payload });
			} catch (error) {
				clearTimeout(timer);
				this.pending.delete(id);
				reject(error);
				this._idle();
			}
		});
	}

	/** @returns {Promise<{hits:any[], meta:any}>} */
	search(payload) {
		return this._call("search", payload);
	}

	/**
	 * 写入记忆（嵌入 + 事务 + 条件重聚类）也在子进程里做。
	 * 主线程里 node:sqlite 是同步 API，条目一多或刚好撞上 rebuild，就会把事件循环
	 * 按住几百毫秒，SSE 一卡就是 "transport failed"。
	 */
	add(payload) {
		return this._call("add", payload);
	}

	/** 查询侧嵌入（会自动加 bge 的检索前缀，在子进程里做） */
	embedQuery(modelDir, text) {
		return this._call("embed", { modelDir, query: true, text });
	}

	/** 文档侧嵌入（批量，在子进程里做） */
	embedPassages(modelDir, texts) {
		return this._call("embed", { modelDir, query: false, texts });
	}

	async ping() {
		return this._call("ping", {});
	}

	async close() {
		const child = this.child;
		this.child = null;
		this.disposed = true;
		if (!child) return;
		try {
			child.send({ id: ++this.seq, type: "close", payload: {} });
		} catch {
			/* ignore */
		}
		// 子进程收到 close 会关库；disconnect 让它的 process.on("disconnect") 自己退
		try {
			child.disconnect?.();
		} catch {
			/* ignore */
		}
		await new Promise((resolve) => {
			try {
				child.ref?.();
				child.channel?.ref?.();
			} catch {
				/* ignore */
			}
			const timer = setTimeout(() => {
				try {
					child.kill();
				} catch {
					/* ignore */
				}
				resolve(undefined);
			}, 1500);
			child.once("exit", () => {
				clearTimeout(timer);
				resolve(undefined);
			});
		});
	}
}

/**
 * 进程级单例。
 *
 * 为什么要单例：插件热重载（改 cordis.patch.yml、改配置）会重新 apply 一次，
 * 每次都 new 一个 RecallWorker 会在同一进程里留下多个连向同一批 sqlite 的句柄，
 * 还会反复预热模型。所以子进程只建一次，挂在 globalThis 上跨 fiber 复用；
 * 插件卸载时**不销毁**它，退出时随宿主一起走。
 */
const SHARED_KEY = Symbol.for("fast-ivf-rag:shared-service");

export function getSharedWorker(options = {}) {
	const now = stampOf();
	const existing = globalThis[SHARED_KEY];
	if (existing && !existing.disposed && existing.kind === "fork") {
		if (existing.stamp && now && existing.stamp !== now) {
			// 代码变了：子进程里跑的还是旧版本，关掉重建（模型下次用时重新加载）
			existing.logger?.info?.("[fast-ivf-rag] 召回子进程的代码变了，重启它");
			void existing.close().catch(() => {});
		} else {
			existing.logger = options.logger ?? existing.logger;
			existing.timeoutMs = options.timeoutMs ?? existing.timeoutMs;
			return existing;
		}
	}
	const created = new RecallWorker(options);
	created.stamp = now;
	Object.defineProperty(globalThis, SHARED_KEY, { value: created, configurable: true, writable: true });
	return created;
}
