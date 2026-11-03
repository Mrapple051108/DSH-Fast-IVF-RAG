import { existsSync, mkdirSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";
import { IndexDb } from "./store.js";
import { RagLibrary } from "./library.js";
import { createEmbedder } from "./embed.js";
import { createWorkerEmbedder } from "./worker-embedder.js";
import { createReranker } from "./rerank.js";
import { defaultDataDir, defaultModelDir, defaultRerankDir, indexPath, libDir, libFile } from "./paths.js";

/**
 * 多库管理：一个全局索引库登记 library 与 session→library 绑定，
 * 每个库一个 sqlite 文件，按需打开、LRU 缓存句柄。
 */
export class RagManager {
	constructor(options = {}) {
		this.dataDir = options.dataDir || defaultDataDir();
		this.modelDir = options.modelDir || defaultModelDir();
		this.logger = options.logger ?? null;
		this.maxK = options.maxK ?? 512;
		this.rebuildEvery = options.rebuildEvery ?? 50;
		this.probe = options.probe ?? 3;
		this.condenseMaxLen = options.condenseMaxLen ?? 160;
		/** 语义去重阈值：新节点跟已有节点余弦 ≥ 这个值就丢弃（0 = 关） */
		this.dedupeThreshold = options.dedupeThreshold ?? 0.92;
		this.rerankEnabled = options.rerankEnabled ?? true;
		this.rerankModelDir = options.rerankModelDir || defaultRerankDir();
		this.rerankTopN = options.rerankTopN ?? 10;
		// 量化版本决定 transformers.js 读哪个 onnx 文件（q8 / fp16 / fp32 / q4f16 / q4）
		this.rerankDtype = options.rerankDtype || 'q8';
		/** 召回/嵌入的工作线程池；给了就说明连嵌入都不在主线程做 */
		this.workerPool = options.workerPool ?? null;
		this._index = null;
		this._embedder = null;
		this._reranker = null;
		this._libs = new Map();
		this._embedderPromise = null;
	}

	/** 运行期改配置：立刻作用到已打开的库，不用重启（嵌入器/数据目录这种还得重启）。 */
	configure(patch = {}) {
		if ("rerankEnabled" in patch) this.rerankEnabled = Boolean(patch.rerankEnabled);
		if ("rerankModelDir" in patch) this.rerankModelDir = patch.rerankModelDir || defaultRerankDir();
		if ("rerankTopN" in patch) this.rerankTopN = Number(patch.rerankTopN) || 10;
		if ("rerankDtype" in patch) this.rerankDtype = String(patch.rerankDtype || 'q8');
		if ("condenseMaxLen" in patch) this.condenseMaxLen = Number(patch.condenseMaxLen) || 160;
		if ("dedupeThreshold" in patch) {
			const v = Number(patch.dedupeThreshold);
			this.dedupeThreshold = Number.isFinite(v) && v >= 0 && v <= 1 ? v : 0.92;
		}
		this._reranker = null;
		const r = this.reranker();
		for (const lib of this._libs.values()) {
			lib.setReranker(r);
			lib.condenseMaxLen = this.condenseMaxLen;
			lib.dedupeThreshold = this.dedupeThreshold;
		}
		return this.reranker()?.status?.() ?? { enabled: false };
	}

	/** 精排器是可选件：关掉或没模型时返回 null，检索照常走 RRF。 */
	reranker() {
		// 有工作线程时，精排一律在 worker 里建：同进程两份 onnxruntime 会话会
		// 撞 V8 HandleScope 锁直接 FATAL，所以主线程一个 ONNX 对象都不许碰。
		if (this.workerPool) return null;
		if (!this.rerankEnabled) return null;
		if (this._reranker) return this._reranker;
		this._reranker = createReranker({ modelDir: this.rerankModelDir, logger: this.logger, topN: this.rerankTopN, dtype: this.rerankDtype });
		return this._reranker;
	}

	/** 给 /api/status 用：线程模式下精排状态由 worker 侧报告，这里只报开关。 */
	rerankStatus() {
		if (this.workerPool) return { enabled: this.rerankEnabled, threaded: true, ready: this.rerankEnabled };
		return this.reranker()?.status?.() ?? { enabled: false };
	}

	ensureDirs() {
		mkdirSync(libDir(this.dataDir), { recursive: true });
	}

	get index() {
		if (!this._index) {
			this.ensureDirs();
			this._index = new IndexDb(indexPath(this.dataDir));
		}
		return this._index;
	}

	async embedder() {
		if (this._embedder) return this._embedder;
		if (!this._embedderPromise) {
			// 有工作线程时，嵌入也交给它：宿主主线程里绝不再加载第二份 ORT。
			// （同一个进程里同时存在两个 onnxruntime 会话会撞 V8 HandleScope 锁，直接 FATAL。）
			this._embedderPromise = (
				this.workerPool
					? Promise.resolve(createWorkerEmbedder(this.workerPool, this.modelDir))
					: createEmbedder({ modelDir: this.modelDir, logger: this.logger })
			).then((e) => {
				this._embedder = e;
				return e;
			});
		}
		return this._embedderPromise;
	}

	listLibraries() {
		const rows = this.index.listLibraries();
		return rows.map((r) => {
			const file = libFile(this.dataDir, r.id);
			let bytes = 0;
			for (const f of [file, file + "-wal"]) {
				try {
					bytes += statSync(f).size;
				} catch {
					/* ignore */
				}
			}
			const binds = this.index.bindingsOf(r.id);
			const open = this._libs.get(r.id);
			return {
				...r,
				dims: r.dims ?? (open ? open.dims : null),
				bytes,
				sessions: binds.map((b) => b.session),
				bound: binds.length,
				exists: existsSync(file),
				open: Boolean(open),
			};
		});
	}

	createLibrary(name) {
		const now = Date.now();
		const base = String(name || "记忆库").trim() || "记忆库";
		const id = `${base.replace(/[^\w\u4e00-\u9fff-]+/g, "-").slice(0, 24)}-${now.toString(36)}`;
		this.ensureDirs();
		const row = { id, name: base, model: this.modelDir, dims: null, created: now, updated: now, last_used: now };
		this.index.upsertLibrary(row);
		return row;
	}

	deleteLibrary(id) {
		const lib = this._libs.get(id);
		if (lib) {
			lib.close();
			this._libs.delete(id);
		}
		this.index.deleteLibrary(id);
		const file = libFile(this.dataDir, id);
		for (const f of [file, file + "-wal", file + "-shm"]) {
			try {
				rmSync(f, { force: true });
			} catch {
				/* ignore */
			}
		}
	}

	/** 打开/新建库对象（同步，嵌入器异步补） */
	library(id) {
		if (!id) return null;
		const cached = this._libs.get(id);
		if (cached) return cached;
		let row = this.index.getLibrary(id);
		if (!row) {
			// 磁盘上有文件但索引没登记（手工拷入）→ 补登记
			const file = libFile(this.dataDir, id);
			if (!existsSync(file)) return null;
			row = { id, name: id, created: Date.now() };
			this.index.upsertLibrary(row);
		}
		const lib = new RagLibrary({
			id,
			file: libFile(this.dataDir, id),
			embedder: this._embedder ?? lazyEmbedder(this),
			rebuildEvery: this.rebuildEvery,
			maxK: this.maxK,
			probe: this.probe,
			onChange: (patch) => {
				try {
					this.index.upsertLibrary({ id, name: this.index.getLibrary(id)?.name ?? id, ...patch });
				} catch {
					/* ignore */
				}
			},
		});
		this._libs.set(id, lib);
		lib.condenseMaxLen = this.condenseMaxLen;
		lib.dedupeThreshold = this.dedupeThreshold;
		lib.reranker = this.reranker();
		this.index.touchLibrary(id);
		return lib;
	}

	async libraryAsync(id) {
		const lib = this.library(id);
		if (!lib) return null;
		lib.embedder = await this.embedder();
		return lib;
	}

	/**
	 * 解析会话默认库：mode='new' 每次新库，'reuse' 沿用上次/最近一个。
	 * autoMount=false 时，**从没绑过库**的会话直接返回 null（不建库、不挂载）——
	 * 已绑定的会话照旧，面板手动绑定也照旧。
	 */
	async resolveSession(sessionId, mode = "reuse", sessionLabel = "", autoMount = true) {
		const existing = this.index.binding(sessionId);
		if (existing && this.index.getLibrary(existing)) {
			this.index.bind(sessionId, existing);
			return this.libraryAsync(existing);
		}
		if (!autoMount) return null;
		if (mode === "reuse") {
			const list = this.index.listLibraries();
			if (list.length) {
				this.index.bind(sessionId, list[0].id);
				return this.libraryAsync(list[0].id);
			}
		}
		const row = this.createLibrary(sessionLabel ? `${sessionLabel}-记忆` : "会话记忆");
		this.index.bind(sessionId, row.id);
		return this.libraryAsync(row.id);
	}

	bind(sessionId, libId) {
		this.index.bind(sessionId, libId);
		this.index.touchLibrary(libId);
	}

	allBindings() {
		return this.index.allBindings();
	}

	async statsFor(id) {
		const lib = await this.libraryAsync(id);
		if (!lib) return null;
		const row = this.index.getLibrary(id);
		return { ...lib.info(), name: row?.name ?? id, created: row?.created, lastUsed: row?.last_used };
	}

	close() {
		for (const lib of this._libs.values()) lib.close();
		this._libs.clear();
		this._index?.close();
		this._index = null;
	}
}

/** 打开库时嵌入器可能还没加载完，先给个按需等待的壳 */
function lazyEmbedder(manager) {
	return {
		kind: "lazy",
		dims: 384,
		available: () => Boolean(manager._embedder),
		status: () => manager._embedder?.status() ?? { kind: "lazy", ready: false },
		embedQuery: async (t) => (await manager.embedder()).embedQuery(t),
		embedPassages: async (ts) => (await manager.embedder()).embedPassages(ts),
	};
}

