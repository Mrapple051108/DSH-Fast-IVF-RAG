/**
 * 召回服务（工作线程 / 子进程两用）。
 *
 * 主线程只做「发消息 / 收结果」，向量的 SQLite 读取、512 维点积、BM25、精排全在这里跑。
 * 为什么必须离线：宿主事件循环里同时挂着 SSE 流，主线程被几百毫秒同步计算占住，
 * socket 就读不动，表现就是 "DeepSeek Messages transport failed"。
 *
 * 为什么最后用了**子进程**而不是 worker_threads：
 * onnxruntime 在同一个 V8 进程里极其敏感 —— 主线程与 worker 各持一份会话时，
 * 先后踩到两个必崩：
 *   FATAL ERROR: v8::HandleScope::CreateHandle() Cannot create a handle without a HandleScope
 *   The requested API version [30] is not available, only API versions [1, 21] are supported
 * 都是整个宿主进程直接退出。子进程是独立 V8，崩也只崩它自己，主进程照常。
 *
 * 传输层自适应：worker_threads 用 parentPort，child_process 用 process IPC。
 */
import { parentPort } from "node:worker_threads";

import { createEmbedder } from "./embed.js";
import { RagLibrary } from "./library.js";
import { libFile } from "./paths.js";

const transport = parentPort
	? {
			send: (message) => parentPort.postMessage(message),
			on: (handler) => parentPort.on("message", handler),
		}
	: typeof process.send === "function"
		? {
				send: (message) => process.send(message),
				on: (handler) => process.on("message", handler),
			}
		: null;

/** @type {Map<string, RagLibrary>} */
const libs = new Map();
/** @type {any} */
let embedder = null;

function send(message) {
	try {
		transport?.send(message);
	} catch {
		/* 对端没了 */
	}
}

async function ensureEmbedder(modelDir) {
	if (!embedder) {
		embedder = await createEmbedder({ modelDir, logger: console });
	}
	return embedder;
}

async function libraryFor({ dataDir, modelDir, libId }) {
	const key = `${dataDir}::${libId}`;
	const cached = libs.get(key);
	if (cached) return cached;
	const emb = await ensureEmbedder(modelDir);
	const lib = new RagLibrary({
		id: libId,
		file: libFile(dataDir, libId),
		embedder: emb,
		onChange: null,
	});
	libs.set(key, lib);
	return lib;
}

transport?.on(async (msg) => {
	const { id, type } = msg ?? {};
	const payload = msg?.payload ?? {};
	try {
		if (type === "ping") {
			send({ id, ok: true, result: "pong" });
			return;
		}
		if (type === "close") {
			for (const lib of libs.values()) {
				try {
					lib.db?.close?.();
				} catch {
					/* 关不掉也无所谓，进程要退了 */
				}
			}
			libs.clear();
			send({ id, ok: true, result: true });
			return;
		}
		if (type === "embed") {
			const emb = await ensureEmbedder(payload.modelDir);
			const dims = emb?.dims ?? null;
			const kind = emb?.kind ?? "unknown";
			if (payload.query) {
				const vector = Array.from(await emb.embedQuery(String(payload.text ?? "")));
				send({ id, ok: true, result: { vector, dims, kind } });
				return;
			}
			const vectors = (await emb.embedPassages((payload.texts ?? []).map((t) => String(t)))).map((v) => Array.from(v));
			send({ id, ok: true, result: { vectors, dims, kind } });
			return;
		}
		if (type === "add") {
			const lib = await libraryFor(payload);
			if (Number.isFinite(payload.rebuildEvery)) lib.rebuildEvery = payload.rebuildEvery;
			const items = (payload.items ?? []).map((it) => ({
				text: String(it?.text ?? ""),
				kind: it?.kind ?? "note",
				src: it?.src ?? null,
				ts: it?.ts ?? Date.now(),
			}));
			const started = Date.now();
			const result = await lib.add(items);
			send({ id, ok: true, result: { ...result, workerMs: Date.now() - started } });
			return;
		}
		if (type === "search") {
			const lib = await libraryFor(payload);
			if (payload.rerank && payload.rerankModelDir && !lib.reranker) {
				try {
					const { createReranker } = await import("./rerank.js");
					lib.reranker = await createReranker({
						modelDir: payload.rerankModelDir,
						logger: console,
						topN: payload.rerankTopN ?? 10,
						// 量化版本必须跟下载的那个文件对上，否则找不到权重
						dtype: payload.rerankDtype || "q8",
					});
				} catch (error) {
					console.warn(`[recall-service] 精排不可用：${error?.message ?? error}`);
					lib.reranker = null;
				}
			}
			if (Number.isFinite(payload.condenseMaxLen)) lib.condenseMaxLen = payload.condenseMaxLen;
			const started = Date.now();
			const result = await lib.search(payload.query, {
				topK: payload.topK,
				probe: payload.probe,
				assoc: payload.assoc,
				assocCount: payload.assocCount,
				assocPerHit: payload.assocPerHit,
				assocMode: payload.assocMode,
				minScore: payload.minScore,
				minScoreMode: payload.minScoreMode,
				minScoreFloor: payload.minScoreFloor,
				minScoreRatio: payload.minScoreRatio,
				mmrLambda: payload.mmrLambda,
				rerank: Boolean(payload.rerank) && Boolean(lib.reranker),
			});
			result.meta = { ...result.meta, workerMs: Date.now() - started };
			send({ id, ok: true, result });
			return;
		}
		send({ id, ok: false, error: `unknown message type: ${String(type)}` });
	} catch (error) {
		send({ id, ok: false, error: String(error?.stack ?? error?.message ?? error) });
	}
});

if (!parentPort && typeof process.send === "function") {
	// 子进程：别因为父进程没发消息就退出，IPC 通道会保持事件循环
	process.on("disconnect", () => process.exit(0));
}
