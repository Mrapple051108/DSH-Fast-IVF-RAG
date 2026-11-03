import { join, dirname, basename } from "node:path";

/**
 * 嵌入器：bge-small-zh（384 维，cls+normalize）为主，模型缺失时退到字符 n-gram 词法嵌入（512 维）。
 * 契约与 qqbot 的 local-embed 对齐：查询侧加前缀，文档侧不加。
 */
export const QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";
const ASSETS = ["onnx/model_quantized.onnx", "tokenizer.json", "config.json"];

export function createLexicalEmbedder(dims = 512) {
	return {
		kind: "lexical",
		dims,
		allDims: [dims],
		modelDir: null,
		async warmup() {},
		available() {
			return true;
		},
		status() {
			return { kind: "lexical", ready: true, dims, modelDir: null };
		},
		async embedQuery(text) {
			return lexicalVec(text, dims);
		},
		async embedPassages(texts) {
			return texts.map((t) => lexicalVec(t, dims));
		},
	};
}

function hash32(s) {
	let h = 2166136261;
	for (let i = 0; i < s.length; i++) {
		h ^= s.charCodeAt(i);
		h = Math.imul(h, 16777619);
	}
	return h >>> 0;
}

export function lexicalVec(text, dims = 512) {
	const out = new Float32Array(dims);
	const s = String(text ?? "").toLowerCase();
	const grams = [];
	for (let i = 0; i < s.length; i++) {
		const c = s[i];
		if (/\s/.test(c)) continue;
		grams.push(c);
		if (i + 1 < s.length && !/\s/.test(s[i + 1])) grams.push(s.slice(i, i + 2));
	}
	for (const g of grams) {
		const h = hash32(g);
		const idx = h % dims;
		const sign = (h >>> 31) & 1 ? -1 : 1;
		out[idx] += sign;
	}
	let n = 0;
	for (let i = 0; i < dims; i++) n += out[i] * out[i];
	if (n > 0) {
		const r = 1 / Math.sqrt(n);
		for (let i = 0; i < dims; i++) out[i] *= r;
	}
	return out;
}

function pickDims(tensorList) {
	const arr = Array.isArray(tensorList) ? tensorList : [tensorList];
	if (!arr.length) return 0;
	return Array.isArray(arr[0]) ? arr[0].length : arr[0]?.dims?.at(-1) ?? 0;
}

export async function createEmbedder({ modelDir, logger } = {}) {
	let extractor = null;
	let failedAt = 0;
	let failure = null;
	let pending = null;
	const fallback = createLexicalEmbedder(512);

	const log = (level, msg) => {
		try {
			logger?.[level]?.(msg);
		} catch {
			/* ignore */
		}
	};

	async function ensure() {
		if (extractor) return extractor;
		if (!modelDir) throw new Error("fast-ivf-rag: modelDir is empty");
		if (failedAt && Date.now() - failedAt < 60_000) throw failure ?? new Error("embedder cooling down");
		if (pending) return pending;
		pending = (async () => {
			const mod = await import("@huggingface/transformers");
			mod.env.allowRemoteModels = false;
			mod.env.allowLocalModels = true;
			mod.env.localModelPath = dirname(modelDir);
			const pipe = await mod.pipeline("feature-extraction", basename(modelDir), { dtype: "q8" });
			extractor = pipe;
			failure = null;
			log("info", `[fast-ivf-rag] 嵌入模型已加载：${basename(modelDir)}`);
			return pipe;
		})().catch((error) => {
			failedAt = Date.now();
			failure = error;
			log("warn", `[fast-ivf-rag] 嵌入模型不可用，降级词法嵌入：${error?.message ?? error}`);
			throw error;
		}).finally(() => {
			pending = null;
		});
		return pending;
	}

	// bge-small-zh-v1.5 实测 hidden_size = 512（config.json 里写死的），不是常见的 384
	let dims = 512;
	const api = {
		kind: "bge",
		modelDir,
		get dims() {
			return dims;
		},
		get allDims() {
			return [dims, 512];
		},
		async warmup() {
			try {
				await ensure();
				const v = await api.embedQuery("warmup");
				dims = v.length;
				return true;
			} catch {
				return false;
			}
		},
		available() {
			return Boolean(extractor);
		},
		status() {
			return { kind: extractor ? "bge" : "pending", ready: Boolean(extractor), dims, modelDir, failure: failure?.message ?? null };
		},
		async embedQuery(text) {
			try {
				const pipe = await ensure();
				const out = await pipe([QUERY_PREFIX + (text ?? "")], { pooling: "cls", normalize: true });
				const list = out.tolist();
				const vec = Float32Array.from(list[0]);
				dims = vec.length;
				return vec;
			} catch {
				return fallback.embedQuery(text);
			}
		},
		async embedPassages(texts) {
			if (!texts.length) return [];
			try {
				const pipe = await ensure();
				const out = await pipe(texts, { pooling: "cls", normalize: true });
				const list = out.tolist();
				const vecs = list.map((row) => Float32Array.from(row));
				dims = vecs[0].length;
				return vecs;
			} catch {
				return fallback.embedPassages(texts);
			}
		},
		reset() {
			extractor = null;
			failure = null;
			failedAt = 0;
		},
	};

	// 启动即预热，失败也不阻断（下次调用会重试）
	void api.warmup();
	return api;
}

export function checkModelAssets(modelDir) {
	const missing = ASSETS.filter((a) => {
		try {
			return !statSync(join(modelDir, a)).isFile();
		} catch {
			return true;
		}
	});
	return { ok: missing.length === 0, missing, modelDir };
}

import { statSync } from "node:fs";
