/**
 * 可选精排：本地 cross-encoder（bge-reranker-v2-m3 / bge-reranker-base 一类，ONNX 量化版）。
 * 没有模型就静默跳过 —— 精排是加分项，不能挡住检索主链路。
 *
 * 为什么不用 pipeline("text-classification")：reranker 的 config 是单标签
 * （id2label 只有 LABEL_0），transformers.js 的 pipeline 会对 1 个 logit 做 softmax，
 * 结果永远 1.0 —— 排不出差别。这里直接跑 AutoModel，读原始 logits 自己 sigmoid。
 */
import { existsSync, readdirSync } from "node:fs";
import { basename, dirname, join } from "node:path";

const ASSETS = ["tokenizer.json", "config.json"];
const BATCH = 8;
const MAX_LEN = 512;

function listOnnx(dir) {
	try {
		return readdirSync(dir).some((f) => f.endsWith(".onnx"));
	} catch {
		return false;
	}
}

/** 目录里有没有可用的 reranker 权重（onnx 目录下任意 .onnx 即可）。 */
export function checkRerankAssets(modelDir) {
	if (!modelDir) return { ok: false, missing: ["(modelDir 未配置)"], modelDir };
	const missing = ASSETS.filter((f) => !existsSync(join(modelDir, f)));
	const hasOnnx = ["onnx", ""].some((d) => listOnnx(join(modelDir, d)));
	if (!hasOnnx) missing.push("onnx/*.onnx");
	return { ok: missing.length === 0, missing, modelDir };
}

/**
 * @param {{modelDir?:string, logger?:any, topN?:number, dtype?:string}} opts
 *   `dtype` 决定 transformers.js 去读哪个权重文件：
 *   q8 → model_quantized.onnx / fp16 → model_fp16.onnx / fp32 → model.onnx / q4f16 → model_q4f16.onnx …
 *   （2026-11-03 姐姐要的"下拉框选量化版本"—— 选的版本和这里必须一致，否则下完了加载不起来）
 */
export function createReranker({ modelDir, logger, topN = 10, dtype = "q8" } = {}) {
	let tokenizer = null;
	let model = null;
	let failedAt = 0;
	let failure = null;
	let pending = null;
	let logged = false;

	const log = (level, msg) => {
		try {
			logger?.[level]?.(msg);
		} catch {
			/* ignore */
		}
	};

	let lastCheck = { at: 0, res: checkRerankAssets(modelDir) };

	/** 每次真正要用时重新看盘：用户可能刚把模型拷进来。 */
	function assetsNow() {
		const now = Date.now();
		if (now - lastCheck.at < 5000) return lastCheck.res;
		lastCheck = { at: now, res: checkRerankAssets(modelDir) };
		return lastCheck.res;
	}

	async function ensure() {
		if (model && tokenizer) return { model, tokenizer };
		if (!modelDir) throw new Error("rerank modelDir is empty");
		const st = assetsNow();
		if (!st.ok) throw new Error(`rerank 模型缺文件：${st.missing.join("、")}`);
		if (failedAt && Date.now() - failedAt < 300_000) throw failure ?? new Error("reranker cooling down");
		if (pending) return pending;
		pending = (async () => {
			const mod = await import("@huggingface/transformers");
			mod.env.allowRemoteModels = false;
			mod.env.allowLocalModels = true;
			mod.env.localModelPath = dirname(modelDir);
			const name = basename(modelDir);
			const tok = await mod.AutoTokenizer.from_pretrained(name);
			const mdl = await mod.AutoModelForSequenceClassification.from_pretrained(name, { dtype });
			tokenizer = tok;
			model = mdl;
			failure = null;
			log("info", `[fast-ivf-rag] 精排模型已加载：${name}`);
			return { model: mdl, tokenizer: tok };
		})()
			.catch((error) => {
				failedAt = Date.now();
				failure = error;
				if (!logged) {
					logged = true;
					log("info", `[fast-ivf-rag] 精排模型没加载起来，跳过精排：${error?.message ?? error}`);
				}
				throw error;
			})
			.finally(() => {
				pending = null;
			});
		return pending;
	}

	/** logits（单标签）→ 0..1 */
	function sigmoid(x) {
		return 1 / (1 + Math.exp(-x));
	}

	const api = {
		kind: "cross-encoder",
		modelDir,
		topN,
		available() {
			return Boolean(model);
		},
		assetsOk() {
			return assetsNow().ok;
		},
		status() {
			return {
				enabled: true,
				kind: api.kind,
				modelDir,
				topN,
				assetsOk: assetsNow().ok,
				missing: assetsNow().missing,
				ready: Boolean(model),
				failure: failure?.message ?? null,
			};
		},
		/** 对 docs 打分并返回与 docs 等长的分数数组；不可用返回 null。 */
		async score(query, docs) {
			if (!docs?.length) return null;
			try {
				const { model: mdl, tokenizer: tok } = await ensure();
				const out = new Array(docs.length).fill(0);
				for (let i = 0; i < docs.length; i += BATCH) {
					const chunk = docs.slice(i, i + BATCH);
					let logits;
					try {
						const inputs = await tok(chunk.map(() => query), {
							text_pair: chunk,
							padding: true,
							truncation: true,
							max_length: MAX_LEN,
						});
						logits = (await mdl(inputs)).logits;
					} catch {
						// 有的分词器不吃数组入参，退回逐条
						for (let j = 0; j < chunk.length; j += 1) {
							const inputs = await tok(query, { text_pair: chunk[j], padding: true, truncation: true, max_length: MAX_LEN });
							const l = (await mdl(inputs)).logits;
							out[i + j] = now1(l.data?.[0] ?? 0);
						}
						continue;
					}
					const data = logits?.data ?? [];
					for (let j = 0; j < chunk.length; j += 1) out[i + j] = now1(data[j] ?? data[0] ?? 0);
				}
				return out;
			} catch {
				return null;
			}
		},
		async warmup() {
			const r = await api.score("warmup", ["warmup"]);
			return Boolean(r);
		},
	};

	return api;
}

function now1(raw) {
	const x = Number(raw);
	if (!Number.isFinite(x)) return 0;
	return Number((1 / (1 + Math.exp(-x))).toFixed(6));
}

