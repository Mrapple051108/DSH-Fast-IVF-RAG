/**
 * 一键拉取模型（2026-11-03 姐姐要的「点哪个下哪个」）。
 *
 * 记忆库要用两个**可选**的本地模型：
 *   · 嵌入 bge-small-zh-v1.5 —— 没它检索就退回字面/关键词，语义召回全废；
 *   · 精排 bge-reranker-base —— 没它也能跑（RRF 直接出结果），只是排序差一点。
 *
 * 以前只能自己去 HuggingFace 翻仓库、手动摆文件、还要摆对 `onnx/model_quantized.onnx`
 * 这个名字（transformers.js 的 `dtype:'q8'` 只认它）。这里把「仓库 + 该有哪几个文件 +
 * 放哪个目录」写死成一份清单，设置页点一下就行。
 *
 * 只做三件事：查状态、下载、报进度。不碰模型加载（那是 embed.js / rerank.js 的事）。
 */
import { existsSync, mkdirSync, renameSync, rmSync, statSync, createWriteStream } from "node:fs";
import { basename, dirname, join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { defaultModelDir, defaultRerankDir } from "./paths.js";

/** 国内默认走镜像；想用官方就改配置里的 base（或设 HF_ENDPOINT）。 */
export const DEFAULT_DOWNLOAD_BASE = "https://hf-mirror.com";

/**
 * 两个模型的清单。
 * `files` 是 transformers.js 本地加载**真正会读**的那几个 —— 少一个就报「缺文件」。
 *
 * 2026-11-03 23:1x：精排给了**量化版本可选**（姐姐要的下拉框）。
 * 权重文件名和 `dtype` 是一一对应的，transformers.js 按 dtype 去挑文件：
 *   q8 → model_quantized.onnx / fp16 → model_fp16.onnx / fp32 → model.onnx …
 * 所以选哪个版本，`rerankDtype` 就得跟着改，否则下完了加载不起来。
 * 大小来自 HF 仓库的 blob 元数据（`Xenova/bge-reranker-base`）。
 */
export const MODEL_SPECS = {
	embed: {
		key: "embed",
		label: "嵌入模型 bge-small-zh-v1.5",
		note: "q8 量化，约 24MB。没有它语义召回用不了，只能退回关键词。",
		repo: "Xenova/bge-small-zh-v1.5",
		dir: defaultModelDir,
		files: ["onnx/model_quantized.onnx", "tokenizer.json", "tokenizer_config.json", "config.json"],
	},
	rerank: {
		key: "rerank",
		label: "精排模型 bge-reranker-base",
		note: "可选件 —— 没有它也能检索，只是排序粗一点。版本越小越快，越大越准。",
		repo: "Xenova/bge-reranker-base",
		dir: defaultRerankDir,
		// 公共文件（哪个版本都要）
		files: ["tokenizer.json", "tokenizer_config.json", "config.json"],
		defaultVariant: "q8",
		variants: {
			q8: { file: "onnx/model_quantized.onnx", size: 279301077, label: "q8 · 266MB", note: "默认，最快，精度略降一点点" },
			fp16: { file: "onnx/model_fp16.onnx", size: 556462150, label: "fp16 · 531MB", note: "半精度，速度和精度折中" },
			fp32: { file: "onnx/model.onnx", size: 1112459588, label: "fp32 · 1.04GB", note: "全精度，最准，也最慢最占内存" },
			q4f16: { file: "onnx/model_q4f16.onnx", size: 434321936, label: "q4f16 · 414MB", note: "4bit 半精度混合，省内存" },
			q4: { file: "onnx/model_q4.onnx", size: 825849673, label: "q4 · 788MB", note: "4bit，比 q8 小不了多少，一般不选" },
		},
	},
};

/** 这个模型在当前配置下该有哪几个文件（精排要按版本算）。 */
function filesFor(spec, variant) {
	if (!spec.variants) return spec.files;
	const key = spec.variants[variant] ? variant : spec.defaultVariant;
	return [...spec.files, spec.variants[key].file];
}

function sizeOf(file) {
	try {
		const st = statSync(file);
		return st.isFile() ? st.size : 0;
	} catch {
		return 0;
	}
}

/**
 * 这个模型现在什么状态。
 * @param {'embed'|'rerank'} kind
 * @param {{dir?: string, variant?: string}} [opts] 目录覆盖 / 精排的量化版本
 */
export function modelStatus(kind, { dir, variant } = {}) {
	const spec = MODEL_SPECS[kind];
	if (!spec) return null;
	const root = String(dir ?? "").trim() || spec.dir();
	const want = spec.variants ? (spec.variants[variant] ? variant : spec.defaultVariant) : "";
	const need = filesFor(spec, want);
	const missing = [];
	let bytes = 0;
	for (const f of need) {
		const n = sizeOf(join(root, f));
		if (n > 0) bytes += n;
		else missing.push(f);
	}
	// 目录里已经装了哪些版本（前端用来标"已装"，也方便看出「这儿躺着的是别的版本」）
	let installed = [];
	if (spec.variants) {
		installed = Object.entries(spec.variants)
			.filter(([, v]) => sizeOf(join(root, v.file)) > 0)
			.map(([k]) => k);
	}
	return {
		kind,
		label: spec.label,
		note: spec.note,
		repo: spec.repo,
		dir: root,
		variant: want,
		variants: spec.variants
			? Object.entries(spec.variants).map(([k, v]) => ({
					key: k,
					label: v.label,
					note: v.note,
					file: v.file,
					size: v.size,
					installed: installed.includes(k),
				}))
			: undefined,
		installed,
		ready: missing.length === 0,
		missing,
		bytes,
	};
}

export function allModelStatus(opts = {}) {
	return {
		embed: modelStatus("embed", { dir: opts.embedDir }),
		rerank: modelStatus("rerank", { dir: opts.rerankDir, variant: opts.rerankVariant }),
	};
}

/** 正在跑/刚跑完的下载任务：kind -> job。 */
const jobs = new Map();

export function modelJob(kind) {
	return jobs.get(kind) ?? null;
}

export function allModelJobs() {
	return [...jobs.values()];
}

/**
 * 开始下载（后台跑，立刻返回 job，UI 轮询看进度）。
 *
 * @param {'embed'|'rerank'} kind
 * @param {{base?:string, dir?:string, force?:boolean}} [opts]
 * @returns {object} job
 */
export function startModelDownload(kind, opts = {}) {
	const spec = MODEL_SPECS[kind];
	if (!spec) throw new Error(`未知模型：${kind}`);
	const running = jobs.get(kind);
	if (running?.state === "running") return running;

	const base = String(opts.base ?? "").trim() || DEFAULT_DOWNLOAD_BASE;
	const root = String(opts.dir ?? "").trim() || spec.dir();
	const force = Boolean(opts.force);
	const variant = spec.variants ? (spec.variants[opts.variant] ? opts.variant : spec.defaultVariant) : "";
	const need = filesFor(spec, variant);

	// 两道保护，都是 2026-11-03 想明白的：
	//   ① 已经就绪还点下载 → 让它明确点「重新下载」（force），别一不小心把能用的模型冲掉；
	//   ② 精排目录里可能躺着别的模型（比如用户自己放的 bge-reranker-v2-m3，561MB），
	//      往那个目录写 bge-reranker-base 的权重会**直接毁掉**它 —— 名字里不带 base 就拒绝覆盖。
	const now = modelStatus(kind, { dir: root, variant });
	if (now.ready && !force) {
		throw new Error(`${root} 里已经有可用的模型了。真要覆盖请点「重新下载」`);
	}
	if (force && kind === "rerank" && !/base/i.test(basename(root)) && now.ready) {
		throw new Error(
			`${basename(root)} 里已经有一套精排模型，不覆盖它 —— 要下 bge-reranker-base，` +
				`请先在「插件设置 → 精排模型目录」里换一个路径（比如 …/models/bge-reranker-base）`,
		);
	}
	const job = {
		kind,
		label: spec.label,
		state: "running",
		variant,
		dir: root,
		total: need.length,
		done: 0,
		bytes: 0,
		current: "",
		error: "",
		startedAt: Date.now(),
		finishedAt: 0,
	};

	jobs.set(kind, job);

	job.promise = (async () => {
		for (const f of need) {
			job.current = f;
			const target = join(root, f);
			if (!force && sizeOf(target) > 0) {
				job.bytes += sizeOf(target);
				job.done += 1;
				continue;
			}
			mkdirSync(dirname(target), { recursive: true });
			const url = `${base.replace(/\/+$/, "")}/${spec.repo}/resolve/main/${f}`;
			const res = await fetch(url, { redirect: "follow" });
			if (!res.ok || !res.body) throw new Error(`下载 ${f} 失败：HTTP ${res.status}`);
			const tmp = `${target}.part`;
			try {
				await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp));
				renameSync(tmp, target);
			} catch (err) {
				try {
					rmSync(tmp, { force: true });
				} catch {
					/* 清不掉就算了 */
				}
				throw err;
			}
			job.bytes += sizeOf(target);
			job.done += 1;
		}
		job.state = "done";
		job.current = "";
		job.finishedAt = Date.now();
	})().catch((err) => {
		job.state = "error";
		job.error = String(err?.message ?? err);
		job.finishedAt = Date.now();
	});

	return job;
}

/** 给 UI 看的精简版（别把 promise 往 JSON 里塞）。 */
export function serializeJob(job) {
	if (!job) return null;
	return {
		kind: job.kind,
		label: job.label,
		variant: job.variant ?? "",
		state: job.state,
		dir: job.dir,
		total: job.total,
		done: job.done,
		bytes: job.bytes,
		current: job.current,
		error: job.error,
		startedAt: job.startedAt,
		finishedAt: job.finishedAt,
	};
}
