import z from "@deepseek-ai/schemastery";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { RagManager } from "./manager.js";
import { getSharedWorker } from "./recall-pool.js";
import { defaultDataDir, defaultModelDir, defaultRerankDir } from "./paths.js";
import {
	MODEL_SPECS,
	DEFAULT_DOWNLOAD_BASE,
	allModelStatus,
	allModelJobs,
	startModelDownload,
	serializeJob,
} from "./model-fetch.js";
import { DEFAULT_SUMMARY_PROMPT, buildSummaryPrompt, parseSummary } from "./summarize.js";
import { textHash } from "./library.js";
import { dot } from "./kmeans.js";
import { QUERY_PREFIX } from "./embed.js";

const name = "fast-ivf-rag";
const inject = ["tools", "systemPrompt", "llm", "webServer"];

const UI_FILE = join(dirname(fileURLToPath(import.meta.url)), "..", "ui", "index.html");

/** 启动自检：写一行日志，用来判断模块导入 / apply 是否真的跑到了。 */
function mark(msg) {
	try {
		mkdirSync(defaultDataDir(), { recursive: true });
		appendFileSync(join(defaultDataDir(), "boot.log"), `${new Date().toISOString()} ${msg}\n`);
	} catch {
		/* ignore */
	}
}

mark(`module imported (pid ${process.pid})`);

const Config = z.object({
	enabled: z.boolean().default(true).description("总开关").volatile(),
	dataDir: z.string().default("").description("数据目录，留空 = {DSH_HOME}/fast-ivf-rag").volatile(),
	embedModelDir: z.string().default("").description("本地嵌入模型目录，留空 = {DSH_HOME}/models/bge-small-zh").volatile(),
	modelDownloadBase: z
		.string()
		.default("")
		.description("一键拉模型的下载源，留空 = https://hf-mirror.com（国内镜像）；想用官方就填 https://huggingface.co")
		.volatile(),
	libraryMode: z
		.union([z.const("new"), z.const("reuse")])
		.default("reuse")
		.description("新会话的向量库策略：new=每个新会话建新库，reuse=沿用最近一个库")
		.volatile(),
	autoMount: z
		.boolean()
		.default(true)
		.description("新会话自动挂载到记忆库；关掉后没绑过的会话就是「没记忆」（已绑定的会话不受影响，面板手动绑定照常有效）")
		.volatile(),
	autoRecall: z.boolean().default(true).description("自动召回：每轮把相关记忆注入系统提示").volatile(),
	recallTopK: z.number().min(1).max(20).default(5).description("自动召回条数").volatile(),
	recallMaxChars: z.number().min(100).max(4000).default(900).description("自动召回注入的最大字符数").volatile(),
	workerRecall: z.boolean().default(true).description("召回跑在工作线程（关掉则回主线程，会卡事件循环）").volatile(),
	autoSummarize: z.boolean().default(true).description("自动总结：把离开活跃区的对话压成记忆").volatile(),
	activeWindow: z.number().min(2).max(400).default(16).description("活跃区条数：最近 N 条原文不参与总结").volatile(),
	summarizeIdleMinutes: z.number().min(0).max(1440).default(0).description("已弃用（现在缓冲够数就压）").volatile(),
	summarizeMinMessages: z.number().min(1).max(200).default(5).description("缓冲积压多少条就激活成节点").volatile(),
	summarizeMaxTake: z.number().min(1).max(200).default(24).description("一次激活最多处理多少条消息").volatile(),
	rawCapture: z
		.boolean()
		.default(false)
		.description("原文也入库（会进检索池）。默认关：原文只当压缩输入，召回靠上下文里本就完整的最近对话")
		.volatile(),
	rawPackSize: z.number().min(1).max(10).default(3).description("原文通道开着时，同一说话人连续几条合成一个节点").volatile(),
	summarizePrompt: z
		.string()
		.default("")
		.description("总结 prompt，留空 = 用代码内置的（推荐）。填了会把这份文本写进配置，代码里再改 prompt 就盖不过它")
		.volatile(),
	summarizeModel: z.string().default("").description("总结用模型，格式 provider/model，留空则跟随当前会话").volatile(),
	useAssociation: z.boolean().default(false).description("检索时附带一跳联想（同簇近邻）").volatile(),
	recallAssocCount: z
		.number()
		.min(0)
		.max(20)
		.default(3)
		.description("联想池总容量：最多带几条联想（0=完全不带）")
		.volatile(),
	recallAssocPerHit: z
		.number()
		.min(1)
		.max(10)
		.default(1)
		.description("每条直接命中最多贡献几条联想（even 模式用；weighted 模式按分数分配）")
		.volatile(),
	assocMode: z
		.union([z.const("weighted"), z.const("even")])
		.default("weighted")
		.description("联想名额分配：weighted=按直命中分数分配，even=每条均分")
		.volatile(),
	recallMinScore: z
		.number()
		.min(0)
		.max(1)
		.step(0.05)
		.default(0.35)
		.description("固定阈值（minScoreMode=fixed 时用；0=不过滤）")
		.volatile(),
	minScoreMode: z
		.union([z.const("dynamic"), z.const("fixed")])
		.default("dynamic")
		.description("阈值模式：dynamic=max(下限, top1×比例)，fixed=用固定值")
		.volatile(),
	minScoreFloor: z.number().min(0).max(1).step(0.05).default(0.35).description("动态阈值的下限").volatile(),
	minScoreRatio: z.number().min(0).max(1).step(0.05).default(0.55).description("动态阈值占比：top1 × 这个比例").volatile(),
	mmrLambda: z
		.number()
		.min(0)
		.max(1)
		.step(0.05)
		.default(0.7)
		.description("MMR 多样性重排的 λ（0=关闭）：score = λ·相关度 − (1−λ)·和已选的最大相似度")
		.volatile(),
	rerankEnabled: z.boolean().default(true).description("精排开关：有本地 cross-encoder 模型时用它重排检索结果").volatile(),
	rerankModelDir: z.string().default("").description("精排模型目录，留空 = {DSH_HOME}/models/bge-reranker-base").volatile(),
	rerankTopN: z.number().min(2).max(100).default(10).description("送去精排的候选条数").volatile(),
	rerankDtype: z
		.union([z.const("q8"), z.const("fp16"), z.const("fp32"), z.const("q4f16"), z.const("q4")])
		.default("q8")
		.description("精排量化版本：既决定下载哪个权重文件，也决定加载时的 dtype（改了要重启）")
		.volatile(),
	condenseMaxLen: z.number().min(40).max(800).default(160).description("精排前把记忆压到多少字（0 效果=不压）").volatile(),
	dedupeThreshold: z
		.number()
		.min(0)
		.max(1)
		.step(0.01)
		.default(0.92)
		.description("语义去重阈值：新节点跟库里已有节点的余弦 ≥ 这个值就丢弃（0=关）。换个措辞说的同一件事不该再存一份")
		.volatile(),
	hydrateLookbackMs: z
		.number()
		.min(0)
		.max(86_400_000)
		.default(7_200_000)
		.description("会话第一次被插件看到时，只回读最近这么久的对话补进缓冲（默认 2 小时，0=不回读）。防止重启后把昨天的旧对话重新压一遍")
		.volatile(),
	rebuildEvery: z.number().min(1).max(5000).default(50).description("每新增 N 条重聚类一次").volatile(),
	probe: z.number().min(1).max(32).default(3).description("IVF 探测簇数").volatile(),
	maxK: z.number().min(2).max(4096).default(512).description("聚类上限").volatile(),
});

/** 面板「设置」页用的字段说明（与上面的 Config 一一对应，改 Config 记得同步这儿）。 */
const FIELDS = [
	{ key: "enabled", label: "总开关", type: "bool", hint: "关掉后不召回、不总结、不响应消息事件" },
	{ key: "libraryMode", label: "新会话建库策略", type: "enum", options: ["reuse", "new"], hint: "reuse=沿用最近一个库，new=每个新会话开新库" },
	{
		key: "autoMount",
		label: "新会话自动挂载记忆",
		type: "bool",
		hint: "关掉后新会话不连记忆库（不召回、不总结、不写入）；要在「会话绑定」页手动绑才有记忆。已绑定的会话不受影响",
	},
	{ key: "autoRecall", label: "自动召回注入", type: "bool", hint: "每轮把相关记忆注入系统提示（不占历史）" },
	{ key: "recallTopK", label: "召回条数", type: "num", min: 1, max: 20 },
	{ key: "recallMaxChars", label: "召回转义上限（字符）", type: "num", min: 100, max: 4000 },
	{ key: "workerRecall", label: "召回走工作线程", type: "bool", hint: "关掉会在主线程算，事件循环被占住时 SSE 流可能断" },
	{ key: "autoSummarize", label: "自动总结入库", type: "bool", hint: "把离开活跃区的对话压成记忆" },
	{ key: "activeWindow", label: "活跃区条数", type: "num", min: 2, max: 400, hint: "最近 N 条原文永不参与总结；超出的进未激活缓冲" },
	{ key: "summarizeIdleMinutes", label: "空闲多少分钟后总结", type: "num", min: 0, max: 1440, hint: "已弃用：现在缓冲够数就立刻激活，不再等空闲" },
	{ key: "summarizeMinMessages", label: "缓冲多少条就激活", type: "num", min: 1, max: 200, hint: "缓冲区积压到这个数就成节点" },
	{ key: "summarizeMaxTake", label: "一次激活多少条", type: "num", min: 1, max: 200, hint: "单批上限，越小节点产出越连续" },
	{
		key: "rawCapture",
		label: "原文也入库（会进检索池）",
		type: "bool",
		hint: "默认关。关掉时原文只当压缩输入，激活后只有摘要成为节点；最近对话本来就在上下文里，不用召回原文",
	},
	{ key: "rawPackSize", label: "同人几条合成一个节点", type: "num", min: 1, max: 10, hint: "仅原文通道打开时生效；不同说话人永远拆成不同节点" },
	{ key: "summarizePrompt", label: "总结 prompt", type: "text", hint: "留空 = 用代码内置的（推荐）。填了会被面板保存写死，代码里改 prompt 就失效" },
	{ key: "summarizeModel", label: "总结用模型", type: "text", hint: "provider/model，留空跟随当前会话" },
	{ key: "useAssociation", label: "检索带一跳联想", type: "bool", hint: "联想单独计数，不占直接召回的名额" },
	{
		key: "recallAssocCount",
		label: "联想池总容量",
		type: "num",
		min: 0,
		max: 20,
		hint: "最多带几条联想；0 = 完全不带。直接召回数看上面的「召回条数」，两者互不挤占",
	},
	{ key: "recallAssocPerHit", label: "每条命中贡献几条联想", type: "num", min: 1, max: 10, hint: "只在 even 模式下生效" },
	{
		key: "assocMode",
		label: "联想名额分配",
		type: "enum",
		options: ["weighted", "even"],
		hint: "weighted=按直命中分数分（分数高的多联想、低的不联想）；even=每条均分再补位",
	},
	{
		key: "minScoreMode",
		label: "阈值模式",
		type: "enum",
		options: ["dynamic", "fixed"],
		hint: "dynamic = max(下限, top1 × 比例)，跟着本次查询最高分走；fixed = 用下面那个固定值",
	},
	{
		key: "recallMinScore",
		label: "固定阈值",
		type: "num",
		min: 0,
		max: 1,
		hint: "只在 fixed 模式生效，0=不过滤。低于阈值的命中直接丢：不占名额、不给联想、不进注入",
	},
	{ key: "minScoreFloor", label: "动态阈值下限", type: "num", min: 0, max: 1, hint: "dynamic 模式下阈值不会低于它（默认 0.35）" },
	{ key: "minScoreRatio", label: "动态阈值占比", type: "num", min: 0, max: 1, hint: "阈值 = top1 × 这个比例（默认 0.55）" },
	{
		key: "mmrLambda",
		label: "MMR 多样性 λ",
		type: "num",
		min: 0,
		max: 1,
		hint: "0=关闭。score = λ·相关度 − (1−λ)·与已选的最大相似度；0.7 是去重的甜点，低于 0.5 会开始捞无关条目",
	},
	{ key: "rerankEnabled", label: "精排开关", type: "bool", hint: "有本地 cross-encoder 模型才生效，没有就自动跳过" },
	{ key: "rerankModelDir", label: "精排模型目录", type: "text", hint: "留空 = {DSH_HOME}/models/bge-reranker-base（可选件，没有也能用）" },
	{ key: "rerankTopN", label: "精排候选条数", type: "num", min: 2, max: 100, hint: "每多 10 条约 +190ms（bge-reranker-v2-m3 int8 实测）" },
	{ key: "condenseMaxLen", label: "精排前压缩字数", type: "num", min: 40, max: 800, hint: "长记忆先按查询压成短句再精排，省算力也更准" },
	{
		key: "dedupeThreshold",
		label: "语义去重阈值",
		type: "num",
		min: 0,
		max: 1,
		hint: "新节点跟已有节点的余弦 ≥ 这个值就丢弃（0=关）。0.92 只挡「换个说法的同一件事」，调到 0.85 更狠、0.97 几乎只挡复读",
	},
	{
		key: "hydrateLookbackMs",
		label: "回读历史的时间窗（毫秒）",
		type: "num",
		min: 0,
		max: 86400000,
		hint: "会话第一次被插件看到时只补最近这么久（默认 2 小时）。调大=重启后补更多，但会把压过的旧对话再压一遍",
	},
	{ key: "rebuildEvery", label: "每 N 条重聚类", type: "num", min: 1, max: 5000 },
	{ key: "probe", label: "IVF 探测簇数", type: "num", min: 1, max: 32 },
	{ key: "maxK", label: "聚类上限", type: "num", min: 2, max: 4096 },
	{ key: "dataDir", label: "数据目录", type: "text", hint: "留空 = {DSH_HOME}/fast-ivf-rag（改这里要重启）" },
	{ key: "embedModelDir", label: "嵌入模型目录", type: "text", hint: "留空 = {DSH_HOME}/models/bge-small-zh（改这里要重启）" },
];

const val = (v, d) => {
	if (v === undefined || v === null) return d;
	if (typeof v?.get === "function") {
		const x = v.get();
		return x === undefined || x === null ? d : x;
	}
	return v;
};

const strOf = (v, d = "") => String(val(v, d) ?? d);

/** 系统注入文本的行首特征：runtime context、采样时间、文件策略……这些不是人话，
 *  拿去检索是纯噪音（还会把无意义查询塞进召回台账），进总结缓冲更是污染 prompt。 */
const NOISE_LINE = [
	/^current runtime context/i,
	/^time sampled while/i,
	/^elapsed since the preceding/i,
	/^current dsh file policy/i,
	/^approval requests are auto-granted/i,
	/^current working directory/i,
	/^current deepseek harness/i,
	/^browser time zone/i,
	/^（更早的对话已按/,
	/^<system-reminder>$/i,
	/^<\/system-reminder>$/i,
];

function stripSystemNoise(text) {
	let s = String(text ?? "");
	if (!s) return "";
	// 整块删 <system-reminder>…</system-reminder>；没有闭合就砍到结尾
	s = s.replace(/<system-reminder>[\s\S]*?(<\/system-reminder>|$)/gi, " ");
	return s
		.split(/\r?\n/)
		.filter((line) => {
			const t = line.trim();
			if (!t) return true;
			return !NOISE_LINE.some((re) => re.test(t));
		})
		.join("\n")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

/**
 * 事件 data → message 本体。
 *
 * ⚠️ 2026-11-02 修：宿主给两种事件的形状**不一样**（dsh-session/lib/index.js:221-228）：
 *   case "user/message":      return event.data;            ← data 本身就是 message
 *   case "assistant/message": return event.data.message;    ← 包了一层 { turn, step, message, usage, stream }
 * 以前 textOfMessage 只读 data.content，于是 assistant 每一条都取到 undefined → 返回 ""，
 * 再被 pushMessage 的 `if (!t) return` 当空消息丢掉。
 * 现象就是"别人的话全记着，我自己的回应一条没有"，下次再问同一件事，态度跟上次对不上。
 * loadHistory() 回读历史走的是同一个函数，所以重启也补不回来 —— 两头一起丢。
 */
function unwrapMessage(data) {
	if (!data || typeof data !== "object") return data;
	// 已经有 content 的就是 message 本体（user/message 走这条），绝不往下挖：
	// 万一某个 user message 自带 message 字段，挖了就把正文挖没了。
	if (data.content !== undefined) return data;
	const inner = data.message;
	if (typeof inner === "string") return { content: inner };
	if (inner && typeof inner === "object") return inner;
	return data;
}

function textOfMessage(data) {
	const msg = unwrapMessage(data);
	const content = msg?.content;
	if (typeof content === "string") return stripSystemNoise(content);
	if (!Array.isArray(content)) return "";
	// 带 tool-call 的 step 是过程步（"我查一下" + 工具），不是最终回复。
	// 一句行动预告记进记忆没有信息量，还会挤掉真正的回答 —— 只留纯文本 step。
	// 宿主 agent-loop:1180-1183 里 toolCalls.length===0 才算 completed，
	// 所以带 tool-call 的必然不是最后一步，丢掉是安全的。
	if (content.some((p) => p?.type === "tool-call")) return "";
	const raw = content
		// 只认 type==="text" 的块：思考(reasoning/thinking)、工具调用(tool-call)、
		// 工具结果(tool-result)、图片/文件一律不进总结。
		// 以前这里还带 `|| typeof p.text === "string"`，会把带 text 字段的
		// 思考块一起吞进来，等于把推理过程也写进记忆 —— 已经收紧。
		.filter((p) => p && p.type === "text" && typeof p.text === "string")
		.map((p) => String(p.text))
		.join("\n");
	return stripSystemNoise(raw);
}

/**
 * 环境快照 / 压缩替身 / 插件注入：**都不是人说的话**，一律不进记忆。
 *
 * 2026-11-03 12:2x 姐姐在面板上又看到旧事被翻出来（`#281 … 2026-11-02 老大说 …`），
 * 追下去两条路一起漏：
 *   ① `loadHistory` 读的是 `ev.at`，可宿主的事件信封只有 `type/seq/time/data/…`
 *      （dsh-session/lib/index.js assertSessionEventEnvelope）—— `at` 恒 undefined，
 *      于是 `newest` 恒 0（水位永远存不进去）、`at > 0` 恒 false（时间闸永远不过滤），
 *      每次 hydrate 都把尾部 600 条事件整段重灌一遍缓冲区；
 *   ② runtime-context 也是 `user/message`，内容却是工作目录 / 沙箱策略 / 记忆召回 / dsh-qq 动态段。
 *      它被当成"用户说的话"推进缓冲区 → 总结成节点 → 下次又被召回 → 再被总结。
 *      召回文本自我繁殖，这才是"过往记录被捞上来"的主因。
 */
export const SRC_KIND_SKIP = new Set([
	"compact-checkpoint", // dsh 官方压缩 / dsh-context-ctl 裁剪的替身
	"plugin", // 插件注入的规矩、notice
	"runtime-context", // 宿主每轮的环境快照（cwd / 策略 / 召回 / 动态段）
]);

function sourceKindOf(data) {
	return String(data?.source?.kind ?? "");
}

/** 这条 user/message 值不值得记忆：压缩替身 / 插件注入 / 环境快照一律不要。 */
export function isMemoryWorthyUser(data) {
	return !SRC_KIND_SKIP.has(sourceKindOf(data));
}

/**
 * 事件时间戳。
 * ⚠️ 宿主用 `time`（不是 `at`）。`at` 留着只为兼容早期自造对象。
 */
export function timeOfEvent(ev) {
	const t = Number(ev?.time ?? ev?.at ?? ev?.data?.time ?? 0);
	return Number.isFinite(t) && t > 0 ? t : 0;
}

/**
 * 真正**发出去**的话：`qq_send` / `qq_reply` 的参数就是发到 QQ 上的正文。
 *
 * 2026-11-03 姐姐要求：「只总结一轮对话里实际发出去的（工具调用 send 出去的）文本」。
 * 这一条很关键 —— dsh-qq 的 assistant 正文只要这一轮调过 qq_reply，就会被 outbound 挡掉
 * 不再重复转发；此时会话里的 assistant/message 虽然有 text，屏幕上却没有它。
 * 反过来，带 tool-call 的那个 step 在 textOfMessage 里是**整步丢弃**的，
 * 于是"我"真正说出口的那句话两头都不进记忆 —— 记忆里只剩对方在说话。
 * 这里从 tool/call 的参数里把它捞回来，只认发消息的两个工具。
 */
export const SELF_SEND_TOOLS = new Set(["qq_send", "qq_reply"]);

export function sentTextOfToolCall(ev) {
	const data = ev?.data ?? ev?.payload ?? ev?.message ?? {};
	const name = String(data?.name ?? data?.toolName ?? "").trim();
	if (!SELF_SEND_TOOLS.has(name)) return "";
	let args = data?.arguments ?? data?.args ?? data?.input;
	if (typeof args === "string") {
		try {
			args = JSON.parse(args);
		} catch {
			return "";
		}
	}
	if (!args || typeof args !== "object") return "";
	const text = args.text ?? args.message ?? args.content;
	return typeof text === "string" ? text.trim().slice(0, 1200) : "";
}

/** 取一个可能没在 inject 里声明的服务：cordis 对未 inject 的服务直接抛错，这里吞掉当"不可用"。 */
function softService(ctx, id) {
	try {
		return ctx.get(id) ?? null;
	} catch {
		return null;
	}
}

function sessionIdOf(...candidates) {
	for (const c of candidates) {
		if (!c) continue;
		const id = c.session?.id ?? c.sessionId ?? c.agent?.session?.id ?? c.scope?.session?.id;
		if (id) return String(id);
		const direct = c.id;
		if (typeof direct === "string" && direct && c.session === undefined && c.content === undefined) {
			// 落在 Agent/Session 对象上：Agent.id === session.id
			if (c.constructor?.name === "Session" || c.append !== undefined) return direct;
			return direct;
		}
	}
	return null;
}

function json(res, code, body) {
	const data = typeof body === "string" ? body : JSON.stringify(body);
	res.writeHead(code, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
	res.end(data);
}

async function readBody(req) {
	const chunks = [];
	for await (const c of req) chunks.push(c);
	const raw = Buffer.concat(chunks).toString("utf8");
	if (!raw) return {};
	try {
		return JSON.parse(raw);
	} catch {
		return {};
	}
}

function apply(ctx, config) {
	mark("apply() entered");
	try {
		return body(ctx, config);
	} catch (error) {
		mark(`apply() THREW ${error?.stack ?? error}`);
		throw error;
	}
}

function body(ctx, config) {
	const cfg = config ?? {};
	const dataDir = strOf(cfg.dataDir) || defaultDataDir();
	const modelDir = strOf(cfg.embedModelDir) || defaultModelDir();

	const recallWorker = getSharedWorker({ logger: ctx.logger ?? null, timeoutMs: 20_000 });
	const useWorker = Boolean(val(cfg.workerRecall, true));

	const manager = new RagManager({
		dataDir,
		modelDir,
		logger: ctx.logger ?? null,
		workerPool: useWorker ? recallWorker : null,
		maxK: Number(val(cfg.maxK, 512)),
		rebuildEvery: Number(val(cfg.rebuildEvery, 50)),
		probe: Number(val(cfg.probe, 3)),
		condenseMaxLen: Number(val(cfg.condenseMaxLen, 160)),
		dedupeThreshold: Number(val(cfg.dedupeThreshold, 0.92)),
		rerankEnabled: Boolean(val(cfg.rerankEnabled, true)),
		rerankModelDir: strOf(cfg.rerankModelDir) || defaultRerankDir(),
		rerankTopN: Number(val(cfg.rerankTopN, 10)),
		rerankDtype: strOf(cfg.rerankDtype) || "q8",
	});

	const recallCache = new Map();
	const trackers = new Map();
	let sectionBound = false;

	/* ───────── 把嵌入器暴露给别的插件（2026-11-02） ───────── */
	// dsh-qq-onebot 的表情包语义检索要用同一份 bge-small-zh。
	// **绝不能让那边自己再加载一份 ONNX Runtime** —— 同进程两份会话会撞 V8 HandleScope
	// 锁，直接 FATAL 把整个 DSH 带走（rag 自己就是靠 worker 躲开这个的）。
	// 所以这里把现有实例（可能是 worker 代理）架成服务，谁要用谁 ctx.get('fastIvfEmbed')。
	try {
		const embedService = {
			/** 现在能不能用；不能用就把原因一起给出来，调用方好显示。 */
			async status() {
				try {
					const e = await manager.embedder();
					const st = typeof e?.status === "function" ? e.status() : null;
					const dims = Number(e?.dims) || Number(st?.dims) || 0;
					const available = Boolean(e) && (typeof e.available !== "function" || e.available() !== false);
					return {
						available,
						dims,
						kind: String(st?.kind ?? e?.kind ?? "unknown"),
						modelDir: String(st?.modelDir ?? e?.modelDir ?? modelDir ?? ""),
						reason: available ? "" : "嵌入器还没就绪",
					};
				} catch (err) {
					return {
						available: false,
						dims: 0,
						kind: "error",
						modelDir: String(modelDir ?? ""),
						reason: String(err?.message ?? err),
					};
				}
			},
			/** 查询侧：**要加** bge 的前缀（文档侧不加，这个不对称必须照做）。 */
			async embedQuery(text) {
				const e = await manager.embedder();
				return e.embedQuery(String(text ?? ""));
			},
			/** 文档侧：不加前缀。 */
			async embedPassages(texts) {
				const e = await manager.embedder();
				const list = Array.isArray(texts) ? texts.map((t) => String(t ?? "")) : [];
				return e.embedPassages(list);
			},
			/** 这个模型要求的查询前缀，暴露出去免得调用方自己猜。 */
			queryPrefix: QUERY_PREFIX,
		};
		ctx.provide("fastIvfEmbed", embedService);
		mark("已暴露嵌入服务 fastIvfEmbed（供 dsh-qq-onebot 的语义匹配复用）");
	} catch (error) {
		// 已经有人占了这个名字 / 宿主不支持 provide —— 都不影响 rag 自己工作
		mark(`暴露嵌入服务失败（不影响本插件）: ${error?.message ?? error}`);
	}
	let timer = null;
	const disposers = [];

	const on = Boolean(val(cfg.enabled, true));

	// 总开关关掉 = 真的什么都不做：不注册工具、不挂提示段、不开定时器、不订阅会话事件。
	// （以前这里只是"各功能自己检查一遍"，插件照样挂着一堆东西，没法拿它做对照实验。）
	if (!on) {
		mark("apply(): enabled=false，插件整体停用，只保留一个「重新启用」的极简页");
		try {
			const ws = softService(ctx, "webServer");
			ws?.register({
				kind: "prefix",
				path: "/fast-ivf-rag",
				handler: (req, res) => {
					const send = (code, body, type = "application/json") => {
						res.writeHead(code, { "content-type": type });
						res.end(typeof body === "string" ? body : JSON.stringify(body));
					};
					const url = String(req.url ?? "");
					if (req.method === "POST" && url.includes("/api/config")) {
						let raw = "";
						req.on("data", (c) => {
							raw += c;
						});
						req.on("end", async () => {
							try {
								const patch = JSON.parse(raw || "{}");
								await softService(ctx, "settings").update("fast-ivf-rag", { ...patch, enabled: true });
								send(200, { ok: true });
							} catch (error) {
								send(500, { error: String(error?.message ?? error) });
							}
						});
						return;
					}
					send(
						200,
						`<!doctype html><meta charset="utf-8"><body style="background:#111;color:#ddd;font:14px/1.7 system-ui;padding:28px">
<h2 style="margin:0 0 8px">Fast IVF RAG 已停用</h2>
<p style="color:#888">设置里的 <code>enabled</code> 现在是 false，插件没有注册任何工具、提示段、定时器或事件订阅。</p>
<button id="on" style="margin-top:14px;padding:8px 16px;border-radius:8px;border:1px solid #444;background:#1b1b1b;color:#eee;cursor:pointer">重新启用</button>
<p id="msg" style="color:#6c6"></p>
<script>document.getElementById('on').onclick=async()=>{const r=await fetch('/fast-ivf-rag/api/config',{method:'POST',headers:{'content-type':'application/json'},body:'{}'});document.getElementById('msg').textContent=r.ok?'已启用，重启 DSH 后生效':'启用失败'}</script>`,
						"text/html",
					);
				},
			});
		} catch (error) {
			mark(`disabled-mode panel FAILED: ${error?.message ?? error}`);
		}
		return;
	}

	/* ───────── 会话 → 库 ───────── */

	async function libFor(sessionId, label) {
		if (!sessionId) return null;
		const mode = strOf(cfg.libraryMode, "reuse");
		return manager.resolveSession(sessionId, mode, label ?? "", Boolean(val(cfg.autoMount, true)));
	}

	/* ───────── 自动召回 ───────── */

	/**
	 * 所有检索都走这里：能上线程就上线程。
	 * 之所以连精排也塞进 worker：同一个进程里同时存在两份 onnxruntime 会话会撞
	 * V8 HandleScope 锁，直接 FATAL —— 所以 ONNX 只允许活在 worker 里。
	 */
	async function searchLibrary(lib, query, options = {}) {
		const wantRerank = options.rerank !== false && Boolean(val(cfg.rerankEnabled, true));
		if (!useWorker) return lib.search(query, { ...options, rerank: false });
		try {
			return await recallWorker.search({
				dataDir: manager.dataDir,
				modelDir: manager.modelDir,
				libId: lib.id,
				query,
				topK: options.topK,
				probe: options.probe ?? Number(val(cfg.probe, 3)),
				assoc: options.assoc,
				assocCount: options.assocCount,
				assocPerHit: options.assocPerHit,
				assocMode: options.assocMode,
				minScore: options.minScore,
				minScoreMode: options.minScoreMode,
				minScoreFloor: options.minScoreFloor,
				minScoreRatio: options.minScoreRatio,
				mmrLambda: options.mmrLambda,
				rerank: wantRerank,
				rerankModelDir: wantRerank ? strOf(cfg.rerankModelDir) || defaultRerankDir() : null,
				rerankTopN: Number(val(cfg.rerankTopN, 10)),
				rerankDtype: strOf(cfg.rerankDtype) || "q8",
				condenseMaxLen: Number(val(cfg.condenseMaxLen, 160)),
			});
		} catch (error) {
			mark(`worker search FAILED, fallback inline: ${error?.message ?? error}`);
			return lib.search(query, { ...options, rerank: false });
		}
	}

	/**
	 * 找出语义上几乎重合的记忆分组（一键去冗余的后端）。
	 *
	 * 做什么：同一簇里按「代表 + 跟代表余弦 ≥ 阈值」贪心分组，每组留一条、其余标记待删。
	 * 为什么按簇分组而不是全库两两比：3000 条两两比是 450 万次点积，主线程会被按住几秒；
	 *   同簇成员本来就语义相近，漏掉的跨簇重复只是"少删几条"，用户把阈值调低再跑一次就行。
	 * 留哪条：
	 *   newest（默认）—— 时间最新的。姐姐的用法里"后来又记住一次"通常就是更完整的版本。
	 *   longest       —— 字数最多的。适合摘要被压得忽长忽短的情况。
	 *
	 * 分批（2026-11-01 加）：`clusterBatch > 0` 时只扫 `fromCluster` 起的这么多个簇，
	 * 返回 `nextCluster` 让前端接着要下一批 —— 面板右侧那根进度条就是靠这个动的。
	 * 一次性扫完（`clusterBatch = 0`）留给真删那一次：那时候要的是一份完整、确定的名单。
	 */
	function findDuplicateGroups(lib, { threshold = 0.92, keep = "newest", fromCluster = 0, clusterBatch = 0 } = {}) {
		const rows = lib.db.allVectors();
		const clusters = lib.db.clusters();
		const totalClusters = clusters.length;
		if (rows.length < 2) {
			return { groups: [], scanned: rows.length, nextCluster: totalClusters, totalClusters, done: true, total: rows.length };
		}
		const byCluster = new Map();
		const lonely = [];
		for (const r of rows) {
			if (!r.vec) continue;
			if (r.cluster === null || r.cluster === undefined) {
				lonely.push(r);
				continue;
			}
			const arr = byCluster.get(r.cluster) ?? [];
			arr.push(r);
			byCluster.set(r.cluster, arr);
		}
		const groups = [];
		const scan = (list) => {
			if (list.length < 2) return;
			const items = [];
			for (const r of list) {
				const row = lib.db.get(r.id);
				if (!row) continue;
				items.push({ id: r.id, vec: r.vec, text: String(row.text ?? ""), ts: Number(row.ts) || 0, kind: row.kind });
			}
			if (items.length < 2) return;
			items.sort((a, b) =>
				keep === "longest" ? b.text.length - a.text.length || b.ts - a.ts : b.ts - a.ts || b.text.length - a.text.length,
			);
			const reps = [];
			for (const it of items) {
				let hit = null;
				for (const rep of reps) {
					if (rep.vec.length !== it.vec.length) continue;
					if (dot(it.vec, rep.vec) >= threshold) {
						hit = rep;
						break;
					}
				}
				if (hit) hit.members.push(it);
				else reps.push({ ...it, members: [] });
			}
			for (const rep of reps) {
				if (!rep.members.length) continue;
				groups.push({
					keep: { id: rep.id, text: rep.text, ts: rep.ts, kind: rep.kind },
					drop: rep.members.map((m) => ({
						id: m.id,
						text: m.text,
						ts: m.ts,
						kind: m.kind,
						sim: Math.round(dot(m.vec, rep.vec) * 1000) / 1000,
					})),
				});
			}
		};

		const from = Math.max(0, Math.min(totalClusters, Number(fromCluster) || 0));
		const to = clusterBatch > 0 ? Math.min(totalClusters, from + clusterBatch) : totalClusters;
		let scanned = 0;
		for (let ci = from; ci < to; ci += 1) {
			const list = byCluster.get(clusters[ci].id);
			if (!list) continue;
			scanned += list.length;
			scan(list);
		}
		const done = to >= totalClusters;
		// 没聚过类的（新库、节点很少）：最后一轮顺手全量扫一遍。
		// 太多就跳过 —— 那是几千条两两比，会把主线程按住。
		if (done && lonely.length && lonely.length <= 800) {
			scanned += lonely.length;
			scan(lonely);
		}
		return {
			groups: groups.sort((a, b) => b.drop.length - a.drop.length),
			scanned,
			nextCluster: to,
			totalClusters,
			done,
			total: rows.length,
		};
	}

	/**
	 * 写入记忆统一入口：嵌入 + sqlite 事务 + 条件重聚类都推给子进程。
	 * 主线程的 node:sqlite 是同步 API，插入一多、或刚好撞上一次 k-means 重建
	 * （1000 条约 380ms），事件循环就被按住，同进程的 SSE 读不动 → transport failed。
	 */
	async function addItems(lib, items) {
		if (!useWorker) return lib.add(items);
		try {
			return await recallWorker.add({
				dataDir: manager.dataDir,
				modelDir: manager.modelDir,
				libId: lib.id,
				items,
				rebuildEvery: Number(val(cfg.rebuildEvery, 50)),
			});
		} catch (error) {
			mark(`worker add FAILED, fallback inline: ${error?.message ?? error}`);
			return lib.add(items);
		}
	}

	/**
	 * 会话 id → 标题。面板「会话绑定」只给一串 32 位 id 根本分不清谁是谁，
	 * 所以列表和绑定都要带上标题（标题来自 session 日志里的 session/title 事件）。
	 */
	async function sessionTitleMap(ids) {
		const out = new Map();
		const uniq = [...new Set((ids ?? []).map(String).filter(Boolean))];
		if (!uniq.length) return out;
		const q = softService(ctx, "sessionQuery");
		if (!q?.readTitleSnapshots) return out;
		try {
			// projectMany 保序：结果数组与输入唯一 id 一一对应，别去猜 header 的字段名
			const rows = await q.readTitleSnapshots(uniq);
			rows?.forEach((row, i) => {
				if (row?.status === "rejected") return;
				const v = row?.value ?? row;
				const raw = v?.title;
				const title = typeof raw === "string" ? raw : typeof raw?.title === "string" ? raw.title : "";
				if (title.trim()) out.set(uniq[i], title.trim().slice(0, 60));
			});
		} catch (error) {
			mark(`sessionTitleMap FAILED: ${error?.message ?? error}`);
		}
		return out;
	}

	/** 给绑定列表补上标题字段 */
	async function withTitles(bindings) {
		const list = bindings ?? [];
		const titles = await sessionTitleMap(list.map((b) => b.session));
		return list.map((b) => ({ ...b, title: titles.get(String(b.session)) ?? "" }));
	}

	/** 所有会话（含标题），给面板的绑定下拉用。带了 60s 缓存：
	 *  readTitleSnapshots 要把每个会话的日志翻开折标题，会话一多就是几秒起步，
	 *  面板每次刷新都现算会直接把请求拖超时。 */
	let sessionsCache = { at: 0, data: [] };
	async function listSessionsBrief({ refresh = false } = {}) {
		if (!refresh && Date.now() - sessionsCache.at < 60_000 && sessionsCache.data.length) return sessionsCache.data;
		const q = softService(ctx, "sessionQuery");
		if (!q?.listSessions) return [];
		let rows = [];
		try {
			rows = (await q.listSessions()) ?? [];
		} catch (error) {
			mark(`listSessions FAILED: ${error?.message ?? error}`);
			return [];
		}
		// 带上 cwd / createdAt / live：面板要按工作区分组，并且要能区分「活着」和「只剩日志」。
		// 排序：活着的在前，再按最近创建。老会话掉到后面不碍事，用户还是看得到。
		const briefs = rows
			.map((r) => {
				const id = String(r?.header?.id ?? r?.id ?? "");
				if (!id) return null;
				return {
					id,
					cwd: String(r?.header?.cwd ?? ""),
					createdAt: Number(r?.header?.createdAt) || 0,
					live: Boolean(r?.live),
				};
			})
			.filter(Boolean)
			.sort((a, b) => Number(b.live) - Number(a.live) || b.createdAt - a.createdAt)
			.slice(0, 120);
		const titles = await sessionTitleMap(briefs.map((b) => b.id));
		const data = briefs.map((b) => ({
			...b,
			title: titles.get(b.id) ?? "",
			bound: manager.index.binding(b.id) ?? "",
		}));
		sessionsCache = { at: Date.now(), data };
		return data;
	}

	/**
	 * 召回台账：每个会话留最近 200 次召回（内存 + 落盘 JSONL），
	 * 面板里能看到"到底检了没有、检出了什么、哪几条老是被翻出来"。
	 */
	const recallLog = new Map();
	const RECALL_KEEP = 200;

	function recordRecall(entry) {
		try {
			const list = recallLog.get(entry.session) ?? [];
			list.push({ ...entry, ts: Date.now() });
			if (list.length > RECALL_KEEP) list.splice(0, list.length - RECALL_KEEP);
			recallLog.set(entry.session, list);
			appendFileSync(join(dataDir, "recall-log.jsonl"), `${JSON.stringify({ ...entry, ts: Date.now() })}\n`);
		} catch {
			/* 记台账失败不影响召回 */
		}
	}

	/** 最近 N 次 + 高频统计（按记忆 id 聚合，带上库和当前权重）。
	 *  传 session 看单个会话；传 lib 看整个库（跨所有会话）。 */
	function recallReport({ session = "", lib = "", limit = 20 } = {}) {
		const pick = (e) => (lib ? e.lib === lib : true);
		let all = [];
		if (session) {
			all = (recallLog.get(session) ?? []).filter(pick);
		} else {
			for (const list of recallLog.values()) all.push(...list.filter(pick));
			all.sort((a, b) => (a.ts ?? 0) - (b.ts ?? 0));
		}
		const items = all.slice(-limit).reverse();
		const freq = new Map();
		for (const e of all) {
			for (const h of e.hits ?? []) {
				const k = `${e.lib}::${h.id}`;
				const cur = freq.get(k) ?? { lib: e.lib, id: h.id, text: h.text, kind: h.kind, count: 0, lastAt: 0, weight: h.weight ?? 1 };
				cur.count += 1;
				cur.lastAt = Math.max(cur.lastAt, e.ts ?? 0);
				freq.set(k, cur);
			}
		}
		const top = [...freq.values()].sort((a, b) => b.count - a.count).slice(0, 15);
		return { items, top };
	}

	/**
	 * 阈值参数打包：默认全部跟设置走，调用方（面板/工具）可以临时覆盖单项。
	 * dynamic = max(下限, top1 × 比例)，fixed = 直接用一个固定值。
	 */
	function minScoreOpts(over = {}) {
		const mode = String(over.mode ?? strOf(cfg.minScoreMode, "dynamic")) === "fixed" ? "fixed" : "dynamic";
		const fixed = Number.isFinite(Number(over.fixed)) ? Number(over.fixed) : Number(val(cfg.recallMinScore, 0.35));
		return {
			minScoreMode: mode,
			minScore: fixed,
			minScoreFloor: Number(val(cfg.minScoreFloor, 0.35)),
			minScoreRatio: Number(val(cfg.minScoreRatio, 0.55)),
		};
	}

	async function refreshRecall(sessionId, queryText) {
		if (!on || !Boolean(val(cfg.autoRecall, true))) return;
		const q = String(queryText ?? "").trim();
		if (q.length < 2) return;
		const key = textHash(q);
		const cached = recallCache.get(sessionId);
		if (cached?.key === key) return;
		const t0 = Date.now();
		try {
			const lib = await libFor(sessionId);
			if (!lib) return;
			const topK = Number(val(cfg.recallTopK, 5));
			// 召回走的是"用户刚发消息"的热路径：这里绝不能碰精排模型。
			// cross-encoder 一加载就是 500MB+ 的 ONNX，CPU 和内存一起抖，
			// 宿主同一个事件循环里正在跑 SSE 流，读不动 socket 就会被掐成
			// "DeepSeek Messages transport failed"。精排只留给 rag_search 显式调用。
			const r = await searchLibrary(lib, q, {
				topK,
				assoc: Boolean(val(cfg.useAssociation, false)),
				assocCount: Number(val(cfg.recallAssocCount, 3)),
				assocPerHit: Number(val(cfg.recallAssocPerHit, 1)),
				assocMode: strOf(cfg.assocMode, "weighted"),
				...minScoreOpts(),
				mmrLambda: Number(val(cfg.mmrLambda, 0.7)),
				rerank: false,
			});
			if (!r.hits.length) {
				recallCache.set(sessionId, { key, text: "" });
				return;
			}
			const maxChars = Number(val(cfg.recallMaxChars, 900));
			const line = (h) => `- [${new Date(h.ts).toISOString().slice(0, 10)}·${h.kind}] ${String(h.text).replace(/\s+/g, " ").slice(0, 160)}`;
			const lines = [];
			let used = 0;
			for (const h of r.hits) {
				const l = line(h);
				if (used + l.length > maxChars) break;
				lines.push(l);
				used += l.length;
			}
			// 联想单独成段、单独吃预算（直命中预算的一半）：它是"顺带看看"，
			// 抢了直命中的位置就本末倒置了。
			const assocLines = [];
			const assocBudget = Math.min(Math.round(maxChars / 2), 320);
			let assocUsed = 0;
			for (const a of r.assoc ?? []) {
				const l = `- [${a.kind}·#${a.id}] ${String(a.text).replace(/\s+/g, " ").slice(0, 110)}`;
				if (assocUsed + l.length > assocBudget) break;
				assocLines.push(l);
				assocUsed += l.length;
			}
			const parts = [];
			if (lines.length) parts.push(`【长期记忆 · 直命中 ${lines.length} 条】\n${lines.join("\n")}`);
			if (assocLines.length) {
				parts.push(`【相关联想 · 另外 ${assocLines.length} 条（不占上面名额，只是参照）】\n${assocLines.join("\n")}`);
			}
			const text = parts.length
				? `${parts.join("\n")}\n（两段分开计数，别相加、别逐条复述，只说和当前问题相关的部分。）`
				: "";
			recallCache.set(sessionId, { key, text, lib: lib.id });
			const ms = Date.now() - t0;
			recordRecall({
				session: sessionId,
				lib: lib.id,
				q,
				ms,
				hits: r.hits.map((h) => ({
					id: h.id,
					text: String(h.text).replace(/\s+/g, " ").slice(0, 120),
					kind: h.kind,
					score: h.score,
					weight: h.weight ?? 1,
				})),
				assoc: (r.assoc ?? []).map((a) => ({ id: a.id, from: a.from, sim: a.sim, kind: a.kind })),
			});
			if (ms > 120) mark(`recall slow ${ms}ms (hits=${r.hits.length} assoc=${(r.assoc ?? []).length})`);
		} catch (error) {
			ctx.logger?.warn?.(`[fast-ivf-rag] 自动召回失败：${error?.message ?? error}`);
			mark(`recall FAILED after ${Date.now() - t0}ms: ${error?.message ?? error}`);
		}
	}

	/**
	 * 召回挂在 **runtime-context**，不挂 system。
	 *
	 * 2026-11-02 晚改：以前是 `sp.section({ order: 9500 })`，也就是把召回写进系统提示词。
	 * host 的 system 每轮是「整块替换第一节点」（agent-loop:1073，`inHistory: false`），
	 * 召回一变 system 就变 → 前缀缓存从第 0 字节起全部失效，连历史一起作废。
	 * 省下来的不是钱，是每次全量 prefill。
	 *
	 * 改挂 `sp.context`：宿主把它渲染成一条 `user/message`（source.kind = runtime-context），
	 * 位置固定在「真实 user 消息之前」（agent-loop:1083），每轮重建，system 一个字不动。
	 * 缓存逻辑（refreshRecall）、预算、联想段全部照旧，只是换了绑定点。
	 *
	 * 老宿主没有 `context()` 就退回 section —— 难看但能跑。
	 */
	function ensureSection() {
		if (sectionBound) return;
		const sp = ctx.get("systemPrompt") ?? ctx.systemPrompt;
		if (!sp?.section) return;
		const read = (c) => {
			// 每轮 assemble 时同步执行，任何异常都会把整轮请求打成失败，这里必须吞干净
			try {
				if (!Boolean(val(cfg.autoRecall, true))) return "";
				const sid = c?.agent?.session?.id ?? c?.agent?.id;
				if (!sid) return "";
				const hit = recallCache.get(String(sid));
				return typeof hit?.text === "string" ? hit.text : "";
			} catch (error) {
				mark(`recall text FAILED: ${error?.stack ?? error}`);
				return "";
			}
		};
		const useContext = typeof sp.context === "function";
		try {
			const dispose = useContext
				? sp.context({ name: "fast-ivf-rag:recall", order: 400, text: read })
				: sp.section({ name: "fast-ivf-rag:recall", order: 9500, text: read });
			if (typeof dispose === "function") disposers.push(dispose);
			sectionBound = true;
			ctx.logger?.info?.(
				useContext
					? "[fast-ivf-rag] 已挂载自动召回环境段（runtime-context，不进 system）"
					: "[fast-ivf-rag] 已挂载自动召回提示段（宿主无 systemPrompt.context，退回 system）",
			);
		} catch (error) {
			mark(`recall mount FAILED: ${error?.stack ?? error}`);
		ctx.logger?.warn?.(`[fast-ivf-rag] 挂载召回段失败：${error?.message ?? error}`);
		}
	}

	/* ───────── 自动总结（未激活缓冲 → 激活成节点） ───────── */

	/** 未激活缓冲上限：LLM 卡死时也不能让内存无限涨，超了丢最老的。 */
	const MAX_PENDING = 600;
	const DEFAULT_PACK = 3;

	function tracker(sessionId) {
		let t = trackers.get(sessionId);
		if (!t) {
			t = {
				msgs: [], // 活跃窗口：最近 N 条原文，永不参与压缩
				pending: [], // 未激活缓冲区（原项目 nr_nodes）：已出窗口、等着被激活成节点
				lastAt: Date.now(),
				busy: false,
				label: "",
				schedule: null, // 事件驱动激活的 debounce 句柄
				nextAt: 0, // 冷却截止时间（空结果后短暂退避）
				emptyStreak: 0, // 连续空结果次数
				lastTake: 0,
				lastStored: 0,
				lastRaw: 0,
				lastActivateAt: 0,
			};
			trackers.set(sessionId, t);
		}
		return t;
	}

	/** 发言人名字：QQ 群里就是群昵称，纯 DSH 会话里一般没有，取不到就返回空。 */
	function senderNameOf(data) {
		// 和 textOfMessage 一样要往下挖一层：assistant/message 的 source 在 data.message.source 里
		const msg = unwrapMessage(data);
		const cands = [msg?.name, msg?.senderName, msg?.source?.name, msg?.meta?.senderName, msg?.meta?.from];
		for (const c of cands) if (typeof c === "string" && c.trim()) return c.trim();
		return "";
	}

	function pushMessage(sessionId, role, text, senderName = "") {
		if (!Boolean(val(cfg.autoSummarize, true))) return;
		const t = String(text ?? "").trim();
		if (!t) return;
		const rec = tracker(sessionId);
		// 相邻同文去重：qq_send / qq_reply 已经把话发出去之后，
		// 那一轮的 assistant 正文有时也会走到这里（selfSent 没挡住的边界情况）。
		// 同一个人连着说两句一模一样的话没有记忆价值，只会让总结重复写。
		const last = rec.msgs[rec.msgs.length - 1];
		if (last && last.role === role && last.text === t.slice(0, 1200)) return;
		const who = typeof senderName === "string" ? senderName.trim().slice(0, 24) : "";
		rec.msgs.push({ role, text: t.slice(0, 1200), at: Date.now(), ...(who ? { name: who } : {}) });
		rec.lastAt = Date.now();
		// 满了就立刻溢到缓冲区：一条消息只走 msgs → pending → 节点 这一条单向路，压过就出队
		drainToPending(sessionId);
	}


	/**
	 * 活跃窗口 → 未激活缓冲区。
	 * 窗口固定保留最近 active 条原文；更老的部分按原顺序搬进 pending。
	 * 搬完顺手看一眼缓冲够不够数，够了就立刻排一次激活（不等 60 秒定时器）。
	 */
	function drainToPending(sessionId, { schedule = true } = {}) {
		const rec = trackers.get(sessionId);
		if (!rec) return 0;
		const active = Math.max(2, Number(val(cfg.activeWindow, 16)));
		const overflow = rec.msgs.length - active;
		if (overflow <= 0) return 0;
		rec.pending.push(...rec.msgs.splice(0, overflow));
		if (rec.pending.length > MAX_PENDING) {
			const drop = rec.pending.length - MAX_PENDING;
			rec.pending.splice(0, drop);
			mark(`pending 溢出，丢弃最老 ${drop} 条`);
		}
		if (schedule) maybeActivate(sessionId);
		return overflow;
	}

	/** 缓冲够数 → 350ms 后激活（debounce：连续刷屏只触发一次）。 */
	function maybeActivate(sessionId, { force = false } = {}) {
		const rec = trackers.get(sessionId);
		if (!rec || rec.schedule) return;
		const minMsgs = Math.max(1, Number(val(cfg.summarizeMinMessages, 5)));
		if (!force && rec.pending.length < minMsgs) return;
		rec.schedule = setTimeout(() => {
			rec.schedule = null;
			void activate(sessionId, { force, reason: "auto" });
		}, 350);
		if (typeof rec.schedule.unref === "function") rec.schedule.unref();
	}

	/** 纯表情 / 纯标点的消息不值得单独成节点。 */
	const EMOJI_ONLY = /^[\p{Extended_Pictographic}\p{Emoji_Presentation}\s\p{P}\p{S}]+$/u;

	/**
	 * 按说话人拆包：**不同说话人绝不成同一个节点**，同一个人的连续发言每 packSize 条合成一个。
	 * 这是"输入好几条包含多个说话人就拆成好几条分别存"的落点。
	 */
	function packBySpeaker(batch, packSize = DEFAULT_PACK) {
		const size = Math.max(1, Math.min(10, packSize));
		const groups = [];
		let cur = null;
		for (const m of batch) {
			const text = String(m.text ?? "").trim();
			if (text.length < 2 || EMOJI_ONLY.test(text)) continue;
			const who = String(m.name || (m.role === "assistant" ? "我" : m.role === "user" ? "对方" : "系统")).slice(0, 24);
			if (!cur || cur.who !== who || cur.items.length >= size) {
				cur = { who, items: [] };
				groups.push(cur);
			}
			cur.items.push(text.slice(0, 300));
		}
		return groups
			.filter((g) => g.items.length)
			.map((g) => ({ who: g.who, text: `${g.who}：${g.items.join("；")}`, n: g.items.length }));
	}

	const hydrated = new Set();

	/**
	 * 回读水位线：每个会话「已经补到哪个时间点」。
	 *
	 * 为什么非得落盘：`hydrated` 是个内存 Set，进程一重启就空。
	 * 不记水位的话，每次重启都从零开始补 —— 而补进来的第 601 条到第 600 条
	 * 早就压成摘要进过库了，于是同一件事被反复写一遍。
	 * 姐姐 2026-11-01 在面板上看到的 `#811 我记得2026-10-31有人问我…`
	 * 就是这么被"翻上来"的：不是检索出问题，是它被重新写了一次。
	 */
	const wmFile = join(dataDir, "hydrate-watermarks.json");
	let wmCache = null;
	function watermarks() {
		if (wmCache) return wmCache;
		try {
			const raw = JSON.parse(readFileSync(wmFile, "utf8"));
			wmCache = raw && typeof raw === "object" && !Array.isArray(raw) ? raw : {};
		} catch {
			wmCache = {};
		}
		return wmCache;
	}
	function saveWatermarks() {
		if (!wmCache) return;
		try {
			mkdirSync(dirname(wmFile), { recursive: true });
			writeFileSync(wmFile, JSON.stringify(wmCache, null, 2), "utf8");
		} catch (error) {
			mark(`watermark 落盘失败: ${error?.message ?? error}`);
		}
	}

	/** 每个会话第一次见到的消息：回读一次历史，把"插件启动前的老对话"接进缓冲。 */
	async function hydrateOnce(sessionId) {
		if (hydrated.has(sessionId)) return;
		hydrated.add(sessionId);
		const wm = watermarks();
		const since = Number(wm[sessionId]) || 0;
		const lookbackMs = Math.max(0, Number(val(cfg.hydrateLookbackMs, 7_200_000)));
		const hist = await loadHistory(sessionId, { since, lookbackMs }).catch(() => null);
		if (!hist?.items?.length) return;
		const rec = tracker(sessionId);
		const before = rec.msgs.length;
		// 合并而不是覆盖：磁盘历史 + 内存里还没落盘的最新几条都要保住。
		// 以前是 rec.msgs = hist，刚落盘的那条用户消息会被冲掉，等于"最新几条被丢"。
		const seen = new Set(hist.items.map((m) => `${m.role}\u0000${m.text}`));
		const extra = rec.msgs.filter((m) => !seen.has(`${m.role}\u0000${m.text}`));
		if (hist.items.length + extra.length > before) {
			rec.msgs = hist.items.concat(extra);
			const scope = since > 0 ? "水位线之后" : `最近 ${Math.round(lookbackMs / 60000)} 分钟`;
			ctx.logger?.info?.(
				`[fast-ivf-rag] 会话 ${sessionId.slice(0, 8)} 回读 ${hist.items.length} 条历史（+${extra.length} 条内存，只补${scope}）`,
			);
			drainToPending(sessionId);
		}
		if (hist.newest > since) {
			wm[sessionId] = hist.newest;
			saveWatermarks();
		}
	}

	async function pickModel(llm, sessionId) {
		const override = strOf(cfg.summarizeModel);
		if (override.includes("/")) {
			const [p, ...rest] = override.split("/");
			return { provider: p, model: rest.join("/") };
		}
		// 最靠谱的来源：这个会话自己上一轮请求用的 provider/model
		// （session/event 里的 request/header 快照，Session.requestHeader().config）。
		try {
			const sessions = softService(ctx, "sessions");
			const live = sessions?.get?.(sessionId);
			const conf = live?.requestHeader?.()?.config;
			if (conf?.provider && conf?.model) {
				mark(`pickModel from session = ${conf.provider}/${conf.model}`);
				return { provider: conf.provider, model: conf.model };
			}
		} catch (error) {
			mark(`pickModel session FAILED: ${error?.message ?? error}`);
		}
		let list = [];
		try {
			list = llm.listProviders?.() ?? [];
		} catch {
			list = [];
		}
		for (const p of list) {
			const pid = typeof p === "string" ? p : p?.name ?? p?.id ?? p?.provider;
			if (!pid) continue;
			// listProviders() 只给 provider 路由，模型清单得单独问（可能是异步的）
			let models = [];
			try {
				models = (await llm.listModels?.(pid)) ?? [];
			} catch {
				models = [];
			}
			const first = models[0];
			const mid = typeof first === "string" ? first : first?.id ?? first?.name;
			mark(`pickModel provider=${pid} models=${JSON.stringify(models).slice(0, 200)} pick=${mid}`);
			if (mid) return { provider: pid, model: mid };
		}
		return { provider: undefined, model: undefined };
	}

	async function callLlm(prompt, sessionId) {
		const llm = ctx.get("llm") ?? ctx.llm;
		if (!llm?.stream) throw new Error("llm 服务不可用");
		const pick = await pickModel(llm, sessionId);
		let out = "";
		const seen = [];
		const stream = llm.stream({
			provider: pick.provider,
			model: pick.model,
			messages: [{ role: "user", content: [{ type: "text", text: prompt }] }],
		});
		for await (const chunk of stream) {
			// 宿主的 StreamChunk 用 `kind` 做判别字段（dsh-llm 的 types.ts）：
			// block-start / text-delta / usage / finish。以前这里只认 `type`，
			// 结果一个字都收不到，总结永远是空串。
			const kind = chunk?.kind ?? chunk?.type;
			if (seen.length < 6) seen.push(kind);
			if (kind === "text-delta") out += chunk.text ?? chunk.delta ?? chunk.textDelta ?? "";
			else if (kind === "block-delta" && typeof chunk.text === "string") out += chunk.text;
			else if (kind === "error" || (kind === "finish" && chunk.kind === "error")) {
				throw new Error(`llm stream error: ${chunk.failure?.code ?? chunk.code ?? "unknown"}`);
			}
		}
		mark(`callLlm pick=${pick.provider}/${pick.model} kinds=${JSON.stringify(seen)} chars=${out.length}`);
		return out.trim();
	}

	/**
	 * 回读会话当前的模型表面（只读，不碰历史）。用于补上"插件启动前的老对话"，
	 * 否则自动总结永远只能看见本次启动之后的消息。
	 *
	 * ⚠️ 2026-11-01 修：这个函数是「昨天的内容被重新压一遍」的源头。
	 *   原来无条件回读尾部 600 个事件，插件每重启一次就把已经压过的旧对话
	 *   再压一遍，写成一条新的摘要 —— 姐姐在面板上看到的就是这个。
	 *   现在必须带 `since`（水位线）和 `lookbackMs`（时间窗）两道闸，
	 *   只补真正"还没见过"的那一段。
	 */
	/**
	 * 环境快照 / 压缩替身 / 插件注入：**都不是人说的话**，一律不进记忆。
	 *
	 * 2026-11-03 12:2x 姐姐在面板上又看到旧事被翻出来（`#281 … 2026-11-02 老大说 …`），
	 * 追下去两条路一起漏：
	 *   ① `loadHistory` 读的是 `ev.at`，可宿主的事件信封只有 `type/seq/time/data/…`
	 *      （dsh-session/lib/index.js:assertSessionEventEnvelope）—— `at` 恒 undefined，
	 *      于是 `newest` 恒 0（水位永远存不进去）、`at > 0` 恒 false（时间闸永远不过滤），
	 *      每次 hydrate 都把尾部 600 条事件整段重灌一遍缓冲区；
	 *   ② runtime-context 是 `user/message`，内容却是工作目录 / 沙箱策略 / 记忆召回 / dsh-qq 动态段。
	 *      它被当成"用户说的话"推进缓冲区 → 总结成节点 → 下次又被召回 → 再被总结。
	 *      召回文本自我繁殖，这才是"过往记录被捞上来"的主因。
	 */
	async function loadHistory(sessionId, { since = 0, lookbackMs = 0 } = {}) {
		const q = softService(ctx, "sessionQuery");
		if (!q?.readSurface) return null;
		const surface = await q.readSurface(sessionId);
		// 大会话别把整条日志翻一遍：只取尾部
		const all = surface?.events ?? [];
		const events = all.length > 3000 ? all.slice(-600) : all;
		// 两道闸取更严格的那个
		const floor = Math.max(Number(since) || 0, lookbackMs > 0 ? Date.now() - lookbackMs : 0);
		const out = [];
		let newest = 0;
		for (const ev of events) {
			const type = ev?.type ?? "";
			const data = ev?.data ?? ev?.payload ?? {};
			let role = "";
			let text = "";
			if (type === "user/message") {
				// 环境快照不是对话：既不算时间水位，也不进缓冲区
				if (!isMemoryWorthyUser(data)) continue;
				role = "user";
				text = textOfMessage(data);
			} else if (type === "assistant/message") {
				role = "assistant";
				text = textOfMessage(data);
			} else if (type === "tool/call") {
				// 真正发出去的那句话在工具参数里（带 tool-call 的 step 在 textOfMessage 里是整步丢弃的）
				role = "assistant";
				text = sentTextOfToolCall(ev);
			} else continue;
			if (!text) continue;
			const at = timeOfEvent(ev);
			if (at > newest) newest = at;
			// at 真的缺失（0）时不过滤 —— 宁可多补一条，也别因为没时间戳丢了新消息
			if (floor > 0 && at > 0 && at < floor) continue;
			// 相邻同文去重：工具发过的话 + 随后的正文，经常是同一句
			const prev = out[out.length - 1];
			if (prev && prev.role === role && prev.text === text.slice(0, 1200)) continue;
			out.push({ role, text: text.slice(0, 1200), at });
		}
		return { items: out, newest };
	}

	/**
	 * 激活：把未激活缓冲区里的一批消息一次性转成节点。双通道：
	 *   ① 原文通道（rawCapture）——按说话人拆包，**先入库**，节点数跟消息数挂钩，完全不等 LLM；
	 *   ② 压缩通道——同一批交给 LLM 压成 summary 节点。
	 * 关键约定：**一批消息只压一遍**。出队即消费，空结果也不回填、不重压，冷却 45 秒后继续跑新消息。
	 * （以前空结果会把整段留在原地重压三遍，白烧 token 还占着 5 分钟。）
	 */
	async function activate(sessionId, { force = false, reason = "auto" } = {}) {
		const rec = trackers.get(sessionId);
		if (!rec || rec.busy) return null;
		const minMsgs = Math.max(1, Number(val(cfg.summarizeMinMessages, 5)));
		const now = Date.now();
		if (!force) {
			if (rec.nextAt && now < rec.nextAt) return null;
			if (rec.pending.length < minMsgs) return null;
		}
		if (!rec.pending.length) return null;

		// 先找库、再摘缓冲：没挂库（autoMount 关了 / 还没绑定）时不能把消息摘出来丢掉，
		// 原地留着，等绑上之后再压。
		const lib = await libFor(sessionId).catch(() => null);
		if (!lib) {
			rec.nextAt = Date.now() + 60_000; // 一分钟内别再试，省得每次 drain 都白跑一趟
			return null;
		}

		// 连续空结果 → 缩小批量：短片段模型更容易吐东西，长片段它干脆说"无"
		const shrink = rec.emptyStreak >= 3 ? 0.5 : 1;
		const cap = Math.max(minMsgs, Math.round(Number(val(cfg.summarizeMaxTake, 24)) * shrink));
		const batch = rec.pending.splice(0, Math.min(rec.pending.length, cap));
		if (!batch.length) return null;

		rec.busy = true;
		const t0 = Date.now();
		try {
			// ① 原文通道（默认关）：开着时先把原文落库，节点增长不等 LLM
			let rawIds = [];
			if (Boolean(val(cfg.rawCapture, false))) {
				try {
					const packs = packBySpeaker(batch, Number(val(cfg.rawPackSize, DEFAULT_PACK)));
					if (packs.length) {
						const r = await addItems(
							lib,
							packs.map((p) => ({ text: p.text, kind: "chat", src: `session:${sessionId}`, ts: Date.now() })),
						);
						rawIds = r?.ids ?? [];
					}
				} catch (error) {
					mark(`raw 通道 FAILED: ${error?.message ?? error}`);
				}
			}

			// ② 压缩通道
			const prompt = buildSummaryPrompt(strOf(cfg.summarizePrompt, DEFAULT_SUMMARY_PROMPT), {
				messages: batch,
				sessionLabel: sessionId,
				range: `${batch[0]?.at ?? ""}~${batch[batch.length - 1]?.at ?? ""}`,
				today: new Date().toISOString().slice(0, 10),
			});
			let items = [];
			let raw = "";
			try {
				raw = await callLlm(prompt, sessionId);
				items = parseSummary(raw);
			} catch (error) {
				mark(`压缩通道 FAILED: ${error?.message ?? error}`);
			}
			if (items.length) {
				try {
					await addItems(
						lib,
						items.map((t) => ({ text: t, kind: "summary", src: `session:${sessionId}`, ts: Date.now() })),
					);
					// 这批原文已经被压成摘要 → 原文节点降权，别在检索里跟摘要抢位置
					for (const id of rawIds) {
						try {
							lib.db.setWeight(id, 0.5);
						} catch (error) {
							mark(`setWeight FAILED: ${error?.message ?? error}`);
							break;
						}
					}
					rec.emptyStreak = 0;
					rec.nextAt = 0;
				} catch (error) {
					mark(`summary 入库 FAILED: ${error?.message ?? error}`);
				}
			} else {
				rec.emptyStreak += 1;
				// 单次"无"不拦下一批（下一批是新消息，不该陪着一起等）；
				// 连续空才短暂冷却，免得 prompt 出问题时把额度烧光。
				rec.nextAt = rec.emptyStreak >= 2 ? Date.now() + 30_000 : 0;
			}
			rec.lastTake = batch.length;
			rec.lastStored = items.length;
			rec.lastRaw = rawIds.length;
			mark(
				`activate(${reason}) 取${batch.length}条 → 原文${rawIds.length}节点 / 摘要${items.length}条 ` +
					`llm=${Date.now() - t0}ms 空=${rec.emptyStreak} 剩${rec.pending.length} 原文头=${JSON.stringify(String(raw ?? "").slice(0, 80))}`,
			);
			return { summarized: batch.length, stored: items.length, raw: rawIds.length, items };
		} finally {
			rec.busy = false;
			rec.lastActivateAt = Date.now();
			// 缓冲还剩够数的，接着排下一批，不必等定时器
			if (rec.pending.length >= Math.max(1, Number(val(cfg.summarizeMinMessages, 5)))) {
				maybeActivate(sessionId);
			}
		}
	}

	/** 工具入口 `rag_summarize`：先把窗口外的推进缓冲，再立刻激活一次（可以不等门槛）。 */
	async function summarize(sessionId, { force = false } = {}) {
		let rec = trackers.get(sessionId);
		if (!hydrated.has(sessionId)) {
			const h0 = Date.now();
			// 走同一套水位线 + 时间窗（别再复制一份回读逻辑：两份迟早走岔）
			await hydrateOnce(sessionId);
			if (Date.now() - h0 > 200) mark(`hydrate ${Date.now() - h0}ms`);
			rec = tracker(sessionId);
		}
		if (!rec) return null;
		if (rec.busy) return null;
		drainToPending(sessionId, { schedule: false });
		return activate(sessionId, { force, reason: "tool" });
	}

	/** 兜底定时器：主路径是事件驱动（缓冲够数 350ms 后就激活），这里只负责补扫窗口溢出。 */
	function startTimer() {
		if (timer) return;
		timer = setInterval(() => {
			if (!Boolean(val(cfg.autoSummarize, true))) return;
			for (const [sid, rec] of trackers) {
				try {
					if (rec.busy) continue;
					drainToPending(sid); // 里面会顺手 maybeActivate
					if (rec.pending.length) maybeActivate(sid);
				} catch (error) {
					mark(`timer tick FAILED: ${error?.message ?? error}`);
				}
			}
		}, 30_000);
		if (typeof timer.unref === "function") timer.unref();
	}

	/* ───────── 会话事件 ───────── */

	try {
		const off = ctx.on("session/event", (session, event) => {
			// 这条链跑在 session.append 的同步路径上：抛出去会把用户发消息打成 "prompt rejected"
			try {
				if (!on) return;
				const sid = String(session?.id ?? "");
				if (!sid) return;
				const type = event?.type ?? "";
				const data = event?.data ?? event?.payload ?? {};
				if (type === "user/message") {
					// 压缩替身 / 插件注入的 notice / 宿主每轮的环境快照，都不是人说的话：
					// 推进缓冲区会污染记忆（还会拿「（更早的对话已省略）」「Current runtime context」
					// 这样的句子当召回 query）。判据集中在 SRC_KIND_SKIP。
					if (!isMemoryWorthyUser(data)) return;
					const text = textOfMessage(data);
					pushMessage(sid, "user", text, senderNameOf(data));
					// 所有耗时活都挪出 append 的同步栈：先让这条用户消息落盘、LLM 请求发出去，
					// 下一轮 tick 再算召回。ONNX 推理/读历史一旦挤在这条栈上，
					// 宿主会连 socket 都读不动，SSE 被掐成 transport failed。
					setImmediate(() => {
						try {
							void hydrateOnce(sid);
							ensureSection();
							if (text) void refreshRecall(sid, text);
						} catch (error) {
							mark(`deferred recall FAILED: ${error?.stack ?? error}`);
						}
					});
				} else if (type === "assistant/message") {
					pushMessage(sid, "assistant", textOfMessage(data));
				} else if (type === "tool/call") {
					// 真正发出去的那句话在 qq_send / qq_reply 的参数里
					const sent = sentTextOfToolCall(event);
					if (sent) pushMessage(sid, "assistant", sent);
				} else if (type === "session/disposed" || type === "session/closed") {
					recallCache.delete(sid);
					const dead = trackers.get(sid);
					if (dead?.schedule) clearTimeout(dead.schedule);
					trackers.delete(sid);
					hydrated.delete(sid);
				}
			} catch (error) {
				mark(`session event FAILED: ${error?.stack ?? error}`);
			}
		});
		if (typeof off === "function") disposers.push(off);
	} catch (error) {
		mark(`session-subscribe FAILED: ${error?.stack ?? error}`);
		ctx.logger?.warn?.(`[fast-ivf-rag] 订阅会话事件失败：${error?.message ?? error}`);
	}
	startTimer();

	/* ───────── 诊断：把 transport failed 的真凶挖出来 ───────── */
	// 宿主 agent-loop 的真实调用链是 llm.prepareCall(config, signal) → preparedCall.stream(request)
	// （dsh-agent-loop/lib/index.js:1217 与 :1100），只有拿不到 preparedCall 时才回退 llm.stream。
	// 宿主对象是**冻结**的，直接赋值会抛 "Cannot assign to read only property"，所以这里只用
	// Proxy 包一层返回值，绝不去写别人的属性。
	try {
		const llmSvc = softService(ctx, "llm");
		const wrapGenerator = (inner, label) =>
			(async function* () {
				try {
					yield* inner;
				} catch (error) {
					mark(
						`${label} FAILED code=${error?.code ?? "?"} msg=${error?.message ?? error} ` +
							`cause=${error?.cause?.stack ?? error?.cause ?? "none"}`,
					);
					throw error;
				}
			})();

		const wrapPreparedCall = (call) => {
			if (!call || typeof call.stream !== "function") return call;
			try {
				// 宿主返回的是 Object.freeze({...})，而且 stream 是不可配置的数据属性：
				// 既不能赋值（严格模式抛错），也不能用 Proxy 返回替身（违反 Proxy 不变式）。
				// 唯一安全的做法是**复制成一个新的可写对象**，再挂上包了日志的 stream。
				const copy = {};
				for (const key of Object.keys(call)) copy[key] = call[key];
				copy.stream = (options) => wrapGenerator(call.stream(options), "preparedCall.stream");
				return copy;
			} catch (error) {
				mark(`preparedCall copy FAILED: ${error?.message ?? error}`);
				return call;
			}
		};

		if (llmSvc && typeof llmSvc.prepareCall === "function" && !llmSvc.__fastIvfRagPatched) {
			const originalPrepare = llmSvc.prepareCall.bind(llmSvc);
			const patched = async (...args) => {
				let call;
				try {
					call = await originalPrepare(...args);
				} catch (error) {
					mark(
						`llm.prepareCall FAILED code=${error?.code ?? "?"} msg=${error?.message ?? error} ` +
							`cause=${error?.cause?.stack ?? error?.cause ?? "none"}`,
					);
					throw error;
				}
				return wrapPreparedCall(call);
			};
			try {
				Object.defineProperty(llmSvc, "prepareCall", { value: patched, configurable: true, writable: true });
				mark("llm.prepareCall patched (diagnostics on)");
			} catch (error) {
				mark(`llm.prepareCall patch skipped (frozen): ${error?.message ?? error}`);
			}
		}
	} catch (error) {
		mark(`llm diagnostics wrap FAILED: ${error?.message ?? error}`);
	}

	/* ───────── 工具 ───────── */

	const tools = ctx.get("tools") ?? ctx.tools;
	const textOut = {
		schema: { type: "string" },
		render: (_args, value) => [{ type: "text", text: String(value ?? "") }],
	};

	function registerTool(definition) {
		try {
			const dispose = tools.register(definition);
			if (typeof dispose === "function") disposers.push(dispose);
		} catch (error) {
			mark(`tool ${definition.name} FAILED: ${error?.stack ?? error}`);
			ctx.logger?.warn?.(`[fast-ivf-rag] 注册工具 ${definition.name} 失败：${error?.message ?? error}`);
		}
	}

	if (tools?.register) {
		registerTool({
			name: "rag_add",
			description: "把一段内容写进当前会话的向量记忆库（长期记忆）。适合记住结论、偏好、踩坑、文件/项目名等以后可能用到的信息。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					texts: { type: "array", items: { type: "string" }, description: "要记住的内容，一条一句，最多 20 条" },
					kind: { type: "string", description: "分类标签，如 chat / code / fact / summary" },
					library: { type: "string", description: "指定库 id，留空用当前会话绑定的库" },
				},
				required: ["texts"],
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				const libId = args.library || (sid ? manager.index.binding(sid) : null);
				const lib = libId ? await manager.libraryAsync(libId) : await libFor(sid);
				if (!lib) return "没有可用的向量库：请先用 rag_libraries 查看或创建。";
				const items = (args.texts ?? []).slice(0, 20).map((t) => ({ text: t, kind: args.kind ?? "note" }));
				const res = await addItems(lib, items);
				return `已写入 ${res.added} 条（跳过重复 ${res.skipped} 条）→ 库 ${lib.id}，当前共 ${lib.db.count()} 条。`;
			},
		});

		registerTool({
			name: "rag_edit",
			description: "改一条记忆的正文（会重新嵌入并重建倒排索引）。记忆 id 从 rag_recent / rag_search 的 #id 里取。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					id: { type: "number", description: "记忆 id" },
					text: { type: "string", description: "新的正文" },
					kind: { type: "string", description: "顺带改分类，留空保持原样" },
					library: { type: "string", description: "指定库 id，留空用当前会话绑定的库" },
				},
				required: ["id", "text"],
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				const libId = args.library || (sid ? manager.index.binding(sid) : null);
				const lib = libId ? await manager.libraryAsync(libId) : await libFor(sid);
				if (!lib) return "没有可用的向量库。";
				const r = await lib.update(Number(args.id), args.text, args.kind ?? null);
				if (r.error) return `改失败：${r.error}`;
				recallCache.clear();
				return `已改 #${r.id}：${String(args.text).slice(0, 60)}`;
			},
		});

		registerTool({
			name: "rag_search",
			description: "在当前会话的向量记忆库里做混合检索（向量 + BM25 + RRF）。用于回忆以前聊过的事、写过的代码、做过的决定。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					query: { type: "string", description: "检索语句，用自然语言描述你想找什么" },
					topK: { type: "number", description: "直接命中条数，默认 6（联想不占这个名额）" },
					library: { type: "string", description: "指定库 id" },
					assoc: { type: "boolean", description: "是否附带一跳联想（单独计数，混不进直命中）" },
					assocCount: { type: "number", description: "联想池总容量，默认跟随设置（3）" },
					assocPerHit: { type: "number", description: "每条直接命中最多贡献几条联想，默认跟随设置（1，仅 even 模式）" },
					assocMode: { type: "string", enum: ["weighted", "even"], description: "联想分配方式，默认跟随设置（weighted）" },
					minScore: { type: "number", description: "固定阈值（0~1），只在 minScoreMode=fixed 时生效" },
					minScoreMode: { type: "string", enum: ["dynamic", "fixed"], description: "阈值模式，默认跟随设置（dynamic）" },
					mmrLambda: { type: "number", description: "MMR 多样性 λ（0=关闭，默认跟随设置 0.7）" },
					rerank: { type: "boolean", description: "是否用 cross-encoder 精排（默认跟随设置；首次会加载模型，稍慢）" },
				},
				required: ["query"],
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				const libId = args.library || (sid ? manager.index.binding(sid) : null);
				const lib = libId ? await manager.libraryAsync(libId) : await libFor(sid);
				if (!lib) return "没有可用的向量库。";
				const wantAssoc = Boolean(args.assoc ?? val(cfg.useAssociation, false));
				const r = await searchLibrary(lib, String(args.query ?? ""), {
					topK: Number(args.topK ?? 6),
					assoc: wantAssoc,
					assocCount: Number(args.assocCount ?? val(cfg.recallAssocCount, 3)),
					assocPerHit: Number(args.assocPerHit ?? val(cfg.recallAssocPerHit, 1)),
					assocMode: args.assocMode ?? strOf(cfg.assocMode, "weighted"),
					...minScoreOpts({ mode: args.minScoreMode, fixed: args.minScore }),
					mmrLambda: Number(args.mmrLambda ?? val(cfg.mmrLambda, 0.7)),
					rerank: args.rerank ?? Boolean(val(cfg.rerankEnabled, true)),
				});
				if (!r.hits.length) return `库里没有命中（共 ${r.meta.total} 条）。`;
				const lines = r.hits.map((h, i) => {
					const day = new Date(h.ts).toISOString().slice(0, 16).replace("T", " ");
					return `${i + 1}. [${day} · ${h.kind} · #${h.id}] ${h.text}`;
				});
				const assoc = r.assoc.length
					? `\n\n联想要点（${r.assoc.length} 条 · 不占上面 ${lines.length} 个名额）：\n` +
						r.assoc.map((a) => `- [${a.kind} · #${a.id} ← #${a.from}] ${a.text}（相似度 ${a.sim}）`).join("\n")
					: "";
				return `库 ${lib.id}（共 ${r.meta.total} 条，${r.meta.strategy} 策略，精排 ${r.meta.rerank ?? "off"}）· 直命中 ${lines.length} 条${r.assoc.length ? ` · 联想 ${r.assoc.length} 条` : ""}：\n${lines.join("\n")}${assoc}`;
			},
		});

		registerTool({
			name: "rag_libraries",
			description: "列出所有向量库及其大小、条数、被哪些会话绑定。",
			parameters: { type: "object", additionalProperties: false, properties: {} },
			output: textOut,
			async execute() {
				const list = manager.listLibraries();
				if (!list.length) return "还没有任何向量库。";
				return (
					`共 ${list.length} 个库：\n` +
					list
						.map(
							(l) =>
								`- ${l.id}（${l.name}）${l.dims ?? "?"} 维 / ${(l.bytes / 1024 / 1024).toFixed(2)}MB / 绑定 ${l.bound} 个会话${l.sessions.length ? "：" + l.sessions.slice(0, 3).join(", ") : ""}`,
						)
						.join("\n")
				);
			},
		});

		registerTool({
			name: "rag_use",
			description: "把当前会话绑定到指定向量库（之后所有回忆/写入都走这个库）。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: { library: { type: "string", description: "库 id，或用 new 新建一个" } },
				required: ["library"],
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				if (!sid) return "拿不到当前会话 id。";
				if (String(args.library) === "new") {
					const row = manager.createLibrary("新库");
					manager.bind(sid, row.id);
					recallCache.delete(sid);
					return `已新建并绑定：${row.id}`;
				}
				const row = manager.index.getLibrary(String(args.library));
				if (!row) return `没有这个库：${args.library}`;
				manager.bind(sid, row.id);
				recallCache.delete(sid);
				return `会话 ${sid} 已绑定到库 ${row.id}（${row.name}）。`;
			},
		});

		registerTool({
			name: "rag_stats",
			description: "查看某个库（默认当前会话的库）的统计：条数、簇数、边数、占用空间、分类分布。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: { library: { type: "string" } },
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				const libId = args.library || (sid ? manager.index.binding(sid) : null) || manager.listLibraries()[0]?.id;
				if (!libId) return "还没有任何向量库。";
				const st = await manager.statsFor(libId);
				if (!st) return `库不存在：${libId}`;
				return [
					`库 ${st.id}（${st.name}）`,
					`条数 ${st.mem} / 词条 ${st.terms} / 边 ${st.edges} / 簇 ${st.clusters}`,
					`维度 ${st.dims}，占用 ${(st.bytes / 1024).toFixed(0)}KB，平均每条 ${(st.bytesPerMem / 1024).toFixed(1)}KB`,
					`分类：${st.kinds.map((k) => `${k.kind}=${k.n}`).join(" ") || "无"}`,
				].join("\n");
			},
		});

		registerTool({
			name: "rag_recent",
			description: "按时间倒序看某个库最近写入的记忆条目。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					library: { type: "string" },
					limit: { type: "number", description: "默认 15" },
					offset: { type: "number" },
				},
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				const libId = args.library || (sid ? manager.index.binding(sid) : null) || manager.listLibraries()[0]?.id;
				const lib = libId ? await manager.libraryAsync(libId) : null;
				if (!lib) return "没有可用的向量库。";
				const rows = lib.recent(Number(args.limit ?? 15), Number(args.offset ?? 0));
				if (!rows.length) return "库是空的。";
				return rows
					.map((r) => {
						const day = new Date(r.ts).toISOString().slice(0, 16).replace("T", " ");
						return `#${r.id} [${day} · ${r.kind} · 簇${r.cluster ?? "-"}] ${r.text}`;
					})
					.join("\n");
			},
		});

		registerTool({
			name: "rag_forget",
			description: "按 id 删除记忆条目。",
			parameters: {
				type: "object",
				additionalProperties: false,
				properties: {
					library: { type: "string" },
					ids: { type: "array", items: { type: "number" }, description: "要删除的记忆 id" },
				},
				required: ["ids"],
			},
			output: textOut,
			async execute(args, exec) {
				const sid = sessionIdOf(exec);
				const libId = args.library || (sid ? manager.index.binding(sid) : null);
				const lib = libId ? await manager.libraryAsync(libId) : null;
				if (!lib) return "没有可用的向量库。";
				const n = lib.forget(args.ids ?? []);
				return `已删除 ${n} 条，剩余 ${lib.db.count()} 条。`;
			},
		});

		registerTool({
			name: "rag_summarize",
			description: "立刻把未激活缓冲里的对话激活成节点（原文按说话人拆包 + 摘要压缩，各一次，不等计时）。",
			parameters: { type: "object", additionalProperties: false, properties: {} },
			output: textOut,
			async execute(_args, exec) {
				const sid = sessionIdOf(exec);
				if (!sid) return "拿不到当前会话 id。";
				const r = await summarize(sid, { force: true });
				if (!r) return "缓冲里没有待激活的对话（也可能正在压缩中，稍等再试）。";
				if (r.error) return `激活失败：${r.error}`;
				return `已激活 ${r.summarized} 条对话 → 原文 ${r.raw ?? 0} 个节点 + 摘要 ${r.stored} 条记忆。\n${(r.items ?? []).map((i) => "- " + i).join("\n")}`;
			},
		});
	}

	/* ───────── HTTP 面板 ───────── */

	const webServer = ctx.get("webServer") ?? ctx.webServer;
	if (webServer?.register) {
		try {
			const dispose = webServer.register({
				kind: "prefix",
				path: "/fast-ivf-rag",
				handler: async (req, res) => {
					try {
						const url = new URL(req.url ?? "/", "http://localhost");
						const p = url.pathname;
						if (p === "/fast-ivf-rag" || p === "/fast-ivf-rag/" || p === "/fast-ivf-rag/index.html") {
							const html = readFileSync(UI_FILE, "utf8");
							res.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
							res.end(html);
							return;
						}
						if (!p.startsWith("/fast-ivf-rag/api/")) {
							res.writeHead(404, { "content-type": "text/plain" });
							res.end("not found");
							return;
						}
						const api = p.slice("/fast-ivf-rag/api/".length);
						const q = url.searchParams;

						if (api === "libraries") {
							json(res, 200, { libraries: manager.listLibraries(), bindings: await withTitles(manager.allBindings()) });
							return;
						}

						// ── 一键拉模型（2026-11-03）──
						// 姐姐要的：设置页点一下就把嵌入 / 精排模型下到该放的地方，点哪个下哪个。
						// GET 看状态（含正在跑的任务），POST 起一个后台下载。
						if (api === "models") {
							json(res, 200, {
								ok: true,
								base: strOf(cfg.modelDownloadBase) || DEFAULT_DOWNLOAD_BASE,
								models: allModelStatus({
									embedDir: strOf(cfg.embedModelDir) || undefined,
									rerankDir: strOf(cfg.rerankModelDir) || undefined,
									rerankVariant: strOf(cfg.rerankDtype) || "q8",
								}),
								jobs: allModelJobs().map(serializeJob),
							});
							return;
						}
						if (api === "models/download") {
							if (req.method !== "POST") return json(res, 405, { ok: false, error: "用 POST" });
							let body = {};
							try {
								body = await readBody(req);
							} catch (err) {
								return json(res, 400, { ok: false, error: String(err?.message ?? err) });
							}
							const which = String(body?.which ?? "").trim();
							if (!MODEL_SPECS[which]) return json(res, 400, { ok: false, error: `未知模型：${which || "(空)"}` });
							const variant = String(body?.variant ?? "").trim();
							try {
								const job = startModelDownload(which, {
									base: String(body?.base ?? "").trim() || strOf(cfg.modelDownloadBase) || DEFAULT_DOWNLOAD_BASE,
									dir: which === "embed" ? strOf(cfg.embedModelDir) || undefined : strOf(cfg.rerankModelDir) || undefined,
									variant: variant || strOf(cfg.rerankDtype) || undefined,
									force: Boolean(body?.force),
								});
								// 下完要能加载：把 rerankDtype 一起改成刚选的那个版本（只对 embed 以外生效）
								if (which === "rerank" && job.variant && job.variant !== (strOf(cfg.rerankDtype) || "q8")) {
									try {
										const settings = softService(ctx, "settings");
										if (settings?.update) await settings.update({ rerankDtype: job.variant });
										cfg.rerankDtype = job.variant;
										ctx.logger?.info?.(`[fast-ivf-rag] 精排版本切到 ${job.variant}（重启后按新 dtype 加载）`);
									} catch (err) {
										ctx.logger?.warn?.(`[fast-ivf-rag] 没能自动切 rerankDtype：${err?.message ?? err}`);
									}
								}
								ctx.logger?.info?.(`[fast-ivf-rag] 开始下载模型 ${which}${job.variant ? `/${job.variant}` : ""} → ${job.dir}`);
								return json(res, 200, { ok: true, job: serializeJob(job) });
							} catch (err) {
								return json(res, 500, { ok: false, error: String(err?.message ?? err) });
							}
						}
						// 一键去冗余：把语义上几乎同一条的记忆并成一条。
						// GET 只算不删（面板拿去做确认框），POST 才真删。
						//
						// 2026-11-01 姐姐发现的：同一个"花块了"能查出四五条，都是换个说法的同一件事。
						// 根因是 hydrate 把旧对话重压了一遍（已在水位线那边修掉），但库里已经堆下的要去。
						if (api === "dedupe") {
							const libId = String(q.get("lib") || "").trim();
							if (!libId) {
								json(res, 400, { ok: false, error: "缺 lib 参数" });
								return;
							}
							let lib = null;
							try {
								lib = await manager.libraryAsync(libId);
							} catch (error) {
								json(res, 500, { ok: false, error: error?.message ?? String(error) });
								return;
							}
							if (!lib) {
								json(res, 404, { ok: false, error: "库不存在" });
								return;
							}
							const threshold = Math.min(0.999, Math.max(0.5, Number(q.get("threshold")) || 0.92));
							const keep = q.get("keep") === "longest" ? "longest" : "newest";
							const cursor = Math.max(0, Number(q.get("cursor")) || 0);
							const batch = Math.max(0, Math.min(64, Number(q.get("batch")) || 4));
							let found = null;
							try {
								found = findDuplicateGroups(lib, {
									threshold,
									keep,
									fromCluster: cursor,
									// 预览分批扫（进度条要动）；真删那一次一次扫完，要一份确定完整的名单
									clusterBatch: req.method === "POST" ? 0 : batch,
								});
							} catch (error) {
								json(res, 500, { ok: false, error: `扫描失败：${error?.message ?? error}` });
								return;
							}
							const groups = found.groups;
							const totalDrop = groups.reduce((a, g) => a + g.drop.length, 0);
							if (req.method !== "POST") {
								json(res, 200, {
									ok: true,
									dryRun: true,
									lib: libId,
									threshold,
									keep,
									total: found.total,
									totalGroups: groups.length,
									totalDrop,
									groups: groups.slice(0, 100),
									// 进度用：前端拿 nextCluster 接着要下一批
									scanned: found.scanned,
									nextCluster: found.nextCluster,
									totalClusters: found.totalClusters,
									done: found.done,
								});
								return;
							}
							let removed = 0;
							const failed = [];
							for (const g of groups) {
								for (const d of g.drop) {
									try {
										lib.db.delete(d.id);
										removed += 1;
									} catch (error) {
										failed.push({ id: d.id, error: error?.message ?? String(error) });
									}
								}
							}
							// 删完必须重建簇和边，否则簇中心还带着被删的条目
							if (removed > 0) {
								try {
									await lib.rebuild();
								} catch (error) {
									mark(`dedupe rebuild FAILED: ${error?.message ?? error}`);
								}
							}
							mark(`dedupe lib=${libId} threshold=${threshold} removed=${removed} failed=${failed.length}`);
							json(res, 200, {
								ok: true,
								dryRun: false,
								removed,
								failed,
								remaining: lib.db.count(),
							});
							return;
						}
						// 回收没有会话绑定的库：GET 只算不删（面板拿去弹确认框），POST 才真删
						if (api === "purge-unbound") {
							let all = [];
							try {
								all = manager.listLibraries();
							} catch (error) {
								json(res, 500, { ok: false, error: error?.message ?? String(error) });
								return;
							}
							const dead = all
								.filter((l) => !(Number(l.bound) || (Array.isArray(l.sessions) ? l.sessions.length : 0)))
								.map((l) => ({
									id: l.id,
									name: l.name,
									count: Number(l.count ?? 0),
									bytes: Number(l.bytes ?? 0),
								}));
							if (req.method !== "POST") {
								json(res, 200, { ok: true, dryRun: true, total: dead.length, libs: dead });
								return;
							}
							const removed = [];
							const failed = [];
							for (const l of dead) {
								try {
									manager.deleteLibrary(l.id);
									removed.push(l);
								} catch (error) {
									failed.push({ ...l, error: error?.message ?? String(error) });
								}
							}
							mark(`purge-unbound: removed=${removed.length} failed=${failed.length}`);
							json(res, 200, { ok: true, dryRun: false, removed, failed });
							return;
						}
						if (api === "sessions") {
							json(res, 200, { sessions: await listSessionsBrief({ refresh: q.get("refresh") === "1" }) });
							return;
						}
						if (api === "status") {
							const emb = await manager.embedder().catch(() => null);
							json(res, 200, {
								dataDir,
								modelDir,
								embedder: emb?.status?.() ?? { kind: "unknown" },
								rerank: manager.rerankStatus?.() ?? { enabled: false },
								worker: { enabled: useWorker },
								config: {
									libraryMode: strOf(cfg.libraryMode, "reuse"),
									autoRecall: Boolean(val(cfg.autoRecall, true)),
									autoSummarize: Boolean(val(cfg.autoSummarize, true)),
									activeWindow: Number(val(cfg.activeWindow, 16)),
									summarizeMinMessages: Number(val(cfg.summarizeMinMessages, 5)),
									summarizeMaxTake: Number(val(cfg.summarizeMaxTake, 24)),
									rawCapture: Boolean(val(cfg.rawCapture, false)),
									rawPackSize: Number(val(cfg.rawPackSize, DEFAULT_PACK)),
									recallAssocCount: Number(val(cfg.recallAssocCount, 3)),
									recallAssocPerHit: Number(val(cfg.recallAssocPerHit, 1)),
									assocMode: strOf(cfg.assocMode, "weighted"),
									recallMinScore: Number(val(cfg.recallMinScore, 0.35)),
									minScoreMode: strOf(cfg.minScoreMode, "dynamic"),
									minScoreFloor: Number(val(cfg.minScoreFloor, 0.35)),
									minScoreRatio: Number(val(cfg.minScoreRatio, 0.55)),
									mmrLambda: Number(val(cfg.mmrLambda, 0.7)),
									useAssociation: Boolean(val(cfg.useAssociation, false)),
									recallTopK: Number(val(cfg.recallTopK, 5)),
									rerankEnabled: Boolean(val(cfg.rerankEnabled, true)),
									rerankTopN: Number(val(cfg.rerankTopN, 10)),
									rerankDtype: strOf(cfg.rerankDtype) || "q8",
									condenseMaxLen: Number(val(cfg.condenseMaxLen, 160)),
								},
							});
							return;
						}
						if (api === "config") {
							const settings = softService(ctx, "settings");
							if (req.method === "POST") {
								const body = await readBody(req);
								const patch = body?.patch ?? {};
								if (!settings?.update) return json(res, 400, { error: "settings 服务不可用：请到 设置 → 插件 → Fast IVF RAG 里改" });
								try {
									await settings.update("fast-ivf-rag", patch);
									// 改了精排相关开关要立刻作用到已打开的库，不然得等重启
									if ("rerankEnabled" in patch || "rerankTopN" in patch || "rerankModelDir" in patch || "condenseMaxLen" in patch) {
										manager.configure({
											rerankEnabled: Boolean(val(cfg.rerankEnabled, true)),
											rerankTopN: Number(val(cfg.rerankTopN, 10)),
											rerankDtype: strOf(cfg.rerankDtype) || "q8",
											rerankModelDir: strOf(cfg.rerankModelDir) || defaultRerankDir(),
											condenseMaxLen: Number(val(cfg.condenseMaxLen, 160)),
										});
									}
									recallCache.clear();
									return json(res, 200, { ok: true, patched: Object.keys(patch) });
								} catch (error) {
									return json(res, 400, { error: String(error?.message ?? error) });
								}
							}
							const values = {};
							for (const f of FIELDS) values[f.key] = val(cfg[f.key], null);
							return json(res, 200, { ns: "fast-ivf-rag", fields: FIELDS, values, canWrite: Boolean(settings?.update) });
						}
						if (api === "detail") {
							const id = q.get("id");
							const st = await manager.statsFor(id);
							if (!st) return json(res, 404, { error: "库不存在" });
							const lib = await manager.libraryAsync(id);
							return json(res, 200, { info: st, recent: lib ? lib.recent(30) : [], bindings: await withTitles(manager.index.bindingsOf(id)) });
						}
						if (api === "summarize-state") {
							const now = Date.now();
							const rows = [];
							for (const [sid, rec] of trackers) {
								rows.push({
									session: sid,
									msgs: rec.msgs.length,
									pending: rec.pending.length,
									idleSec: Math.round((now - rec.lastAt) / 1000),
									busy: Boolean(rec.busy),
									hydrated: hydrated.has(sid),
									emptyStreak: rec.emptyStreak,
									lastTake: rec.lastTake,
									lastStored: rec.lastStored,
									lastRaw: rec.lastRaw,
									lastActivateAt: rec.lastActivateAt || null,
									cooldownSec: rec.nextAt && rec.nextAt > now ? Math.round((rec.nextAt - now) / 1000) : 0,
									lib: recallCache.get(sid)?.lib ?? manager.index.binding(sid) ?? "",
								});
							}
							rows.sort((a, b) => (b.lastActivateAt ?? 0) - (a.lastActivateAt ?? 0));
							return json(res, 200, {
								activeWindow: Number(val(cfg.activeWindow, 16)),
								minMsgs: Number(val(cfg.summarizeMinMessages, 5)),
								maxTake: Number(val(cfg.summarizeMaxTake, 24)),
								rawCapture: Boolean(val(cfg.rawCapture, false)),
								rawPackSize: Number(val(cfg.rawPackSize, DEFAULT_PACK)),
								autoSummarize: Boolean(val(cfg.autoSummarize, true)),
								sessions: rows,
							});
						}
						if (api === "recalls") {
							const sid = String(q.get("session") ?? "");
							const libQ = String(q.get("lib") ?? "");
							if (!sid && !libQ) return json(res, 400, { error: "需要 session 或 lib" });
							const rep = recallReport({ session: sid, lib: libQ, limit: Number(q.get("limit") ?? 20) });
							// 把当前的权重补进去，面板才能显示"这条现在被压到什么程度"
							const byLib = new Map();
							for (const t of rep.top) {
								if (!byLib.has(t.lib)) byLib.set(t.lib, []);
								byLib.get(t.lib).push(t.id);
							}
							for (const [libId, ids] of byLib) {
								const lib = await manager.libraryAsync(libId).catch(() => null);
								if (!lib) continue;
								const wm = lib.db.weightsOf(ids);
								for (const t of rep.top) if (t.lib === libId) t.weight = wm.get(t.id) ?? 1;
							}
							return json(res, 200, rep);
						}
						if (api === "weight") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const lib = await manager.libraryAsync(body.lib);
							if (!lib) return json(res, 404, { error: "库不存在" });
							const id = Number(body.id);
							const w = body.w === undefined ? 1 : Number(body.w);
							lib.db.setWeight(id, w);
							return json(res, 200, { ok: true, id, w });
						}
						if (api === "vis") {
							const lib = await manager.libraryAsync(q.get("id"));
							if (!lib) return json(res, 404, { error: "库不存在" });
							const sample = Number(q.get("sample") ?? 240);
							const vis = lib.visualize({ sample });
							// 主题用全量算（可视化只是抽样），两件事分开做
							vis.topics = lib.clusterTopics({ topTerms: Number(q.get("terms") ?? 5) });
							return json(res, 200, vis);
						}
						if (api === "bind") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const session = String(body.session ?? q.get("session") ?? "");
							const lib = String(body.lib ?? q.get("lib") ?? "");
							if (!session || !lib) return json(res, 400, { error: "需要 session 和 lib" });
							if (!manager.index.getLibrary(lib)) return json(res, 404, { error: "库不存在" });
							manager.bind(session, lib);
							recallCache.delete(session);
							return json(res, 200, { ok: true });
						}
						if (api === "unbind") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const session = String(body.session ?? q.get("session") ?? "");
							manager.index.unbind(session);
							recallCache.delete(session);
							return json(res, 200, { ok: true });
						}
						if (api === "create") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const row = manager.createLibrary(String(body.name ?? q.get("name") ?? "记忆库"));
							return json(res, 200, { library: row });
						}
						if (api === "delete") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const id = String(body.id ?? q.get("id") ?? "");
							if (!id) return json(res, 400, { error: "需要 id" });
							manager.deleteLibrary(id);
							return json(res, 200, { ok: true });
						}
						if (api === "add") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const lib = await manager.libraryAsync(body.lib);
							if (!lib) return json(res, 404, { error: "库不存在" });
							const res2 = await addItems(lib, [{ text: body.text, kind: body.kind ?? "note" }]);
							return json(res, 200, res2);
						}
						if (api === "update") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const lib = await manager.libraryAsync(body.lib);
							if (!lib) return json(res, 404, { error: "库不存在" });
							const r = await lib.update(Number(body.mem), body.text, body.kind || null);
							if (r.error) return json(res, 400, { error: r.error });
							recallCache.clear();
							return json(res, 200, r);
						}
						if (api === "one") {
							const lib = await manager.libraryAsync(q.get("lib"));
							if (!lib) return json(res, 404, { error: "库不存在" });
							const row = lib.db.get(Number(q.get("mem")));
							if (!row) return json(res, 404, { error: "记忆不存在" });
							return json(res, 200, { mem: row });
						}
						if (api === "search") {
							const lib = await manager.libraryAsync(q.get("id"));
							if (!lib) return json(res, 404, { error: "库不存在" });
							// 检索测试：直命中数、联想池容量、每条命中贡献数都可现场调
							const r = await searchLibrary(lib, q.get("q") ?? "", {
								topK: Number(q.get("topK") ?? 6),
								assoc: q.get("assoc") !== "0",
								assocCount: Number(q.get("assocCount") ?? val(cfg.recallAssocCount, 3)),
								assocPerHit: Number(q.get("assocPerHit") ?? val(cfg.recallAssocPerHit, 1)),
								assocMode: q.get("assocMode") ?? strOf(cfg.assocMode, "weighted"),
								...minScoreOpts({
									mode: q.get("minScoreMode"),
									fixed: q.get("minScore") === null ? undefined : Number(q.get("minScore")),
								}),
								mmrLambda: q.get("mmrLambda") === null ? Number(val(cfg.mmrLambda, 0.7)) : Number(q.get("mmrLambda")),
							});
							return json(res, 200, r);
						}
						if (api === "forget") {
							const body = req.method === "POST" ? await readBody(req) : {};
							const lib = await manager.libraryAsync(body.lib);
							if (!lib) return json(res, 404, { error: "库不存在" });
							const n = lib.forget(body.ids ?? []);
							return json(res, 200, { removed: n });
						}
						if (api === "rebuild") {
							const lib = await manager.libraryAsync(q.get("id"));
							if (!lib) return json(res, 404, { error: "库不存在" });
							return json(res, 200, await lib.rebuild());
						}
						json(res, 404, { error: "unknown api" });
					} catch (error) {
						json(res, 500, { error: String(error?.message ?? error) });
					}
				},
			});
			if (typeof dispose === "function") disposers.push(dispose);
			ctx.logger?.info?.("[fast-ivf-rag] 面板已挂载：/fast-ivf-rag");
		} catch (error) {
			mark(`panel FAILED: ${error?.stack ?? error}`);
			ctx.logger?.warn?.(`[fast-ivf-rag] 挂载面板失败：${error?.message ?? error}`);
		}
	}

	/* ───────── 收尾 ───────── */

	try {
		ctx.on("dispose", () => {
			if (timer) clearInterval(timer);
			for (const d of disposers) {
				try {
					d();
				} catch {
					/* ignore */
				}
			}
			manager.close();
			// 故意不销毁召回线程：onnxruntime 一个进程只能有一个环境，
			// 卸载时 terminate 会让 OrtSession 在销毁/重建之间交叠，直接崩宿主
			// （FATAL ERROR: v8::HandleScope::CreateHandle()）。线程是 unref 的，
			// 进程退出时自然结束。下次 apply 会从 globalThis 取回同一个实例。
			mark("dispose: recall worker kept alive (shared singleton)");
		});
	} catch {
		/* ignore */
	}

	ctx.logger?.info?.(`[fast-ivf-rag] 已启动，数据目录 ${dataDir}`);
	mark("apply() done");
}

export { Config, apply, inject, name, textOfMessage, unwrapMessage };





