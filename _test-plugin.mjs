import * as plugin from "./lib/index.js";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// 用临时目录跑，不往仓库/本机固定路径里写
const dataDir = join(tmpdir(), "fast-ivf-rag-selftest");
rmSync(dataDir, { recursive: true, force: true });

const tools = {};
const handlers = {};
const routes = {};
let section = null;

const llm = {
	listProviders: () => [{ name: "fake-provider", models: [{ id: "fake-model" }] }],
	async *stream(req) {
		console.log("[llm] provider=", req.provider, "model=", req.model, "promptLen=", req.messages[0].content[0].text.length);
		yield { type: "text-delta", text: "- 用户在做 fast-ivf-rag 插件\n" };
		yield { type: "text-delta", text: "- 用 sqlite 存向量，聚类后 IVF 检索\n" };
		yield { type: "finish" };
	},
};

const HIST = [];
for (let i = 0; i < 6; i++) {
	HIST.push({ type: "user/message", data: { content: [{ type: "text", text: `历史问题 ${i}：我们在聊什么` }] } });
	HIST.push({ type: "assistant/message", data: { content: [{ type: "text", text: `历史回答 ${i}：在聊 RAG 插件` }] } });
}
const sessionQuery = {
	async readSurface() {
		return { events: HIST };
	},
	async readTitleSnapshots(ids) {
		return ids.map((id) => ({ status: "fulfilled", value: { session: { id }, title: `标题-${String(id).slice(0, 6)}` } }));
	},
	async listSessions() {
		return [{ header: { id: "sess-test-1" } }, { header: { id: "sess-hist-9" } }, { header: { id: "sess-other-3" } }];
	},
};

const ctx = {
	logger: { info: (m) => console.log("[info]", m), warn: (m) => console.log("[WARN]", m), debug: () => {} },
	get(name) {
		if (name === "tools") return { register: (d) => { tools[d.name] = d; return () => {}; } };
		if (name === "systemPrompt") return { section: (s) => { section = s; return () => {}; } };
		if (name === "webServer") return { register: (r) => { routes[r.path] = r; return () => {}; } };
		if (name === "llm") return llm;
		if (name === "settings") return { update: async (ns, patch) => { console.log("[settings.update]", ns, JSON.stringify(patch)); } };
		if (name === "sessionQuery") return sessionQuery;
		return undefined;
	},
	on(ev, fn) { handlers[ev] = fn; return () => {}; },
};

console.log("== plugin meta ==", plugin.name, JSON.stringify(plugin.inject));
console.log("== schema ok ==", typeof plugin.Config === "function" || typeof plugin.Config === "object");

plugin.apply(ctx, {
	dataDir,
	embedModelDir: "C:/Users/17189/.dsh/models/bge-small-zh",
	activeWindow: 3,
	summarizeMinMessages: 2,
	summarizeIdleMinutes: 0,
});

console.log("== tools ==", Object.keys(tools).join(","));
console.log("== routes ==", Object.keys(routes).join(","));
console.log("== section bound ==", section ? section.name : "(未挂载，等首条消息)");

const session = { id: "sess-test-1" };
const exec = { session: { id: "sess-test-1" } };

// 1) 造 6 条对话（活跃区 3 → 3 条溢出）
const conv = [
	["user", "我想给 dsh 写个 RAG 插件"],
	["assistant", "可以，用 sqlite 存向量，聚类做 IVF 粗筛"],
	["user", "聚类用什么算法"],
	["assistant", "kmeans++ 初始化，点积当相似度"],
	["user", "检索要混合 BM25 吗"],
	["assistant", "BM25 + 向量 RRF 融合效果更好"],
];
for (const [role, text] of conv) {
	handlers["session/event"](session, {
		type: role === "user" ? "user/message" : "assistant/message",
		data: { role, content: [{ type: "text", text }] },
	});
}
console.log("== section after msg ==", section ? section.name : "none");

await new Promise((r) => setTimeout(r, 1500));
const recall = section?.text?.({ agent: { session: { id: "sess-test-1" } } });
console.log("== recall text ==", JSON.stringify(recall ?? "").slice(0, 240));

// 2) 工具冒烟
console.log("\n== rag_add ==");
console.log(await tools.rag_add.execute({ texts: ["用户喜欢冰美式不加糖", "项目叫 fast-ivf-rag"] }, exec));
console.log("\n== rag_search ==");
console.log(await tools.rag_search.execute({ query: "用户在做什么插件" }, exec));
console.log("\n== rag_libraries ==");
console.log(await tools.rag_libraries.execute({}, exec));
console.log("\n== rag_stats ==");
console.log(await tools.rag_stats.execute({}, exec));
console.log("\n== rag_summarize ==");
console.log(await tools.rag_summarize.execute({}, exec));
// 2.5) 历史补录：新会话、缓冲为空，靠 sessionQuery 回读 12 条历史再总结
console.log("\n== rag_summarize（回读历史）==");
const histExec = { session: { id: "sess-hist-9" } };
console.log(await tools.rag_summarize.execute({}, histExec));
console.log("== 再压一次（应无溢出）==");
console.log(await tools.rag_summarize.execute({}, histExec));

// 2.6) 改一条记忆
console.log("\n== rag_edit ==");
console.log(await tools.rag_edit.execute({ id: 1, text: "用户喜欢冰美式，不加糖，夏天要加冰" }, exec));
console.log("== rag_search（改后）==");
console.log(await tools.rag_search.execute({ query: "他喝什么咖啡", topK: 2 }, exec));

console.log("\n== rag_recent ==");
console.log(tools.rag_recent.output.render({ limit: 5 }, await tools.rag_recent.execute({ limit: 5 }, exec)).map((b) => b.text).join("\n"));

// 库里有货之后再发一条用户消息 → 自动召回该生效了
handlers["session/event"](session, {
	type: "user/message",
	data: { role: "user", content: [{ type: "text", text: "我们那个插件检索用什么算法来着" }] },
});
await new Promise((r) => setTimeout(r, 900));
console.log("\n== recall after memories ==");
console.log(JSON.stringify(section?.text?.({ agent: { session: { id: "sess-test-1" } } }) ?? ""));

// 3) HTTP 面板
const route = routes["/fast-ivf-rag"];
async function call(path) {
	return new Promise((resolve) => {
		const res = {
			code: 0,
			body: "",
			writeHead(c) { this.code = c; },
			end(b) { this.body = String(b ?? ""); resolve({ code: this.code, body: this.body }); },
		};
		route.handler({ url: path, method: "GET" }, res);
	});
}
const r1 = await call("/fast-ivf-rag/api/libraries");
console.log("\n== GET libraries ==", r1.code, r1.body.slice(0, 300));
{
	const parsed = JSON.parse(r1.body);
	const withTitle = (parsed.bindings ?? []).filter((b) => b.title);
	console.log("== bindings 带标题 ==", `${withTitle.length}/${(parsed.bindings ?? []).length}`, withTitle[0] ? `例: [${withTitle[0].title}]` : "（无绑定）");
	const rs = await call("/fast-ivf-rag/api/sessions");
	const sj = JSON.parse(rs.body);
	console.log("== GET sessions ==", rs.code, `共 ${(sj.sessions ?? []).length} 个`, (sj.sessions ?? []).map((s) => `[${s.title || s.id.slice(0, 6)}]`).join(" "));
}
const libId = JSON.parse(r1.body).libraries[0]?.id;
const r2 = await call(`/fast-ivf-rag/api/vis?id=${encodeURIComponent(libId)}&sample=50`);
console.log("== GET vis ==", r2.code, r2.body.slice(0, 200));
const r3 = await call("/fast-ivf-rag/api/detail?id=" + encodeURIComponent(libId));
console.log("== GET detail ==", r3.code, r3.body.slice(0, 300));
const r4 = await call("/fast-ivf-rag");
console.log("== GET ui ==", r4.code, r4.body.slice(0, 60).replace(/\s+/g, " "), "len", r4.body.length);
const r5 = await call("/fast-ivf-rag/api/config");
console.log("== GET config ==", r5.code, r5.body.slice(0, 200));
const w1 = await post("/fast-ivf-rag/api/config", { patch: { autoRecall: false, activeWindow: 12, dataDir: "" } });
console.log("== POST config ==", w1.code, w1.body);

async function post(path, body) {
	const raw = JSON.stringify(body ?? {});
	const req = {
		url: path,
		method: "POST",
		[Symbol.asyncIterator]() {
			let done = false;
			return {
				async next() {
					if (done) return { done: true, value: undefined };
					done = true;
					return { done: false, value: Buffer.from(raw, "utf8") };
				},
			};
		},
	};
	return new Promise((resolve) => {
		const res = {
			code: 0,
			body: "",
			writeHead(c) { this.code = c; },
			end(b) { this.body = String(b ?? ""); resolve({ code: this.code, body: this.body }); },
		};
		route.handler(req, res);
	});
}

// 4) 增 / 改 / 查单条 / 删
const add = await post("/fast-ivf-rag/api/add", { lib: libId, text: "面板手动加的一条记忆", kind: "note" });
console.log("== POST add ==", add.code, add.body);
const memId = JSON.parse(add.body).ids?.[0];
const one = await call(`/fast-ivf-rag/api/one?lib=${encodeURIComponent(libId)}&mem=${memId}`);
console.log("== GET one ==", one.code, one.body.slice(0, 160));
const upd = await post("/fast-ivf-rag/api/update", { lib: libId, mem: memId, text: "面板改过的一条记忆" });
console.log("== POST update ==", upd.code, upd.body);
const one2 = await call(`/fast-ivf-rag/api/one?lib=${encodeURIComponent(libId)}&mem=${memId}`);
console.log("== GET one（改后）==", one2.code, one2.body.slice(0, 160));
const del = await post("/fast-ivf-rag/api/forget", { lib: libId, ids: [memId] });
console.log("== POST forget ==", del.code, del.body);

// 5) 精排 + condense：没装模型时必须是 model-missing，且检索照常返回
const { createReranker } = await import("./lib/rerank.js");
const { condense } = await import("./lib/condense.js");
const rk = createReranker({ modelDir: "C:/definitely/not/here" });
if (rk.available() !== false) throw new Error("rerank 未加载时 available() 应为 false");
if (await rk.score("q", ["d"]) !== null) throw new Error("模型缺失时 score 应返回 null");
const srch = await call(`/fast-ivf-rag/api/search?id=${encodeURIComponent(libId)}&q=${encodeURIComponent("插件")}&topK=3`);
const sjson = JSON.parse(srch.body);
console.log("== search meta.rerank ==", sjson.meta.rerank);
if (!["off", "model-missing", "unavailable", "on"].includes(sjson.meta.rerank)) throw new Error("meta.rerank 取值非法");
const st = JSON.parse((await call("/fast-ivf-rag/api/status")).body);
console.log("== status.rerank ==", JSON.stringify(st.rerank));
if (!st.rerank || st.rerank.enabled === undefined) throw new Error("status 缺 rerank 状态");
const cnd = condense("先说了一堆无关的话。然后明天要去环球影城，问我要不要一起。最后说喜欢我。", "明天去哪玩", { maxLen: 40 });
console.log("== condense ==", cnd);
if (!cnd || !cnd.includes("环球影城")) throw new Error("condense 没保住相关从句");

console.log("\nALL OK");
process.exit(0);
