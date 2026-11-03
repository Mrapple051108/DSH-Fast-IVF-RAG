/**
 * 回归测试：模拟一段带昵称的群聊，走「格式化 → 总结 prompt → 解析 → 入节点 → 检索」全链路。
 *
 * 关注四件事：
 *  1. 昵称和日期有没有正确拼进 prompt（模型得看得见是谁说的、哪天说的）
 *  2. 思考块 / 工具调用 / 工具结果 会不会漏进 prompt（不该进）
 *  3. 总结返回的 `- ` 条目能不能被 parseSummary 拆成节点
 *  4. 节点真的写进库以后，能不能被检索/聚到簇
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { formatMessages, buildSummaryPrompt, parseSummary, DEFAULT_SUMMARY_PROMPT } from "./lib/summarize.js";
import { RagLibrary } from "./lib/library.js";
import { createLexicalEmbedder } from "./lib/embed.js";

const DAY = (d) => new Date(d).toISOString().slice(0, 10);

/** 一条 user/message 事件的样子（和宿主 session/event 的 data 同构） */
const userEvent = (name, id, text, iso) => ({
	name,
	id,
	at: iso,
	content: [{ type: "text", text: `[${name} #${id}] ${text}` }],
});

/** 一条 assistant/message：思考块 + 工具调用 + 真正的文本块 */
const assistantEvent = (text, iso) => ({
	at: iso,
	content: [
		{ type: "reasoning", text: "（这是思考块，不该进总结）先想一下他们问的是什么……" },
		{ type: "tool-call", name: "rag_search", input: { query: "环球影城" } },
		{ type: "text", text },
	],
});

/** 模拟宿主里 textOfMessage 的行为（只认 type === "text"） */
function textOfMessage(data) {
	const content = data?.content;
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((p) => p && p.type === "text" && typeof p.text === "string")
		.map((p) => String(p.text))
		.join("\n")
		.trim();
}

const T = (h, m) => new Date(2026, 9, 31, h, m).toISOString();

const events = [
	userEvent("苹果三块一斤", "10311", "明天去环球影城玩，早上九点集合", T(9, 12)),
	assistantEvent("记住了，明天（2026-11-01）九点环球影城，我提醒你。", T(9, 13)),
	userEvent("小云", "20011", "我不去，我要在家写作业，周三要交", T(9, 20)),
	assistantEvent("好，小云在家写作业，周三交。", T(9, 21)),
	userEvent("阿水", "30022", "我的猫叫团子，昨天又把我耳机咬坏了", T(10, 14, 2)),
	assistantEvent("团子记下了，耳机咬坏这事你上次也提过一次。", T(10, 14, 3)),
	userEvent("苹果三块一斤", "10311", "对了以后叫我老苹就行，别叫全名", T(16, 5)),
	assistantEvent("行，以后叫老苹。", T(16, 6)),
	userEvent("小云", "20011", "帮我把颈椎疼那个事记一下，热敷了三天好点了", T(16, 22)),
	assistantEvent("记下了：颈椎疼，热敷三天见好。", T(16, 23)),
];

const msgs = [];
for (let i = 0; i < events.length; i += 2) {
	const u = events[i];
	const a = events[i + 1];
	msgs.push({ role: "user", name: u.name, at: u.at, text: textOfMessage(u) });
	msgs.push({ role: "assistant", at: a.at, text: textOfMessage(a) });
}

const fail = [];
const ok = (cond, label, extra = "") => {
	console.log(`${cond ? "  ✓" : "  ✗"} ${label}${extra ? ` — ${extra}` : ""}`);
	if (!cond) fail.push(label);
};

console.log("== 1. 格式化 ==");
const formatted = formatMessages(msgs);
console.log(formatted.split("\n").map((l) => `   ${l}`).join("\n"));
ok(formatted.includes("苹果三块一斤"), "保留了昵称");
ok(formatted.includes("2026-10-31"), "带上了日期");
ok(!formatted.includes("思考块"), "思考块没漏进来");
ok(!formatted.includes("rag_search"), "工具调用没漏进来");
ok(formatted.includes("团子"), "正文抓到了");

console.log("\n== 2. 总结 prompt ==");
const prompt = buildSummaryPrompt(DEFAULT_SUMMARY_PROMPT, {
	messages: msgs,
	sessionLabel: "群聊 9686D572（回归测试）",
	range: "2026-10-31 09:12 → 16:23",
	today: "2026-10-31",
});
ok(prompt.includes("我"), "prompt 是第一人称口吻");
ok(prompt.includes("2026-10-31"), "注入了 {{today}}");
ok(prompt.includes("苹果三块一斤 [2026-10-31]"), "每条消息带 昵称 + 日期");
ok(!prompt.includes("思考块"), "prompt 里没有思考内容");
ok(prompt.includes("明天（2026-11-01）"), "相对时间换算要求已写明");
console.log(`   长度 ${prompt.length} 字符`);

console.log("\n== 3. 解析模型输出 ==");
const llmOut = [
	"- 老苹（苹果三块一斤）说明天（2026-11-01）九点去环球影城，让我提醒他",
	"- 小云不去环球影城，周三要交作业",
	"- 阿水的猫叫团子，昨天咬坏了耳机",
	"- 老苹让我以后别叫全名，叫老苹",
	"- 小云颈椎疼，热敷三天后见好",
	"- 我答应提醒老苹明天的集合时间",
].join("\n");
const parsed = parseSummary(llmOut);
console.log(`   解析出 ${parsed.length} 条：`);
parsed.forEach((p) => console.log(`   - ${p}`));
ok(parsed.length === 6, "六条全部解析出来");
ok(parsed.every((p) => !p.startsWith("-")), "前缀 `- ` 被剥掉");
ok(parseSummary("- 无").length === 0, "「- 无」被过滤");

console.log("\n== 4. 入库 + 建节点 ==");
const dir = mkdtempSync(join(tmpdir(), "fir-sum-"));
const lib = new RagLibrary({
	id: "regress-" + Date.now(),
	file: join(dir, "lib.sqlite"),
	embedder: createLexicalEmbedder(512),
	rebuildEvery: 2,
});

const written = await lib.add(parsed.map((text) => ({ text, kind: "summary", src: "session:regress" })));
ok(written.added === 6, "写入 6 条 summary", `added=${written.added} skipped=${written.skipped}`);

const again = await lib.add([{ text: parsed[0], kind: "summary", src: "session:regress" }]);
ok(again.added === 0 && again.skipped === 1, "重复内容被 hash 去重");

await lib.rebuild();
const info = lib.info();
console.log(`   info: ${info.mem} 条 / 簇 ${info.clusters} / 边 ${info.edges} / ${(info.bytes / 1024).toFixed(1)}KB`);
ok(info.mem === 6, "库里有 6 条");
ok(info.clusters >= 2, "聚类出了至少 2 个簇", `k=${info.clusters}`);
ok(info.edges > 0, "建了邻居边", `${info.edges} 条`);

console.log("\n== 5. 检索验证 ==");
for (const q of ["老苹明天几点集合", "团子是谁的猫", "小云颈椎怎么样了"]) {
	const r = await lib.search(q, { topK: 2 });
	const top = r.hits[0];
	console.log(`   「${q}」→ ${top ? top.text.slice(0, 34) : "（无命中）"}`);
	ok(Boolean(top), `命中：${q}`);
}

console.log("\n== 6. 第一人称归属 ==");
const mine = parsed.filter((p) => p.startsWith("我"));
const theirs = parsed.filter((p) => !p.startsWith("我"));
console.log(`   我做的事：${mine.length} 条；记录别人的话：${theirs.length} 条`);
ok(mine.length >= 1 && theirs.length >= 1, "第一人称与第三人称都出现了");

lib.close();
rmSync(dir, { recursive: true, force: true });

console.log(fail.length ? `\nFAILED: ${fail.join(" | ")}` : "\nALL OK");
process.exit(fail.length ? 1 : 0);


