/**
 * 回归测试：assistant 的回应必须能进缓冲区。
 *
 * 踩过的坑（2026-11-02）：宿主两种会话事件的形状不一样
 *   dsh-session/lib/index.js:221-228
 *     case "user/message":      return event.data;           ← data 本身就是 message
 *     case "assistant/message": return event.data.message;   ← 包了一层 { turn, step, message, usage, stream }
 * 以前 textOfMessage 只读 data.content，assistant 每条都取到 undefined → "" →
 * pushMessage 里 `if (!t) return` 直接丢弃。群里就是"别人的话全记着，小肥鱼的回应一条没有"。
 *
 * 本测试锁四件事：
 *  1. user / assistant 两种形状都要抠得出正文
 *  2. 带 tool-call 的过程步（"我查一下"+工具）不能当回答记进去
 *  3. 思考块 / 系统噪音照旧不进缓冲
 *  4. 回应进 prompt 时带「我」标签，模型能分清谁说的
 */
import { textOfMessage, unwrapMessage } from "./lib/index.js";
import { formatMessages, buildSummaryPrompt, parseSummary } from "./lib/summarize.js";

const fail = [];
function ok(cond, msg, extra = "") {
	console.log(`  ${cond ? "✔" : "✘"} ${msg}${extra ? `  (${extra})` : ""}`);
	if (!cond) fail.push(msg);
}
const eq = (a, b) => a === b;

/** user/message 的 data：message 本体 */
const userData = (text) => ({ role: "user", content: [{ type: "text", text }] });

/** assistant/message 的 data：包了一层，content 在 .message.content */
const assistantData = (blocks) => ({
	turn: 3,
	step: 2,
	message: { role: "assistant", content: blocks, source: { provider: "deepseek", model: "deepseek-flash" } },
	usage: { input: 1, output: 1 },
	stream: { frames: [] },
});

console.log("\n== 1. user/message 形状 ==");
ok(eq(textOfMessage(userData("你们好吵啊")), "你们好吵啊"), "user 取到正文");
ok(eq(textOfMessage({ ...userData("你"), message: "不该被挖出来" }), "你"), "user 自带 message 字段时不误挖");
ok(eq(textOfMessage({ role: "user", content: "纯字符串" }), "纯字符串"), "content 是字符串也吃");

console.log("\n== 2. assistant/message 形状（本次修的核心）==");
const a1 = textOfMessage(assistantData([{ type: "text", text: "哼，才不是特意为你查的呢" }]));
ok(eq(a1, "哼，才不是特意为你查的呢"), "assistant 取到正文", `got=${JSON.stringify(a1)}`);
ok(a1.length > 0, "assistant 不再返回空串（这就是被丢掉的根因）");

console.log("\n== 3. 过程步必须排除 ==");
const mixed = assistantData([
	{ type: "text", text: "我查一下" },
	{ type: "tool-call", toolName: "fs_read", args: {} },
]);
ok(eq(textOfMessage(mixed), ""), "带 tool-call 的 step 不算回答");
ok(eq(textOfMessage(assistantData([{ type: "tool-call", toolName: "web_search" }])), ""), "纯 tool-call 的 step 也不记");
const lastStep = assistantData([{ type: "text", text: "结论是端口被占用了" }]);
ok(eq(textOfMessage(lastStep), "结论是端口被占用了"), "最后一步（无 tool-call）正常记下");

console.log("\n== 4. 思考块 / 系统噪音 ==");
ok(
	eq(textOfMessage(assistantData([{ type: "reasoning", text: "用户可能想让" }, { type: "text", text: "行，给你" }])), "行，给你"),
	"reasoning 块不进来",
);
ok(
	eq(textOfMessage(userData("正事\nCurrent working directory: D:\\x\n继续")), "正事\n继续"),
	"runtime context 噪音行照旧剥掉",
);

console.log("\n== 5. unwrapMessage 本身 ==");
ok(unwrapMessage(userData("a")).content[0].text === "a", "user 不展开");
ok(Array.isArray(unwrapMessage(assistantData([])).content), "assistant 展开到 message");
ok(eq(unwrapMessage({ message: "裸字符串" }).content, "裸字符串"), "message 是字符串时兜住");
ok(eq(unwrapMessage(null), null), "null 不炸");
ok(Object.keys(unwrapMessage({})).length === 0, "无 message 时原样返回");

console.log("\n== 6. 拼进 prompt 时的身份标签 ==");
const batch = [
	{ role: "user", text: "你帮我看看这个报错", name: "老苹果", at: Date.parse("2026-11-02T01:00:00Z") },
	{ role: "assistant", text: "啧，端口被占用了，关掉再开", at: Date.parse("2026-11-02T01:00:20Z") },
];
const body = formatMessages(batch);
console.log(`   ${body.replace(/\n/g, "\n   ")}`);
ok(/"老苹果.*：你帮我看看这个报错|老苹果.*: 你帮我看看这个报错/.test(body), "user 带昵称");
ok(/我.*: 啧，端口被占用了/.test(body), "assistant 标成「我」");

console.log("\n== 7. 端到端：回应能一路走到节点 ==");
const prompt = buildSummaryPrompt("", { messages: batch, range: "x~y", today: "2026-11-02" });
ok(prompt.includes("啧，端口被占用了"), "prompt 里看得见我的回应");
ok(prompt.includes("态度和立场"), "prompt 要求保留我的态度");
const nodes = parseSummary("- 我嫌这需求烦，但还是帮他查了\n- 老苹果在 2026-11-02 问过端口报错");
ok(nodes.length === 2 && nodes[0].startsWith("我"), "回应压成了可检索的节点", `n=${nodes.length}`);

console.log(fail.length ? `\nFAILED: ${fail.join(" | ")}` : "\nALL OK");
process.exit(fail.length ? 1 : 0);
