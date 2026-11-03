/**
 * 自动总结：把「已经离开活跃区」的对话压成一条可检索的记忆。
 * 默认 prompt 用第一人称（"我" = 这个助手），面向 bge-small-zh 检索：短事实句、带绝对日期。
 *
 * 注意两个已经踩过的坑：
 * 1) prompt 里的示例会被模型原样抄进输出（实测入过库："用户说喜欢我""群友A问我 Python 装饰器怎么写"），
 *    所以示例必须标成「格式示意、严禁照抄」，parseSummary 里还留了一道兜底过滤。
 * 2) rule 6 原来写「没有值得记的就只输出 - 无」，实测一半批次返回 "- 无"，节点根本不涨。
 *    现在改成"默认必须写、只有整段全是寒暄才允许无"。
 */

export const DEFAULT_SUMMARY_PROMPT = `你是「我」的记忆压缩器。「我」就是正在和你说话的这个助手；片段里 user 是对方，assistant 是「我」自己。群聊里 user 前面会带说话人昵称。

要求：
1. 一律用第一人称「我」来写，站在我的视角记。
   下面两条只是**格式示意**，内容全是假的，严禁原样出现在输出里：
   - ××说他明天要搬家
   - 我建议他把存储引擎换成 WAL 模式
2. 每条一行，以 "- " 开头，一条只讲一件事，控制在 40 字内。
   **不同说话人的事必须分开写成不同条目，不同话题也不许合成一条**；一条里只出现一个人名。
3. 重点保留：对方的偏好/情绪/诉求、我做过的决定和给出的结论、答应过的事、人名/群友名/项目名/文件名/数字。
   同时保留**我当时的态度和立场**：嫌弃、敷衍、认真、拒绝过什么、承诺过什么，每条都要写成事实。
   以后再遇到同一件事，我的反应要和这里记的一致，不许忽冷忽热。
4. 时间必须落成绝对日期（精确到天）：把"明天""后天""下周三"换成 YYYY-MM-DD，并保留原话线索。今天是 {{today}}。
5. 丢弃寒暄、重复、纯表情、临场情绪、已经被后文推翻的内容。
6. 输出 3~8 条（片段越长写得越多）。**只要片段里出现具体信息（人名/称呼/偏好/决定/约定/时间/数字/项目名/踩过的坑），就必须写下来**；只有整段真的全是寒暄、确实没有任何可复用信息时，才输出 "- 无"。
7. 不要编造，不要加标题，不要写成一段话，不要输出上面格式示意里的句子。

对话片段（{{range}}）：
{{messages}}`;

const ROLE_LABEL = { user: "用户", assistant: "我", system: "系统", tool: "工具" };

/** 示例句兜底黑名单：模型偶尔还是会照抄，归一化后比对，命中就丢弃。 */
const EXAMPLE_NOISE = new Set(
	[
		"用户说喜欢我",
		"用户表达了对我的喜爱",
		"群友a问我python装饰器怎么写",
		"群友a问了我python装饰器怎么写",
		"我建议他把sqlite换成wal模式",
		"xx说他明天要搬家",
		"我建议他把存储引擎换成wal模式",
	].map(norm),
);

function norm(s) {
	return String(s ?? "")
		.toLowerCase()
		.replace(/[\s\p{P}\p{S}]/gu, "");
}

export function formatMessages(messages) {
	return messages
		.map((m) => {
			const who = m.name ? String(m.name) : ROLE_LABEL[m.role] ?? m.role ?? "?";
			const text = String(m.text ?? "").trim().replace(/\s+/g, " ");
			const at = m.at ? ` [${new Date(m.at).toISOString().slice(0, 10)}]` : "";
			return text ? `${who}${at}: ${text}` : "";
		})
		.filter(Boolean)
		.join("\n");
}

export function buildSummaryPrompt(template, { messages, sessionLabel = "", range = "", today = "" }) {
	const tpl = String(template && template.trim() ? template : DEFAULT_SUMMARY_PROMPT);
	const body = formatMessages(messages);
	const day = today || new Date().toISOString().slice(0, 10);
	return tpl
		.replaceAll("{{messages}}", body)
		.replaceAll("{{session}}", sessionLabel)
		.replaceAll("{{range}}", range)
		.replaceAll("{{today}}", day);
}

/** 从模型输出里抽出 "- " 条目；抽不到就整段当一条。 */
export function parseSummary(text) {
	const raw = String(text ?? "").trim();
	if (!raw) return [];
	// 模型被要求「没有可记的就输出 - 无」——那是哨兵，不是一条记忆。
	// 注意别把它当成没有结构的散文再兜底成一条。
	const structured = /^\s*(?:[-*•]|\d+[.)])\s*/m.test(raw);
	const lines = raw
		.split(/\r?\n/)
		.map((l) => l.replace(/^\s*(?:[-*•]|\d+[.)])\s*/, "").trim())
		.filter((l) => l && l !== "无" && l !== "无。" && l !== "(无)" && l !== "（无）")
		// 示例句照抄兜底：整句归一化后命中黑名单就丢
		.filter((l) => !EXAMPLE_NOISE.has(norm(l)));
	const picked = lines.filter((l) => l.length >= 4).slice(0, 12);
	if (picked.length) return picked;
	if (structured) return [];
	const fallback = [raw.slice(0, 240)];
	return EXAMPLE_NOISE.has(norm(fallback[0])) ? [] : fallback;
}
