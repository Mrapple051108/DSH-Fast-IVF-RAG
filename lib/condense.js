/**
 * 精排前的压缩（对应原项目 association_pipline.py 的 CONDENSE_BEFORE_RERANK）。
 * 思路：长记忆里真正相关的只有一两个从句，先按查询关键词把句子筛出来，
 * 既让 cross-encoder 判得更准，也省它的 token/算力。
 */
import { tokenize } from "./tokenize.js";

/** 情感/态度词：命中一个就算"有信息量"的从句。 */
const EMOTION = new Set([
	"喜欢", "爱", "讨厌", "烦", "生气", "开心", "高兴", "难受", "难过", "累", "爽", "怕", "担心",
	"期待", "失望", "焦虑", "兴奋", "感谢", "抱歉", "抱歉", "生气", "心疼", "舍不得", "太棒", "垃圾",
	"牛逼", "离谱", "操", "卧槽", "笑死", "哈哈", "撒娇", "哄", "夸",
]);
const EMOTION_CHARS = new Set("喜怒哀乐爱恨怕累爽烦气".split(""));

function hasEmotion(words) {
	for (const w of words) {
		if (EMOTION.has(w)) return true;
		if (w.length === 1 && EMOTION_CHARS.has(w)) return true;
	}
	return false;
}

/**
 * 按查询把长文本压成短句。
 * 规则（对齐原项目 condense）：保留「命中 ≥2 个查询词」或「命中 1 个且带情感」的从句，按原序拼。
 * 没有命中的从句就退化成截断，绝不返回空串。
 */
export function condense(text, query = "", { maxLen = 160 } = {}) {
	const raw = String(text ?? "").trim();
	if (!raw) return "";
	if (raw.length <= maxLen) return raw;
	const qwords = new Set(tokenize(String(query ?? "")));
	const clauses = raw
		.split(/[。！？；\n，、,;!?]/)
		.map((s) => s.trim())
		.filter(Boolean);
	const keep = [];
	for (const c of clauses) {
		const words = tokenize(c);
		let hits = 0;
		for (const w of words) if (qwords.has(w)) hits += 1;
		if (hits >= 2 || (hits >= 1 && hasEmotion(words))) keep.push(c);
	}
	let out = "";
	for (const c of keep) {
		if (!out && c.length > maxLen) return c.slice(0, maxLen);
		if (out.length + c.length + 1 > maxLen) break;
		out += (out ? "，" : "") + c;
	}
	if (out) return out;
	return raw.slice(0, maxLen).trim() + "…";
}
