/**
 * 极简中英混合分词：拉丁词 + CJK 单字/双字。
 * 用于 BM25 词法通道 —— FTS5 的 trigram 只吃 >=3 字，中文两字词检索不到，
 * 所以这里自己切词、自己建 posting 表。
 */
const CJK_RUN = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/g;
const LATIN = /[a-z0-9_+#.]+/g;

const STOP = new Set([
	"的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "和", "与", "就", "都", "也",
	"不", "有", "这", "那", "个", "上", "下", "去", "来", "说", "吗", "呢", "吧", "啊", "呀",
	"the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were", "be", "it", "this",
	"that", "in", "on", "for", "with", "as", "at", "by", "i", "you", "we", "they",
]);

export function tokenize(text) {
	const s = String(text ?? "").toLowerCase();
	const out = [];
	for (const m of s.matchAll(LATIN)) {
		const w = m[0];
		if (w.length > 1 && !STOP.has(w)) out.push(w);
	}
	for (const m of s.matchAll(CJK_RUN)) {
		const run = m[0];
		if (run.length === 1) {
			if (!STOP.has(run)) out.push(run);
			continue;
		}
		for (let i = 0; i < run.length; i++) {
			const c = run[i];
			if (!STOP.has(c)) out.push(c);
			if (i + 1 < run.length) out.push(run.slice(i, i + 2));
		}
	}
	return out;
}

/** 词频表。 */
export function termFreq(text) {
	const tf = new Map();
	const len = tokenize(text);
	for (const t of len) tf.set(t, (tf.get(t) ?? 0) + 1);
	return { tf, len: len.length };
}
