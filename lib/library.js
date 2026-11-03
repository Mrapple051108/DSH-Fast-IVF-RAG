import { createHash } from "node:crypto";
import { statSync } from "node:fs";
import { LibraryDb } from "./store.js";
import { chooseK, dot, kmeans } from "./kmeans.js";
import { termFreq, tokenize } from "./tokenize.js";
import { condense } from "./condense.js";

const K1 = 1.5;
const B = 0.75;
const RRF_K = 60;

/**
 * 按直命中终分分配联想名额。
 *
 * 思路：终分归一到最大值为 1 → 平方（放大头部差距）→ 低于 0.2 的直接归零，
 * 剩下的按 w/Σw 分 capacity 个名额；向下取整后的余数按小数部分从大到小补。
 *
 * 例：capacity=8，终分 [0.9, 0.6, 0.35, 0.3]
 *   norm [1, .667, .389, .333] → w [1, .444, .151, .111]
 *   精确配额 [4.69, 2.08, 0.71, 0.52] → 取整 [4,2,0,0] → 余 2 条补小数大的 → [4,2,1,1]
 */
function quotaByScore(hits, capacity) {
	const n = hits.length;
	const scores = hits.map((h) => Math.max(0, Number(h.final ?? 0)));
	const max = Math.max(...scores, 0);
	if (!(max > 0) || capacity <= 0) return new Array(n).fill(0);
	const w = scores.map((s) => {
		const norm = s / max;
		return norm < 0.2 ? 0 : norm * norm; // 分数太低的直接不联想
	});
	const sum = w.reduce((a, b) => a + b, 0);
	if (!(sum > 0)) return new Array(n).fill(0);
	const exact = w.map((x) => (capacity * x) / sum);
	const quota = exact.map((x) => Math.floor(x));
	let left = capacity - quota.reduce((a, b) => a + b, 0);
	// 余数分两步走：
	// ① 先把"一条都没分到"的头部补到 1 条 —— 分数略低但够格的也该有联想，
	//    否则容量一大就全被第一名吃掉；
	// ② 还有剩就按标准最大余数法（小数部分大的先补）。
	const starved = exact
		.map((x, i) => ({ i, frac: x - Math.floor(x), w: w[i] }))
		.filter((o) => quota[o.i] === 0 && o.w > 0)
		.sort((a, b) => b.w - a.w || b.frac - a.frac);
	for (const o of starved) {
		if (left <= 0) break;
		quota[o.i] += 1;
		left -= 1;
	}
	if (left > 0) {
		const byFrac = exact
			.map((x, i) => ({ i, frac: x - Math.floor(x), w: w[i] }))
			.filter((o) => o.w > 0)
			.sort((a, b) => b.frac - a.frac || b.w - a.w);
		let guard = 0;
		while (left > 0 && byFrac.length && guard++ < 100) {
			for (const o of byFrac) {
				if (left <= 0) break;
				quota[o.i] += 1;
				left -= 1;
			}
		}
	}
	return quota;
}

/**
 * MMR 多样性重排（Maximal Marginal Relevance）。
 *
 * 每轮挑「和查询够像、但和已选条目不像」的那条：
 *   score = λ·sim(q,d) − (1−λ)·max(sim(d, 已选))
 * λ=1 退化成按相关度排；λ 越小越惩罚重复。默认 0.7 —— 比单纯砍 topK 更省，
 * 因为它是在**同样的候选池**里换掉冗余项，而不是把整批裁短。
 */
function mmrSelect(pool, k, lambda, vecById) {
	const rest = pool.slice();
	const chosen = [];
	// 预先归一化：候选向量在库里通常已归一，但 fake/降级路径不一定，自己算一遍更稳
	const unit = new Map();
	for (const p of pool) {
		const v = vecById.get(p.row.id);
		if (!v) continue;
		let n = 0;
		for (let i = 0; i < v.length; i++) n += v[i] * v[i];
		n = Math.sqrt(n) || 1;
		const u = new Float32Array(v.length);
		for (let i = 0; i < v.length; i++) u[i] = v[i] / n;
		unit.set(p.row.id, u);
	}
	const cos = (a, b) => {
		if (!a || !b || a.length !== b.length) return 0;
		let s = 0;
		for (let i = 0; i < a.length; i++) s += a[i] * b[i];
		return s;
	};
	while (chosen.length < k && rest.length) {
		let bestAt = 0;
		let bestScore = -Infinity;
		for (let i = 0; i < rest.length; i++) {
			const cand = rest[i];
			const uv = unit.get(cand.row.id);
			let maxRedundant = 0;
			for (const c of chosen) {
				const s = cos(uv, unit.get(c.row.id));
				if (s > maxRedundant) maxRedundant = s;
			}
			const score = lambda * cand.final - (1 - lambda) * maxRedundant;
			if (score > bestScore) {
				bestScore = score;
				bestAt = i;
			}
		}
		chosen.push(rest.splice(bestAt, 1)[0]);
	}
	return chosen;
}

export function textHash(text) {
	return createHash("sha1").update(String(text).trim().replace(/\s+/g, " ")).digest("hex").slice(0, 24);
}

const CJK_CHAR = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;

/** 主题词里没信息量的高频碎片（tokenize 里那套 STOP 只管单字，这里补二字虚词）。 */
const THEME_STOP = new Set([
	"今天", "明天", "昨天", "然后", "一个", "可以", "什么", "就是", "这个", "那个", "我们", "你们",
	"他们", "时候", "因为", "所以", "但是", "还是", "已经", "没有", "不是", "一下", "真的", "自己",
	"现在", "知道", "觉得", "应该", "东西", "事情", "问题", "有点", "好像", "怎么", "这些", "那些",
	"用户", "对方", "记录", "内容", "消息", "这条", "那条", "上面", "下面", "里面", "之前", "之后",
	"一直", "可能", "需要", "记得", "说过", "说要", "让我", "我的", "你的", "他的", "就是", "还是",
	"一起", "有点", "的话", "了吧", "了吗", "是在", "是不", "不在", "不要", "是不是", "用户说", "我在这",
]);

const CJK_RUN = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/g;
const LATIN_RUN = /[a-z0-9_+#.-]{2,}/gi;

/**
 * 一条记忆的主题候选：拉丁词 + 2~4 字中文 n-gram。
 * 用 n-gram 而不是 tokenize 的双字：双字会把"三角洲行动"切成
 * 三角/角洲/洲行/行动 四个碎片，看不出主题。
 */
function themeTermsOf(text) {
	const s = String(text ?? "");
	const out = new Set();
	for (const m of s.matchAll(LATIN_RUN)) {
		const w = m[0].toLowerCase();
		if (/^[\d._+#-]+$/.test(w)) continue; // 2026-10-31 / 3.14 这类日期和数字不是主题
		out.add(w);
	}
	// 只取"贴着标点/句首句尾"的片段：跨词边界的碎片（"斤让我""包发出"）天然被排除，
	// 而真正的实体几乎总会在某条记忆里以边界形式出现至少一次。
	for (const m of s.matchAll(CJK_RUN)) {
		const run = m[0];
		for (let n = 2; n <= 6; n++) {
			for (let i = 0; i + n <= run.length; i++) {
				if (i !== 0 && i + n !== run.length) continue;
				const g = run.slice(i, i + n);
				if (THEME_STOP.has(g)) continue;
				out.add(g);
			}
		}
	}
	return out;
}

/** 最长公共子串长度（只比短串，够判"是不是同一个主题的错位切片"了）。 */
function commonRunLength(a, b) {
	const x = String(a);
	const y = String(b);
	if (x.length > y.length) return commonRunLength(y, x);
	let best = 0;
	for (let i = 0; i < x.length; i++) {
		for (let j = 0; j < y.length; j++) {
			let k = 0;
			while (i + k < x.length && j + k < y.length && x[i + k] === y[j + k]) k += 1;
			if (k > best) best = k;
		}
	}
	return best;
}

/**
 * 一个向量库的门面。逻辑对应 Cherry 版 association_pipline.py：
 * 写入 → 聚类分区（IVF）→ 簇内 dense+BM25 → RRF 融合 → 可选一跳联想。
 */
export class RagLibrary {
	constructor({ id, file, embedder, rebuildEvery = 50, maxK = 512, probe = 3, onChange = null }) {
		this.id = id;
		this.file = file;
		this.embedder = embedder;
		this.rebuildEvery = rebuildEvery;
		this.maxK = maxK;
		this.probeDefault = probe;
		this.onChange = onChange;
		this.neighborK = 5;
		this.edgeSample = 200;
		this.reranker = null;
		this.condenseMaxLen = 160;
		/** 语义去重阈值（0 = 关）。见 _nearDuplicate。 */
		this.dedupeThreshold = 0.92;
		this.db = new LibraryDb(file);
	}

	get dims() {
		return Number(this.db.meta("dims")) || this.embedder?.dims || 384;
	}

	/** 改一条记忆的正文：重新嵌入 + 重排倒排表。返回 {ok} 或 {error} */
	async update(id, text, kind) {
		const t = String(text ?? "").trim();
		if (!t) return { error: "内容不能为空" };
		const cur = this.db.get(id);
		if (!cur) return { error: "记忆不存在" };
		const hash = textHash(t);
		const dup = this.db.idByHash(hash);
		if (dup !== null && dup !== undefined && Number(dup) !== Number(id)) return { error: "已存在内容相同的记忆" };
		const vecs = await this.embedder.embedPassages([t]);
		this.db.updateMem(id, { text: t, kind: kind ?? cur.kind, hash, vec: vecs[0], ts: Date.now() });
		const { tf, len } = termFreq(t);
		this.db.addTerms(id, tf, len);
		this.onChange?.({ dims: vecs[0]?.length ?? this.dims, updated: Date.now() });
		return { ok: true, id };
	}

	/**
	 * 找一条"几乎是同一句话"的已有节点。
	 *
	 * 为什么需要它（2026-11-01 姐姐发现的）：
	 *   原来 add() 只有 `idByHash` —— 文本**逐字相同**才拦。
	 *   可摘要是 LLM 重写的：「我记得2026-10-31有人问我是不是又惹到花块了了」
	 *   换个措辞 hash 就变了，同一件事在库里堆成好几条，检索时一起被翻出来。
	 *
	 * 做法：先用簇中心粗筛（取最像的 3 个簇），再在簇成员里算精确余弦。
	 * 没聚过类就退回全量 —— 节点少的时候全量也就几毫秒。
	 */
	_nearDuplicate(vec, threshold) {
		if (!vec || !(threshold > 0)) return null;
		let rows = [];
		const cs = this.db.clusters();
		if (cs.length) {
			const top = cs
				.filter((c) => Array.isArray(c.centroid) && c.centroid.length === vec.length)
				.map((c) => ({ id: c.id, sim: dot(vec, c.centroid) }))
				.sort((a, b) => b.sim - a.sim)
				.slice(0, 3)
				.map((c) => c.id);
			if (top.length) rows = this.db.membersOf(top);
		}
		if (!rows.length) rows = this.db.allVectors();
		let bestId = null;
		let bestSim = 0;
		for (const r of rows) {
			if (!r.vec || r.vec.length !== vec.length) continue;
			const s = dot(vec, r.vec);
			if (s > bestSim) {
				bestSim = s;
				bestId = r.id;
			}
		}
		return bestSim >= threshold ? { id: bestId, sim: bestSim } : null;
	}

	/** 批量写入；返回 {added, skipped, ids, deduped} */
	async add(items) {
		const clean = [];
		for (const it of items) {
			const text = String(it?.text ?? "").trim();
			if (!text) continue;
			const hash = textHash(text);
			if (this.db.idByHash(hash)) continue;
			clean.push({ ...it, text, hash });
		}
		if (!clean.length) return { added: 0, skipped: items.length, ids: [], deduped: 0 };
		const vecs = await this.embedder.embedPassages(clean.map((c) => c.text));

		// ── 语义去重（2026-11-01 加）──
		// 逐字相同的上面已经用 hash 挡掉了，这里挡「换了个说法的同一件事」。
		// 两处都要看：库里已有的，和这一批里刚收下的（一次压 24 条，前三条可能就是同一件事）。
		const thr = Number(this.dedupeThreshold) || 0;
		let kept = clean;
		let keptVecs = vecs;
		let deduped = 0;
		if (thr > 0) {
			kept = [];
			keptVecs = [];
			const accepted = [];
			for (let i = 0; i < clean.length; i++) {
				const v = vecs[i];
				if (v) {
					const hit = this._nearDuplicate(v, thr);
					if (hit) {
						deduped += 1;
						continue;
					}
					let same = false;
					for (const a of accepted) {
						if (!a.vec || a.vec.length !== v.length) continue;
						if (dot(v, a.vec) >= thr) {
							same = true;
							break;
						}
					}
					if (same) {
						deduped += 1;
						continue;
					}
				}
				kept.push(clean[i]);
				keptVecs.push(v);
				if (v) accepted.push({ vec: v });
			}
			if (!kept.length) {
				return { added: 0, skipped: items.length, ids: [], deduped };
			}
		}
		const useVecs = keptVecs;
		const useClean = kept;

		// 已聚过类 → 新节点在线路由进最近簇（不等全量重建），并把簇中心按滑动均值往前推
		const online = this._onlineCentroids();
		const ids = [];
		this.db.db.exec("BEGIN");
		try {
			for (let i = 0; i < useClean.length; i++) {
				const c = useClean[i];
				const id = this.db.insertMem({
					text: c.text,
					kind: c.kind ?? "note",
					src: c.src ?? null,
					ts: c.ts ?? Date.now(),
					hash: c.hash,
					vec: useVecs[i],
				});
				if (!id) continue;
				ids.push(id);
				const { tf, len } = termFreq(c.text);
				this.db.addTerms(id, tf, len);
				if (online && useVecs[i]) this._assignOnline(online, id, useVecs[i]);
			}
			this.db.setMeta("dims", useVecs[0]?.length ?? this.dims);
			if (!this.db.meta("embed_kind")) this.db.setMeta("embed_kind", this.embedder?.kind ?? "unknown");
			this.db.db.exec("COMMIT");
		} catch (error) {
			this.db.db.exec("ROLLBACK");
			throw error;
		}
		const lastAt = Number(this.db.meta("cluster_at") ?? 0);
		this.onChange?.({ dims: useVecs[0]?.length ?? this.dims, updated: Date.now(), model: this.embedder?.modelDir ?? null });
		if (this.db.count() - lastAt >= this.rebuildEvery || (this.db.count() >= 8 && this.db.stats().clusters === 0)) {
			await this.rebuild();
		}
		return { added: ids.length, skipped: items.length - ids.length, ids, deduped };
	}

	/**
	 * 增量路径：把新节点直接路由进最近簇（argmax 余弦），簇中心按滑动均值推进。
	 * 对应原项目 association_pipline.py 的 _incremental_update / MiniBatchKMeans.partial_fit。
	 * 返回 null 表示"现在不该走增量"（没聚过类 / 存的是别的嵌入模型的向量）。
	 */
	_onlineCentroids() {
		const storedKind = this.db.meta("embed_kind");
		const curKind = this.embedder?.kind ?? "unknown";
		if (storedKind && storedKind !== curKind && curKind !== "lazy") return null;
		const cs = this.db.clusters().filter((c) => c.centroid && c.centroid.length);
		if (!cs.length) return null;
		return cs.map((c) => ({ id: c.id, centroid: c.centroid, size: Number(c.size ?? 0) }));
	}

	_assignOnline(state, memId, vec) {
		let best = state[0];
		let bestSim = -Infinity;
		for (const c of state) {
			const s = dot(vec, c.centroid);
			if (s > bestSim) {
				bestSim = s;
				best = c;
			}
		}
		this.db.setClusterOf([memId], [best.id]);
		const size = best.size + 1;
		const merged = new Float32Array(best.centroid.length);
		for (let i = 0; i < merged.length; i++) merged[i] = (best.centroid[i] * best.size + (vec[i] ?? 0)) / size;
		let norm = 0;
		for (let i = 0; i < merged.length; i++) norm += merged[i] * merged[i];
		norm = Math.sqrt(norm) || 1;
		for (let i = 0; i < merged.length; i++) merged[i] /= norm;
		best.centroid = merged;
		best.size = size;
		this.db.bumpCluster(best.id, merged, size);
		// 邻居边也增量补上（原项目 _incremental_update 第 2 步），否则联想要等全量 rebuild
		const cand = this.db.membersOfSample(best.id, this.edgeSample);
		if (cand.length > 1) {
			const scored = [];
			for (const c of cand) {
				if (c.id === memId) continue;
				scored.push([c.id, dot(vec, c.vec)]);
			}
			scored.sort((a, b) => b[1] - a[1]);
			const pairs = [];
			for (const [other, sim] of scored.slice(0, this.neighborK)) {
				pairs.push([memId, other, sim], [other, memId, sim]);
			}
			this.db.addEdges(pairs);
		}
	}

	needsRebuild() {
		const at = Number(this.db.meta("cluster_at") ?? 0);
		return this.db.count() - at >= this.rebuildEvery;
	}

	/** 全量重聚类 + 重建邻居边。N 大时用 IVFFlat 式随机抽样训练，再全量指派。 */
	async rebuild({ maxK = this.maxK, neighbors = 5 } = {}) {
		const rows = this.db.allVectors();
		const n = rows.length;
		if (n === 0) {
			this.db.clearClusters();
			this.db.setMeta("cluster_at", 0);
			return { n: 0, k: 0 };
		}
		const ids = rows.map((r) => r.id);
		const vecs = rows.map((r) => r.vec);
		const k = Math.min(chooseK(n, maxK), n);
		const { labels, centroids } = kmeans(vecs, k, { iters: 10 });
		this.db.db.exec("BEGIN");
		try {
			this.db.clearClusters();
			this.db.setClusterOf(ids, labels);
			const sizes = new Array(centroids.length).fill(0);
			for (let i = 0; i < labels.length; i++) sizes[labels[i]]++;
			// 训练集抽样（最多 4000 条）用于建边，避免 N^2
			const cap = 4000;
			const step = Math.max(1, Math.ceil(n / cap));
			const pairs = [];
			const byCluster = new Map();
			for (let i = 0; i < n; i += step) {
				const c = labels[i];
				if (!byCluster.has(c)) byCluster.set(c, []);
				byCluster.get(c).push(i);
			}
			for (const [c, idxs] of byCluster) {
				const members = idxs.slice(0, 400);
				const pool = [];
				for (const c2 of [c - 1, c + 1]) if (byCluster.has(c2)) pool.push(...byCluster.get(c2).slice(0, 200));
				const cand = members.concat(pool).slice(0, 600);
				for (const i of members) {
					const sims = [];
					for (const j of cand) {
						if (i === j) continue;
						sims.push([j, dot(vecs[i], vecs[j])]);
					}
					sims.sort((a, b) => b[1] - a[1]);
					for (const [j, sim] of sims.slice(0, neighbors)) {
						pairs.push([ids[i], ids[j], sim]);
					}
				}
			}
			const labelOf = (c) => {
				const idx = byCluster.get(c)?.[0];
				if (idx === undefined) return `簇${c}`;
				const row = this.db.get(ids[idx]);
				return (row?.text ?? `簇${c}`).slice(0, 12);
			};
			this.db.writeClusters(centroids, sizes, centroids.map((_, i) => labelOf(i)));
			this.db.writeEdges(pairs);
			this.db.setMeta("cluster_at", n);
			this.db.setMeta("k", k);
			this.db.db.exec("COMMIT");
		} catch (error) {
			this.db.db.exec("ROLLBACK");
			throw error;
		}
		this.onChange?.({ dims: this.dims, updated: Date.now() });
		return { n, k };
	}

	/** 词法分：BM25（自建 posting 表，中文单/双字 + 拉丁词） */
	_lexScores(queryText, candidateSet) {
		const terms = [...new Set(tokenize(queryText))];
		if (!terms.length) return new Map();
		const postings = this.db.postingsFor(terms);
		if (!postings.length) return new Map();
		const st = this.db.stats();
		const N = Math.max(1, st.docs);
		const avg = st.avgLen || terms.length || 1;
		const lens = new Map();
		for (const r of this.db._st.docLen.all()) lens.set(r.mem, r.len);
		const scores = new Map();
		for (const p of postings) {
			if (candidateSet && !candidateSet.has(p.memId)) continue;
			const len = lens.get(p.memId) ?? avg;
			const idf = Math.log(1 + (N - p.df + 0.5) / (p.df + 0.5));
			const tfPart = (p.tf * (K1 + 1)) / (p.tf + K1 * (1 - B + (B * len) / avg));
			scores.set(p.memId, (scores.get(p.memId) ?? 0) + idf * tfPart);
		}
		return scores;
	}

	static _rrf(ranked) {
		const fused = new Map();
		for (const list of ranked) {
			list.forEach((id, rank) => {
				fused.set(id, (fused.get(id) ?? 0) + 1 / (RRF_K + rank + 1));
			});
		}
		return fused;
	}

	/**
	 * 混合检索。useClusters 为 true 时只探 top-probe 个簇（IVF），否则全库暴力。
	 * 返回 {hits:[{id,text,kind,ts,cluster,score,dense,lex}], meta:{...}}
	 */
	async search(queryText, options = {}) {
		const topK = Math.max(1, options.topK ?? 6);
		const probe = Math.max(1, options.probe ?? this.probeDefault);
		const useClusters = options.useClusters !== false;
		const qv = await this.embedder.embedQuery(queryText);
		const total = this.db.count();
		// 库里存的是 bge 向量，但当前退化成词法嵌入时，点积没有意义 → 只走词法通道
		const storedKind = this.db.meta("embed_kind");
		const curKind = this.embedder?.kind ?? "unknown";
		const denseUsable = !storedKind || storedKind === curKind || curKind === "lazy";
		if (!total) return { hits: [], meta: { total: 0, strategy: "empty", dims: qv.length } };

		let candidates = null;
		let strategy = "brute";
		let probed = [];
		const clusters = useClusters && denseUsable ? this.db.clusters() : [];
		if (clusters.length) {
			const sims = clusters.map((c) => ({ id: c.id, sim: c.centroid ? dot(qv, c.centroid) : -1 }));
			sims.sort((a, b) => b.sim - a.sim);
			probed = sims.slice(0, probe);
			strategy = "ivf";
			const members = this.db.membersOf(probed.map((p) => p.id));
			candidates = members;
		}
		if (!candidates || !candidates.length) {
			candidates = this.db.allVectors(20000);
			strategy = "brute";
		}

		const candSet = new Set(candidates.map((c) => c.id));
		const dense = new Map();
		for (const c of candidates) dense.set(c.id, denseUsable ? dot(qv, c.vec) : 0);
		const lex = this._lexScores(queryText, candSet);

		const denseRank = [...dense.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
		const lexRank = [...lex.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id);
		const fused = RagLibrary._rrf([denseRank, lexRank]);

		const ranked = [...fused.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, Math.max(topK * 3, topK))
			.map(([id, score]) => ({ id, score }));

		// 手动权重：面板里可以给某条记忆降权（0.5）或加权（1.5），0 权重以外一律在打分上乘一下。
		// 放在精排之前，让"我不想再看这条"能真的沉下去；精排分也乘同一份权重，避免精排把它顶回来。
		const wMap = this.db.weightsOf(ranked.map((r) => r.id));
		for (const r of ranked) {
			const w = wMap.get(r.id) ?? 1;
			r.weight = w;
			if (w !== 1) r.score = r.score * w;
		}
		if (wMap.size) ranked.sort((a, b) => b.score - a.score);

		// 精排：RRF 只保证"大致对"，cross-encoder 判 (query, doc) 才能把真正相关的顶上来。
		// 输入先用 condense 压过（原项目 CONDENSE_BEFORE_RERANK），模型缺失就整段跳过。
		let rerankState = "off";
		if (this.reranker && options.rerank !== false && ranked.length > 1) {
			const topN = Math.min(Number(this.reranker.topN ?? 20), ranked.length);
			const head = ranked.slice(0, topN);
			const docs = head.map((r) => condense(this.db.get(r.id)?.text ?? "", queryText, { maxLen: this.condenseMaxLen }));
			const scores = await this.reranker.score(queryText, docs);
			if (scores && scores.length === head.length) {
				head.forEach((r, i) => {
					r.rerank = Number((scores[i] ?? 0).toFixed(6));
					if ((r.weight ?? 1) !== 1) r.rerank = Number((r.rerank * r.weight).toFixed(6));
				});
				head.sort((a, b) => b.rerank - a.rerank);
				ranked.splice(0, topN, ...head);
				rerankState = "on";
			} else {
				rerankState = this.reranker.assetsOk?.() === false ? "model-missing" : "unavailable";
			}
		}

		// 直命中：先算一个 0~1 的"终分"再拿阈值卡。
		// 有精排分就用精排分（sigmoid 后天然 0~1）；没有精排的热路径退回 dense 余弦，负数当 0。
		// 阈值两种模式：
		//   fixed   —— 用传进来的固定值
		//   dynamic —— max(下限, top1 × 比例)，跟着这次查询自己的最高分走，
		//              查得到就收紧、查不到就放宽，比一个死数字更耐话题漂移
		// 低于阈值的直接丢：不占 topK 名额、不给联想配额、不进提示注入 —— 宁可少几条也不污染认知。
		const clamp01 = (x) => Math.max(0, Math.min(1, Number(x) || 0));
		const scoreMode = options.minScoreMode === "fixed" ? "fixed" : "dynamic";
		const floor = clamp01(options.minScoreFloor ?? 0.35);
		const ratio = clamp01(options.minScoreRatio ?? 0.55);
		const finalOf = (r) => {
			if (r.rerank != null) return Math.max(0, Math.min(1, r.rerank));
			return Math.max(0, Math.min(1, dense.get(r.id) ?? 0));
		};
		let threshold = clamp01(options.minScore ?? 0);
		let top1 = 0;
		if (scoreMode === "dynamic") {
			const scan = ranked.slice(0, Math.max(topK * 3, topK, 12));
			for (const r of scan) top1 = Math.max(top1, finalOf(r));
			threshold = Math.max(floor, top1 * ratio);
		}
		const hits = [];
		let dropped = 0;
		const mmrLambda = Math.max(0, Math.min(1, Number(options.mmrLambda ?? 0)));
		// 先收齐"过了阈值"的候选，再交给 MMR 挑 —— 阈值和多样性是两件事，别互相干扰
		const pool = [];
		for (const r of ranked) {
			const row = this.db.get(r.id);
			if (!row) continue;
			const final = finalOf(r);
			if (threshold > 0 && final < threshold) {
				dropped += 1;
				continue;
			}
			// MMR 会把"跟谁都不像"的候选留下（重复惩罚为 0），所以这里必须挡住零相关的，
			// 否则 λ 一低就会把驴唇不对马嘴的条目捞上来。
			if (mmrLambda > 0 && final <= 0) {
				dropped += 1;
				continue;
			}
			pool.push({ r, row, final });
		}
		let ordered = pool;
		if (mmrLambda > 0 && pool.length > 2) {
			const vecById = new Map(candidates.map((c) => [c.id, c.vec]));
			ordered = mmrSelect(pool, topK, mmrLambda, vecById);
		}
		for (const { r, row, final } of ordered) {
			if (hits.length >= topK) break;
			hits.push({
				id: row.id,
				text: row.text,
				kind: row.kind,
				ts: row.ts,
				cluster: row.cluster,
				score: Number(r.score.toFixed(6)),
				weight: r.weight ?? 1,
				rerank: r.rerank ?? null,
				dense: Number((dense.get(r.id) ?? 0).toFixed(4)),
				lex: Number((lex.get(r.id) ?? 0).toFixed(4)),
				final: Number(final.toFixed(4)),
			});
		}

		// 联想层：和直接命中完全分开 —— 不占 topK 名额，不参与打分排序。
		// 两种分配方式：
		//   weighted（默认）：按直命中的终分归一化后分名额，分数高的多联想、低的不联想
		//   even：每条均分 + 补位（老行为）
		let assoc = [];
		const assocMode = options.assocMode === "even" ? "even" : "weighted";
		if (options.assoc) {
			const capacity = Math.max(0, Number(options.assocCount ?? 3));
			const perHit = Math.max(1, Number(options.assocPerHit ?? 1));
			if (capacity > 0 && hits.length) {
				const seen = new Set(hits.map((h) => h.id));
				const edgesBy = new Map(hits.map((h) => [h.id, this.db.edgesOf([h.id])])); // 每条命中自己的一跳边，按 sim 降序
				// 每条命中的实际配额
				const quota =
					assocMode === "weighted"
						? quotaByScore(hits, capacity)
						: new Array(hits.length).fill(perHit);
				// 按配额领取；某条边不够就把空出来的名额让给下一轮（最多 8 轮）
				let round = 0;
				let progressed = true;
				while (assoc.length < capacity && progressed && round < 8) {
					progressed = false;
					for (let i = 0; i < hits.length; i++) {
						if (assoc.length >= capacity) break;
						const h = hits[i];
						const limit = assocMode === "weighted" ? quota[i] : perHit;
						let taken = 0;
						for (const e of edgesBy.get(h.id) ?? []) {
							if (taken >= limit || assoc.length >= capacity) break;
							if (seen.has(e.dst)) continue;
							seen.add(e.dst);
							const row = this.db.get(e.dst);
							if (!row) continue;
							assoc.push({ ...row, sim: Number(e.sim.toFixed(4)), from: e.src });
							taken += 1;
							progressed = true;
						}
						// weighted 模式：这轮没领够不影响下一轮，配额归零的条目不参与
						if (assocMode === "weighted" && taken >= limit) quota[i] = 0;
					}
					round += 1;
				}
			}
		}

		return {
			hits,
			assoc,
			meta: {
				total,
				strategy,
				dense: denseUsable,
				embedKind: curKind,
				probed: probed.map((p) => ({ cluster: p.id, sim: Number(p.sim.toFixed(4)) })),
				candidates: candidates.length,
				threshold: Number(threshold.toFixed(4)),
				minScoreMode: scoreMode,
				top1: Number(top1.toFixed(4)),
				dropped,
				mmr: mmrLambda > 0 ? Number(mmrLambda.toFixed(2)) : 0,
				lexTerms: [...new Set(tokenize(queryText))].length,
				rerank: rerankState,
				dims: qv.length,
			},
		};
	}

	/** 2D 可视化：对簇中心做 PCA（幂迭代），样本点投影到同一平面。 */
	visualize({ sample = 240 } = {}) {
		const clusters = this.db.clusters();
		const rows = this.db.allVectors(sample);
		if (!clusters.length && !rows.length) return { centroids: [], points: [], dims: 0 };
		const dims = rows[0]?.vec?.length ?? clusters[0]?.centroid?.length ?? 0;
		const mean = new Float64Array(dims);
		const all = rows.map((r) => r.vec).concat(clusters.map((c) => c.centroid).filter(Boolean));
		for (const v of all) for (let d = 0; d < dims; d++) mean[d] += v[d] / all.length;

		const power = (seedIdx, exclude) => {
			let v = new Float64Array(dims);
			v[seedIdx % dims] = 1;
			for (let it = 0; it < 24; it++) {
				const out = new Float64Array(dims);
				for (const x of all) {
					let s = 0;
					for (let d = 0; d < dims; d++) s += (x[d] - mean[d]) * v[d];
					for (let d = 0; d < dims; d++) out[d] += s * (x[d] - mean[d]);
				}
				for (let d = 0; d < dims; d++) out[d] -= exclude ? exclude[d] * dotArr(out, exclude) : 0;
				let n = 0;
				for (let d = 0; d < dims; d++) n += out[d] * out[d];
				if (n <= 1e-9) break;
				const r = 1 / Math.sqrt(n);
				for (let d = 0; d < dims; d++) v[d] = out[d] * r;
			}
			return v;
		};
		const pc1 = power(0, null);
		const pc2 = power(1, pc1);
		const project = (vec) => {
			let x = 0;
			let y = 0;
			for (let d = 0; d < dims; d++) {
				const c = vec[d] - mean[d];
				x += c * pc1[d];
				y += c * pc2[d];
			}
			return [Number(x.toFixed(4)), Number(y.toFixed(4))];
		};
		const centroids = clusters.map((c) => {
			const [x, y] = c.centroid ? project(c.centroid) : [0, 0];
			return { id: c.id, x, y, size: c.size, label: c.label };
		});
		const points = rows.map((r) => {
			const [x, y] = project(r.vec);
			return { id: r.id, x, y, cluster: r.cluster };
		});
		return { centroids, points, dims, sampled: rows.length };
	}

	/**
	 * 每个簇在聊什么：簇内高频 n-gram × 全库稀有度，再附两条最短的样例记忆。
	 * 纯词频统计，不走模型 —— 面板一打开就能看，不花一分算力。
	 */
	clusterTopics({ topTerms = 5, samples = 2, scan = 5000 } = {}) {
		const rows = this.db.recent(scan, 0);
		if (!rows.length) return [];
		const total = rows.length;
		const perRow = [];
		const dfGlobal = new Map();
		for (const r of rows) {
			const terms = themeTermsOf(r.text);
			perRow.push(terms);
			for (const t of terms) dfGlobal.set(t, (dfGlobal.get(t) ?? 0) + 1);
		}
		const groups = new Map();
		rows.forEach((r, i) => {
			const cid = r.cluster;
			if (cid === null || cid === undefined || cid < 0) return;
			if (!groups.has(cid)) groups.set(cid, []);
			groups.get(cid).push(i);
		});
		const out = [];
		for (const [cid, idxs] of [...groups.entries()].sort((a, b) => b[1].length - a[1].length)) {
			const size = idxs.length;
			const dfLocal = new Map();
			for (const i of idxs) for (const t of perRow[i]) dfLocal.set(t, (dfLocal.get(t) ?? 0) + 1);
			// 边界片段本来就稀少，门槛放低：簇够大才要求出现两次
			const minHit = size >= 8 ? 2 : 1;
			const scored = [];
			const scoredAll = [];
			// 二字中文太泛（"用户""时候""记录"），得在簇里占到三分之一才认
			const twoCharNeed = Math.max(2, Math.ceil(size * 0.34));
			for (const [t, f] of dfLocal) {
				const g = dfGlobal.get(t) ?? 1;
				// 簇内越集中、全库越少见 → 越像这个簇的主题词
				const s = f * Math.log(1 + total / (1 + g));
				scoredAll.push([t, s]);
				const isTwoCJK = [...t].length === 2 && CJK_CHAR.test(t);
				if (f >= minHit && (!isTwoCJK || f >= twoCharNeed)) scored.push([t, s]);
			}
			// 长词略占优：同样的簇内集中度，"三角洲行动" 比 "三角" 更像个主题
			const weight = ([w, sc]) => sc * (1 + 0.18 * Math.min([...w].length, 6));
			const byScore = (a, b) => weight(b) - weight(a) || b[0].length - a[0].length;
			scored.sort(byScore);
			scoredAll.sort(byScore);
			const terms = []; // [{ w, s }]，长词可以把先入的短碎片顶掉
			const conflictOf = (t) =>
				terms.find(({ w: x }) => {
					if (x === t || x.includes(t) || t.includes(x)) return true;
					const a = [...x].length;
					const b = [...t].length;
					// 同一个实体的错位切片（"苹果三块一斤" vs "块一斤说"）算重复
					if (x.includes(t) || t.includes(x)) return true;
					const lcs = commonRunLength(x, t);
					if (lcs >= Math.min(3, a, b)) return true;
					// 三字以上的候选，只要和已选词共享 len-1 个连续字，就是同一处的碎片
					return b >= 3 && lcs >= 2;
				});
			const take = (t, s) => {
				const hit = conflictOf(t);
				if (!hit) {
					if (terms.length < topTerms) terms.push({ w: t, s });
					return;
				}
				// 已选的被更长的词包含、分数又没差太多 → 换成长词（"精排" → "精排模型"）
				if ([...t].length > [...hit.w].length && s >= hit.s * 0.7) {
					hit.w = t;
					hit.s = s;
				}
			};
			for (const [t, s] of scored) take(t, s);
			// 词不够就放宽到只出现一次的词（小库、新库常见）
			if (terms.length < Math.min(3, topTerms)) for (const [t, s] of scoredAll) take(t, s);
			const picked = idxs
				.map((i) => rows[i])
				.filter((r) => String(r.text ?? "").trim().length >= 6)
				.sort((a, b) => String(a.text).length - String(b.text).length)
				.slice(0, samples)
				.map((r) => ({ id: r.id, text: String(r.text).replace(/\s+/g, " ").slice(0, 70) }));
			const words = terms.map((x) => x.w);
			out.push({ id: cid, size, terms: words, label: words.join(" · "), samples: picked });
		}
		return out;
	}

	sizeBytes() {		try {
			const a = statSync(this.file).size;
			let wal = 0;
			try {
				wal = statSync(this.file + "-wal").size;
			} catch {
				/* ignore */
			}
			return a + wal;
		} catch {
			return 0;
		}
	}

	info() {
		const st = this.db.stats();
		return {
			id: this.id,
			dims: this.dims,
			bytes: this.sizeBytes(),
			bytesPerMem: st.mem ? Math.round(this.sizeBytes() / st.mem) : 0,
			...st,
			kinds: this.db.countByKind(),
			rerank: this.rerankStatus(),
		};
	}

	/** 精排器状态：null / model-missing / ready / 失败原因，面板与 /api/status 都用它。 */
	rerankStatus() {
		if (!this.reranker) return { enabled: false };
		try {
			const s = this.reranker.status?.() ?? {};
			return { enabled: true, ...s, condenseMaxLen: this.condenseMaxLen };
		} catch (error) {
			return { enabled: true, ready: false, failure: String(error?.message ?? error) };
		}
	}

	setReranker(reranker) {
		this.reranker = reranker ?? null;
		return this;
	}

	recent(limit = 20, offset = 0) {
		return this.db.recent(limit, offset);
	}

	searchText(q, limit = 20) {
		return this.db.searchText(q, limit);
	}

	forget(ids) {
		for (const id of ids) this.db.delete(id);
		return ids.length;
	}

	close() {
		this.db.close();
	}
}

function dotArr(a, b) {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}
