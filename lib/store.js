import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

/**
 * SQLite 层。两套库：
 *  - index.sqlite（全局）：library 列表 + session→library 绑定 + meta
 *  - libraries/<id>.sqlite（每库）：mem / cluster / edge / term / posting / docstat
 * 用 node:sqlite（Node 22.5+ 内置，当前 v24.14.1 实测可用）。
 */

const SCHEMA_INDEX = `
CREATE TABLE IF NOT EXISTS library (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  model TEXT,
  dims INTEGER,
  created INTEGER NOT NULL,
  updated INTEGER,
  last_used INTEGER,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS bind (
  session TEXT PRIMARY KEY,
  lib TEXT NOT NULL,
  updated INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
`;

const SCHEMA_LIB = `
CREATE TABLE IF NOT EXISTS mem (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'note',
  src TEXT,
  ts INTEGER NOT NULL,
  cluster INTEGER,
  hash TEXT,
  vec BLOB
);
CREATE UNIQUE INDEX IF NOT EXISTS mem_hash_uq ON mem(hash);
CREATE INDEX IF NOT EXISTS mem_ts ON mem(ts);
CREATE INDEX IF NOT EXISTS mem_cluster ON mem(cluster);
CREATE INDEX IF NOT EXISTS mem_kind ON mem(kind);
CREATE TABLE IF NOT EXISTS cluster (
  id INTEGER PRIMARY KEY,
  centroid BLOB,
  size INTEGER NOT NULL DEFAULT 0,
  label TEXT,
  updated INTEGER
);
CREATE TABLE IF NOT EXISTS edge (
  src INTEGER NOT NULL,
  dst INTEGER NOT NULL,
  sim REAL NOT NULL,
  PRIMARY KEY (src, dst)
);
CREATE INDEX IF NOT EXISTS edge_dst ON edge(dst);
CREATE TABLE IF NOT EXISTS term (term TEXT PRIMARY KEY, df INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS posting (
  term TEXT NOT NULL,
  mem INTEGER NOT NULL,
  tf INTEGER NOT NULL,
  PRIMARY KEY (term, mem)
);
CREATE INDEX IF NOT EXISTS posting_mem ON posting(mem);
CREATE TABLE IF NOT EXISTS docstat (mem INTEGER PRIMARY KEY, len INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS weight (mem INTEGER PRIMARY KEY, w REAL NOT NULL, ts INTEGER);
`;

export function toBlob(vec) {
	const f = vec instanceof Float32Array ? vec : Float32Array.from(vec);
	return Buffer.from(f.buffer, f.byteOffset, f.byteLength);
}

export function fromBlob(buf) {
	if (!buf) return null;
	const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
	return new Float32Array(ab);
}

export function openDb(file) {
	mkdirSync(dirname(file), { recursive: true });
	return new DatabaseSync(file);
}

/* ───────────────────────── 全局索引库 ───────────────────────── */

export class IndexDb {
	constructor(file) {
		this.file = file;
		this.db = openDb(file);
		this.db.exec("PRAGMA journal_mode = WAL;");
		// 召回在 worker 线程里持有自己的连接：读写并发时别直接 SQLITE_BUSY 掉。
		this.db.exec("PRAGMA busy_timeout = 4000;");
		this.db.exec(SCHEMA_INDEX);
	}

	listLibraries() {
		return this.db.prepare("SELECT * FROM library ORDER BY COALESCE(last_used, updated, created) DESC").all();
	}

	getLibrary(id) {
		return this.db.prepare("SELECT * FROM library WHERE id = ?").get(id) ?? null;
	}

	upsertLibrary(row) {
		this.db
			.prepare(
				`INSERT INTO library (id, name, model, dims, created, updated, last_used, notes)
				 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
				 ON CONFLICT(id) DO UPDATE SET
				   name = excluded.name,
				   model = COALESCE(excluded.model, library.model),
				   dims = COALESCE(excluded.dims, library.dims),
				   updated = COALESCE(excluded.updated, library.updated),
				   last_used = COALESCE(excluded.last_used, library.last_used),
				   notes = COALESCE(excluded.notes, library.notes)`,
			)
			.run(
				row.id,
				row.name ?? row.id,
				row.model ?? null,
				row.dims ?? null,
				row.created ?? Date.now(),
				row.updated ?? null,
				row.last_used ?? null,
				row.notes ?? null,
			);
	}

	deleteLibrary(id) {
		this.db.prepare("DELETE FROM library WHERE id = ?").run(id);
		this.db.prepare("DELETE FROM bind WHERE lib = ?").run(id);
	}

	touchLibrary(id, ts = Date.now()) {
		this.db.prepare("UPDATE library SET last_used = ? WHERE id = ?").run(ts, id);
	}

	/** 某库被哪些会话绑定 */
	bindingsOf(lib) {
		return this.db.prepare("SELECT session, updated FROM bind WHERE lib = ? ORDER BY updated DESC").all(lib);
	}

	bind(session, lib) {
		this.db
			.prepare(
				`INSERT INTO bind (session, lib, updated) VALUES (?, ?, ?)
				 ON CONFLICT(session) DO UPDATE SET lib = excluded.lib, updated = excluded.updated`,
			)
			.run(session, lib, Date.now());
	}

	unbind(session) {
		this.db.prepare("DELETE FROM bind WHERE session = ?").run(session);
	}

	binding(session) {
		return this.db.prepare("SELECT lib FROM bind WHERE session = ?").get(session)?.lib ?? null;
	}

	allBindings() {
		return this.db.prepare("SELECT session, lib, updated FROM bind ORDER BY updated DESC").all();
	}

	meta(k) {
		return this.db.prepare("SELECT v FROM meta WHERE k = ?").get(k)?.v ?? null;
	}

	setMeta(k, v) {
		this.db
			.prepare("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v")
			.run(k, String(v));
	}

	close() {
		try {
			this.db.close();
		} catch {
			/* ignore */
		}
	}
}

/* ───────────────────────── 单库 ───────────────────────── */

export class LibraryDb {
	constructor(file) {
		this.file = file;
		this.db = openDb(file);
		this.db.exec("PRAGMA journal_mode = WAL;");
		this.db.exec("PRAGMA synchronous = NORMAL;");
		this.db.exec("PRAGMA busy_timeout = 4000;");
		this.db.exec(SCHEMA_LIB);
		this._st = {
			insertMem: this.db.prepare(
				"INSERT OR IGNORE INTO mem (text, kind, src, ts, cluster, hash, vec) VALUES (?, ?, ?, ?, NULL, ?, ?)",
			),
			insTerm: this.db.prepare(
				"INSERT INTO term (term, df) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET df = df + 1",
			),
			insPosting: this.db.prepare("INSERT OR REPLACE INTO posting (term, mem, tf) VALUES (?, ?, ?)"),
			insDocstat: this.db.prepare("INSERT OR REPLACE INTO docstat (mem, len) VALUES (?, ?)"),
			byHash: this.db.prepare("SELECT id FROM mem WHERE hash = ?"),
			getMem: this.db.prepare("SELECT id, text, kind, src, ts, cluster, hash FROM mem WHERE id = ?"),
			delMem: this.db.prepare("DELETE FROM mem WHERE id = ?"),
			delPost: this.db.prepare("DELETE FROM posting WHERE mem = ?"),
			delStat: this.db.prepare("DELETE FROM docstat WHERE mem = ?"),
			delEdge: this.db.prepare("DELETE FROM edge WHERE src = ? OR dst = ?"),
			count: this.db.prepare("SELECT COUNT(*) AS n FROM mem"),
			termDf: this.db.prepare("SELECT df FROM term WHERE term = ?"),
			postOf: this.db.prepare("SELECT mem, tf FROM posting WHERE term = ?"),
			avgLen: this.db.prepare("SELECT AVG(len) AS a, COUNT(*) AS n FROM docstat"),
			docLen: this.db.prepare("SELECT mem, len FROM docstat"),
		};
	}

	insertMem({ text, kind = "note", src = null, ts = Date.now(), hash = null, vec }) {
		const r = this._st.insertMem.run(text, kind, src, ts, hash, vec ? toBlob(vec) : null);
		return Number(r.lastInsertRowid);
	}

	idByHash(hash) {
		if (!hash) return null;
		return this._st.byHash.get(hash)?.id ?? null;
	}

	/** 改一条记忆的正文（顺带换向量与倒排；簇归到 NULL，等下次重聚类再归位）。 */
	updateMem(id, { text, kind = null, hash = null, vec = null, ts = null }) {
		const cur = this.get(id);
		if (!cur) return false;
		this.db
			.prepare("UPDATE mem SET text = ?, kind = ?, hash = ?, vec = ?, ts = ?, cluster = NULL WHERE id = ?")
			.run(text ?? cur.text, kind ?? cur.kind, hash ?? cur.hash, vec ? toBlob(vec) : null, ts ?? cur.ts, id);
		this._st.delPost.run(id);
		this._st.delStat.run(id);
		return true;
	}

	addTerms(memId, tfMap, len) {
		for (const [term, tf] of tfMap) {
			this._st.insTerm.run(term);
			this._st.insPosting.run(term, memId, tf);
		}
		this._st.insDocstat.run(memId, len);
	}

	count() {
		return this._st.count.get()?.n ?? 0;
	}

	get(id) {
		return this._st.getMem.get(id) ?? null;
	}

	delete(id) {
		this._st.delPost.run(id);
		this._st.delStat.run(id);
		this._st.delEdge.run(id, id);
		this._st.delMem.run(id);
	}

	/** 全量向量（可视化 / 暴力检索用），返回 [{id, vec, cluster}] */
	allVectors(limit = 0) {
		const sql = "SELECT id, vec, cluster FROM mem WHERE vec IS NOT NULL" + (limit > 0 ? " LIMIT " + limit : "");
		return this.db.prepare(sql).all().map((r) => ({ id: r.id, vec: fromBlob(r.vec), cluster: r.cluster }));
	}

	setClusterOf(ids, labels) {
		const st = this.db.prepare("UPDATE mem SET cluster = ? WHERE id = ?");
		for (let i = 0; i < ids.length; i++) st.run(labels[i], ids[i]);
	}

	clearClusters() {
		this.db.exec("DELETE FROM cluster");
		this.db.exec("UPDATE mem SET cluster = NULL");
		this.db.exec("DELETE FROM edge");
	}

	writeClusters(centroids, sizes, labels) {
		this.db.exec("DELETE FROM cluster");
		const st = this.db.prepare("INSERT INTO cluster (id, centroid, size, label, updated) VALUES (?, ?, ?, ?, ?)");
		const now = Date.now();
		for (let i = 0; i < centroids.length; i++) {
			st.run(i, toBlob(centroids[i]), sizes[i] ?? 0, labels?.[i] ?? null, now);
		}
	}

	/** 增量路径用：改一个簇的中心与成员数（MiniBatchKMeans 的 partial_fit 语义）。 */
	bumpCluster(id, centroid, size) {
		this.db.prepare("UPDATE cluster SET centroid = ?, size = ?, updated = ? WHERE id = ?").run(toBlob(centroid), size, Date.now(), id);
	}

	clusters() {
		return this.db
			.prepare("SELECT id, centroid, size, label FROM cluster ORDER BY id")
			.all()
			.map((r) => ({ id: r.id, centroid: fromBlob(r.centroid), size: r.size, label: r.label }));
	}

	membersOf(clusterIds) {
		if (!clusterIds.length) return [];
		const marks = clusterIds.map(() => "?").join(",");
		return this.db
			.prepare(`SELECT id, vec FROM mem WHERE cluster IN (${marks}) AND vec IS NOT NULL`)
			.all(...clusterIds)
			.map((r) => ({ id: r.id, vec: fromBlob(r.vec) }));
	}

	membersOfClustersLimited(clusterIds, limit = 4000) {
		return this.membersOf(clusterIds).slice(0, limit);
	}

	/** 增量建边：只追加，不清表（rebuild 的 writeEdges 才是全量重建）。 */
	addEdges(pairs) {
		if (!pairs.length) return;
		const st = this.db.prepare("INSERT OR REPLACE INTO edge (src, dst, sim) VALUES (?, ?, ?)");
		for (const [a, b, sim] of pairs) st.run(a, b, sim);
	}

	/** 抽样取某个簇的成员（增量建边时用，避免把整簇拉进内存）。 */
	membersOfSample(clusterId, limit = 200) {
		return this.db
			.prepare("SELECT id, vec FROM mem WHERE cluster = ? AND vec IS NOT NULL ORDER BY RANDOM() LIMIT ?")
			.all(clusterId, limit)
			.map((r) => ({ id: r.id, vec: fromBlob(r.vec) }));
	}

	writeEdges(pairs) {
		this.db.exec("DELETE FROM edge");
		const st = this.db.prepare("INSERT OR REPLACE INTO edge (src, dst, sim) VALUES (?, ?, ?)");
		for (const [a, b, sim] of pairs) st.run(a, b, sim);
	}

	edgesOf(ids) {
		if (!ids.length) return [];
		const marks = ids.map(() => "?").join(",");
		return this.db
			.prepare(`SELECT src, dst, sim FROM edge WHERE src IN (${marks}) ORDER BY sim DESC`)
			.all(...ids);
	}

	/** 词法通道：返回 [{memId, tf, df}] */
	postingsFor(terms) {
		const out = [];
		for (const t of terms) {
			const df = this._st.termDf.get(t)?.df ?? 0;
			if (!df) continue;
			for (const row of this._st.postOf.all(t)) out.push({ memId: row.mem, tf: row.tf, df, term: t });
		}
		return out;
	}

	stats() {
		const avg = this._st.avgLen.get();
		return {
			mem: this.count(),
			terms: this.db.prepare("SELECT COUNT(*) AS n FROM term").get()?.n ?? 0,
			postings: this.db.prepare("SELECT COUNT(*) AS n FROM posting").get()?.n ?? 0,
			edges: this.db.prepare("SELECT COUNT(*) AS n FROM edge").get()?.n ?? 0,
			clusters: this.db.prepare("SELECT COUNT(*) AS n FROM cluster").get()?.n ?? 0,
			avgLen: avg?.a ?? 0,
			docs: avg?.n ?? 0,
		};
	}

	meta(k) {
		return this.db.prepare("SELECT v FROM meta WHERE k = ?").get(k)?.v ?? null;
	}

	setMeta(k, v) {
		this.db
			.prepare("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v")
			.run(k, String(v));
	}

	countByKind() {
		return this.db.prepare("SELECT kind, COUNT(*) AS n FROM mem GROUP BY kind ORDER BY n DESC").all();
	}

	recent(limit = 20, offset = 0) {
		return this.db
			.prepare("SELECT id, text, kind, src, ts, cluster FROM mem ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?")
			.all(limit, offset);
	}

	searchText(q, limit = 20) {
		return this.db
			.prepare("SELECT id, text, kind, src, ts, cluster FROM mem WHERE text LIKE ? ORDER BY ts DESC LIMIT ?")
			.all(`%${q}%`, limit);
	}

	/** 取一批记忆的权重（缺省 1）。面板手动降权/加权用。 */
	weightsOf(ids) {
		const out = new Map();
		const list = (ids ?? []).filter((n) => Number.isInteger(n));
		if (!list.length) return out;
		const marks = list.map(() => "?").join(",");
		for (const r of this.db.prepare(`SELECT mem, w FROM weight WHERE mem IN (${marks})`).all(...list)) {
			out.set(r.mem, Number(r.w));
		}
		return out;
	}

	/** 设权重：1 = 恢复默认（直接删行，别留垃圾） */
	setWeight(mem, w) {
		const id = Number(mem);
		const val = Number(w);
		if (!Number.isInteger(id) || !Number.isFinite(val)) return false;
		if (val <= 0) {
			this.db.prepare("DELETE FROM weight WHERE mem = ?").run(id);
			return true;
		}
		this.db
			.prepare("INSERT INTO weight (mem, w, ts) VALUES (?, ?, ?) ON CONFLICT(mem) DO UPDATE SET w = excluded.w, ts = excluded.ts")
			.run(id, Math.max(0.05, Math.min(5, val)), Date.now());
		return true;
	}

	allWeights() {
		return this.db.prepare("SELECT mem, w, ts FROM weight ORDER BY ts DESC").all();
	}

	close() {
		try {
			this.db.close();
		} catch {
			/* ignore */
		}
	}
}
