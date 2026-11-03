/**
 * 主线程用的「嵌入器代理」。
 *
 * 它长得跟 lib/embed.js 的 createEmbedder 一模一样，但所有计算都转发给工作线程：
 * 主线程里不再加载 onnxruntime —— 这样既不会卡事件循环，也不会撞上
 * 「同进程两份 ORT 会话」的 V8 HandleScope FATAL。
 *
 * kind 先报 "lazy"（library 的判据里 lazy = 向量可用），第一次真正嵌入后
 * 用 worker 回报的真实 kind/dims 覆盖它。
 */
export function createWorkerEmbedder(pool, modelDir) {
	let realKind = "lazy";
	let realDims = 512;

	const apply = (result) => {
		if (result?.kind) realKind = result.kind;
		if (result?.dims) realDims = result.dims;
	};

	return {
		kind: "lazy",
		get dims() {
			return realDims;
		},
		modelDir,
		ready: true,
		get kindName() {
			return realKind;
		},
		status() {
			return { kind: realKind, dims: realDims, modelDir, ready: true, threaded: true };
		},
		async warmup() {
			const r = await pool.embedQuery(modelDir, "warmup");
			apply(r);
			return true;
		},
		async embedQuery(text) {
			const result = await pool.embedQuery(modelDir, String(text ?? ""));
			apply(result);
			this.kind = realKind;
			return result.vector;
		},
		async embedPassages(texts) {
			const list = (texts ?? []).map((t) => String(t));
			if (!list.length) return [];
			const result = await pool.embedPassages(modelDir, list);
			apply(result);
			this.kind = realKind;
			return result.vectors;
		},
	};
}
