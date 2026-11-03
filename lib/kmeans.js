/**
 * 球形 k-means（向量已 L2 归一化，点积即余弦）。
 * 对应 Cherry 版 association_pipline.py 的 MiniBatchKMeans 角色：
 * 只做「粗筛分区」，不需要极致精度，10 轮 argmax 足够。
 */

export function chooseK(n, maxK = 512) {
	if (n <= 0) return 0;
	return Math.max(2, Math.min(maxK, Math.round(Math.sqrt(n) / 2)));
}

function lcg(seed) {
	let s = seed >>> 0;
	return () => {
		s = (s * 1664525 + 1013904223) >>> 0;
		return s / 4294967296;
	};
}

export function dot(a, b) {
	let s = 0;
	for (let i = 0; i < a.length; i++) s += a[i] * b[i];
	return s;
}

export function normalizeInPlace(v) {
	let s = 0;
	for (let i = 0; i < v.length; i++) s += v[i] * v[i];
	if (s <= 0) return v;
	const r = 1 / Math.sqrt(s);
	for (let i = 0; i < v.length; i++) v[i] *= r;
	return v;
}

/** kmeans++ 初始化（用点积当相似度，选离已选中心最远的点）。 */
function initCentroids(vecs, k, rand) {
	const n = vecs.length;
	const picked = [Math.floor(rand() * n)];
	const best = new Float64Array(n).fill(-2);
	while (picked.length < k) {
		const last = vecs[picked[picked.length - 1]];
		let sum = 0;
		for (let i = 0; i < n; i++) {
			const sim = dot(vecs[i], last);
			if (sim > best[i]) best[i] = sim;
			const d = 1 - best[i];
			sum += d * d;
		}
		if (sum <= 1e-12) {
			picked.push(Math.floor(rand() * n));
			continue;
		}
		let r = rand() * sum;
		let idx = 0;
		for (let i = 0; i < n; i++) {
			const d = 1 - best[i];
			r -= d * d;
			if (r <= 0) {
				idx = i;
				break;
			}
		}
		picked.push(idx);
	}
	return picked.map((i) => Float32Array.from(vecs[i]));
}

export function kmeans(vecs, k, options = {}) {
	const iters = options.iters ?? 10;
	const rand = lcg(options.seed ?? 42);
	const n = vecs.length;
	if (n === 0) return { labels: new Int32Array(0), centroids: [], inertia: 0 };
	const kk = Math.max(1, Math.min(k, n));
	const dims = vecs[0].length;
	let centroids = initCentroids(vecs, kk, rand);
	const labels = new Int32Array(n);

	for (let it = 0; it < iters; it++) {
		let moved = 0;
		for (let i = 0; i < n; i++) {
			let best = 0;
			let bestSim = -2;
			for (let c = 0; c < kk; c++) {
				const sim = dot(vecs[i], centroids[c]);
				if (sim > bestSim) {
					bestSim = sim;
					best = c;
				}
			}
			if (labels[i] !== best) {
				labels[i] = best;
				moved++;
			}
		}
		const sums = Array.from({ length: kk }, () => new Float64Array(dims));
		const counts = new Int32Array(kk);
		for (let i = 0; i < n; i++) {
			const c = labels[i];
			counts[c]++;
			const v = vecs[i];
			const s = sums[c];
			for (let d = 0; d < dims; d++) s[d] += v[d];
		}
		for (let c = 0; c < kk; c++) {
			if (counts[c] === 0) {
				// 空簇：抢一个随机点，避免死簇
				centroids[c] = Float32Array.from(vecs[Math.floor(rand() * n)]);
				continue;
			}
			const s = sums[c];
			const out = new Float32Array(dims);
			for (let d = 0; d < dims; d++) out[d] = s[d] / counts[c];
			centroids[c] = normalizeInPlace(out);
		}
		if (moved === 0 && it > 0) break;
	}

	let inertia = 0;
	for (let i = 0; i < n; i++) inertia += 1 - dot(vecs[i], centroids[labels[i]]);
	return { labels, centroids, inertia };
}
