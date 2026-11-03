import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/** DSH_HOME 优先，其次 ~/.dsh —— 与 qqbot / 模型目录保持同一套约定。 */
export function dshHome() {
	const fromEnv = (process.env.DSH_HOME ?? "").trim();
	return fromEnv || join(homedir(), ".dsh");
}

/** 插件数据根：{DSH_HOME}/fast-ivf-rag */
export function defaultDataDir() {
	return join(dshHome(), "fast-ivf-rag");
}

/** 默认嵌入模型目录：{DSH_HOME}/models/bge-small-zh（实测 512 维，q8 量化 24MB） */
export function defaultModelDir() {
	return join(dshHome(), "models", "bge-small-zh");
}

/**
 * 默认精排模型目录：{DSH_HOME}/models/ 下任意名字带 reranker 且带 tokenizer+config 的目录，
 * 都没找到就退回 bge-reranker-base（可选件 —— 不存在时精排自动跳过，检索仍走 RRF）。
 */
export function defaultRerankDir() {
	const base = join(dshHome(), "models");
	try {
		const hit = readdirSync(base).find(
			(n) => /rerank/i.test(n) && existsSync(join(base, n, "config.json")) && existsSync(join(base, n, "tokenizer.json")),
		);
		if (hit) return join(base, hit);
	} catch {
		/* 目录不存在就算了 */
	}
	return join(base, "bge-reranker-base");
}

export function libDir(dataDir) {
	return join(dataDir, "libraries");
}

export function libFile(dataDir, id) {
	return join(libDir(dataDir), `${id}.sqlite`);
}

export function indexPath(dataDir) {
	return join(dataDir, "index.sqlite");
}
