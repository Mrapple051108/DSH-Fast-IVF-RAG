/**
 * 一键拉模型：状态判断 + 下载失败要能落到 error 上。
 *   node test/model-fetch.test.mjs
 *
 * 不真的下大模型（那是用户点按钮的事），只钉三件：
 *   ① 缺文件时能列清楚缺哪几个；
 *   ② 文件齐了判 ready；
 *   ③ 下载源不通时任务要停在 error，不能一直 running。
 */
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { MODEL_SPECS, modelStatus, startModelDownload, serializeJob } from '../lib/model-fetch.js'

let pass = 0
const ok = (label) => {
  pass += 1
  console.log(`${label}  ✅`)
}

const tmp = mkdtempSync(join(tmpdir(), 'modelfetch-'))
try {
  /* ① 空目录：缺什么要列出来 */
  {
    const dir = join(tmp, 'empty-embed')
    mkdirSync(dir, { recursive: true })
    const st = modelStatus('embed', { dir })
    assert.equal(st.ready, false)
    assert.deepEqual(st.missing, MODEL_SPECS.embed.files, '四个文件都该在缺失列表里')
    ok('① 目录空着 → ready=false，缺文件列全')
  }

  /* ② 文件齐了（每个非空）才 ready */
  {
    const dir = join(tmp, 'full-embed')
    for (const f of MODEL_SPECS.embed.files) {
      const p = join(dir, f)
      mkdirSync(join(p, '..'), { recursive: true })
      writeFileSync(p, 'x'.repeat(64))
    }
    const st = modelStatus('embed', { dir })
    assert.equal(st.ready, true)
    assert.equal(st.missing.length, 0)
    assert.ok(st.bytes >= MODEL_SPECS.embed.files.length * 64, '字节数要累加')

    // 空文件不算数：0 字节的 onnx 会让 transformers 加载失败，不能判 ready
    writeFileSync(join(dir, 'onnx/model_quantized.onnx'), '')
    const st2 = modelStatus('embed', { dir })
    assert.equal(st2.ready, false, '0 字节文件不能算就绪')
    assert.deepEqual(st2.missing, ['onnx/model_quantized.onnx'])
    ok('② 文件齐全才 ready；0 字节不算')
  }

  /* ③ 下载源不通 → job 停在 error（不能一直转圈） */
  {
    const dir = join(tmp, 'dl-fail')
    const job = startModelDownload('embed', { dir, base: 'http://127.0.0.1:9/nowhere', force: true })
    await job.promise
    assert.equal(job.state, 'error', `该是 error，实际 ${job.state}`)
    assert.ok(job.error, '要带错误原因')
    const slim = serializeJob(job)
    assert.equal('promise' in slim, false, 'JSON 里不能带 promise')
    assert.equal(slim.state, 'error')
    ok('③ 源不通 → error，且序列化不带 promise')
  }

  /* ④ 未知模型直接抛 */
  {
    assert.throws(() => startModelDownload('nope', {}), /未知模型/)
    ok('④ 未知模型名直接拒')
  }

  /* ⑤ 精排量化版本：选哪个就缺哪个文件 */
  {
    const dir = join(tmp, 'rerank-empty')
    mkdirSync(dir, { recursive: true })
    const q8 = modelStatus('rerank', { dir, variant: 'q8' })
    assert.equal(q8.variant, 'q8', '默认 q8')
    assert.ok(q8.missing.includes('onnx/model_quantized.onnx'), 'q8 要的是 model_quantized.onnx')

    const fp16 = modelStatus('rerank', { dir, variant: 'fp16' })
    assert.ok(fp16.missing.includes('onnx/model_fp16.onnx'), 'fp16 要的是 model_fp16.onnx')
    assert.equal(fp16.missing.includes('onnx/model_quantized.onnx'), false, '不该再要 q8 的文件')

    // 五个版本都该列出来，且都标未装
    assert.equal(fp16.variants.length, 5)
    assert.ok(fp16.variants.every((v) => v.installed === false))

    // 乱填的版本名回落到默认
    assert.equal(modelStatus('rerank', { dir, variant: 'nope' }).variant, 'q8')

    // 下载任务也要带上版本
    const job = startModelDownload('rerank', { dir, variant: 'fp16', base: 'http://127.0.0.1:9/x', force: true })
    assert.equal(job.variant, 'fp16')
    assert.equal(job.total, 4, 'fp16：3 个公共文件 + 1 个权重')
    await job.promise
    ok('⑤ 量化版本决定缺哪个文件 / 下哪个权重')
  }
} finally {
  rmSync(tmp, { recursive: true, force: true })
}

console.log(`\n全部通过（${pass} 组）`)
