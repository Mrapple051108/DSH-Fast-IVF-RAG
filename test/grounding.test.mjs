/**
 * 记忆库「只记真发生过的事」自检。
 *
 *   node test/grounding.test.mjs
 *
 * 钉的是 2026-11-03 那次事故的三条判据：
 *   ① 时间字段是 `time` 不是 `at`（宿主事件信封只有 type/seq/time/data/…）
 *   ② runtime-context / compact-checkpoint / plugin 都不是人说的话
 *   ③ 真正发出去的话在 qq_send / qq_reply 的参数里；思考块永远不进
 */
import assert from 'node:assert/strict'
import {
	textOfMessage,
	timeOfEvent,
	isMemoryWorthyUser,
	sentTextOfToolCall,
	SRC_KIND_SKIP,
} from '../lib/index.js'

let pass = 0
const ok = (label) => {
	pass += 1
	console.log(`${label}  ✅`)
}

/* ① 时间戳：宿主用 time */
{
	const at = 1_760_000_000_000
	assert.equal(timeOfEvent({ type: 'user/message', time: at, data: {} }), at, '要认 time')
	assert.equal(timeOfEvent({ at }), at, '老对象上的 at 也要认出')
	assert.equal(timeOfEvent({}), 0, '没有时间就是 0')
	assert.equal(timeOfEvent({ time: 'abc' }), 0, '垃圾值不能变成 NaN')
	ok('① 时间戳认 time（不是 at）')
}

/* ② 哪些 user/message 不算人话 */
{
	assert.deepEqual(
		[...SRC_KIND_SKIP].sort(),
		['compact-checkpoint', 'plugin', 'runtime-context'],
		'跳过名单少了会重新污染记忆',
	)
	for (const kind of ['compact-checkpoint', 'plugin', 'runtime-context']) {
		assert.equal(isMemoryWorthyUser({ content: [{ type: 'text', text: '环境快照' }], source: { kind } }), false, `${kind} 不该记`)
	}
	assert.equal(isMemoryWorthyUser({ content: [{ type: 'text', text: '姐姐说今晚吃面' }], source: { kind: 'user' } }), true, '真人说的话要记')
	assert.equal(isMemoryWorthyUser({ content: [{ type: 'text', text: '没 source 也算人话' }] }), true)
	ok('② 环境快照 / 压缩替身 / 插件注入都跳过')
}

/* ③ 真正发出去的话在工具参数里 */
{
	const call = (name, args) => ({ type: 'tool/call', data: { name, arguments: args } })
	assert.equal(sentTextOfToolCall(call('qq_send', { text: '晚安，睡了' })), '晚安，睡了')
	assert.equal(sentTextOfToolCall(call('qq_reply', { text: '这条是引用回复' })), '这条是引用回复')
	assert.equal(sentTextOfToolCall(call('qq_send', JSON.stringify({ text: '参数是字符串也要认' }))), '参数是字符串也要认')
	assert.equal(sentTextOfToolCall(call('qq_send_image', { file: 'D:/x.png', text: '配图' })), '', '只发图不算说话')
	assert.equal(sentTextOfToolCall(call('write', { path: 'a.md', text: '文件正文' })), '', '写文件的参数不是话')
	assert.equal(sentTextOfToolCall(call('pwsh', { command: 'ls' })), '')
	assert.equal(sentTextOfToolCall({ type: 'tool/call', data: { name: 'qq_send', arguments: '{坏JSON' } }), '')
	assert.equal(sentTextOfToolCall({ type: 'tool/call', data: {} }), '')
	ok('③ 只认 qq_send / qq_reply 的 text')
}

/* ④ 思考、工具步、图片都不进文本 */
{
	const assistant = (content) => ({ turn: 1, step: 1, message: { role: 'assistant', content }, usage: {}, stream: {} })
	assert.equal(
		textOfMessage(assistant([{ type: 'text', text: '晚点再说' }])),
		'晚点再说',
		'assistant 的 content 包在 data.message 里，要挖出来',
	)
	assert.equal(
		textOfMessage(assistant([{ type: 'reasoning', text: '我先想想她为什么这么说……' }])),
		'',
		'思考块绝不能进记忆',
	)
	assert.equal(
		textOfMessage(assistant([{ type: 'reasoning', text: '想一下' }, { type: 'text', text: '好' }])),
		'好',
		'有思考也有正文时只取正文',
	)
	assert.equal(
		textOfMessage(assistant([{ type: 'text', text: '我查一下' }, { type: 'tool-call', id: 'c1', name: 'qq_send', arguments: {} }])),
		'',
		'带 tool-call 的过程步整步丢弃',
	)
	assert.equal(textOfMessage(assistant([{ type: 'image', path: 'D:/a.png' }])), '', '图片不是话')
	ok('④ 思考 / 工具步 / 图片都不进记忆')
}

console.log(`\n全部通过（${pass} 组）`)
