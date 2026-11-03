/* fast-ivf-rag — 浏览器半边（手写 __ModuleLoader__ bundle，无需构建步骤）
 *
 * 作用：在 DSH 设置左侧菜单里加一个「向量记忆」页。
 * 页面本体不在这里重复实现 —— 直接内嵌同源面板 /fast-ivf-rag?embed=1，
 * 面板（ui/index.html）已经包含库列表 / 库详情可视化 / 会话改绑 / 检索测试四个页签。
 *
 * 数据走同源 HTTP：面板内部自己 fetch('/fast-ivf-rag/api/...')。
 * 任何一步失败都必须降级为 console.warn，不能让整条 client fiber 判 FAILED
 * （那会让 Web UI 只剩 "Failed to load plugins"）。
 */
window.__ModuleLoader__.load({
  id: 'fast-ivf-rag',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports
    var React = require('react')
    var h = React.createElement

    var PANEL_SRC = '/fast-ivf-rag?embed=1'

    function Panel(props) {
      var ref = React.useRef(null)
      var box = { width: '100%', height: '100%', minHeight: '62vh', display: 'flex', flexDirection: 'column', gap: 8 }
      var head = { display: 'flex', alignItems: 'center', gap: 10, fontSize: 12.5, opacity: 0.72 }
      var link = { color: 'inherit', textDecoration: 'underline', cursor: 'pointer' }
      var frame = { flex: 1, width: '100%', minHeight: '58vh', border: 'none', borderRadius: 10, background: 'transparent', colorScheme: 'normal' }

      function openExternal() {
        try { window.open('/fast-ivf-rag', '_blank') } catch (e) { /* 被弹窗拦截就算了 */ }
      }
      function reload() {
        try { if (ref.current) ref.current.src = ref.current.src } catch (e) { /* ignore */ }
      }

      return h(
        'div',
        { style: box },
        h(
          'div',
          { style: head },
          h('span', null, '库列表 / 聚类可视化 / 会话绑定 / 检索测试'),
          h('a', { style: link, onClick: openExternal }, '新标签打开'),
          h('a', { style: link, onClick: reload }, '刷新')
        ),
        h('iframe', {
          ref: ref,
          src: PANEL_SRC,
          style: frame,
          title: 'fast-ivf-rag 面板',
          allow: 'clipboard-write'
        })
      )
    }

    function apply(ctx) {
      var safe = function (label, fn) {
        try {
          return fn()
        } catch (e) {
          try {
            console.warn('[fast-ivf-rag] ' + label + ' degraded: ' + String((e && e.message) || e))
          } catch (x) { /* ignore */ }
        }
      }
      var slots = ctx && ctx.slots
      if (!slots) {
        try { console.warn('[fast-ivf-rag] slots unavailable, settings page skipped') } catch (e) {}
        return
      }
      safe('settings section', function () {
        slots.inject('settings.section', function () {
          return slots.register(
            { name: 'settings.section', id: 'fast-ivf-rag', order: 34, label: '向量记忆' },
            function (props) { return h(Panel, { close: props && props.close }) }
          )
        })
      })
    }

    exports.inject = ['slots']
    exports.apply = apply
    return module.exports
  }
})
