/* HTML 번역 실행기 — 화면에 뜨는 일본어를 그 자리에서 한국어로 바꿔 보여 주는 도우미 (실행기 화면에 끼워 넣어짐)
   원래 글자는 투명하게만 하고(실행기는 그대로 돌아감), 같은 자리에 한국어를 겹쳐 그림 */
(() => {
  if (window.__trOn) return; window.__trOn = true;
  const GID = (location.pathname.match(/^\/g\/([^/]+)\//) || [])[1];
  if (!GID) return;
  const JP = /[぀-ヿ㐀-鿿ｦ-ﾟ]/;          // 히라가나·가타카나·한자
  const KANA = /[぀-ヿｦ-ﾟ]/;
  const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "INPUT", "SELECT", "OPTION", "RT", "RP", "TITLE", "CODE", "PRE", "TEMPLATE", "XMP", "SVG", "CANVAS", "IFRAME"]);
  const MAXLEN = 1500;                   // 화면 대사는 길어야 몇백 자. 이보다 긴 글자 덩어리(숨겨 둔 대본 등)는 아예 안 봄
  const okText = n => n.nodeValue.length <= MAXLEN && JP.test(n.nodeValue);
  const WAIT = 150;                      // 글자가 한 자씩 찍히는 동안은 기다렸다가, 멈추면 번역
  const st = new Map();                  // 글자 상자 → 상태
  const ctx = [];                        // 앞 대사 (말투·이름 맞추기용)
  const mem = new Map();                 // 이 화면에서 이미 받은 번역
  let on = true;

  const css = document.createElement("style");
  css.textContent = `
  .__trh, .__trh *:not(.__tro):not(.__tro *) { color: transparent !important; text-shadow: none !important;
     -webkit-text-stroke-color: transparent !important; -webkit-text-fill-color: transparent !important; }
  .__tro { position: absolute; pointer-events: none; z-index: 2147483000; white-space: pre-wrap; word-break: keep-all;
     overflow-wrap: anywhere; box-sizing: border-box; -webkit-text-fill-color: currentColor !important; }
  #__trbar { position: fixed; right: 8px; bottom: 8px; z-index: 2147483600; font: 12px/1.4 "Malgun Gothic", sans-serif;
     background: rgba(20,20,20,.72); color: #f3c27a; border: 1px solid rgba(243,194,122,.5); border-radius: 8px;
     padding: 4px 9px; cursor: pointer; user-select: none; -webkit-text-fill-color: #f3c27a !important; }
  #__trbar.off { color: #aaa; -webkit-text-fill-color: #aaa !important; }`;
  (document.head || document.documentElement).appendChild(css);

  const isOurs = n => { for (let e = n.nodeType === 1 ? n : n.parentElement; e; e = e.parentElement) { if (e.classList && (e.classList.contains("__tro") || e.id === "__trbar")) return true; } return false; };
  const disp = e => { try { return getComputedStyle(e).display; } catch (x) { return "block"; } };
  const inlineish = e => { const d = disp(e); return d === "inline" || d === "contents" || d === "inline-block" || d === "inline-flex" || d === "ruby" || d === "ruby-base" || d === "ruby-text"; };

  // 글자 조각이 속한 '글자 상자' (대사창 한 칸): 위로 올라가며 처음 만나는 블록. 줄마다 블록이 나뉘어 있으면 그 위로 한 번 더
  function boxOf(textNode) {
    let e = textNode.parentElement;
    while (e && e !== document.body && e !== document.documentElement && inlineish(e)) e = e.parentElement;
    if (!e || e === document.body || e === document.documentElement) return textNode.parentElement;
    const p = e.parentElement;
    if (p && p !== document.body && p.children.length > 1 && p.children.length <= 12 && !inlineish(p)) {
      const kids = [...p.children];
      const textKids = kids.filter(k => (k.textContent || "").length <= MAXLEN * 2 && JP.test(textOf(k)));
      const avg = textKids.reduce((a, k) => a + textOf(k).length, 0) / Math.max(1, textKids.length);
      if (textKids.length > 1 && avg >= 8 && kids.every(k => !k.querySelector("img,canvas,video,button,input") && (k.children.length < 200))) return p;
    }
    return e;
  }

  // 상자 안 글자 (후리가나·우리 겹친 글자 빼고, <br> 은 줄바꿈)
  function textOf(el) {
    let s = "";
    const walk = n => {
      if (s.length > MAXLEN * 2) return;
      if (n.nodeType === 3) { if (n.nodeValue.length <= MAXLEN) s += n.nodeValue; return; }
      if (n.nodeType !== 1 || SKIP.has(n.tagName) || (n.classList && n.classList.contains("__tro"))) return;
      if (n.tagName === "BR") { s += "\n"; return; }
      for (const c of n.childNodes) walk(c);
    };
    walk(el);
    return s.replace(/[ \t　]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  // 원래 글자 모양 (첫 글자 조각 기준)
  function firstTextEl(el) {
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: n => (JP.test(n.nodeValue) && !isOurs(n) && !n.parentElement.closest("rt,rp")) ? 1 : 3 });
    const n = w.nextNode();
    return n ? n.parentElement : el;
  }

  // 원문 글자가 실제로 차지한 자리 (상자 안 좌표). 실행기가 화면을 확대·축소해 둔 것도 맞춰서
  function textBox(el) {
    const er = el.getBoundingClientRect();
    const ratio = (er.width / (el.offsetWidth || er.width)) || 1;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: n => (n.nodeValue.trim() && !isOurs(n) && !n.parentElement.closest("rt,rp,script,style")) ? 1 : 3 });
    const rg = document.createRange();
    let n; while ((n = w.nextNode())) {
      rg.selectNodeContents(n);
      for (const r of rg.getClientRects()) { if (r.width < 0.5) continue; x0 = Math.min(x0, r.left); y0 = Math.min(y0, r.top); x1 = Math.max(x1, r.right); y1 = Math.max(y1, r.bottom); }
    }
    if (x1 < x0) return null;
    return { l: (x0 - er.left) / ratio - el.clientLeft, t: (y0 - er.top) / ratio - el.clientTop, w: (x1 - x0) / ratio, h: (y1 - y0) / ratio };
  }

  function show(el, s) {
    let o = s.ov;
    if (!o || !o.isConnected) {
      o = document.createElement("div"); o.className = "__tro"; s.ov = o;
      if (getComputedStyle(el).position === "static") { el.style.position = "relative"; s.posSet = true; }
      el.appendChild(o);
    }
    const cs = getComputedStyle(el), fe = getComputedStyle(firstTextEl(el));
    const bx = textBox(el) || { l: 0, t: 0, w: el.clientWidth, h: el.clientHeight };
    s.size = el.clientWidth + "x" + el.clientHeight;
    Object.assign(o.style, {
      left: bx.l + "px", top: bx.t + "px", width: Math.max(bx.w, 8) + "px", height: "auto", padding: "0",
      textAlign: cs.textAlign, writingMode: cs.writingMode,
      fontFamily: fe.fontFamily + ', "Malgun Gothic", "Apple SD Gothic Neo", sans-serif',
      fontSize: fe.fontSize, fontWeight: fe.fontWeight, fontStyle: fe.fontStyle, lineHeight: fe.lineHeight,
      letterSpacing: "normal", color: fe.color, textShadow: fe.textShadow,
      webkitTextStroke: fe.webkitTextStrokeWidth + " " + fe.webkitTextStrokeColor,
    });
    o.textContent = s.ko;
    el.classList.add("__trh");
    // 원문이 차지했던 줄 수 안에 들어가게: 넘치면 먼저 옆으로 조금 넓혀 보고, 그래도 넘치면 글자를 조금씩 줄임 (원래의 55%까지)
    const f0 = parseFloat(fe.fontSize) || 16;
    const lh0 = parseFloat(fe.lineHeight) || f0 * 1.4;
    const budget = Math.max(1, Math.round(bx.h / lh0)) * lh0 * 1.05;
    if (o.offsetHeight > budget) {
      const avail = Math.max(bx.w, el.clientWidth - bx.l);
      o.style.width = Math.min(avail, bx.w * 1.25) + "px";
    }
    let fsz = f0, k = 0;
    while (k++ < 12 && o.offsetHeight > budget && fsz > f0 * 0.55) {
      fsz *= 0.93; o.style.fontSize = fsz + "px"; o.style.lineHeight = (lh0 * fsz / f0) + "px";
    }
  }

  function hide(el, s) {
    el.classList.remove("__trh");
    if (s.ov) { s.ov.remove(); s.ov = null; }
  }

  // 번역 요청: 잠깐 모았다가 한 번에
  let queue = [], qt = null;
  function request(src) {
    return new Promise(res => { queue.push({ src, res }); clearTimeout(qt); qt = setTimeout(flush, 40); });
  }
  async function flush() {
    const q = queue; queue = [];
    if (!q.length) return;
    const items = [...new Set(q.map(x => x.src))];
    try {
      const r = await fetch(`/__tr/${GID}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ items, ctx: ctx.slice(-8) }) });
      const d = await r.json();
      const ko = d.ko || [];
      items.forEach((t, i) => { if (ko[i]) { mem.set(t, ko[i]); if (t.length > 6) { ctx.push([t, ko[i]]); if (ctx.length > 16) ctx.shift(); } } });
    } catch (e) { bar("번역 서버와 연결이 끊겼어요"); }
    q.forEach(x => x.res(mem.get(x.src) || ""));
  }

  async function settle(el) {
    const s = st.get(el); if (!s) return;
    if (!el.isConnected) { st.delete(el); return; }
    const src = textOf(el);
    if (src && src === s.src && s.ko) { if (on) show(el, s); return; }   // 같은 대사면 번역 그대로 (다시 붙이기만)
    if (!visible(el)) return;
    if (!src || !JP.test(src)) { hide(el, s); s.src = ""; return; }
    if (src === s.src && s.ko) { if (on) show(el, s); return; }
    s.src = src; s.ko = "";
    if (!on) return;
    let ko = mem.get(src);
    if (ko === undefined) {                               // 번역 글자가 만들어지는 대로 바로바로 찍음
      pending++; bar();
      ko = "";
      try {
        const r = await fetch(`/__trs/${GID}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ item: src, ctx: ctx.slice(-3) }) });
        const rd = r.body.getReader(), dec = new TextDecoder();
        let last = 0;
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          ko += dec.decode(value, { stream: true });
          if (s.src !== src || !el.isConnected) { try { rd.cancel(); } catch (e) {} break; }
          const now = performance.now();
          if (on && ko.trim() && now - last > 40) { s.ko = ko; show(el, s); last = now; }
        }
      } catch (e) { bar("번역 서버와 연결이 끊겼어요"); }
      pending--; bar();
      ko = ko.trim();
      if (ko) { mem.set(src, ko); if (src.length > 6) { ctx.push([src, ko]); if (ctx.length > 16) ctx.shift(); } }
    }
    if (s.src !== src || !el.isConnected) return;           // 그 사이 다음 대사로 넘어갔으면 버림
    s.ko = ko;
    if (ko && on) show(el, s); else hide(el, s);
  }

  function touch(el) {
    let s = st.get(el);
    if (!s) { s = { src: "", ko: "", ov: null, t: null }; st.set(el, s); try { io.observe(el); } catch (e) {} }
    if (s.ov) { const now = textOf(el); if (now && now !== s.src && JP.test(now)) hide(el, s); }   // 다른 일본어 대사가 찍히기 시작할 때만 이전 번역을 치움
    clearTimeout(s.t); s.t = setTimeout(() => settle(el), WAIT);
  }

  // 바뀐 글자 조각은 모아 두었다가 조금씩 처리 (글자가 아주 많은 화면에서도 버벅이지 않게)
  const todoNodes = new Set(); let todoT = null;
  const boxMemo = new WeakMap();
  function queueNode(n) { todoNodes.add(n); if (!todoT) todoT = setTimeout(drain, 30); }
  function drain() {
    todoT = null;
    const t0 = performance.now();
    for (const n of todoNodes) {
      todoNodes.delete(n);
      const pe = n.parentElement;
      if (!pe || !n.isConnected || SKIP.has(pe.tagName) || !okText(n) || isOurs(n)) continue;
      let b = boxMemo.get(pe);
      if (!b || !b.isConnected) { b = boxOf(n); boxMemo.set(pe, b); }
      touch(b);
      if (performance.now() - t0 > 12) break;              // 한 번에 너무 오래 붙잡지 않음
    }
    if (todoNodes.size) todoT = setTimeout(drain, 16);
  }
  function scan(root) {
    if (!root || isOurs(root)) return;
    if (root.nodeType === 3) { queueNode(root); return; }
    if (root.nodeType !== 1 || SKIP.has(root.tagName)) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n, k = 0; while ((n = w.nextNode()) && k < 3000) if (okText(n)) { queueNode(n); k++; }
  }

  const io = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting && st.has(e.target)) touch(e.target); });   // 화면에 나타나면 그때 번역

  // 지금 화면에 보이는 상자만 번역 (숨겨 둔 대본·지난 대사 기록 등은 안 건드림 → 보일 때 번역)
  function visible(el) {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== "hidden" && cs.display !== "none" && parseFloat(cs.opacity || "1") > 0.05;
  }

  let el_keep = 0;
  const mo = new MutationObserver(ms => {
    for (const m of ms) {
      if (isOurs(m.target)) continue;
      if (m.type === "characterData") { scan(m.target); continue; }
      for (const n of m.addedNodes) scan(n);
      if (m.removedNodes.length) {                        // 글자가 지워졌으면 (다음 장) 그 상자 다시 보기
        for (let e = m.target; e && e !== document.body; e = e.parentElement) if (st.has(e)) {
          const s = st.get(e);
          if (on && s.ko && textOf(e) === s.src) { show(e, s); el_keep++; }   // 실행기가 같은 대사를 다시 그리며 한국어를 지웠으면 바로 다시 붙임
          else touch(e);
          break;
        }
      }
      if (m.type === "attributes" && st.has(m.target)) {   // 실행기가 대사창 class 를 바꿔 우리 표시가 빠졌으면 다시
        const s = st.get(m.target);
        if (on && s.ko && !m.target.classList.contains("__trh") && textOf(m.target) === s.src) m.target.classList.add("__trh");
      }
    }
  });

  // 작은 상태 표시 (누르면 번역 켜기/끄기 — 원문 확인용)
  let pending = 0, barEl = null;
  function bar(msg) {
    if (!barEl) {
      barEl = document.createElement("div"); barEl.id = "__trbar";
      barEl.onclick = e => { e.stopPropagation(); e.preventDefault(); on = !on; for (const [el, s] of st) { if (on) settle(el); else hide(el, s); } bar(); };
      ["mousedown", "mouseup", "pointerdown", "pointerup", "touchstart", "touchend"].forEach(t => barEl.addEventListener(t, e => e.stopPropagation()));
      document.body.appendChild(barEl);
    }
    barEl.classList.toggle("off", !on);
    barEl.textContent = msg || (!on ? "번역 꺼짐 (누르면 켜기)" : pending ? "번역 중…" : "한국어 번역 켜짐 (누르면 원문)");
  }

  function start() {
    if (!document.body) { setTimeout(start, 50); return; }
    mo.observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["class"] });
    scan(document.body);
    bar();
    setInterval(() => {                                   // 실행기가 대사창을 다시 그리며 한국어를 지워 버려도, 같은 대사면 다시 붙임
      for (const [el, s] of st) {
        if (!el.isConnected) { st.delete(el); continue; }
        if (!on || !s.ko) continue;
        if (!s.ov || !s.ov.isConnected || !el.classList.contains("__trh") || s.size !== el.clientWidth + "x" + el.clientHeight) {
          if (textOf(el) === s.src) show(el, s);
        }
      }
    }, 300);
  }
  start();
})();
