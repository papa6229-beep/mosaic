/* HTML 번역 실행기 — 화면에 뜨는 일본어를 그 자리에서 한국어로 바꿔 보여 주는 도우미 (실행기 화면에 끼워 넣어짐)
   원래 글자는 투명하게만 하고(실행기는 그대로 돌아감), 같은 자리에 한국어를 겹쳐 그림 */
(() => {
  if (window.__trOn) return; window.__trOn = true;
  const GID = (location.pathname.match(/^\/g\/([^/]+)\//) || [])[1];
  if (!GID) return;
  const JP = /[぀-ヿ㐀-鿿ｦ-ﾟ]/;          // 히라가나·가타카나·한자
  const KANA = /[぀-ヿｦ-ﾟ]/;
  const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEXTAREA", "INPUT", "SELECT", "OPTION", "RT", "RP", "TITLE", "CODE", "PRE"]);
  const WAIT = 280;                      // 글자가 한 자씩 찍히는 동안은 기다렸다가, 멈추면 번역
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
      const textKids = kids.filter(k => JP.test(textOf(k)));
      const avg = textKids.reduce((a, k) => a + textOf(k).length, 0) / Math.max(1, textKids.length);
      if (textKids.length > 1 && avg >= 8 && kids.every(k => !k.querySelector("img,canvas,video,button,input") && (k.children.length < 200))) return p;
    }
    return e;
  }

  // 상자 안 글자 (후리가나·우리 겹친 글자 빼고, <br> 은 줄바꿈)
  function textOf(el) {
    let s = "";
    const walk = n => {
      if (n.nodeType === 3) { s += n.nodeValue; return; }
      if (n.nodeType !== 1 || SKIP.has(n.tagName) || (n.classList && n.classList.contains("__tro"))) return;
      if (n.tagName === "BR") { s += "\n"; return; }
      for (const c of n.childNodes) walk(c);
    };
    walk(el);
    return s.replace(/[ \t　]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  }

  // 원래 글자 모양 (첫 글자 조각 기준)
  function firstTextEl(el) {
    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: n => (JP.test(n.nodeValue) && !isOurs(n)) ? 1 : 3 });
    const n = w.nextNode();
    return n ? n.parentElement : el;
  }

  function show(el, s) {
    let o = s.ov;
    if (!o || !o.isConnected) {
      o = document.createElement("div"); o.className = "__tro"; s.ov = o;
      if (getComputedStyle(el).position === "static") { el.style.position = "relative"; s.posSet = true; }
      el.appendChild(o);
    }
    const cs = getComputedStyle(el), fe = getComputedStyle(firstTextEl(el));
    Object.assign(o.style, {
      left: "0px", top: "0px", width: el.clientWidth + "px", height: el.clientHeight + "px",
      padding: cs.padding, textAlign: cs.textAlign, writingMode: cs.writingMode,
      fontFamily: fe.fontFamily + ', "Malgun Gothic", "Apple SD Gothic Neo", sans-serif',
      fontSize: fe.fontSize, fontWeight: fe.fontWeight, fontStyle: fe.fontStyle, lineHeight: fe.lineHeight,
      letterSpacing: "normal", color: fe.color, textShadow: fe.textShadow,
      webkitTextStroke: fe.webkitTextStrokeWidth + " " + fe.webkitTextStrokeColor,
    });
    o.textContent = s.ko;
    el.classList.add("__trh");
    let fsz = parseFloat(fe.fontSize) || 16, k = 0;          // 한국어가 길어 넘치면 조금씩 줄임
    while (k++ < 8 && el.clientHeight > 0 && o.scrollHeight > el.clientHeight * 1.04 && fsz > 9) { fsz *= 0.92; o.style.fontSize = fsz + "px"; }
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
    if (!src || !JP.test(src)) { hide(el, s); s.src = ""; return; }
    if (src === s.src && s.ko) { if (on) show(el, s); return; }
    s.src = src; s.ko = "";
    if (!on) return;
    let ko = mem.get(src);
    if (ko === undefined) { pending++; bar(); ko = await request(src); pending--; bar(); }
    if (s.src !== src || !el.isConnected) return;           // 그 사이 다음 대사로 넘어갔으면 버림
    s.ko = ko;
    if (ko && on) show(el, s); else hide(el, s);
  }

  function touch(el) {
    let s = st.get(el);
    if (!s) { s = { src: "", ko: "", ov: null, t: null }; st.set(el, s); }
    if (s.ov && textOf(el) !== s.src) { hide(el, s); }   // 새 대사가 찍히기 시작하면 이전 번역은 바로 치움
    clearTimeout(s.t); s.t = setTimeout(() => settle(el), WAIT);
  }

  function scan(root) {
    if (!root || isOurs(root)) return;
    if (root.nodeType === 3) { if (JP.test(root.nodeValue) && root.parentElement && !SKIP.has(root.parentElement.tagName)) touch(boxOf(root)); return; }
    if (root.nodeType !== 1 || SKIP.has(root.tagName)) return;
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n; while ((n = w.nextNode())) {
      if (JP.test(n.nodeValue) && n.parentElement && !SKIP.has(n.parentElement.tagName) && !isOurs(n)) touch(boxOf(n));
    }
  }

  const mo = new MutationObserver(ms => {
    for (const m of ms) {
      if (isOurs(m.target)) continue;
      if (m.type === "characterData") { scan(m.target); continue; }
      for (const n of m.addedNodes) scan(n);
      if (m.removedNodes.length) {                        // 글자가 지워졌으면 (다음 장) 그 상자 다시 보기
        for (let e = m.target; e && e !== document.body; e = e.parentElement) if (st.has(e)) { touch(e); break; }
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
    mo.observe(document.body, { childList: true, subtree: true, characterData: true });
    scan(document.body);
    bar();
    setInterval(() => { for (const [el, s] of st) { if (!el.isConnected) { st.delete(el); continue; } if (s.ov && s.ko && on) { if (s.ov.style.width !== el.clientWidth + "px" || s.ov.style.height !== el.clientHeight + "px") show(el, s); } } }, 500);
  }
  start();
})();
