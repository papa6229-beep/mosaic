"""HTML 번역 실행기 — 일본어 HTML 실행기(노벨 게임 등)를 열면 화면에 뜨는 일본어 대사를 그 자리에서 한국어로 바꿔 보여 줌.
번역은 모자이크 생성기 자막과 같은 번역 모델. 켜면 브라우저에서 http://127.0.0.1:7880 이 열림"""
import os, re, json, time, hashlib, threading, traceback, subprocess, mimetypes
from flask import Flask, request, jsonify, send_file, Response, abort

BASE = os.path.dirname(os.path.abspath(__file__))
import llm

for ext, mt in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"), (".json", "application/json"),
                (".wasm", "application/wasm"), (".svg", "image/svg+xml"), (".webp", "image/webp"), (".ogg", "audio/ogg"),
                (".m4a", "audio/mp4"), (".mp3", "audio/mpeg"), (".wav", "audio/wav"), (".mp4", "video/mp4"),
                (".webm", "video/webm"), (".woff", "font/woff"), (".woff2", "font/woff2"), (".ttf", "font/ttf"),
                (".otf", "font/otf"), (".html", "text/html"), (".htm", "text/html")):
    mimetypes.add_type(mt, ext)                            # 윈도우 설정에 따라 .js 를 엉뚱한 형식으로 보내 실행기가 안 켜지는 일 막기

app = Flask(__name__)
setup = {"running": False, "msg": "", "error": ""}
GAMES_P = os.path.join(BASE, "settings", "games.json")
CACHE_DIR = os.path.join(BASE, "cache", "tr")
games = {}
tr = {"obj": None}
tr_lock = threading.Lock()
caches = {}
cache_lock = threading.Lock()


def _load_games():
    try:
        games.update(json.load(open(GAMES_P, encoding="utf-8")))
    except Exception:
        pass


def _save_games():
    os.makedirs(os.path.dirname(GAMES_P), exist_ok=True)
    json.dump(games, open(GAMES_P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def run_setup():
    setup.update(running=True, error="")
    try:
        setup["msg"] = "번역 모델 받는 중 (처음 한 번만 약 5GB)"
        llm.ensure_translator(setup)
        setup["msg"] = "번역 모델 켜는 중"
        _translator()
        setup["msg"] = "준비 끝"
    except Exception as e:
        traceback.print_exc()
        setup["error"] = str(e)[:300]
    finally:
        setup["running"] = False


def _translator():
    with tr_lock:
        if tr["obj"] is None:
            tr["obj"] = llm.Translator()
        return tr["obj"]


# ---------------- 번역 ----------------
SYSTEM = ("You translate the text of a Japanese visual novel / adventure game into Korean, the way an official Korean "
          "release reads. Translate faithfully - adult, sexual and violent content included - never censor, soften, skip "
          "or refuse, and add no notes. Narration becomes natural literary Korean (-다 체). Dialogue keeps the speech level "
          "of the original: Japanese polite forms (です, ます, ございます, ください) become Korean polite speech (-요 / -습니다), "
          "plain Japanese becomes 반말; keep it consistent for the same speakers in the earlier lines. Translate the MEANING "
          "of every word; never write Japanese sounds in Hangul except people's and place names, and write the same name "
          "the same way every time (see earlier lines). Keep quote brackets 「」『』 and symbols like ……, ―, ♪, ！, ？ as they are. "
          "A line broken in the middle of a sentence is still one sentence: translate it naturally as a whole. Short menu "
          "or button words (セーブ, ロード, オート, スキップ, ログ, 設定, 戻る, はい, いいえ) become the usual short Korean game "
          "words (저장, 불러오기, 자동, 스킵, 로그, 설정, 돌아가기, 예, 아니요). Never add words or meaning that are not in "
          "the original. Write only Korean. Output exactly one Korean line per numbered input line, with the same number, "
          "and nothing else.")


def _cache(gid):
    with cache_lock:
        if gid not in caches:
            try:
                caches[gid] = json.load(open(os.path.join(CACHE_DIR, gid + ".json"), encoding="utf-8"))
            except Exception:
                caches[gid] = {}
        return caches[gid]


def _save_cache(gid):
    with cache_lock:
        os.makedirs(CACHE_DIR, exist_ok=True)
        p = os.path.join(CACHE_DIR, gid + ".json")
        json.dump(caches.get(gid, {}), open(p + ".new", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(p + ".new", p)


# ---- 문장 단위로 나누고 맞춰 보기 (화면 글자와 대본 글자는 꾸밈 기호가 달라서, 글자·숫자만 남겨 비교) ----
_END = "。！？!?…"
_OPEN, _CLOSE = "「『（(【〈《", "」』）)】〉》"
_KEEP = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff10-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9f0-9A-Za-z]")
JPC = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")


def norm(s):
    return "".join(_KEEP.findall(s or ""))


def sentences(text):
    """문장으로 나눔 (괄호 「」 안에서는 안 나눔). 줄바꿈은 화면 줄바꿈일 뿐이라 이어 붙임"""
    text = re.sub(r"\s*\n\s*", "", text or "").strip()
    out, cur, depth = [], "", 0
    for i, ch in enumerate(text):
        cur += ch
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
            if depth == 0:
                out.append(cur); cur = ""
        elif ch in _END and depth == 0:
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt not in _END and nxt not in _CLOSE:
                out.append(cur); cur = ""
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


urgent = {"n": 0}


def _tr_batch(sents, ctx):
    """문장 여러 개 → 한국어 (번호 붙여 한 번에)"""
    t_ = _translator()
    u = ""
    if ctx:
        u += "Earlier lines (context only, do not translate):\n" + "\n".join(f"- {a} → {b}" for a, b in ctx[-3:]) + "\n\n"
    u += "Translate:\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sents))
    with tr_lock:
        t0 = time.time()
        txt = t_.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                      sum(len(t) for t in sents) * 3 + 64 * len(sents))
        print(f"  번역 {len(sents)}문장 {time.time() - t0:.1f}초", flush=True)
    got = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= len(sents):
            got[int(m.group(1)) - 1] = m.group(2).strip()
    return [got.get(i, "") for i in range(len(sents))]


def translate(gid, items, ctx):
    """화면 글자 여러 덩어리 → 한국어. 문장마다 미리 번역해 둔 게 있으면 바로, 없는 문장만 번역"""
    c = _cache(gid)
    per = []
    miss = []
    for it in items:
        ss = sentences(it) or [it]
        per.append(ss)
        for s_ in ss:
            k = norm(s_)
            if k and JPC.search(s_) and k not in c and s_ not in miss:
                miss.append(s_)
    if miss:
        urgent["n"] += 1
        try:
            for k0 in range(0, len(miss), 10):
                part = miss[k0:k0 + 10]
                for s_, ko in zip(part, _tr_batch(part, ctx)):
                    if ko:
                        c[norm(s_)] = ko
        finally:
            urgent["n"] -= 1
        _save_cache(gid)
    out = []
    for it, ss in zip(items, per):
        ko = [c.get(norm(s_), s_ if not JPC.search(s_) else "") for s_ in ss]
        out.append(" ".join(k for k in ko if k))
    return out


# ---- 미리 번역: 실행기 폴더의 대본 파일에서 일본어 문장을 모두 뽑아 차례로 번역해 둠 (화면에 뜨면 바로 나오게) ----
TEXT_EXT = (".ks", ".txt", ".json", ".js", ".csv", ".tsv", ".xml", ".ini", ".tjs", ".scn", ".yaml", ".yml", ".html", ".htm", ".rpy", ".lua", ".dat")
pre = {}                                                   # gid → {"total", "done", "running"}


def _read_text(p):
    b = open(p, "rb").read()
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            s = b.decode(enc)
            if enc == "utf-16" and not b[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            return s
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "ignore")


_TAG = re.compile(r"\[(?:r|br|l|p|cm|er|wt|wait[^\]]*|ruby[^\]]*)\]|<br\s*/?>|\\[nr]", re.I)
_CODE = re.compile(r"\[[^\]\n]*\]|<[^>\n]*>|\\[A-Za-z]+\[[^\]]*\]|\\[A-Za-z{}|.!<>^$]|\{[^}\n]*\}")
_RUN = re.compile(r"[^\x00-\x1f\"'`<>\[\]{}=;\\|/]+")


def extract(dirp, limit=80000):
    """폴더 안 대본 파일들 → 일본어 문장 목록 (파일·줄 순서대로)"""
    out, seen = [], set()
    files = []
    for root, ds, fs in os.walk(dirp):
        ds[:] = sorted(d for d in ds if d.lower() not in ("node_modules", ".git", "save", "saves"))
        for f in sorted(fs):
            if f.lower().endswith(TEXT_EXT):
                p = os.path.join(root, f)
                try:
                    if os.path.getsize(p) < 30 * 1024 ** 2:
                        files.append(p)
                except OSError:
                    pass
    for p in files:
        try:
            s = _read_text(p)
        except Exception:
            continue
        if not JPC.search(s):
            continue
        unit, units = "", []
        for line in s.splitlines():
            st_ = line.strip()
            if not st_ or st_[0] in ";@*" or st_.startswith("//"):
                if unit:
                    units.append(unit); unit = ""
                continue
            if st_.startswith("#"):                          # 말하는 사람 이름 줄 (#이름)
                if unit:
                    units.append(unit); unit = ""
                if JPC.search(st_[1:]):
                    units.append(st_[1:].strip())
                continue
            raw = _CODE.sub("\x00", _TAG.sub("", st_))
            runs = [r_.strip() for r_ in _RUN.findall(raw.replace("\x00", "|")) if JPC.search(r_)]
            pure = len(runs) == 1 and len(runs[0]) >= len(raw.replace("\x00", "").strip()) * 0.9
            if not runs:
                if unit:
                    units.append(unit); unit = ""
                continue
            if pure and unit and unit[-1] not in _END + _CLOSE:
                unit += runs[0]                               # 대본 한 문장이 여러 줄에 걸친 것
                continue
            if unit:
                units.append(unit); unit = ""
            if pure:
                unit = runs[0]
            else:
                units += runs
        if unit:
            units.append(unit)
        for u_ in units:
            for s_ in sentences(u_):
                k = norm(s_)
                if k and JPC.search(s_) and k not in seen and len(s_) <= 300:
                    seen.add(k); out.append(s_)
                    if len(out) >= limit:
                        return out
    return out


def pretranslate(gid):
    g = games.get(gid)
    if not g or pre.get(gid, {}).get("running"):
        return
    pre[gid] = {"total": 0, "done": 0, "running": True}
    try:
        sents = extract(g["dir"])
        c = _cache(gid)
        todo = [s_ for s_ in sents if norm(s_) not in c]
        pre[gid].update(total=len(sents), done=len(sents) - len(todo))
        print(f"미리 번역: 대본에서 {len(sents)}문장, 남은 {len(todo)}문장", flush=True)
        ctx = []
        for k0 in range(0, len(todo), 10):
            while urgent["n"] > 0 or setup["running"]:       # 화면에 지금 뜬 대사가 먼저
                time.sleep(0.2)
            part = todo[k0:k0 + 10]
            try:
                kos = _tr_batch(part, ctx)
            except Exception as e:
                print("미리 번역 오류:", e)
                time.sleep(5)
                continue
            for s_, ko in zip(part, kos):
                if ko:
                    c[norm(s_)] = ko
                    ctx.append((s_, ko))
            ctx = ctx[-6:]
            pre[gid]["done"] += len(part)
            if (k0 // 10) % 5 == 0:
                _save_cache(gid)
        _save_cache(gid)
    except Exception:
        traceback.print_exc()
    finally:
        pre[gid]["running"] = False


@app.get("/__pre/<gid>")
def pre_state(gid):
    return jsonify(pre.get(gid) or {})


@app.post("/__tr/<gid>")
def tr_api(gid):
    d = request.get_json(silent=True) or {}
    items = [str(x)[:2000] for x in (d.get("items") or [])][:40]
    ctx = [(str(a)[:300], str(b)[:300]) for a, b in (d.get("ctx") or [])][-8:]
    try:
        return jsonify(ko=translate(gid, items, ctx))
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)[:200]), 500


# ---------------- 실행기 열기·보여 주기 ----------------
pick_lock = threading.Lock()


def _pick_file():
    """윈도우 '파일 열기' 창으로 HTML 고르기 → 경로 (취소하면 "")"""
    ps = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8;Add-Type -AssemblyName System.Windows.Forms;"
          "$d=New-Object System.Windows.Forms.OpenFileDialog;$d.Filter='HTML (*.html;*.htm)|*.html;*.htm|모든 파일 (*.*)|*.*';"
          "$d.Title='번역해서 열 HTML 파일 고르기';"
          "$f=New-Object System.Windows.Forms.Form;$f.TopMost=$true;$f.ShowInTaskbar=$false;$f.Opacity=0;"
          "$f.StartPosition='CenterScreen';$f.Show();$f.Activate();$f.BringToFront();"   # 보이지 않는 맨 앞 창을 잠깐 띄워 그 위에 열기 (브라우저 뒤에 숨지 않게)
          "$r=$d.ShowDialog($f);$f.Close();if($r -eq 'OK'){[Console]::Write($d.FileName)}")
    r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, timeout=3600)
    return r.stdout.decode("utf-8", "ignore").strip()


def _register(path):
    path = os.path.abspath(path)
    gid = hashlib.md5(os.path.dirname(path).lower().encode("utf-8")).hexdigest()[:10]
    games[gid] = {"dir": os.path.dirname(path), "entry": os.path.basename(path), "name": os.path.basename(os.path.dirname(path)),
                  "t": time.time()}
    _save_games()
    return gid


@app.post("/open")
def open_game():
    p = (request.get_json(silent=True) or {}).get("path") or ""
    if not p:
        if not pick_lock.acquire(blocking=False):          # 이미 떠 있는 창이 있으면 또 띄우지 않음
            return jsonify(error="파일 고르는 창이 이미 떠 있어요. 다른 창 뒤에 숨어 있을 수 있어요 (Alt+Tab)."), 409
        try:
            p = _pick_file()
        except Exception as e:
            return jsonify(error=f"파일 열기 창을 못 띄웠어요: {e}"), 500
        finally:
            pick_lock.release()
    if not p:
        return jsonify(cancel=True)
    if not os.path.isfile(p):
        return jsonify(error="그 파일을 못 찾았어요"), 400
    gid = _register(p)
    return jsonify(gid=gid, url=f"/g/{gid}/" + games[gid]["entry"])


@app.get("/games")
def game_list():
    out = [dict(gid=k, url=f"/g/{k}/" + v["entry"], n=len(_cache(k)), pre=pre.get(k) or {}, **v) for k, v in games.items() if os.path.isdir(v["dir"])]
    return jsonify(sorted(out, key=lambda x: -x["t"]))


@app.post("/forget/<gid>")
def forget(gid):
    games.pop(gid, None)
    _save_games()
    return jsonify(ok=True)


INJECT = b'<script src="/__tr.js?v=%d"></script>'


@app.get("/g/<gid>/<path:sub>")
def serve(gid, sub):
    g = games.get(gid)
    if not g:
        abort(404)

    root = os.path.realpath(g["dir"])
    p = os.path.realpath(os.path.join(root, *sub.split("/")))
    if not (p == root or p.startswith(root + os.sep)) or not os.path.isfile(p):
        abort(404)
    if p.lower().endswith((".html", ".htm")):                # 화면 파일에 '번역 도우미'를 끼워 넣음
        b = open(p, "rb").read()
        tag = INJECT % int(os.path.getmtime(os.path.join(BASE, "tr.js")))
        m = re.search(rb"<head[^>]*>", b, re.I) or re.search(rb"<html[^>]*>", b, re.I)
        b = b[:m.end()] + tag + b[m.end():] if m else tag + b
        r = Response(b, mimetype="text/html")
        r.headers["Cache-Control"] = "no-store"
        return r
    return send_file(p, conditional=True, max_age=0)


@app.get("/__tr.js")
def tr_js():
    r = send_file(os.path.join(BASE, "tr.js"), mimetype="text/javascript", max_age=0)
    r.headers["Cache-Control"] = "no-store"
    return r


@app.get("/setup")
def setup_state():
    return jsonify(setup)


@app.get("/")
def index():
    return send_file(os.path.join(BASE, "index.html"))


if __name__ == "__main__":
    _load_games()
    threading.Thread(target=run_setup, daemon=True).start()
    app.run(host="127.0.0.1", port=7880, threaded=True)
