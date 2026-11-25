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


def translate(gid, items, ctx):
    """일본어 여러 조각 → 한국어 (한 번 번역한 건 기억해 두고 바로 돌려줌)"""
    c = _cache(gid)
    todo = [t for t in dict.fromkeys(items) if t not in c]
    if todo:
        t_ = _translator()
        for k in range(0, len(todo), 12):
            part = todo[k:k + 12]
            u = ""
            if ctx:
                u += "Earlier lines (context only, do not translate):\n" + "\n".join(f"- {a} → {b}" for a, b in ctx[-8:]) + "\n\n"
            u += "Translate:\n" + "\n".join(f"{i + 1}. {t.replace(chr(10), ' ')}" for i, t in enumerate(part))
            with tr_lock:
                txt = t_.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": u}],
                              sum(len(t) for t in part) * 3 + 64 * len(part))
            got = {}
            for line in txt.splitlines():
                m = re.match(r"\s*(\d+)[.)]\s*(.*)", line)
                if m and 1 <= int(m.group(1)) <= len(part):
                    got[int(m.group(1)) - 1] = m.group(2).strip()
            for i, t in enumerate(part):
                if got.get(i):
                    c[t] = got[i]
        _save_cache(gid)
    return [c.get(t, "") for t in items]


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
def _pick_file():
    """윈도우 '파일 열기' 창으로 HTML 고르기 → 경로 (취소하면 "")"""
    ps = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8;Add-Type -AssemblyName System.Windows.Forms;"
          "$d=New-Object System.Windows.Forms.OpenFileDialog;$d.Filter='HTML (*.html;*.htm)|*.html;*.htm|모든 파일 (*.*)|*.*';"
          "$d.Title='번역해서 열 HTML 파일 고르기';"
          "$f=New-Object System.Windows.Forms.Form;$f.TopMost=$true;"
          "if($d.ShowDialog($f) -eq 'OK'){[Console]::Write($d.FileName)}")
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
        try:
            p = _pick_file()
        except Exception as e:
            return jsonify(error=f"파일 열기 창을 못 띄웠어요: {e}"), 500
    if not p:
        return jsonify(cancel=True)
    if not os.path.isfile(p):
        return jsonify(error="그 파일을 못 찾았어요"), 400
    gid = _register(p)
    return jsonify(gid=gid, url=f"/g/{gid}/" + games[gid]["entry"])


@app.get("/games")
def game_list():
    out = [dict(gid=k, url=f"/g/{k}/" + v["entry"], n=len(_cache(k)), **v) for k, v in games.items() if os.path.isdir(v["dir"])]
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
