"""이미지 번역기 — 상품·광고 이미지 속 일본어·중국어를 한국어로 바꿔 넣기 (모자이크 생성기와 별개 프로그램)
켜면 브라우저에서 http://127.0.0.1:7870 이 열림"""
import os, re, io, json, time, uuid, zipfile, threading, traceback
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, Response

BASE = os.path.dirname(os.path.abspath(__file__))
for k, sub in (("HF_HOME", "hf"), ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg")):   # 받는 모델은 전부 이 폴더 안에
    os.environ.setdefault(k, os.path.join(BASE, "cache", sub))
import imgtr, llm, craftnet

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3
tr_lock = threading.Lock()
setup = {"running": False, "msg": "", "error": ""}


def run_setup():
    """처음 켤 때 한 번: 필요한 모델을 미리 받아 둠"""
    setup.update(running=True, error="")
    try:
        setup["msg"] = "글자 지우기 모델 받는 중 (처음 한 번만 약 200MB)"
        imgtr.lama()
        setup["msg"] = "휜 글씨 찾기 모델 받는 중 (처음 한 번만 약 80MB)"
        craftnet.net(os.path.join(BASE, "cache", "craft"))
        setup["msg"] = "번역 모델 받는 중 (처음 한 번만 약 5GB)"
        llm.ensure_translator(setup)
        setup["msg"] = "준비 끝"
    except Exception as e:
        traceback.print_exc()
        setup["error"] = str(e)[:300]
    finally:
        setup["running"] = False


@app.get("/setup")
def setup_state():
    return jsonify(setup)


# ---------------- 이미지 번역 (상품·광고 이미지 속 일본어·중국어 → 한국어) ----------------
IMG_DIR = os.path.join(BASE, "jobs")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
img_state = {}


def _safe_name(n):
    n = os.path.basename(n or "").replace("\\", "_")
    return re.sub(r'[<>:"/|?*]', "_", n) or "image.png"


def _img_blocks_public(bl):
    return [{"text": b["text"], "ko": b.get("ko", ""), "vert": b["vert"], "skip": bool(b.get("skip")),
             "box": [b["x0"], b["y0"], b["x1"], b["y1"]], "style": b["style"]} for b in bl]


def _img_job(iid):
    st = img_state[iid]
    d = os.path.join(IMG_DIR, iid)
    names = st["files"]
    try:
        allb = {}
        for k, n in enumerate(names):
            st.update(msg=f"글자 찾고 지우는 중 {k + 1}/{len(names)} — {n}", pct=int(5 + 55 * k / max(1, len(names))))
            rgb, clean, bl = imgtr.analyze(os.path.join(d, "in", n))
            imgtr.save_img(clean, os.path.join(d, "work", n + ".clean.png"), "x.png")
            allb[n] = bl
        texts = [(n, i, b["text"]) for n, bl in allb.items() for i, b in enumerate(bl)]
        if texts:
            st.update(msg="번역 준비 중 (자막 번역이 돌고 있으면 끝나길 기다려요)", pct=62)
            with tr_lock:
                tr = llm.Translator(st)
                try:
                    st.update(msg=f"번역 중 ({len(texts)}곳)", pct=70)
                    kos = imgtr.translate([x[2] for x in texts], tr)
                finally:
                    tr.close()
            for (n, i, _), ko in zip(texts, kos):
                allb[n][i]["ko"] = imgtr.with_tail(allb[n][i], ko)
        for k, n in enumerate(names):
            st.update(msg=f"한국어 쓰는 중 {k + 1}/{len(names)}", pct=int(85 + 14 * k / max(1, len(names))))
            _img_render(iid, n, allb[n])
            st["done"].append(n)
        st.update(msg=f"끝 — {len(names)}장", pct=100)
    except Exception as e:
        traceback.print_exc()
        st["error"] = str(e)[:300]
    finally:
        st["running"] = False


def _img_render(iid, n, bl):
    d = os.path.join(IMG_DIR, iid)
    clean = np.asarray(Image.open(os.path.join(d, "work", n + ".clean.png")).convert("RGB"))
    out = imgtr.render(clean, bl)
    imgtr.save_img(out, os.path.join(d, "out", n), n)
    json.dump(bl, open(os.path.join(d, "work", n + ".json"), "w", encoding="utf-8"), ensure_ascii=False)


@app.get("/")
def page():
    return Response(open(os.path.join(BASE, "index.html"), encoding="utf-8").read(), mimetype="text/html")


@app.post("/imgtr/run")
def imgtr_run():
    fs = [f for f in request.files.getlist("files") if f and os.path.splitext(f.filename or "")[1].lower() in IMG_EXT]
    if not fs:
        return jsonify(error="이미지 파일(png·jpg·webp)을 넣어 주세요"), 400
    iid = time.strftime("%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    d = os.path.join(IMG_DIR, iid)
    for s in ("in", "work", "out"):
        os.makedirs(os.path.join(d, s), exist_ok=True)
    names = []
    for f in fs:
        n = _safe_name(f.filename)
        while n in names:
            n = "1_" + n
        f.save(os.path.join(d, "in", n))
        names.append(n)
    img_state[iid] = {"running": True, "msg": "시작", "pct": 0, "files": names, "done": [], "error": ""}
    threading.Thread(target=_img_job, args=(iid,), daemon=True).start()
    return jsonify(id=iid)


@app.get("/imgtr/state/<iid>")
def imgtr_stateview(iid):
    iid = os.path.basename(iid)
    st = img_state.get(iid)
    d = os.path.join(IMG_DIR, iid)
    if st is None:
        if not os.path.isdir(d):
            return jsonify(error="없는 작업"), 404
        names = sorted(os.listdir(os.path.join(d, "in")))
        st = {"running": False, "msg": "", "pct": 100, "files": names,
              "done": [n for n in names if os.path.exists(os.path.join(d, "out", n))], "error": ""}
    blocks = {}
    for n in st["done"]:
        try:
            blocks[n] = _img_blocks_public(json.load(open(os.path.join(d, "work", n + ".json"), encoding="utf-8")))
        except Exception:
            blocks[n] = []
    return jsonify(dict(st, blocks=blocks))


@app.get("/imgtr/img/<iid>/<kind>/<path:name>")
def imgtr_img(iid, kind, name):
    if kind not in ("in", "out"):
        return "", 404
    p = os.path.join(IMG_DIR, os.path.basename(iid), kind, os.path.basename(name))
    return send_file(p, conditional=True) if os.path.exists(p) else ("", 404)


@app.post("/imgtr/edit/<iid>/<path:name>")
def imgtr_edit(iid, name):
    """화면에서 고친 번역·색·굵기로 다시 그림 (번역 모델 없이 바로)"""
    iid, name = os.path.basename(iid), os.path.basename(name)
    p = os.path.join(IMG_DIR, iid, "work", name + ".json")
    if not os.path.exists(p):
        return jsonify(error="없는 이미지"), 404
    bl = json.load(open(p, encoding="utf-8"))
    for b, e in zip(bl, (request.get_json(silent=True) or {}).get("blocks", [])):
        b["ko"] = e.get("ko", b.get("ko", ""))
        b["skip"] = bool(e.get("skip"))
        for k in ("fill", "outline", "grad", "weight"):
            if k in e:
                b["style"][k] = e[k]
    _img_render(iid, name, bl)
    return jsonify(ok=True)


@app.get("/imgtr/zip/<iid>")
def imgtr_zip(iid):
    d = os.path.join(IMG_DIR, os.path.basename(iid), "out")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for n in sorted(os.listdir(d)):
            z.write(os.path.join(d, n), n)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"번역이미지_{os.path.basename(iid)}.zip", mimetype="application/zip")

if __name__ == "__main__":
    threading.Thread(target=run_setup, daemon=True).start()
    app.run(host="127.0.0.1", port=7870, threaded=True)
