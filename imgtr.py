"""이미지 속 일본어·중국어 글자를 한국어로 바꿔 넣기 (상품·광고 이미지)
  1. 글자 찾기·읽기 — RapidOCR(PP-OCRv6 다국어: 일본어·중국어, 가로·세로)
  2. 줄 묶기 — 한 문단(가로) / 한 단(세로)으로
  3. 번역 — 자막과 같은 번역 모델 (PC 안에서)
  4. 원래 글자 지우기 — LaMa (배경을 자연스럽게 채움)
  5. 같은 자리·방향(가로/세로)·색·테두리·그라데이션·굵기로 한국어 쓰기"""
import os, re, json, math, threading
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(ROOT, "fonts", "NotoSansKR-VF.ttf")
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿ｦ-ﾟ]")

_ocr = None
_lama = None
_lock = threading.Lock()


# ---------------- 1. 글자 찾기·읽기 ----------------
def ocr():
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR
        _ocr = RapidOCR()
    return _ocr


def _geom(poly):
    p = np.array(poly, np.float32)
    w = (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3])) / 2
    h = (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1])) / 2
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    return w, h, float(x0), float(y0), float(x1), float(y1)


def detect(rgb):
    """→ 글자 줄 목록 [{poly, text, vert, x0,y0,x1,y1, size}] (일본어·중국어가 든 줄만)"""
    r = ocr()(rgb)
    out = []
    if r is None or r.boxes is None:
        return out
    for poly, t, s in zip(r.boxes, r.txts, r.scores):
        t = (t or "").strip()
        if float(s) < 0.5 or not t:
            continue
        w, h, x0, y0, x1, y1 = _geom(poly)
        vert = bool(h > 1.3 * w and len(t) >= 2)
        out.append({"poly": [[float(a), float(b)] for a, b in poly], "text": t, "vert": vert, "cjk": bool(CJK.search(t)),
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1, "size": float(w if vert else h)})
    return out


# ---------------- 2. 줄 묶기 ----------------
def _near(b, l):
    """줄 l 이 덩어리 b 에 이어지는가 (같은 방향 · 비슷한 크기 · 바로 아래(가로)/바로 왼쪽(세로) · 비슷한 글자색)"""
    last = b["lines"][-1]
    if not (0.7 < l["size"] / last["size"] < 1.4):
        return False
    if l.get("col") is not None and last.get("col") is not None and np.linalg.norm(np.subtract(l["col"], last["col"])) > 70:
        return False                                       # 글자색이 다르면 다른 문구
    if not b["vert"]:
        gap = l["y0"] - last["y1"]
        ov = min(l["x1"], b["x1"]) - max(l["x0"], b["x0"])
        return -0.3 * l["size"] < gap < 0.8 * l["size"] and ov > 0.3 * min(l["x1"] - l["x0"], b["x1"] - b["x0"])
    gap = last["x0"] - l["x1"]
    ov = min(l["y1"], b["y1"]) - max(l["y0"], b["y0"])
    return -0.3 * l["size"] < gap < 0.8 * l["size"] and ov > 0.3 * min(l["y1"] - l["y0"], b["y1"] - b["y0"])


def _tail(b, l):
    """일본어·중국어가 없는 짧은 조각(！, 100% 같은 것)이 덩어리 끝에 붙어 있는가 → 같이 지우고 같이 번역"""
    s = b["size"]
    if b["vert"]:
        ov = min(l["x1"], b["x1"]) - max(l["x0"], b["x0"])
        return ov > 0.5 * min(l["x1"] - l["x0"], b["x1"] - b["x0"]) and -0.3 * s < l["y0"] - b["y1"] < 0.6 * s
    last = b["lines"][-1]
    ov = min(l["y1"], last["y1"]) - max(l["y0"], last["y0"])
    return ov > 0.5 * min(l["y1"] - l["y0"], last["y1"] - last["y0"]) and -0.3 * s < l["x0"] - last["x1"] < 0.8 * s


def group(lines):
    """가로는 위→아래로 이어지는 줄, 세로는 오른쪽→왼쪽으로 이어지는 단을 한 덩어리로.
    일본어·중국어가 없는 줄은 혼자서는 안 건드리고(브랜드·영어), 덩어리 끝에 붙은 조각만 같이"""
    blocks = []
    hs = sorted([l for l in lines if l["cjk"] and not l["vert"]], key=lambda l: (l["y0"], l["x0"]))
    vs = sorted([l for l in lines if l["cjk"] and l["vert"]], key=lambda l: (-l["x1"], l["y0"]))
    for l in hs + vs:
        for b in blocks:
            if b["vert"] == l["vert"] and _near(b, l):
                _add(b, l)
                break
        else:
            blocks.append(_new(l))
    for l in [l for l in lines if not l["cjk"]]:
        for b in blocks:
            if _tail(b, l):
                b["lines"].append(dict(l, tail=True))
                b["x0"], b["y0"] = min(b["x0"], l["x0"]), min(b["y0"], l["y0"])
                b["x1"], b["y1"] = max(b["x1"], l["x1"]), max(b["y1"], l["y1"])
                break
    for b in blocks:
        fs = [l["fs"] for l in b["lines"] if l.get("fs") and not l.get("tail")]
        b["fs"] = float(np.median(fs)) if fs else b["size"] * 0.6
        b["text"] = "".join(l["text"] for l in b["lines"] if not l.get("tail"))
        tl = [l["text"] for l in b["lines"] if l.get("tail")]
        b["tail"] = "".join("!" if x in ("8", "1", "l", "I", "|", "！", "!") else x for x in tl)   # 끝에 붙은 ！ 같은 것 (8로 잘못 읽힐 때가 있음)
        xs = [l["x0"] for l in b["lines"] if not l.get("tail")]
        b["align"] = "left" if (not b["vert"] and len(xs) > 1 and max(xs) - min(xs) < 0.4 * b["size"]) else "center"
    blocks.sort(key=lambda b: (b["y0"], b["x0"]))
    return blocks


def _new(l):
    return {"vert": l["vert"], "lines": [l], "x0": l["x0"], "y0": l["y0"], "x1": l["x1"], "y1": l["y1"], "size": l["size"]}


def _add(b, l):
    b["lines"].append(l)
    b["x0"], b["y0"] = min(b["x0"], l["x0"]), min(b["y0"], l["y0"])
    b["x1"], b["y1"] = max(b["x1"], l["x1"]), max(b["y1"], l["y1"])
    b["size"] = float(np.median([x["size"] for x in b["lines"]]))


# ---------------- 3. 번역 ----------------
TR_SYSTEM = ("You translate short Japanese or Chinese text printed on product photos and advertising images of an adult "
             "products online shop into natural Korean marketing copy. Keep each line short and punchy, about the same "
             "length as the original. Keep numbers, units, model names, brand names and English words as they are. "
             "Translate the meaning, never transliterate. Output exactly one Korean line per numbered input, as "
             "'N. 번역', and nothing else.")


def with_tail(b, ko):
    """번역 뒤에 덩어리 끝 조각(！ 등)을 붙임"""
    tl = b.get("tail") or ""
    ko = (ko or "").strip()
    return ko + tl if ko and tl and not ko.endswith(tl) else ko


def translate(texts, tr, extra=""):
    """번호 붙인 여러 줄을 한 번에 → 한국어 목록"""
    out = [""] * len(texts)
    for k in range(0, len(texts), 20):
        part = texts[k:k + 20]
        u = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(part))
        txt = tr.chat([{"role": "system", "content": TR_SYSTEM + extra}, {"role": "user", "content": u}], 60 * len(part) + 64)
        for line in txt.splitlines():
            m = re.match(r"\s*(\d+)[.)]\s*(.*)", line)
            if m and 1 <= int(m.group(1)) <= len(part):
                out[k + int(m.group(1)) - 1] = m.group(2).strip()
    return out


# ---------------- 4. 글자 지우기 ----------------
def lama():
    global _lama
    if _lama is None:
        import torch, urllib.request
        p = os.path.join(ROOT, "cache", "lama", "big-lama.pt")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            urllib.request.urlretrieve(LAMA_URL, p + ".part")
            os.replace(p + ".part", p)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = torch.jit.load(p, map_location=dev)
        m.eval()
        _lama = (m, dev)
    return _lama


def block_mask(shape, b):
    m = np.zeros(shape[:2], np.uint8)
    for l in b["lines"]:
        cv2.fillPoly(m, [np.array(l["poly"], np.int32)], 255)
    k = max(3, int(b["size"] * 0.18))                      # 테두리·그림자까지 덮게 조금 넓게
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))


def inpaint(rgb, mask):
    """mask(255) 부분을 배경으로 채움. 글자 둘레만 잘라서 해서 큰 상세페이지도 빠르고 메모리 적게"""
    import torch
    m, dev = lama()
    out = rgb.copy()
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    H, W = mask.shape
    rects = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        pad = max(48, int(max(w, h) * 0.5))
        rects.append([max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)])
    merged = True
    while merged:                                          # 겹치는 자리는 한 번에
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, c = rects[i], rects[j]
                if a[0] < c[2] and c[0] < a[2] and a[1] < c[3] and c[1] < a[3]:
                    rects[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3])]
                    rects.pop(j)
                    merged = True
                    break
            if merged:
                break
    for X0, Y0, X1, Y1 in rects:
        crop, mk = rgb[Y0:Y1, X0:X1], mask[Y0:Y1, X0:X1]
        ring = (cv2.dilate(mk, np.ones((9, 9), np.uint8)) > 0) & (mk == 0)
        if ring.sum() > 50 and crop[ring].std(0).max() < 6:     # 둘레가 거의 한 색(단색 배경) → 그 색으로 채움
            col = np.median(crop[ring], 0)
            a = (cv2.GaussianBlur(mk, (5, 5), 0).astype(np.float32) / 255)[..., None]
            out[Y0:Y1, X0:X1] = (col * a + out[Y0:Y1, X0:X1] * (1 - a)).astype(np.uint8)
            continue
        s = min(1.0, 1024 / max(crop.shape[:2]))
        c2 = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
        m2 = cv2.resize(mk, (c2.shape[1], c2.shape[0]), interpolation=cv2.INTER_NEAREST) if s < 1 else mk
        h2, w2 = c2.shape[:2]
        ph, pw = (8 - h2 % 8) % 8, (8 - w2 % 8) % 8
        xi = np.pad(c2.astype(np.float32) / 255, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mi = np.pad((m2 > 0).astype(np.float32), ((0, ph), (0, pw)))
        with torch.no_grad():
            r = m(torch.from_numpy(xi).permute(2, 0, 1)[None].to(dev), torch.from_numpy(mi)[None, None].to(dev))
        r = r[0].permute(1, 2, 0).float().cpu().numpy()[:h2, :w2]
        r = (r.clip(0, 1) * 255 if r.max() <= 1.5 else r.clip(0, 255)).astype(np.uint8)
        if s < 1:
            r = cv2.resize(r, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
        a = (cv2.GaussianBlur(mk, (5, 5), 0).astype(np.float32) / 255)[..., None]
        out[Y0:Y1, X0:X1] = (r * a + out[Y0:Y1, X0:X1] * (1 - a)).astype(np.uint8)
    return out


# ---------------- 5. 글자 모양 읽기 + 한국어 쓰기 ----------------
def style(rgb, clean, b, pal=True):
    """원래 글자의 색·테두리·위아래 그라데이션·굵기"""
    x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(math.ceil(b["x1"])), int(math.ceil(b["y1"]))
    k = int(b["size"] * 0.15)
    x0, y0, x1, y1 = max(0, x0 - k), max(0, y0 - k), min(rgb.shape[1], x1 + k), min(rgb.shape[0], y1 + k)
    o, c = rgb[y0:y1, x0:x1].astype(np.int32), clean[y0:y1, x0:x1].astype(np.int32)
    bm = block_mask(rgb.shape, b)
    area = bm[y0:y1, x0:x1] > 0                             # 글자 자리 안에서만
    ink = (np.abs(o - c).sum(2) > 70) & area
    # 글자 자리 바로 바깥 배경에 흔한 색이면 글자가 아님 (무늬 있는 배경에서 배경을 글자로 잘못 잡지 않게)
    g = max(3, int(b["fs"] * 0.25))                        # 글자 찾기 상자가 꾸밈 테두리를 다 못 덮을 수 있어서 조금 떨어진 곳
    inner = cv2.dilate(bm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    ring = (cv2.dilate(inner, np.ones((15, 15), np.uint8)) > 0) & (inner == 0)
    rp = rgb[ring].astype(np.float32)
    if pal and len(rp) > 50:
        rp = rp[np.random.default_rng(0).choice(len(rp), min(len(rp), 4000), replace=False)]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, _, pal = cv2.kmeans(rp, min(8, len(rp)), None, crit, 2, cv2.KMEANS_PP_CENTERS)
        dpal = np.min(np.linalg.norm(o[..., None, :].astype(np.float32) - pal[None, None], axis=-1), axis=-1)
        ink &= dpal > 40
    ink = ink.astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    st = {"fill": [255, 255, 255], "outline": None, "grad": None, "weight": 700}
    if ink.sum() < 20:
        bg = c.reshape(-1, 3).mean(0)
        st["fill"] = [20, 20, 20] if bg.mean() > 128 else [245, 245, 245]
        return st
    px = o[ink > 0].astype(np.float32)
    dmap = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    dist = dmap[ink > 0]
    # 두 가지 색으로 나눠 보기: 테두리는 가장자리(거리 작음)에 있고, 글자색은 안쪽
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    d0 = dist[lab == 0].mean() if (lab == 0).any() else 0
    d1 = dist[lab == 1].mean() if (lab == 1).any() else 0
    bnd = (dist <= 1.5)                                    # 글자 덩어리의 맨 가장자리 (배경과 맞닿은 곳)
    b0 = (lab[bnd] == 0).mean() if bnd.any() else 0.5
    oi = 0 if b0 > 0.5 else 1                              # 가장자리를 더 많이 차지한 색 = 테두리 후보
    fc, oc = cen[1 - oi], cen[oi]
    frac_out = (lab == oi).mean()
    bgc = np.median(c[(ink == 0) & area].reshape(-1, 3), 0) if ((ink == 0) & area).any() else c.reshape(-1, 3).mean(0)
    # 그냥 글자 가장자리가 배경과 섞인 색이면 테두리가 아님 (섞인 색은 글자색과 배경색 사이에 있음)
    between = np.linalg.norm(fc - oc) + np.linalg.norm(oc - bgc) < 1.25 * np.linalg.norm(fc - bgc)
    edge = max(b0, 1 - b0) > 0.65                           # 테두리라면 가장자리 대부분을 차지
    if frac_out > 0.15 and np.linalg.norm(fc - oc) > 80 and not between and edge:
        st["outline"] = [int(v) for v in oc]
        fill_px = (lab != oi) & (dist >= 1.5)              # 테두리 색이 아닌 안쪽 = 글자
    elif min(frac_out, 1 - frac_out) > 0.2 and np.linalg.norm(fc - oc) > 80 and not between:
        ii = 0 if d0 >= d1 else 1                          # 두 색이 뚜렷한데 테두리로 확신이 없으면 안쪽(획 가운데) 색을 글자색으로
        fill_px = (lab == ii) & (dist >= 1.5)
    else:
        fill_px = dist >= 1.5                              # 가장자리 섞인 색은 빼고
    if fill_px.sum() < 10:
        fill_px = np.ones(len(dist), bool)
    st["fill"] = [int(v) for v in np.median(px[fill_px], 0)]
    ys = np.nonzero(ink)[0][fill_px]
    if len(ys) > 50:
        top, bot = np.percentile(ys, 25), np.percentile(ys, 75)
        ct, cb = np.median(px[fill_px][ys <= top], 0), np.median(px[fill_px][ys >= bot], 0)
        if np.linalg.norm(ct - cb) > 60:
            st["grad"] = [[int(v) for v in ct], [int(v) for v in cb]]
    # 굵기: 글자 획 두께 ÷ 글자 크기
    fm = np.zeros_like(ink)
    iy, ix = np.nonzero(ink)
    fm[iy[fill_px], ix[fill_px]] = 1
    dd = cv2.distanceTransform(fm, cv2.DIST_L2, 3)[fm > 0]
    th = 2 * float(np.percentile(dd, 90)) / max(1.0, b["fs"]) if len(dd) else 0.1
    st["weight"] = 900 if th > 0.135 else 700 if th > 0.112 else 500   # 본고딕 굵기로 대략 (화면에서 바꿀 수 있음)
    if st["outline"]:
        st["weight"] = max(st["weight"], 700)              # 테두리 있는 광고 글씨는 대부분 굵음
    return st


def line_color(rgb, clean, l):
    """줄 하나의 대략적인 글자색 (덩어리 나눌 때 씀)"""
    x0, y0, x1, y1 = int(l["x0"]), int(l["y0"]), int(math.ceil(l["x1"])), int(math.ceil(l["y1"]))
    o, c = rgb[y0:y1, x0:x1].astype(np.int32), clean[y0:y1, x0:x1].astype(np.int32)
    if o.size == 0:
        return None
    ink = np.abs(o - c).sum(2) > 70
    if ink.sum() < 10:
        return None
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    core = dist >= max(1.0, np.percentile(dist[ink], 60))      # 획 가운데 (테두리 말고)
    prof = ink.sum(1 if not l["vert"] else 0)                  # 글자 실제 높이(가로)/너비(세로) → 글자 크기
    idx = np.nonzero(prof > max(1, prof.max() * 0.05))[0]
    if len(idx):
        l["fs"] = float(idx[-1] - idx[0] + 1) / 0.92
    return [float(v) for v in np.median(o[core], 0)]


def _font(size, weight):
    f = ImageFont.truetype(FONT, max(8, int(size)))
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def _wrap(text, f, maxw, draw):
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    out = []
    for l in lines:                                        # 한 낱말이 너무 길면 글자 단위로 끊음
        while draw.textlength(l, font=f) > maxw and len(l) > 1:
            k = len(l)
            while k > 1 and draw.textlength(l[:k], font=f) > maxw:
                k -= 1
            out.append(l[:k])
            l = l[k:]
        out.append(l)
    return out


def layout(b, ko, st):
    """한국어를 덩어리 자리에 맞춰 놓을 글자 크기와 위치 → [(x, y, 글자)], 크기"""
    W, H = b["x1"] - b["x0"], b["y1"] - b["y0"]
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    base = b["fs"]                                          # 원래 글자 크기
    for s in np.linspace(base, base * 0.35, 25):
        f = _font(s, st["weight"])
        if not b["vert"]:
            lines = _wrap(ko, f, W * 1.02, tmp)
            lh = s * 1.22
            if len(lines) * lh <= H * 1.08 or s <= base * 0.36:
                y = b["y0"] + (H - len(lines) * lh) / 2 + (lh - s) / 2 - s * 0.12
                items = []
                for i, l in enumerate(lines):
                    lw = tmp.textlength(l, font=f)
                    x = b["x0"] if b["align"] == "left" else b["x0"] + (W - lw) / 2
                    items.append((x, y + i * lh, l))
                return items, s
        else:
            chars = [ch for ch in ko if ch != " "]
            step = s * 1.08
            per = max(1, int(H // step))
            cols = math.ceil(len(chars) / per)
            if cols * s * 1.15 <= W * 1.08 or s <= base * 0.36:
                items = []
                total_w = cols * s * 1.15
                xr = b["x0"] + (W + total_w) / 2 - s * 1.15        # 오른쪽 단부터
                for ci in range(cols):
                    col = chars[ci * per:(ci + 1) * per]
                    y = b["y0"] + (H - len(col) * step) / 2 if cols == 1 else b["y0"]
                    for j, ch in enumerate(col):
                        cw = tmp.textlength(ch, font=f)
                        items.append((xr - ci * s * 1.15 + (s * 1.15 - cw) / 2, y + j * step - s * 0.12, ch))
                return items, s
    return [], base


def render(clean, blocks):
    """지운 그림 위에 블록마다 한국어를 그려 넣음"""
    img = Image.fromarray(clean).convert("RGBA")
    for b in blocks:
        ko = (b.get("ko") or "").strip()
        if not ko or b.get("skip"):
            continue
        st = b["style"]
        items, s = layout(b, ko, st)
        if not items:
            continue
        f = _font(s, st["weight"])
        sw = max(1, int(round(s * 0.09))) if st.get("outline") else 0
        W, H = img.size
        fill_m = Image.new("L", (W, H), 0)
        out_m = Image.new("L", (W, H), 0)
        df, do = ImageDraw.Draw(fill_m), ImageDraw.Draw(out_m)
        for x, y, t in items:
            df.text((x, y), t, font=f, fill=255)
            if sw:
                do.text((x, y), t, font=f, fill=255, stroke_width=sw, stroke_fill=255)
        if sw:
            img.paste(Image.new("RGBA", (W, H), tuple(st["outline"]) + (255,)), (0, 0), out_m)
        if st.get("grad"):
            ys = np.linspace(0, 1, H)[:, None]
            bb = fill_m.getbbox() or (0, 0, W, H)
            t = np.clip((np.arange(H) - bb[1]) / max(1, bb[3] - bb[1]), 0, 1)[:, None]
            c0, c1 = np.array(st["grad"][0], np.float32), np.array(st["grad"][1], np.float32)
            g = (c0 * (1 - t) + c1 * t).astype(np.uint8)
            grad = np.repeat(g[:, None, :], W, 1)
            img.paste(Image.fromarray(grad).convert("RGBA"), (0, 0), fill_m)
        else:
            img.paste(Image.new("RGBA", (W, H), tuple(st["fill"]) + (255,)), (0, 0), fill_m)
    return np.asarray(img.convert("RGB"))


# ---------------- 한 장 처리 ----------------
def _residual(rgb, b, st):
    """글자 찾기가 놓친 끝 문장부호(！ 등): 덩어리 바로 뒤에 같은 색 조각이 있으면 그 자리 (마스크, 느낌표처럼 길쭉한가)"""
    H, W = rgb.shape[:2]
    fs = b["fs"]
    if b["vert"]:
        x0, x1, y0, y1 = b["x0"], b["x1"], b["y1"], b["y1"] + 1.2 * fs
    else:
        last = [l for l in b["lines"] if not l.get("tail")][-1]
        x0, x1, y0, y1 = last["x1"], last["x1"] + 1.0 * fs, last["y0"], last["y1"]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, False
    z = rgb[y0:y1, x0:x1].astype(np.float32)
    cols = [st["fill"]] + ([st["outline"]] if st.get("outline") else []) + (st["grad"] if st.get("grad") else [])
    near = np.min([np.linalg.norm(z - np.array(c, np.float32), axis=-1) for c in cols], 0) < 45
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stt, _ = cv2.connectedComponentsWithStats(near)
    m = np.zeros((H, W), np.uint8)
    tall = False
    for i in range(1, n):
        x, y, w, h, a = stt[i]
        if a < fs * fs * 0.02 or h < fs * 0.35:                # 너무 작은 건 배경 무늬
            continue
        if x == 0 and not b["vert"] and (lab[:, 0] == i).sum() > h * 0.5:
            pass                                               # 글자에 붙어 이어진 조각 (테두리)도 같이
        m[y0:y1, x0:x1][lab == i] = 255
        tall = tall or h > 2.2 * w
    return (m if m.any() else None), tall


def analyze(path):
    """글자 찾기 → 지우기 → 줄마다 글자색 → 덩어리로 묶기 → 덩어리 모양 읽기 → 글자 획만 다시 지우기 (번역 전까지)
    → (원본, 지운 그림, 블록들). 배경을 최대한 살리려고, 마지막엔 글자 상자 전체가 아니라 글자 획(+테두리) 자리만 새로 채움"""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    lines = detect(rgb)
    if not any(l["cjk"] for l in lines):
        return rgb, rgb, []
    pre = group(lines)                                     # 지울 곳 (덩어리 끝 조각까지)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for b in pre:
        mask = np.maximum(mask, block_mask(rgb.shape, b))
    clean = inpaint(rgb, mask)                             # 1차: 상자째 지운 그림 (글자 모양·색 읽는 데 씀)
    used = [l for b in pre for l in b["lines"]]
    for l in used:
        l.pop("tail", None)
        l["col"] = line_color(rgb, clean, l)
    blocks = group(used)                                   # 글자색까지 보고 다시 묶기
    ink = ((np.abs(rgb.astype(np.int32) - clean.astype(np.int32)).sum(2) > 60) & (mask > 0)).astype(np.uint8) * 255
    for b in blocks:
        b["style"] = style(rgb, clean, b)
        rm, tall = _residual(rgb, b, b["style"])           # 놓친 ！ 같은 것
        if rm is not None:
            ink = np.maximum(ink, rm)
            if tall and not b.get("tail"):
                b["tail"] = "!"
    # 2차: 글자 획 자리만 (가장자리 섞인 색까지 조금 넓혀서) 원본에서 다시 채움 → 글자 사이·둘레 배경은 원본 그대로
    k = max(3, int(np.median([b["fs"] for b in blocks]) * 0.06))
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    clean = inpaint(rgb, ink)
    for b in blocks:                                       # 글자 획 자리만 새로 채운 그림과 비교하면 글자 픽셀이 정확 → 색·테두리 다시 읽기
        tl = b.get("tail")
        b["style"] = style(rgb, clean, b, pal=False)
        b["tail"] = tl
    return rgb, clean, blocks


def save_img(arr, path, like):
    """원본과 같은 형식으로 저장 (jpg는 높은 화질)"""
    ext = os.path.splitext(like)[1].lower()
    im = Image.fromarray(arr)
    if ext in (".jpg", ".jpeg"):
        im.save(path, quality=95, subsampling=0)
    elif ext == ".webp":
        im.save(path, quality=95)
    else:
        im.save(path)
