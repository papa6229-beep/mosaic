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
SERIF = os.path.join(ROOT, "cache", "fonts", "NotoSerifKR-VF.ttf")      # 명조(본명조) — 처음 켤 때 받음
SERIF_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifkr/NotoSerifKR%5Bwght%5D.ttf"
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
        l = {"poly": [[float(a), float(b)] for a, b in poly], "text": t, "vert": vert, "cjk": bool(CJK.search(t)),
             "x0": x0, "y0": y0, "x1": x1, "y1": y1, "size": float(w if vert else h)}
        q = np.array(poly, np.float32)
        ang = math.degrees(math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]))
        if not vert and 8 < abs(ang) < 80 and l["cjk"]:     # 비스듬한 가로줄 → 같은 기울기로 다시 쓰게
            a, b = (q[0] + q[3]) / 2, (q[1] + q[2]) / 2
            l["curve"] = {"pts": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]], "mode": "rot", "cs": float(h)}
        out.append(l)
    return out


# ---------------- 1-2. 휘거나 비스듬한 글씨 (글자 하나씩 찾아 이어서 읽기) ----------------
def _resample(pts, step=1.0):
    p = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    L = np.concatenate([[0], np.cumsum(seg)])
    ts = np.arange(0, L[-1] + 1e-6, step)
    xs, ys = np.interp(ts, L, p[:, 0]), np.interp(ts, L, p[:, 1])
    P = np.stack([xs, ys], 1)
    T = np.gradient(P, axis=0)
    k = max(3, int(len(P) * 0.08)) | 1                       # 방향은 부드럽게
    T = np.stack([np.convolve(np.pad(T[:, i], k // 2, mode="edge"), np.ones(k) / k, mode="valid") for i in range(2)], 1)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    return P, T, float(L[-1])


def _extend(pts, cs):
    p = np.asarray(pts, np.float64)
    d0 = p[0] - p[1]; d0 /= np.linalg.norm(d0) + 1e-9
    d1 = p[-1] - p[-2]; d1 /= np.linalg.norm(d1) + 1e-9
    return np.vstack([p[0] + d0 * cs * 0.6, p, p[-1] + d1 * cs * 0.6])


def _strip(rgb, pts, cs):
    """경로를 따라 띠 모양으로 잘라 반듯한 가로 줄 그림으로 펴기 (휜 글씨 → 가로 글씨)"""
    P, T, L = _resample(_extend(pts, cs))
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    h = int(cs * 1.6)
    off = np.arange(h) - h / 2
    mx = (P[None, :, 0] + N[None, :, 0] * off[:, None]).astype(np.float32)
    my = (P[None, :, 1] + N[None, :, 1] * off[:, None]).astype(np.float32)
    return cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _upright(rgb, pts, cs):
    """글자는 똑바로 서 있고 줄만 비스듬한 경우: 글자를 하나씩 잘라 가로로 이어 붙임"""
    r = int(cs * 0.6)
    tiles = []
    for x, y in pts:
        x0, y0 = int(x) - r, int(y) - r
        tile = np.full((2 * r, 2 * r, 3), 255, np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(rgb.shape[1], x0 + 2 * r), min(rgb.shape[0], y0 + 2 * r)
        tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
        tiles.append(tile)
    return np.concatenate(tiles, 1) if tiles else None


def _read(img):
    if img is None or img.size == 0:
        return "", 0.0
    r = ocr()(img)
    if r is None or r.boxes is None:
        return "", 0.0
    items = sorted(zip(r.boxes, r.txts, r.scores), key=lambda x: np.mean(np.asarray(x[0])[:, 0]))
    t = "".join(x[1] for x in items)
    s = float(np.mean([x[2] for x in items])) if items else 0.0
    return t, s * min(1.0, len(CJK.findall(t)) / max(1, len(t)) + 0.3)


def curve_lines(rgb, lines):
    """반듯한 줄로 못 찾은 글씨(휘거나 비스듬한)를 글자 하나씩 찾아 이어서 읽음 → 줄 목록(경로 포함)"""
    import craftnet
    hm = craftnet.heat(rgb, os.path.join(ROOT, "cache", "craft"))
    cover = np.zeros(hm.shape, np.uint8)
    for l in lines:                                         # 이미 읽은 줄 자리는 빼고
        cv2.fillPoly(cover, [np.array(l["poly"], np.int32)], 255)
    cover = cv2.dilate(cover, np.ones((9, 9), np.uint8))
    hm[cover > 0] = 0
    blob = (hm > 0.3).astype(np.uint8)
    if blob.sum() < 30:
        return []
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blob)
    pts = []
    for i in range(1, n):
        m = lab == i
        md = float(dist[m].max())
        if md < 3:
            continue
        k = max(3, int(md * 1.3)) | 1
        mx = cv2.dilate(np.where(m, hm, 0).astype(np.float32), np.ones((k, k), np.uint8))
        pk = np.argwhere(m & (hm >= mx - 1e-6) & (hm > 0.5))
        for y, x in pk:
            pts.append((float(x), float(y), max(6.0, float(dist[y, x]) * 2.6)))
    if len(pts) < 3:
        return []
    P = np.array(pts)
    # 가까운 글자끼리 잇기 (최소 연결 나무) → 한 줄씩
    D = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=-1)
    cs = np.maximum(P[:, None, 2], P[None, :, 2])
    ok = (D < 1.9 * cs) & (np.maximum(P[:, None, 2], P[None, :, 2]) < 1.8 * np.minimum(P[:, None, 2], P[None, :, 2]))
    np.fill_diagonal(ok, False)
    seen, out = set(), []
    for s0 in range(len(P)):
        if s0 in seen:
            continue
        comp, stack = [], [s0]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a); comp.append(a)
            stack += [b for b in np.nonzero(ok[a])[0] if b not in seen]
        if len(comp) < 3:
            continue
        # 나무에서 가장 먼 두 끝을 잇는 길 = 글자 순서
        sub = D[np.ix_(comp, comp)] * ok[np.ix_(comp, comp)]

        def far(a):
            dd = {a: 0.0}; prev = {a: None}; q = [a]
            while q:
                u = q.pop()
                for v in range(len(comp)):
                    if sub[u, v] > 0 and v not in dd:
                        dd[v] = dd[u] + sub[u, v]; prev[v] = u; q.append(v)
            e = max(dd, key=dd.get)
            return e, prev
        e1, _ = far(0)
        e2, prev = far(e1)
        path, u = [], e2
        while u is not None:
            path.append(comp[u]); u = prev[u]
        if len(path) < 3:
            continue
        cpts = P[path, :2]
        csz = float(np.median(P[path, 2]))
        best = None
        for mode in ("rot", "up"):
            for order in (cpts, cpts[::-1]):
                img = _strip(rgb, order, csz) if mode == "rot" else _upright(rgb, order, csz)
                txt, sc = _read(img)
                if CJK.search(txt) and (best is None or sc > best[0]):
                    best = (sc, txt, order, mode)
        if not best or best[0] < 0.5:
            continue
        sc, txt, order, mode = best
        Pp, T, _ = _resample(_extend(order, csz))
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        Pp, T, _ = _resample(_extend(_extend(order, csz), csz))   # 끝의 ~ ！ 같은 것까지 지우게 양 끝을 조금 더
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        band = np.vstack([Pp - N * csz * 0.62, (Pp + N * csz * 0.62)[::-1]])
        band = band[:: max(1, len(band) // 60)]
        x0, y0 = band.min(0); x1, y1 = band.max(0)
        out.append({"poly": band.tolist(), "text": txt, "vert": False, "cjk": True,
                    "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "size": csz * 1.4,
                    "curve": {"pts": [[float(a), float(b)] for a, b in order], "mode": mode, "cs": csz}})
    return out


# ---------------- 2. 줄 묶기 ----------------
def _hue_gap(a, b):
    """두 색의 색감 차이 (밝기 빼고, Lab 의 a·b)"""
    la, lb = (cv2.cvtColor(np.uint8([[np.clip(c, 0, 255)]]), cv2.COLOR_RGB2LAB)[0, 0].astype(float) for c in (a, b))
    return float(np.hypot(la[1] - lb[1], la[2] - lb[2]))


def _near(b, l):
    """줄 l 이 덩어리 b 에 이어지는가 (같은 방향 · 비슷한 크기 · 바로 아래(가로)/바로 왼쪽(세로) · 비슷한 글자색)"""
    last = b["lines"][-1]
    if not (0.7 < l["size"] / last["size"] < 1.4):
        return False
    if bool(l.get("plate")) != bool(last.get("plate")):
        return False                                       # 판 위 글씨와 그냥 글씨는 따로
    if l.get("col") is not None and last.get("col") is not None and (
            np.linalg.norm(np.subtract(l["col"], last["col"])) > 70 or _hue_gap(l["col"], last["col"]) > 40):
        return False                                       # 글자색이 다르면 다른 문구 (밝기가 비슷해도 갈색·초록처럼 색감이 다르면)
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
    if len(re.sub(r"\s", "", l["text"])) > 4:
        return False                                       # 긴 영어(로고·브랜드)는 조각이 아님
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
    for l in [l for l in lines if l.get("curve")]:          # 휜 글씨는 그 줄 하나가 한 덩어리
        blocks.append(dict(_new(l), curve=l["curve"]))
    hs = [l for l in hs if not l.get("curve")]
    vs = [l for l in vs if not l.get("curve")]
    for l in hs + vs:
        for b in blocks:
            if b.get("curve"):
                continue
            if b["vert"] == l["vert"] and _near(b, l):
                _add(b, l)
                break
        else:
            blocks.append(_new(l))
    for l in [l for l in lines if not l["cjk"]]:
        for b in blocks:
            if not b.get("curve") and _tail(b, l):
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
    return {"plate": l.get("plate"), "soft": l.get("soft"), "vert": l["vert"], "lines": [l], "x0": l["x0"], "y0": l["y0"], "x1": l["x1"], "y1": l["y1"], "size": l["size"]}


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
        import io
        with open(p, "rb") as fh:                          # 경로에 한글이 있으면 torch 가 파일을 못 열어서, 파이썬으로 읽어 넘김
            m = torch.jit.load(io.BytesIO(fh.read()), map_location=dev)
        m.eval()
        _lama = (m, dev)
    return _lama


def block_mask(shape, b):
    m = np.zeros(shape[:2], np.uint8)
    for l in b["lines"]:
        cv2.fillPoly(m, [np.array(l["poly"], np.int32)], 255)
    k = max(3, int(b["size"] * 0.18))                      # 테두리·그림자까지 덮게 조금 넓게
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))


def fg_mask(rgb, b):
    """큰 장식 글씨용: 상자 전체가 아니라 '바깥 배경에 없는 색' 픽셀(글자 획·무늬)만.
    상자째 지우면 그림 대부분이 비어서 AI가 회색으로 칠해 버림"""
    H, W = rgb.shape[:2]
    box = np.zeros((H, W), np.uint8)
    for l in b["lines"]:
        cv2.fillPoly(box, [np.array(l["poly"], np.int32)], 1)
    g = max(4, int(b["size"] * 0.3))
    ring = (cv2.dilate(box, np.ones((2 * g + 1, 2 * g + 1), np.uint8)) > 0) & (box == 0)
    rp = rgb[ring].astype(np.float32)
    if len(rp) < 50:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(rp, 4, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=4)
    cen = cen[cnt >= 0.1 * len(rp)]                         # 바깥 배경의 주된 색들
    big = cv2.dilate(box, np.ones((g + 1, g + 1), np.uint8))
    px = rgb.astype(np.float32)
    d = np.min(np.stack([np.linalg.norm(px - c, axis=-1) for c in cen]), 0)
    m = ((d > 40) & (big > 0)).astype(np.uint8)
    k = max(3, int(b["size"] * 0.05)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return m * 255


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


# ---------------- 4-2. 정밀 다시 그리기 (RORem: 지우기 전용으로 훈련된 그림 AI) ----------------
RE_REPO = "LetsThink/RORem"
RE_DIR = os.path.join(ROOT, "cache", "rorem")
RE_STEPS = 25
_re = {"pipe": None, "tried": False}
_re_lock = threading.Lock()


def rorem_download(job=None):
    """처음 켤 때 한 번 받음 (약 7GB). 이미 있으면 그냥 넘어감"""
    if os.path.exists(os.path.join(RE_DIR, "model_index.json")) and os.path.exists(os.path.join(RE_DIR, ".done")):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(RE_REPO, local_dir=RE_DIR)
    open(os.path.join(RE_DIR, ".done"), "w").write("ok")


def rorem():
    """정밀 다시 그리기 모델. 없거나(안 받았거나) 그래픽카드가 없으면 None → 기존 지우개만 씀"""
    with _re_lock:
        if _re["tried"]:
            return _re["pipe"]
        if not os.path.exists(os.path.join(RE_DIR, ".done")):
            return None                                    # 아직 받는 중이면 이번엔 기존 지우개로 (받고 나면 다음 작업부터 씀)
        _re["tried"] = True
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            from diffusers import StableDiffusionXLInpaintPipeline
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(RE_DIR, torch_dtype=torch.float16)
            pipe.set_progress_bar_config(disable=True)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if vram < 14:                                  # 8GB 그래픽카드: 쓰는 부분만 그래픽카드에 올림
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            _re["pipe"] = pipe
        except Exception as e:
            print("정밀 다시 그리기 모델을 못 켰어요 (기존 지우개로 계속):", e)
            _re["pipe"] = None
        return _re["pipe"]


def _textured(crop, mk):
    """도려낸 자리 둘레가 무늬·그림인가 (단색·완만한 그라데이션이면 기존 지우개로 충분)"""
    ring = (cv2.dilate(mk, np.ones((15, 15), np.uint8)) > 0) & (mk == 0)
    if ring.sum() < 50:
        return False
    g = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    return float(np.abs(g[ring]).mean()) > RE_TEX


RE_TEX = 4.0


def regen(rgb, base, mask, pipe=None):
    """무늬·그림 위를 도려낸 자리만 RORem 으로 다시 그림 (나머지는 base 그대로).
    도려낸 덩어리마다 둘레를 넉넉히 잘라 1024 크기로 그린 뒤, 도려낸 자리만 부드럽게 붙임"""
    pipe = pipe or rorem()
    if pipe is None or not mask.any():
        return base
    import torch
    out = base.copy()
    H, W = mask.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    rects = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        pad = max(64, int(max(w, h) * 0.6))
        rects.append([max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)])
    merged = True
    while merged:
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
        crop, mk = rgb[Y0:Y1, X0:X1], (mask[Y0:Y1, X0:X1] > 0).astype(np.uint8) * 255
        if not mk.any() or not _textured(crop, mk):
            continue
        h, w = crop.shape[:2]
        s = 1024 / max(h, w)                               # 그림 AI 가 잘 그리는 크기 (긴 쪽 1024, 8의 배수)
        tw, th = max(512, int(round(w * s / 8)) * 8), max(512, int(round(h * s / 8)) * 8)
        tw, th = min(tw, 1536), min(th, 1536)
        src = out[Y0:Y1, X0:X1]                            # 기존 지우개로 먼저 채운 그림을 바탕으로 (밑그림이 자연스러워 결과가 안정적)
        im = Image.fromarray(src).resize((tw, th), Image.LANCZOS)
        mm = Image.fromarray(cv2.dilate(mk, np.ones((5, 5), np.uint8))).resize((tw, th), Image.NEAREST)
        try:
            with torch.inference_mode():
                r = pipe(prompt="", image=im, mask_image=mm, width=tw, height=th, strength=1.0,
                         num_inference_steps=RE_STEPS, guidance_scale=1.0,
                         generator=torch.Generator("cpu").manual_seed(0)).images[0]
        except Exception as e:
            print("정밀 다시 그리기 실패 (이 자리는 기존 결과 그대로):", e)
            continue
        r = np.asarray(r.convert("RGB").resize((w, h), Image.LANCZOS))
        # 붙이는 자리: 도려낸 곳 + 살짝 바깥까지 부드럽게 (경계선 안 보이게). 색이 조금 틀어졌으면 둘레 색에 맞춤
        ring = (cv2.dilate(mk, np.ones((9, 9), np.uint8)) > 0) & (mk == 0)
        if ring.sum() > 30:
            r = np.clip(r.astype(np.float32) + (src[ring].astype(np.float32).mean(0) - r[ring].astype(np.float32).mean(0)), 0, 255)
        a = cv2.GaussianBlur(cv2.dilate(mk, np.ones((5, 5), np.uint8)).astype(np.float32) / 255, (0, 0), 1.5)[..., None]
        out[Y0:Y1, X0:X1] = (r * a + src * (1 - a)).astype(np.uint8)
    return out


# ---------------- 5. 글자 모양 읽기 + 한국어 쓰기 ----------------
def style(rgb, clean, b, pal=True, others=None):
    """원래 글자의 색·테두리·위아래 그라데이션·굵기"""
    x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(math.ceil(b["x1"])), int(math.ceil(b["y1"]))
    k = int(b["size"] * 0.15)
    x0, y0, x1, y1 = max(0, x0 - k), max(0, y0 - k), min(rgb.shape[1], x1 + k), min(rgb.shape[0], y1 + k)
    o, c = rgb[y0:y1, x0:x1].astype(np.int32), clean[y0:y1, x0:x1].astype(np.int32)
    bm = block_mask(rgb.shape, b)
    area = bm[y0:y1, x0:x1] > 0                             # 글자 자리 안에서만
    if others is not None:
        area &= others[y0:y1, x0:x1] == 0                  # 겹친 다른 문구(다른 색 글씨)는 빼고
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
    st = {"fill": [255, 255, 255], "outline": None, "grad": None, "weight": 700, "cols": []}
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
    st["cols"] = [[int(v) for v in cc] for cc in cen]      # 글자에 쓰인 두 색 (놓친 ！ 찾을 때 씀)
    bnd = (dist <= 1.5)                                    # 글자 덩어리의 맨 가장자리 (배경과 맞닿은 곳)
    b0 = (lab[bnd] == 0).mean() if bnd.any() else 0.5
    oi = 0 if b0 > 0.5 else 1                              # 가장자리를 더 많이 차지한 색 = 테두리 후보
    fc, oc = cen[1 - oi], cen[oi]
    frac_out = (lab == oi).mean()
    bgc = np.median(c[(ink == 0) & area].reshape(-1, 3), 0) if ((ink == 0) & area).any() else c.reshape(-1, 3).mean(0)
    # 그냥 글자 가장자리가 배경과 섞인 색이면 테두리가 아님 (섞인 색은 글자색과 배경색 사이에 있음)
    between = np.linalg.norm(fc - oc) + np.linalg.norm(oc - bgc) < 1.25 * np.linalg.norm(fc - bgc)
    do_, df_ = (d0, d1) if oi == 0 else (d1, d0)
    edge = max(b0, 1 - b0) > 0.65 or do_ < 0.6 * df_       # 테두리라면 가장자리 대부분을 차지하거나, 획 가운데서 멂
    if frac_out > 0.15 and np.linalg.norm(fc - oc) > 80 and not between and edge:
        st["outline"] = [int(v) for v in oc]
        st["how"] = "outline"
        fill_px = (lab != oi) & (dist >= 1.5)              # 테두리 색이 아닌 안쪽 = 글자
    elif min(frac_out, 1 - frac_out) > 0.2 and np.linalg.norm(fc - oc) > 80 and not between:
        ii = 0 if d0 >= d1 else 1                          # 두 색이 뚜렷한데 테두리로 확신이 없으면 안쪽(획 가운데) 색을 글자색으로
        st["how"] = "interior"
        thick = (dist[lab != ii] >= 2).mean() if (lab != ii).any() else 0
        if min(d0, d1) < 0.85 * max(d0, d1) and thick > 0.3:   # 다른 색이 바깥쪽에 두께 있게 있으면 테두리 (가장자리 섞인 색은 얇음)
            st["outline"] = [int(v) for v in cen[1 - ii]]
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
    kc = max(3, int(b["fs"] * 0.04)) | 1                   # 질감 있는 장식 글씨는 획에 구멍이 많아 → 메우고 두께 잼
    fm = cv2.morphologyEx(fm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
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
    if l.get("curve"):
        l["fs"] = l["curve"]["cs"] * 0.85
    else:
        prof = ink.sum(1 if not l["vert"] else 0)              # 글자 실제 높이(가로)/너비(세로) → 글자 크기
        idx = np.nonzero(prof > max(1, prof.max() * 0.05))[0]
        if len(idx):
            l["fs"] = float(idx[-1] - idx[0] + 1) / 0.92
    return [float(v) for v in np.median(o[core], 0)]


_FONTS = {}


def _font_file(fid):
    """글꼴 id ("경로|번호", "@serif"=본명조) → (경로, 번호). 없거나 못 쓰면 기본 본고딕"""
    if fid == "@serif":
        return (SERIF, 0) if os.path.isfile(SERIF) else (FONT, 0)
    if fid:
        p, _, i = str(fid).rpartition("|")
        if p and os.path.isfile(p) and p.lower().endswith((".ttf", ".otf", ".ttc", ".otc")):
            return p, int(i or 0)
    return FONT, 0


def _font(size, weight, fid=None):
    """글꼴 + 가짜 굵게 두께. 굵기 조절이 되는 글꼴(기본 본고딕)은 진짜 굵기, 안 되는 글꼴은 획을 조금 덧칠해서 굵게"""
    p, idx = _font_file(fid)
    key = (p, idx, max(8, int(size)), int(weight))
    if key not in _FONTS:
        try:
            f = ImageFont.truetype(p, key[2], index=idx)
        except Exception:
            f = ImageFont.truetype(FONT, key[2])
        fb = 0
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            if weight >= 700:
                fb = max(1, int(round(key[2] * (0.025 if weight < 900 else 0.045))))
        if len(_FONTS) > 300:
            _FONTS.clear()
        _FONTS[key] = (f, fb)
    return _FONTS[key]


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
    nl = max(1, len([l for l in b["lines"] if not l.get("tail")]))   # 원래 줄 수: 되도록 그 줄 수 그대로 (줄이 늘면 자리 밖으로 넘침)
    for s in np.linspace(base, base * 0.35, 25):
        keep = s > base * 0.45                              # 많이 작아지기 전까지는 줄 수를 늘리지 않음
        f = _font(s, st["weight"], st.get("font"))[0]
        if not b["vert"]:
            lines = _wrap(ko, f, W * 1.02, tmp)
            lh = s * 1.22
            if (len(lines) * lh <= H * 1.08 and not (keep and len(lines) > nl)) or s <= base * 0.36:
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
            if (cols * s * 1.15 <= W * 1.08 and not (keep and cols > nl)) or s <= base * 0.36:
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


def _render_curve(img, b, ko, st):
    """휜 글씨: 원래 글자들이 놓인 길을 따라 한 글자씩 (원래가 기울어진 글자면 같이 기울여서)"""
    c = b["curve"]
    P, T, Lx = _resample(_extend(c["pts"], c["cs"]))      # 원래 글자 길 + 양 끝 조금
    e = c["cs"] * 0.6                                      # 앞쪽 늘린 길이 (원래 첫 글자 가운데 위치)
    L = Lx - 2 * e                                         # 원래 첫 글자~끝 글자 가운데 사이 길이
    adv = [0.55 if ch == " " else 0.6 if ch in ",.!?~…、。，！？" else 1.0 for ch in ko]
    total = sum(adv)
    s = min(b["fs"], (L + c["cs"]) / max(1.0, total * 1.02))
    k = float(st.get("scale") or 1.0)                      # 화면에서 키우고 옮긴 것: 길 가운데를 기준으로 글자·간격 같이 커짐 + 통째로 옮김
    dx, dy = float(st.get("dx") or 0), float(st.get("dy") or 0)
    s0 = s = s * k
    start = e + L / 2 - total * s * 1.02 / 2               # 가운데 맞춤 (글자 폭을 다 더한 길이 기준)
    f, fb = _font(s, st["weight"], st.get("font"))
    sw = max(1, int(round(s * float(st.get("ow") or 0.09)))) if st.get("outline") else 0
    box = None
    cum = 0.0
    for ch, a in zip(ko, adv):
        pos = start + (cum + a / 2) * s0 * 1.02            # 이 글자 가운데
        cum += a
        if ch != " ":
            if pos < 0:                                    # 길 밖으로 나가면 끝 방향대로 이어서 (한 자리에 겹치지 않게)
                cx, cy = P[0] + T[0] * pos
                t = 0
            elif pos > len(P) - 1:
                cx, cy = P[-1] + T[-1] * (pos - (len(P) - 1))
                t = len(P) - 1
            else:
                t = int(pos)
                cx, cy = P[t]
            cx, cy = cx + dx, cy + dy
            ang = math.degrees(math.atan2(T[t][1], T[t][0])) if c["mode"] == "rot" else 0.0
            R = int(s * 1.8)
            fm, om = Image.new("L", (R, R), 0), Image.new("L", (R, R), 0)
            ImageDraw.Draw(fm).text((R / 2, R / 2), ch, font=f, fill=255, anchor="mm", stroke_width=fb, stroke_fill=255)
            if sw:
                ImageDraw.Draw(om).text((R / 2, R / 2), ch, font=f, fill=255, anchor="mm", stroke_width=sw + fb, stroke_fill=255)
            tile = Image.new("RGBA", (R, R), (0, 0, 0, 0))
            if sw:
                tile.paste(Image.new("RGBA", (R, R), tuple(st["outline"]) + (255,)), (0, 0), om)
            if st.get("grad"):
                tt = np.linspace(0, 1, R)[:, None]
                g = (np.array(st["grad"][0]) * (1 - tt) + np.array(st["grad"][1]) * tt).astype(np.uint8)
                tile.paste(Image.fromarray(np.repeat(g[:, None, :], R, 1)).convert("RGBA"), (0, 0), fm)
            else:
                tile.paste(Image.new("RGBA", (R, R), tuple(st["fill"]) + (255,)), (0, 0), fm)
            tile = tile.rotate(-ang, resample=Image.BICUBIC, expand=True)
            px, py = int(cx - tile.width / 2), int(cy - tile.height / 2)
            cx0, cy0 = max(0, -px), max(0, -py)            # 그림 밖으로 나간 부분은 잘라서 얹음
            if cx0 < tile.width and cy0 < tile.height and px + tile.width > 0 and py + tile.height > 0:
                t2 = tile.crop((cx0, cy0, min(tile.width, img.width - px), min(tile.height, img.height - py)))
                if t2.width > 0 and t2.height > 0:
                    img.alpha_composite(t2, (px + cx0, py + cy0))
            tb = tile.getbbox()
            if tb:
                r = [px + tb[0], py + tb[1], px + tb[2], py + tb[3]]
                box = r if box is None else [min(box[0], r[0]), min(box[1], r[1]), max(box[2], r[2]), max(box[3], r[3])]
    return s


def _paint(img, st, fill_m, out_m):
    """글자 모양(fill_m)·테두리 모양(out_m)을 색칠해서 붙임"""
    W, H = img.size
    if out_m is not None:
        img.paste(Image.new("RGBA", (W, H), tuple(st["outline"]) + (255,)), (0, 0), out_m)
    if st.get("grad"):
        bb = fill_m.getbbox() or (0, 0, W, H)
        t = np.clip((np.arange(H) - bb[1]) / max(1, bb[3] - bb[1]), 0, 1)[:, None]
        c0, c1 = np.array(st["grad"][0], np.float32), np.array(st["grad"][1], np.float32)
        g = (c0 * (1 - t) + c1 * t).astype(np.uint8)
        img.paste(Image.fromarray(np.concatenate([np.repeat(g[:, None, :], W, 1), np.full((H, W, 1), 255, np.uint8)], 2), "RGBA"), (0, 0), fill_m)
    else:
        img.paste(Image.new("RGBA", (W, H), tuple(st["fill"]) + (255,)), (0, 0), fill_m)


def _compose(img, layer, st, fs):
    """글씨 층을 그림에 얹음: 외부 광선(번짐: 색·크기·진하기) + 글씨 투명도"""
    a = layer.getchannel("A")
    bb = a.getbbox()
    if not bb:
        return img, None
    drawn = list(bb)
    g = st.get("glow")
    if g:
        r = max(1.0, float(st.get("gs") or 0.15) * fs)    # 번짐 크기 (글자 크기 대비)
        pad = int(r * 3) + 2
        W, H = img.size
        x0, y0, x1, y1 = max(0, bb[0] - pad), max(0, bb[1] - pad), min(W, bb[2] + pad), min(H, bb[3] + pad)
        sub = np.asarray(a.crop((x0, y0, x1, y1)), np.float32) / 255
        k = max(1, int(r * 0.35))
        sub = cv2.dilate(sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        sub = cv2.GaussianBlur(sub, (0, 0), r * 0.6)
        sub = np.clip(sub * 1.6, 0, 1) * float(st.get("ga") if st.get("ga") is not None else 0.8)
        gm = Image.fromarray((sub * 255).astype(np.uint8), "L")
        img.paste(Image.new("RGBA", gm.size, tuple(g) + (255,)), (x0, y0), gm)
        sb = gm.getbbox()
        if sb:
            drawn = [min(drawn[0], x0 + sb[0]), min(drawn[1], y0 + sb[1]), max(drawn[2], x0 + sb[2]), max(drawn[3], y0 + sb[3])]
    op = float(st.get("op") if st.get("op") is not None else 1.0)
    if op < 0.999:
        layer.putalpha(a.point(lambda v: int(v * op)))
    img.alpha_composite(layer)
    return img, [int(v) for v in drawn]


def render(clean, blocks):
    """지운 그림 위에 블록마다 한국어를 그려 넣음.
    화면에서 고친 것: 글꼴(font) · 크기 배율(scale) · 위치 옮김(dx, dy) · 테두리 두께(ow) · 외부 광선(glow, gs, ga) · 투명도(op).
    그린 자리는 b["drawn"] 에 남김 (화면에서 마우스로 잡을 상자)"""
    img = Image.fromarray(clean).convert("RGBA")
    W, H = img.size
    for b in blocks:
        b["drawn"] = None
        ko = (b.get("ko") or "").strip()
        if not ko or b.get("skip"):
            continue
        st = b["style"]
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if b.get("curve"):
            fs = _render_curve(layer, b, ko, st)
        else:
            items, s = layout(b, ko, st)
            if not items:
                continue
            k = float(st.get("scale") or 1.0)
            dx, dy = float(st.get("dx") or 0), float(st.get("dy") or 0)
            cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
            fs = s * k
            f, fb = _font(fs, st["weight"], st.get("font"))
            sw = max(1, int(round(fs * float(st.get("ow") or 0.09)))) if st.get("outline") else 0
            fill_m = Image.new("L", (W, H), 0)
            out_m = Image.new("L", (W, H), 0) if sw else None
            df = ImageDraw.Draw(fill_m)
            do = ImageDraw.Draw(out_m) if sw else None
            for x, y, t in items:
                x, y = cx + (x - cx) * k + dx, cy + (y - cy) * k + dy
                df.text((x, y), t, font=f, fill=255, stroke_width=fb, stroke_fill=255)
                if sw:
                    do.text((x, y), t, font=f, fill=255, stroke_width=sw + fb, stroke_fill=255)
            _paint(layer, st, fill_m, out_m)
        img, b["drawn"] = _compose(img, layer, st, fs)
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
    cols = [st["fill"]] + ([st["outline"]] if st.get("outline") else []) + (st["grad"] if st.get("grad") else []) + st.get("cols", [])
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


DBG_PLATE = False
PLATE_EDGE = 0.9
PLATE_BORDER = 0.4


def plate_of(rgb, l):
    """글자가 단색 판(글상자·띠·단색 배경) 위에 있는가 → {color, text, region(글자 자리를 포함한 판 영역)}.
    판 위 글씨는 판을 남기고 글자 획만 판 색으로 메움 (AI로 다시 그리지 않음 → 판이 사라지거나 잘리지 않게)"""
    H, W = rgb.shape[:2]
    pm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(pm, [np.array(l["poly"], np.int32)], 1)
    px = rgb[pm > 0].astype(np.float32)
    if len(px) < 40:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)   # 판 · 글자 · (상자 밖으로 걸친 배경)
    lab = lab.ravel()
    cnt = np.bincount(lab, minlength=3)
    bi = int(cnt.argmax())
    if cnt[bi] / len(lab) < 0.35:
        return None
    bgc = cen[bi]
    if px[lab == bi].std(0).max() > 24:                     # 판이 고른 한 색이어야 (잔무늬 조금은 괜찮음)
        return None
    others = [i for i in range(3) if i != bi and cnt[i] > 0]
    ti = max(others, key=lambda i: np.linalg.norm(cen[i] - bgc) * min(1.0, cnt[i] / (0.08 * len(lab))))   # 판과 가장 다른 색 = 글자
    if np.linalg.norm(cen[ti] - bgc) < 60:
        return None
    b = max(2, int(l["size"] * 0.08))                       # 판이면 글자 상자 테두리에도 판 색이 보여야 (큰 장식 글씨는 글자 자체가 '판'처럼 보임)
    band = (pm > 0) & (cv2.erode(pm, np.ones((2 * b + 1, 2 * b + 1), np.uint8)) == 0)
    if (np.linalg.norm(rgb[band].astype(np.float32) - bgc, axis=-1) < 40).mean() < 0.2:
        return None
    g = int(max(6, l["size"] * 0.6))
    x0, y0, x1, y1 = max(0, int(l["x0"]) - g), max(0, int(l["y0"]) - g), min(W, int(l["x1"]) + g), min(H, int(l["y1"]) + g)
    win = rgb[y0:y1, x0:x1].astype(np.float32)
    near = (np.linalg.norm(win - bgc, axis=-1) < 28).astype(np.uint8)
    k = max(3, int(l["size"] * 0.5)) | 1
    closed = cv2.morphologyEx(near, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab2, st, _ = cv2.connectedComponentsWithStats(closed)
    pw = pm[y0:y1, x0:x1]
    best, ov = 0, 0
    for i in range(1, n):
        o = int(((lab2 == i) & (pw > 0)).sum())
        if o > ov:
            best, ov = i, o
    if not best or ov < 0.6 * pw.sum():
        return None
    reg = (lab2 == best).astype(np.uint8)
    edge = np.concatenate([reg[0], reg[-1], reg[:, 0], reg[:, -1]])
    if DBG_PLATE:
        print("plate edge", l["text"], round(float(edge.mean()), 2))
    if edge.mean() > PLATE_EDGE:                            # 주변이 온통 같은 색 = 그냥 단색 바탕 (판 아님 → 보통 지우기가 더 깔끔)
        return None
    ff = reg.copy()                                         # 판 안의 구멍(글자 자리) 메우기
    h2, w2 = ff.shape
    flood = np.zeros((h2 + 2, w2 + 2), np.uint8)
    inv = (1 - ff).astype(np.uint8)
    for sy, sx in ((0, 0), (0, w2 - 1), (h2 - 1, 0), (h2 - 1, w2 - 1)):
        if inv[sy, sx]:
            cv2.floodFill(inv, flood, (sx, sy), 2)
    reg = ((reg > 0) | (inv == 1)).astype(np.uint8)
    region = np.zeros((H, W), np.uint8)
    region[y0:y1, x0:x1] = reg
    full = region.copy()                                    # 판 전체 (옆 글씨를 지울 때도 이 판은 AI로 다시 그리지 않게)
    border = np.concatenate([full[0], full[-1], full[:, 0], full[:, -1]])
    if DBG_PLATE: print("border", l["text"], round(float(border.mean()), 2))
    if border.mean() > PLATE_BORDER:                        # 그림 가장자리까지 이어진 색 = 판이 아니라 그냥 바탕
        return None
    gd = max(2, int(l["size"] * 0.2))                       # 글자 상자 둘레까지만 (옆 글씨·판은 안 건드리게)
    region &= cv2.dilate(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gd + 1, 2 * gd + 1)))
    tc = cen[ti]
    return {"color": [int(v) for v in bgc], "text": [int(v) for v in tc], "region": region, "full": full}


def soft_of(rgb, l, plates):
    """무늬 있는 판 위 글씨 (단색 판 바로 옆에 이어진 판): 판 색과 가장 다른 색 = 글자 → 글자 획 픽셀만 표시.
    상자째 AI로 지우면 판이 하얗게 뭉개지니, 획 자리만 지움"""
    H, W = rgb.shape[:2]
    h = l["y1"] - l["y0"]
    for p in plates:
        ov = min(l["y1"], p["y1"]) - max(l["y0"], p["y0"])
        gap = max(l["x0"] - p["x1"], p["x0"] - l["x1"])
        if ov < 0.6 * min(h, p["y1"] - p["y0"]) or gap > 2 * max(l["size"], p["size"]):
            continue
        pc = np.array(p["plate"]["color"], np.float32)
        pm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(pm, [np.array(l["poly"], np.int32)], 1)
        px = rgb[pm > 0].astype(np.float32)
        if len(px) < 40:
            return None
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, lab, cen = cv2.kmeans(px, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        cnt = np.bincount(lab.ravel(), minlength=3)
        ok = [i for i in range(3) if cnt[i] >= 0.06 * len(px)]
        ti = max(ok, key=lambda i: np.linalg.norm(cen[i] - pc))
        tc = cen[ti]
        d = float(np.linalg.norm(tc - pc))
        if d < 100:
            return None
        g = max(2, int(l["size"] * 0.15))
        area = cv2.dilate(pm, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
        m = ((np.linalg.norm(rgb.astype(np.float32) - tc, axis=-1) < 0.4 * d) & (area > 0)).astype(np.uint8)
        if m.sum() < 20:
            return None
        return {"text": [int(v) for v in tc], "mask": m}
    return None


def _fs_from(l, tm):
    """글자 픽셀 높이(가로줄) / 너비(세로줄)로 글자 크기"""
    x0, y0, x1, y1 = int(l["x0"]), int(l["y0"]), int(math.ceil(l["x1"])), int(math.ceil(l["y1"]))
    sub = tm[y0:y1, x0:x1]
    prof = sub.sum(1 if not l["vert"] else 0)
    idx = np.nonzero(prof > max(1, prof.max() * 0.05))[0] if prof.size and prof.max() > 0 else []
    if len(idx):
        l["fs"] = float(idx[-1] - idx[0] + 1) / 0.92


GLYPH_TH = 60


def glyph_mask(dfill, poly, b, dout=None):
    """글자 모양 그대로의 마스크: 글자 자리 안에서 글자색에 가까운 픽셀 (+ 그 바로 둘레의 글자 테두리색) + 가장자리 1~2칸.
    뱃지·스티커 테두리·글상자처럼 글자색이 아닌 꾸밈은 안 건드림"""
    fs = b["fs"]
    g = max(2, int(b["size"] * 0.12))
    area = cv2.dilate(poly, np.ones((2 * g + 1, 2 * g + 1), np.uint8))
    m = ((dfill < GLYPH_TH) & (area > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))          # 배경 잡티 빼기
    ys, xs = np.nonzero(poly)                              # 글자 상자 밖으로 크게 뻗어 나간 같은 색 덩어리(띠·바탕)는 글자가 아님
    e = max(2, int(b["size"] * 0.05))
    bx0, by0, bx1, by1 = xs.min() - e, ys.min() - e, xs.max() + e, ys.max() + e
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        yy, xx = np.nonzero(lab[y:y + h, x:x + w] == i)
        keep[i] = ((xx + x >= bx0) & (xx + x <= bx1) & (yy + y >= by0) & (yy + y <= by1)).mean() > 0.9
    m = keep[lab].astype(np.uint8)
    kc = max(3, int(fs * 0.05)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))   # 질감 글씨 구멍
    if dout is not None:                                   # 글자 테두리: 글자 획 바로 둘레에서 테두리색인 곳만
        ko = max(2, int(fs * 0.14))
        ring = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ko + 1, 2 * ko + 1)))
        m = m | ((dout < GLYPH_TH) & (ring > 0)).astype(np.uint8)
    ke = min(6, max(3, int(round(fs * 0.08))))                   # 글자 가장자리 섞인 색까지
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ke + 1, 2 * ke + 1)))


def _curve_style(b, clean):
    """휜 글씨: 배경과 더 다른 색 = 글자색, 다른 하나가 뚜렷이 다르면 테두리 (위아래 그라데이션은 안 봄)"""
    if not (b.get("curve") and len(b["style"].get("cols", [])) == 2 and b["style"].get("how") is None):
        return
    x0, y0, x1, y1 = int(b["x0"]), int(b["y0"]), int(b["x1"]), int(b["y1"])
    bg = np.median(clean[y0:y1, x0:x1].reshape(-1, 3), 0)
    c0, c1 = (np.array(c, np.float32) for c in b["style"]["cols"])
    fc, oc = (c0, c1) if np.linalg.norm(c0 - bg) >= np.linalg.norm(c1 - bg) else (c1, c0)
    b["style"].update(fill=[int(v) for v in fc], grad=None,
                      outline=[int(v) for v in oc] if np.linalg.norm(fc - oc) > 80 else None)
    if b["style"]["outline"]:
        b["style"]["weight"] = max(b["style"]["weight"], 700)


def _fit_box(b, gm):
    """한글을 넣을 자리 = 실제로 도려낸 글자 자리 (글자 찾기 상자는 잎사귀·뱃지까지 넉넉히 잡힐 때가 있음).
    한 줄짜리는 글자 크기도 실제 글자 높이(세로줄은 너비)로"""
    if b.get("curve") or not gm.any():
        return
    ys, xs = np.nonzero(gm)
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)
    if (x1 - x0) * (y1 - y0) < 0.25 * (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]):
        return                                             # 너무 작게 잡혔으면 (글자를 다 못 찾은 것) 원래 상자 그대로
    b["x0"], b["y0"], b["x1"], b["y1"] = max(b["x0"], x0), max(b["y0"], y0), min(b["x1"], x1), min(b["y1"], y1)
    if len([l for l in b["lines"] if not l.get("tail")]) == 1:
        h = (b["x1"] - b["x0"]) if b["vert"] else (b["y1"] - b["y0"])
        b["fs"] = min(b["fs"], h / 0.92)


def _poly_text_color(rgb, l, extra=None):
    """글자색 = 글자 상자 안에는 많고 상자 바로 바깥(배경)에는 드문 색.
    상자 안 색 3묶음(+ 1차로 읽은 색) 중 (안쪽 비율 - 바깥 비율)이 가장 큰 것"""
    H, W = rgb.shape[:2]
    pm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(pm, [np.array(l["poly"], np.int32)], 1)
    g = max(3, int(l["size"] * 0.3))
    ring = (cv2.dilate(pm, np.ones((2 * g + 1, 2 * g + 1), np.uint8)) > 0) & (pm == 0)
    px = rgb[pm > 0].astype(np.float32)
    rp = rgb[ring].astype(np.float32)
    if len(px) < 30 or len(rp) < 30:
        return extra
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cands = [c for c in cen] + ([np.array(extra, np.float32)] if extra is not None else [])
    score = lambda c: (np.linalg.norm(px - c, axis=1) < 40).mean() - (np.linalg.norm(rp - c, axis=1) < 40).mean()
    best = max(cands, key=score)
    if extra is not None and score(cands[-1]) >= 0.5 * score(best):
        return extra                                       # 1차로 읽은 색이 충분히 글자다우면 그대로 (테두리 색을 글자색으로 잘못 고르지 않게)
    return [float(v) for v in best]


def _flat_around(rgb, l, km):
    """번역 안 하는 글씨(영어 띠 등)가 놓인 단색 바탕을 이어진 데까지 전부 → 마스크 (같은 색 띠가 글자로 잘못 지워지지 않게)"""
    px = rgb[km > 0].astype(np.float32)
    if len(px) < 30:
        return np.zeros(km.shape, np.uint8)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(px, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=2)
    bi = int(cnt.argmax())
    if cnt[bi] < 0.5 * len(px) or px[lab.ravel() == bi].std(0).max() > 24:
        return np.zeros(km.shape, np.uint8)
    nearc = (np.linalg.norm(rgb.astype(np.float32) - cen[bi], axis=-1) < 30).astype(np.uint8)
    k = max(3, int(l["size"] * 0.4)) | 1
    nearc = cv2.morphologyEx(nearc, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab2 = cv2.connectedComponents(nearc)
    ids = np.unique(lab2[(km > 0) & (nearc > 0)])
    ids = ids[ids > 0]
    if not len(ids):
        return np.zeros(km.shape, np.uint8)
    m = np.isin(lab2, ids).astype(np.uint8)
    ys, xs = np.nonzero(km)                                # 그 글씨 줄 높이(±글자 크기) 안의 띠만 (위아래 다른 글씨 바탕까지 막지 않게)
    s = int(l["size"])
    strip = np.zeros_like(m)
    if l.get("vert"):
        strip[:, max(0, xs.min() - s):xs.max() + s + 1] = 1
    else:
        strip[max(0, ys.min() - s):ys.max() + s + 1, :] = 1
    return m & strip


def analyze(path):
    """글자 찾기 → 지우기 → 줄마다 글자색 → 덩어리로 묶기 → 덩어리 모양 읽기 → 글자 획만 다시 지우기 (번역 전까지)
    → (원본, 지운 그림, 블록들). 배경을 최대한 살리려고, 마지막엔 글자 상자 전체가 아니라 글자 획(+테두리) 자리만 새로 채움"""
    cv2.setRNGSeed(0)                                      # 같은 그림이면 언제나 같은 결과
    rgb = np.asarray(Image.open(path).convert("RGB"))
    lines = detect(rgb)
    try:
        lines += curve_lines(rgb, lines)                   # 휘거나 비스듬해서 못 찾은 글씨
    except Exception as e:
        print("휜 글씨 찾기 실패:", e)
    if not any(l["cjk"] for l in lines):
        return rgb, rgb, []
    protect = np.zeros(rgb.shape[:2], np.uint8)            # 판(글상자) 자리: AI로 다시 그리지 않음
    for l in lines:
        if l["cjk"] and not l.get("curve"):
            pl = plate_of(rgb, l)
            if pl is not None:
                l["plate"] = pl
                protect |= pl["full"]
    plated = [l for l in lines if l.get("plate")]
    for l in lines:                                        # 무늬 판 (단색 판에 이어진) 위 글씨
        if not l.get("plate") and not l.get("curve") and plated:
            sf = soft_of(rgb, l, plated)
            if sf is not None:
                l["soft"] = sf
    free = [l for l in lines if not l.get("plate") and not l.get("soft")]
    pre = group(free) if any(l["cjk"] for l in free) else []   # 판이 아닌 곳에 있는 글씨 (지울 곳)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for b in pre:
        bm = block_mask(rgb.shape, b)
        if (bm > 0).mean() > 0.12:                         # 그림 대부분을 덮는 큰 글씨 → 획만
            fm = fg_mask(rgb, b)
            if fm is not None:
                bm = fm
        mask = np.maximum(mask, bm)
    mask[protect > 0] = 0
    clean = inpaint(rgb, mask) if mask.any() else rgb.copy()   # 1차: 상자째 지운 그림 (글자 모양·색 읽는 데 씀)
    used = [l for b in pre for l in b["lines"]]
    for l in used:
        l.pop("tail", None)
        l["col"] = _poly_text_color(rgb, l, line_color(rgb, clean, l))   # 1차로 읽은 색과 상자 안 색들 중 글자다운 것
    for l in lines:                                        # 판 위 글씨: 글자색 = 판이 아닌 색, 글자 크기 = 글자 픽셀 높이
        pl = l.get("plate")
        if not pl:
            continue
        l["col"] = pl["text"]
        tm = (pl["region"] > 0) & (np.linalg.norm(rgb.astype(np.float32) - np.array(pl["color"], np.float32), axis=-1) > 45)
        _fs_from(l, tm)
        used.append(l)
    for l in lines:
        sf = l.get("soft")
        if sf:
            l["col"] = sf["text"]
            _fs_from(l, sf["mask"] > 0)
            used.append(l)
    blocks = group(used)                                   # 글자색까지 보고 다시 묶기
    ink = np.zeros(rgb.shape[:2], np.uint8)
    polys = []
    for b in blocks:
        pm = np.zeros(rgb.shape[:2], np.uint8)
        for l in b["lines"]:
            cv2.fillPoly(pm, [np.array(l["poly"], np.int32)], 1)
        polys.append(pm)
    allp = np.sum(polys, 0) if polys else None

    def others(i):
        return None if allp is None else (allp - polys[i]) > 0
    for i, b in enumerate(blocks):
        if b.get("plate") or b.get("soft"):
            continue
        b["style"] = style(rgb, clean, b, others=others(i))
        _curve_style(b, clean)
        rm, tall = _residual(rgb, b, b["style"]) if not b.get("curve") else (None, False)   # 놓친 ！ 같은 것
        if rm is not None:
            kr = max(2, int(b["fs"] * (0.14 if b["style"].get("outline") else 0.04)))
            ink = np.maximum(ink, cv2.dilate(rm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))))
            if tall and not b.get("tail"):
                b["tail"] = "!"
    # 2차: 글자 모양대로만 도려냄 (글자색·글자 테두리색 픽셀만. 뱃지·스티커 테두리·글상자는 남김) → 그 자리만 원본에서 새로 채움
    px = rgb.astype(np.float32)
    dist = lambda cs: np.min(np.stack([np.linalg.norm(px - np.array(c, np.float32), axis=-1) for c in cs]), 0)

    def near(a, c):                                        # 두 덩어리가 붙어 있나 (로고처럼 한 덩어리 안에 다른 색 글자가 섞일 때 색을 같이 씀)
        g = 0.5 * max(a["size"], c["size"])
        return not (a["x1"] + g < c["x0"] or c["x1"] + g < a["x0"] or a["y1"] + g < c["y0"] or c["y1"] + g < a["y0"])
    if True:
        for i, b in enumerate(blocks):
            if b.get("plate") or b.get("soft"):
                continue
            tcols = [l["col"] for c in blocks if (c is b or near(b, c)) and not (c.get("plate") or c.get("soft"))
                     for l in c["lines"] if l.get("col") is not None]
            stb = b.get("style") or {}
            if stb.get("how") in ("outline", "interior"):   # 두 색 글씨(흰 글자+검은 테두리 등): 둘레 배경에 거의 없는 색만 글자로
                rg = (cv2.dilate(polys[i], np.ones((25, 25), np.uint8)) > 0) & (polys[i] == 0)
                rp = px[rg]
                for c in [stb["fill"]] + list(stb.get("cols") or []):
                    if len(rp) and (np.linalg.norm(rp - np.array(c, np.float32), axis=1) < 40).mean() < 0.05:
                        tcols.append(c)
            if not tcols:
                continue
            dfill = dist(tcols)
            st = b.get("style") or {}                      # 글자 테두리 후보 = 글자에 쓰인 색 중 글자색이 아닌 것
            lc = [np.array(l["col"], np.float32) for l in b["lines"] if l.get("col") is not None]
            ocs = [c for c in list(st.get("cols") or []) + ([st["outline"]] if st.get("outline") else [])
                   if lc and min(np.linalg.norm(np.array(c, np.float32) - x) for x in lc) > 80] \
                if (st.get("how") == "outline" or st.get("outline")) else []   # 테두리가 있는 글씨만 (꾸밈 스티커·뱃지는 그대로)
            gm = glyph_mask(dfill, polys[i], b, dist(ocs) if ocs else None)
            ink = np.maximum(ink, gm * 255)
            b["_gm"] = gm
    inb = {id(l) for b in blocks for l in b["lines"]}
    keep = np.zeros(rgb.shape[:2], bool)                   # 번역 안 하는 글씨(영어 로고·띠 등)와 그 둘레: 끝까지 안 건드림
    for l in lines:
        if id(l) not in inb:
            km = np.zeros(rgb.shape[:2], np.uint8)
            cv2.fillPoly(km, [np.array(l["poly"], np.int32)], 1)
            if len(re.sub(r"\s", "", l["text"])) <= 3 and allp is not None and (allp[km > 0] > 0).mean() > 0.2:
                continue                                   # 번역 글씨에 붙은 짧은 조각(！ 등)은 같이 지움
            g = max(2, int(l["size"] * 0.5))
            fl = (cv2.dilate(km, np.ones((2 * g + 1, 2 * g + 1), np.uint8)) > 0) | (_flat_around(rgb, l, km) > 0)
            keep |= fl & ~((allp > 0) if allp is not None else False)   # 그 글씨 둘레 + 놓인 단색 띠·바탕 (번역할 글자 상자 안은 빼고)
    ink[keep] = 0
    for b in blocks:                                       # 한글 넣을 자리 = 실제로 도려낸 자리
        gm = b.pop("_gm", None)
        if gm is not None:
            _fit_box(b, ((gm > 0) & (ink > 0)).astype(np.uint8))
    for l in lines:                                        # 판·무늬 판 위 글씨: 글자 획 자리만 (판 보호 구역이어도)
        if l.get("soft"):
            ink = np.maximum(ink, cv2.dilate(l["soft"]["mask"] * 255, np.ones((5, 5), np.uint8)))
        pl = l.get("plate")
        if pl:
            pc, tc = np.array(pl["color"], np.float32), np.array(pl["text"], np.float32)
            px = rgb.astype(np.float32)                    # 글자색 쪽에 더 가까운 픽셀(글자 획 + 가장자리)만
            dt, dp = np.linalg.norm(px - tc, axis=-1), np.linalg.norm(px - pc, axis=-1)
            tm = ((pl["region"] > 0) & (dt < dp * 1.6) & (dp > 20) & (dt < max(GLYPH_TH, 0.5 * float(np.linalg.norm(tc - pc))))).astype(np.uint8)
            kp = 2 * min(6, max(1, int(round(l.get("fs", l["size"]) * 0.08)))) + 1   # 글자 가장자리 섞인 색까지 (큰 글씨도 몇 칸만)
            ink = np.maximum(ink, cv2.dilate(tm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kp, kp))) * 255)
    ink[keep] = 0
    if ink.any():
        clean = inpaint(rgb, ink)
        clean = regen(rgb, clean, ink)                     # 무늬·그림 위 자리는 정밀 다시 그리기 (모델이 있을 때만)
    else:
        clean = rgb.copy()
    for i, b in enumerate(blocks):                         # 글자 획 자리만 새로 채운 그림과 비교하면 글자 픽셀이 정확 → 색·테두리 다시 읽기
        if b.get("plate"):
            pl = b["plate"]
            tm = ((pl["region"] > 0) & (np.linalg.norm(rgb.astype(np.float32) - np.array(pl["color"], np.float32), axis=-1) > 45)).astype(np.uint8)
            dd = cv2.distanceTransform(tm, cv2.DIST_L2, 3)[tm > 0]
            th = 2 * float(np.percentile(dd, 90)) / max(1.0, b["fs"]) if len(dd) else 0.1
            b["style"] = {"fill": pl["text"], "outline": None, "grad": None, "cols": [],
                          "weight": 900 if th > 0.135 else 700 if th > 0.112 else 500, "plate": pl["color"]}
            continue
        if b.get("soft"):
            tm = np.zeros(rgb.shape[:2], np.uint8)
            for l in b["lines"]:
                if l.get("soft"):
                    tm |= l["soft"]["mask"]
            dd = cv2.distanceTransform(tm, cv2.DIST_L2, 3)[tm > 0]
            th = 2 * float(np.percentile(dd, 90)) / max(1.0, b["fs"]) if len(dd) else 0.1
            b["style"] = {"fill": b["soft"]["text"], "outline": None, "grad": None, "cols": [],
                          "weight": 900 if th > 0.135 else 700 if th > 0.112 else 500}
            continue
        tl = b.get("tail")
        b["style"] = style(rgb, clean, b, pal=False, others=others(i))
        b["tail"] = tl
        _curve_style(b, clean)
    for b in blocks:                                       # 판 영역은 저장할 필요 없음 (크기만 큼)
        b.pop("plate", None)
        b.pop("soft", None)
        for l in b["lines"]:
            l.pop("plate", None)
            l.pop("soft", None)
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
