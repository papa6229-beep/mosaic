"""자막을 영상에 입히기: 자막 스타일(글꼴·크기·위치·색·테두리·상자)과 자막 한 장면을 그림으로 만들어 YUV 프레임에 섞기.
검수 화면 미리보기(subs.html)와 같은 비율 기준(영상 높이 대비)을 씀"""
import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(ROOT, "settings", "sub_style.json")

DEFAULT_STYLE = {
    "font": "NotoSansKR-Medium.otf",   # fonts 폴더 또는 윈도우 글꼴 파일 이름
    "size": 0.046,                     # 글자 크기 = 영상 높이 × size
    "bottom": 0.096,                   # 자막 아래 끝이 영상 아래에서 영상 높이 × bottom 만큼 위
    "color": "#ffffff",
    "outline": "#000000",
    "outline_w": 0.05,                 # 테두리 두께(바깥쪽) = 글자 크기 × outline_w
    "box": False,                      # 모든 자막 뒤에 상자
    "box_color": "#000000",
    "box_alpha": 0.6,
    "nar_box_alpha": 0.85,             # 나레이션 줄(상자 표시한 줄)의 상자 진하기
    "line_gap": 1.38,                  # 줄 간격 (글자 크기 배수)
}

# 윈도우에 흔히 있는 한글 글꼴 (있는 것만 목록에 보임)
WIN_FONTS = [("맑은 고딕", "malgun.ttf"), ("맑은 고딕 굵게", "malgunbd.ttf"), ("나눔고딕", "NanumGothic.ttf"),
             ("나눔고딕 굵게", "NanumGothicBold.ttf"), ("나눔스퀘어", "NanumSquareR.ttf"), ("나눔스퀘어 굵게", "NanumSquareB.ttf"),
             ("나눔바른고딕", "NanumBarunGothic.ttf"), ("굴림", "gulim.ttc"), ("돋움", "dotum.ttc"), ("바탕", "batang.ttc")]


def load_style():
    st = dict(DEFAULT_STYLE)
    try:
        st.update(json.load(open(STYLE_PATH, encoding="utf-8")))
    except Exception:
        pass
    return st


def save_style(st):
    os.makedirs(os.path.dirname(STYLE_PATH), exist_ok=True)
    cur = load_style()
    cur.update({k: v for k, v in st.items() if k in DEFAULT_STYLE})
    json.dump(cur, open(STYLE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return cur


def font_dirs():
    ds = [os.path.join(ROOT, "fonts")]
    w = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    ds.append(w)
    la = os.environ.get("LOCALAPPDATA")
    if la:
        ds.append(os.path.join(la, "Microsoft", "Windows", "Fonts"))
    return ds


def font_path(name):
    for d in font_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "fonts", DEFAULT_STYLE["font"])


def font_list():
    """고를 수 있는 글꼴: 생성기에 들어 있는 글꼴 + 이 PC에 있는 한글 글꼴"""
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "fonts", "*.[ot]tf"))):
        nm = os.path.basename(p)
        out.append({"file": nm, "label": "본고딕(Noto Sans KR) " + nm.split("-")[-1].split(".")[0] if nm.startswith("NotoSansKR") else nm})
    for label, f in WIN_FONTS:
        if any(os.path.exists(os.path.join(d, f)) for d in font_dirs()[1:]):
            out.append({"file": f, "label": label})
    return out


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _wrap(t, n=20):
    """한 줄이 n자를 넘으면 가운데 가까운 띄어쓰기에서 두 줄로 (자막 화면 미리보기와 같은 규칙)"""
    if len(t) <= n or " " not in t:
        return [t]
    sp = [i for i, ch in enumerate(t) if ch == " "]
    i = min(sp, key=lambda x: abs(x - len(t) / 2))
    return [t[:i], t[i + 1:]]


class Overlay:
    """자막 목록(cues: [{start,end,ko,nar}], 초 단위)을 프레임 번호로 찾아 YUV 프레임에 입힘. 같은 자막 그림은 한 번만 만듦"""

    def __init__(self, cues, W, H, fps, style=None):
        self.cues = sorted([c for c in cues if (c.get("ko") or "").strip()], key=lambda c: c["start"])
        self.W, self.H, self.fps = W, H, fps
        self.st = style or load_style()
        self.cache = {}
        self.font = None

    def _font(self):
        if self.font is None:
            from PIL import ImageFont
            self.font = ImageFont.truetype(font_path(self.st["font"]), max(8, int(round(self.H * self.st["size"]))))
        return self.font

    def at(self, i):
        import bisect
        t = (i + 0.5) / self.fps
        if not hasattr(self, "_starts"):
            self._starts = [c["start"] for c in self.cues]
        k = bisect.bisect_right(self._starts, t) - 1
        for j in range(k, max(-1, k - 4), -1):        # 겹친 자막이 있어도 가장 늦게 시작한 것
            if self.cues[j]["start"] <= t < self.cues[j]["end"]:
                return j
        return -1

    def render(self, k):
        """k번째 자막 → (BGR 실수, 투명도, X0, Y0) — 크로마 칸에 맞게 짝수 위치"""
        if k in self.cache:
            return self.cache[k]
        from PIL import Image, ImageDraw
        c, st, W, H = self.cues[k], self.st, self.W, self.H
        f = self._font()
        fs = f.size
        lines = [x for l in c["ko"].strip().split("\n") for x in _wrap(l)]
        dlg = len(lines) > 1 and all(l.startswith("-") for l in lines)      # 두 사람 대사: 줄 왼쪽 맞춤 덩어리를 가운데에
        sw = max(1, int(round(fs * st["outline_w"])))
        lh = int(round(fs * st["line_gap"]))
        tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
        widths = [tmp.textlength(l, font=f) for l in lines]
        bw = int(max(widths)) + 2 * sw
        bh = lh * (len(lines) - 1) + fs + 2 * sw
        box = st["box"] or c.get("nar")
        pad = int(fs * 0.45) if box else 0
        cw, chh = bw + 2 * pad + 4, bh + 2 * pad + int(fs * 0.3) + 4
        img = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if box:
            a = st["nar_box_alpha"] if c.get("nar") else st["box_alpha"]
            d.rectangle([0, 0, cw - 1, chh - 1], fill=_rgb(st["box_color"]) + (int(255 * a),))
        for j, l in enumerate(lines):
            x = pad + sw + 2 + (0 if dlg else (bw - 2 * sw - widths[j]) / 2)
            y = pad + sw + 2 + j * lh
            d.text((x, y), l, font=f, fill=_rgb(st["color"]) + (255,),
                   stroke_width=0 if box and c.get("nar") else sw, stroke_fill=_rgb(st["outline"]) + (255,))
        a = np.asarray(img, np.float32)
        X0 = int((W - cw) / 2)
        Y0 = int(H - H * st["bottom"] - chh + int(fs * 0.3))
        X0, Y0 = max(0, X0 - X0 % 2), max(0, Y0 - Y0 % 2)
        a = a[:max(0, min(chh, H - Y0)), :max(0, min(cw, W - X0))]
        h2, w2 = a.shape[0] - a.shape[0] % 2, a.shape[1] - a.shape[1] % 2
        a = a[:h2, :w2]
        bgr = np.ascontiguousarray(a[..., [2, 1, 0]])
        alpha = np.ascontiguousarray(a[..., 3] / 255.0)
        if len(self.cache) > 40:                     # 너무 쌓이지 않게
            self.cache.clear()
        self.cache[k] = (bgr, alpha, X0, Y0)
        return self.cache[k]

    def apply(self, yf, i):
        """프레임 i에 자막이 있으면 입히고 True"""
        k = self.at(i)
        if k < 0:
            return False
        bgr, alpha, X0, Y0 = self.render(k)
        if alpha.size == 0:
            return False
        yf.blend(X0, Y0, bgr, alpha)
        return True


def cues_of(subs_json):
    """subs.json → 입힐 자막 목록 (한국어가 있는 줄만)"""
    try:
        d = json.load(open(subs_json, encoding="utf-8"))
    except Exception:
        return []
    return [{"start": float(l["start"]), "end": float(l["end"]), "ko": l.get("ko", ""), "nar": bool(l.get("nar"))}
            for l in d.get("lines", []) if (l.get("ko") or "").strip()]
