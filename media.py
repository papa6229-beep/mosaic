"""원본 영상 정보 읽기 + 원본과 같은 형식으로 저장하기

- probe(src): 원본의 압축 방식·색 방식·화질(비트레이트)·화면 방식(인터레이스)·GOP·타임코드·소리 트랙을 읽음
- compose_fmt(info): 블러를 합성할 때 쓸 원본 그대로의 픽셀 형식 (원본 픽셀을 바꾸지 않으려고 YUV 그대로 다룸)
- encode_args(info): 원본과 같은 압축 방식·화질로 저장하는 ffmpeg 인자
- YuvFrame: YUV 프레임에서 블러 영역만 BGR로 꺼내고, 블러를 YUV에 바로 섞어 넣음 (블러 밖 픽셀은 원본 그대로)
"""
import os, re, subprocess
from fractions import Fraction
import numpy as np


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _run(args, timeout=120):
    return subprocess.run([ffmpeg_exe(), "-hide_banner"] + args, capture_output=True, timeout=timeout,
                          text=True, encoding="utf-8", errors="ignore")


# ---------------- 원본 정보 ----------------
def _fps_str(f):
    for num, den in ((24000, 1001), (30000, 1001), (60000, 1001), (24, 1), (25, 1), (30, 1), (50, 1), (60, 1)):
        if abs(f - num / den) < 0.005:
            return f"{num}/{den}"
    fr = Fraction(f).limit_denominator(1001)
    return f"{fr.numerator}/{fr.denominator}"


def probe(src):
    err = _run(["-i", src]).stderr
    info = {"src": src, "ext": os.path.splitext(src)[1].lower(), "codec": None, "pix_fmt": "yuv420p", "bitrate": None,
            "field": "progressive", "matrix": "bt709", "range": "tv", "fps": 30000 / 1001, "fps_str": "30000/1001",
            "W": None, "H": None, "timecode": None, "profile": "", "audio": 0, "vline": ""}
    for line in err.splitlines():
        s = line.strip()
        m = re.match(r"timecode\s*:\s*(\S+)", s)
        if m and not info["timecode"]:
            info["timecode"] = m.group(1)
        if "Stream #" in s and "Audio:" in s:
            info["audio"] += 1
            m = re.search(r"(\d+)\s*kb/s", s)
            info["audio_bps"] = info.get("audio_bps", 0) + (int(m.group(1)) * 1000 if m else 0)
        if "Stream #" in s and "Video:" in s and info["codec"] is None:
            info["vline"] = s
            v = s.split("Video:", 1)[1].strip()
            info["codec"] = v.split()[0].strip(",")
            m = re.match(r"[\w]+\s*\(([^)]*)\)", v)
            info["profile"] = m.group(1) if m else ""
            # 픽셀 형식과 괄호 속 (범위, 색, 필드 순서)
            m = re.search(r",\s*([a-z0-9_]+)(\(([^)]*)\))?,\s*(\d+)x(\d+)", v)
            if m:
                info["pix_fmt"] = m.group(1)
                info["W"], info["H"] = int(m.group(4)), int(m.group(5))
                extra = (m.group(3) or "").lower()
                if "top first" in extra or "top coded first" in extra:
                    info["field"] = "tt"
                elif "bottom first" in extra or "bottom coded first" in extra:
                    info["field"] = "bb"
                if "pc" in [x.strip() for x in extra.split(",")] or info["pix_fmt"].startswith("yuvj"):
                    info["range"] = "pc"
                if "bt601" in extra or "smpte170m" in extra or "bt470bg" in extra:
                    info["matrix"] = "bt601"
                elif "bt709" in extra:
                    info["matrix"] = "bt709"
                elif info["H"] and info["H"] < 720:
                    info["matrix"] = "bt601"
            m = re.search(r"(\d+)\s*kb/s", v)
            if m:
                info["bitrate"] = int(m.group(1)) * 1000
            m = re.search(r"([\d.]+)\s*fps", v)
            if m:
                info["fps"] = float(m.group(1))
                info["fps_str"] = _fps_str(info["fps"])
    m = re.search(r"bitrate:\s*(\d+)\s*kb/s", err)
    total = int(m.group(1)) * 1000 if m else None
    if info["bitrate"] is None and total:                # 영상 비트레이트가 안 나오면 전체에서 소리 몫을 뺌
        info["bitrate"] = max(total - info.get("audio_bps", 0), int(total * 0.5))
    # 실제 영상 비트레이트가 표시값과 거의 같으면 고정 비트레이트(CBR) 원본으로 봄
    info["cbr"] = bool(total and info["bitrate"] and total - info.get("audio_bps", 0) >= info["bitrate"] * 0.95)
    g, b = _gop(src)
    info["gop"], info["bframes"] = g, b
    info["t0"] = _t0(src)
    return info


def _t0(src):
    """첫 프레임의 원래 시간 (프레임 번호를 시간으로 맞출 때 기준)"""
    try:
        err = _run(["-nostats", "-copyts", "-i", src, "-map", "0:v:0", "-frames:v", "1", "-vf", "showinfo=checksum=0",
                    "-f", "null", "-"], timeout=60).stderr
        m = re.search(r"pts_time:\s*([-\d.e]+)", err)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def _gop(src):
    """앞부분 프레임 종류(I/P/B)를 보고 GOP 길이와 연속 B 개수를 알아냄"""
    try:
        err = _run(["-i", src, "-an", "-frames:v", "120", "-vf", "showinfo", "-f", "null", "-"], timeout=120).stderr
        types = re.findall(r"type:([IPB])", err)
    except Exception:
        types = []
    if not types:
        return 15, 2
    ii = [k for k, t in enumerate(types) if t == "I"]
    gop = int(np.median(np.diff(ii))) if len(ii) >= 2 else (len(types) if types.count("I") == 1 else 1)
    run, best = 0, 0
    for t in types:
        run = run + 1 if t == "B" else 0
        best = max(best, run)
    return max(1, gop), best


# ---------------- 합성용 픽셀 형식 ----------------
PLANAR = {  # 형식: (가로 줄임, 세로 줄임, 비트)
    "yuv420p": (2, 2, 8), "yuvj420p": (2, 2, 8), "yuv422p": (2, 1, 8), "yuvj422p": (2, 1, 8),
    "yuv444p": (1, 1, 8), "yuvj444p": (1, 1, 8), "yuv411p": (4, 1, 8),
    "yuv420p10le": (2, 2, 10), "yuv422p10le": (2, 1, 10), "yuv444p10le": (1, 1, 10),
    "yuv420p12le": (2, 2, 12), "yuv422p12le": (2, 1, 12), "yuv444p12le": (1, 1, 12),
}


def compose_fmt(info):
    """원본을 이 형식 그대로 풀어서 블러만 섞고 다시 저장 (원본 픽셀 형식이 이상하면 가장 가까운 것으로)"""
    pf = info.get("pix_fmt") or "yuv420p"
    if pf in PLANAR:
        return pf
    if "422" in pf:
        return "yuv422p10le" if ("10" in pf or "210" in pf) else "yuv422p"
    if "444" in pf or pf.startswith(("rgb", "bgr", "gbr")):
        return "yuv444p"
    return "yuv420p10le" if "10" in pf else "yuv420p"


def frame_bytes(fmt, W, H):
    sx, sy, bits = PLANAR[fmt]
    bps = 1 if bits == 8 else 2
    return (W * H + 2 * (W // sx) * (H // sy)) * bps


# ---------------- 원본과 같은 형식으로 저장 ----------------
_enc_cache = {}


def _encoder_pix_fmts(enc):
    if enc not in _enc_cache:
        try:
            out = _run(["-h", "encoder=" + enc]).stdout
            m = re.search(r"Supported pixel formats:\s*(.*)", out)
            _enc_cache[enc] = m.group(1).split() if m else None
        except Exception:
            _enc_cache[enc] = None
    return _enc_cache[enc]


def _has_encoder(enc):
    return _encoder_pix_fmts(enc) is not None or enc in ("mpeg2video", "libx264", "dnxhd", "prores_ks")


def _pick_pf(enc, want):
    ok = _encoder_pix_fmts(enc)
    if not ok or want in ok:
        return want
    for alt in (want.replace("12le", "10le"), want.replace("10le", ""), want.replace("422", "420"),
                want.replace("444", "422"), "yuv420p"):
        if alt in ok:
            return alt
    return ok[0]


def encode_args(info, fmt):
    """원본과 같은 압축 방식·화질·픽셀 형식·화면 방식으로 저장하는 인자 → (인자 목록, 설명)"""
    codec = (info.get("codec") or "").lower()
    br = info.get("bitrate")
    pf = info.get("pix_fmt") if info.get("pix_fmt") in PLANAR or "p10" in (info.get("pix_fmt") or "") else fmt
    inter = info.get("field") in ("tt", "bb")
    tff = info.get("field") == "tt"
    gop, bf = info.get("gop", 15), info.get("bframes", 2)
    color = []   # 색 정보는 입력 쪽에 붙임 (출력 쪽에 붙이면 ffmpeg가 색을 바꿔 버림): input_args()
    if codec == "mpeg2video":
        pfx = _pick_pf("mpeg2video", pf if pf in ("yuv420p", "yuv422p") else ("yuv422p" if "422" in pf else "yuv420p"))
        a = ["-c:v", "mpeg2video", "-pix_fmt", pfx, "-g", str(gop), "-bf", str(min(bf, 2) if gop > 1 else 0),
             "-flags", "+cgop", "-intra_vlc", "1", "-non_linear_quant", "1", "-qmin", "1", "-qmax", "28", "-dc", "10", "-sc_threshold", "1000000000"]
        if br:
            a += ["-b:v", str(br), "-maxrate", str(br), "-bufsize", str(int(br * (0.36 if pfx == "yuv422p" else 0.5)))]
            if info.get("cbr"):                          # 원본이 고정 비트레이트면 똑같이 (용량·화질 원본과 같게)
                a += ["-minrate", str(br)]
        else:
            a += ["-q:v", "2"]
        if inter:
            a += ["-flags", "+cgop+ildct+ilme", "-alternate_scan", "1", "-vf", "setfield=" + ("tff" if tff else "bff"),
                  "-field_order", "tt" if tff else "bb"]
        return a + color, f"MPEG-2 {pfx}" + (f" {br // 1000000}Mbps" if br else "")
    if codec in ("h264", "hevc"):
        enc = "libx264" if codec == "h264" else ("libx265" if _has_encoder("libx265") else "libx264")
        pfx = _pick_pf(enc, pf)
        a = ["-c:v", enc, "-pix_fmt", pfx, "-preset", "fast", "-g", str(gop), "-bf", str(bf)]
        if br:
            a += ["-b:v", str(br), "-maxrate", str(int(br * 1.2)), "-bufsize", str(int(br * 2))]
        else:
            a += ["-crf", "16"]
        if inter and enc == "libx264":
            a += ["-flags", "+ildct+ilme", "-x264-params", "interlaced=1:" + ("tff=1" if tff else "bff=1"),
                  "-vf", "setfield=" + ("tff" if tff else "bff"), "-field_order", "tt" if tff else "bb"]
        return a + color, f"{'H.264' if enc == 'libx264' else 'H.265'} {pfx}" + (f" {br // 1000000}Mbps" if br else "")
    if codec == "dnxhd":
        pfx = _pick_pf("dnxhd", pf)
        prof = (info.get("profile") or "").lower()
        a = ["-c:v", "dnxhd", "-pix_fmt", pfx]
        if "dnxhr" in prof:
            a += ["-profile:v", prof.replace(" ", "_")]
        elif br:
            a += ["-b:v", str(br)]
        if inter:
            a += ["-flags", "+ildct", "-vf", "setfield=" + ("tff" if tff else "bff"), "-field_order", "tt" if tff else "bb"]
        return a + color, f"DNxHD {pfx}"
    if codec == "prores":
        prof = (info.get("profile") or "").lower()
        pid = 3 if "hq" in prof else 1 if "lt" in prof else 0 if "proxy" in prof else 4 if "4444" in prof else 2
        pfx = _pick_pf("prores_ks", "yuva444p10le" if pid == 4 else "yuv422p10le")
        return ["-c:v", "prores_ks", "-profile:v", str(pid), "-pix_fmt", pfx] + color, "ProRes"
    # 그 밖의 형식: 같은 압축 방식이 있으면 그걸로, 없으면 고화질 H.264
    if codec and _has_encoder(codec):
        a = ["-c:v", codec, "-pix_fmt", _pick_pf(codec, pf)] + (["-b:v", str(br)] if br else ["-q:v", "2"])
        return a + color, codec
    return ["-c:v", "libx264", "-pix_fmt", _pick_pf("libx264", pf), "-crf", "14", "-preset", "fast"] + color, "H.264 (원본 방식 저장 불가)"


def input_args(info):
    """원본 형식 그대로 넘기는 프레임에 원본의 색 정보를 붙임 → ffmpeg가 색을 다시 계산하지 않고 그대로 저장"""
    if info.get("matrix") == "bt601":
        c = ["-colorspace", "smpte170m", "-color_primaries", "smpte170m", "-color_trc", "smpte170m"]
    else:
        c = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    return c + ["-color_range", "pc" if info.get("range") == "pc" else "tv"]


def out_ext(info):
    e = info.get("ext") or ".mp4"
    return e if e in (".mxf", ".mov", ".mp4", ".m4v", ".mkv", ".mts", ".m2ts", ".ts", ".avi", ".mpg", ".mpeg") else ".mp4"


def mux_args(info, dst):
    """최종 파일: 영상(이어 붙인 조각) + 원본 소리 트랙 전부 그대로 + 타임코드"""
    a = []
    if info.get("timecode"):
        a += ["-timecode", info["timecode"]]
    e = os.path.splitext(dst)[1].lower()
    if e == ".mxf":
        a += ["-f", "mxf"]
    elif e in (".mp4", ".m4v", ".mov"):
        a += ["-movflags", "+faststart"]
    return a


# ---------------- YUV 프레임에서 블러만 섞기 ----------------
_K = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}


class YuvFrame:
    """ffmpeg가 원본 형식 그대로 풀어 준 한 프레임. 블러 영역만 BGR로 꺼내고, 블러를 YUV에 직접 섞음"""

    def __init__(self, buf, fmt, W, H, matrix="bt709", rng="tv"):
        sx, sy, bits = PLANAR[fmt]
        dt = np.uint8 if bits == 8 else np.dtype("<u2")
        a = np.frombuffer(buf, dt)
        n0, Wc, Hc = W * H, W // sx, H // sy
        self.Y = a[:n0].reshape(H, W).copy()
        self.U = a[n0:n0 + Wc * Hc].reshape(Hc, Wc).copy()
        self.V = a[n0 + Wc * Hc:n0 + 2 * Wc * Hc].reshape(Hc, Wc).copy()
        self.sx, self.sy, self.bits, self.W, self.H = sx, sy, bits, W, H
        self.sc = float(1 << (bits - 8))
        self.kr, self.kb = _K.get(matrix, _K["bt709"])
        self.full = rng == "pc"
        self.maxv = (1 << bits) - 1

    def tobytes(self):
        return self.Y.tobytes() + self.U.tobytes() + self.V.tobytes()

    def small_gray(self):
        import cv2
        y = (self.Y >> (self.bits - 8)).astype(np.uint8) if self.bits > 8 else self.Y
        g = cv2.resize(y, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        return g if self.full else (g - 16) * (255 / 219)

    def align(self, X0, Y0, X1, Y1):
        X0 -= X0 % self.sx; Y0 -= Y0 % self.sy
        X1 = min(self.W, X1 + (-X1) % self.sx); Y1 = min(self.H, Y1 + (-Y1) % self.sy)
        return X0, Y0, X1, Y1

    def _mats(self):
        """YUV(원본 값) ↔ BGR(0~255) 변환 행렬 (3x4: 곱하고 더하기)"""
        if getattr(self, "_M", None) is None:
            kr, kb = self.kr, self.kb
            kg = 1 - kr - kb
            if self.full:
                ys, yo, cs = 255.0, 0.0, 255.0
            else:
                ys, yo, cs = 219.0, 16.0, 224.0
            sc = self.sc
            # BGR(0~1) → YUV(0~1 정규화) 행렬
            T = np.array([[kr, kg, kb],
                          [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), 0.5],
                          [0.5, -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))]])  # 열 순서 R,G,B
            T = T[:, [2, 1, 0]]                                                     # → B,G,R
            S = np.diag([ys, cs, cs]) * sc / 255.0
            fwd = np.hstack([S @ T, (np.array([yo, 128.0, 128.0]) * sc)[:, None]])  # BGR(0~255) → Y,U,V
            inv = np.linalg.inv(S @ T)
            back = np.hstack([inv, -(inv @ (np.array([yo, 128.0, 128.0]) * sc))[:, None]])  # Y,U,V → BGR(0~255)
            self._M = (fwd.astype(np.float32), back.astype(np.float32))
        return self._M

    def _up(self, P, h, w):
        if self.sx == 1 and self.sy == 1:
            return P
        import cv2
        return cv2.resize(P, (w, h), interpolation=cv2.INTER_NEAREST)

    def bgr(self, X0, Y0, X1, Y1):
        """정렬된 영역을 8비트 BGR로"""
        import cv2
        h, w = Y1 - Y0, X1 - X0
        Yf = self.Y[Y0:Y1, X0:X1].astype(np.float32)
        U = self._up(self.U[Y0 // self.sy:Y1 // self.sy, X0 // self.sx:X1 // self.sx].astype(np.float32), h, w)
        V = self._up(self.V[Y0 // self.sy:Y1 // self.sy, X0 // self.sx:X1 // self.sx].astype(np.float32), h, w)
        out = cv2.transform(cv2.merge([Yf, U, V]), self._mats()[1])
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def blend(self, X0, Y0, soft, alpha):
        """블러 결과(soft: BGR 실수)를 투명도(alpha)만큼 YUV에 섞음. alpha가 0인 곳은 원본 값 그대로 (정수 그대로 되돌아감)"""
        import cv2
        h, w = alpha.shape
        yuv = cv2.transform(np.ascontiguousarray(soft, np.float32), self._mats()[0])
        ys, us, vs = cv2.split(yuv)
        def mix(P, src, a):
            o = P.astype(np.float32)
            r = o + (src - o) * a
            if P.dtype == np.uint8:
                return np.clip(r + 0.5, 0, 255).astype(np.uint8)
            return np.clip(np.rint(r), 0, self.maxv).astype(P.dtype)
        Yr = self.Y[Y0:Y0 + h, X0:X0 + w]
        Yr[...] = mix(Yr, ys, alpha)
        sx, sy = self.sx, self.sy
        hc, wc = h // sy, w // sx
        if sx == 1 and sy == 1:
            ac, uc, vc = alpha, us, vs
        else:
            dn = lambda x: cv2.resize(x[:hc * sy, :wc * sx], (wc, hc), interpolation=cv2.INTER_AREA)   # 칸마다 평균
            ac, uc, vc = dn(alpha), dn(us), dn(vs)
        for P, src in ((self.U, uc), (self.V, vc)):
            Pr = P[Y0 // sy:Y0 // sy + hc, X0 // sx:X0 // sx + wc]
            Pr[...] = mix(Pr, src, ac)
