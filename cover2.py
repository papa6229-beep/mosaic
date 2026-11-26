"""
덮기 엔진 v2 — 납품 블러 방식(타원 하나를 키프레임처럼 천천히 움직이는 방식)을 자동으로

1) 원본 모자이크를 '여러 개' 따로 추적 (Lada가 잠깐 놓쳐도 자리 유지, 한두 번 엉뚱하게 잡은 건 무시)
2) 모자이크마다 '반드시 덮을 곳' = 원본 모자이크 + 붙어 있는 음모 + 근처 항문/성기(노출 탐지)
3) 그 전부를 감싸는 가장 작은 타원 하나 + 여유 → 털 모양을 따라가지 않음(삐져나온 팔 없음)
4) 타원은 '덮을 곳이 안에 있는 동안' 가만히 있음. 벗어나려 할 때만 넓히고, 한동안 너무 크면 천천히 줄임
5) 블러 속은 살색이 아닌 것(털·검은 부분·흰 배경)을 피부로 메운 뒤 강하게 흐림, 프레임 사이에서 천천히만 바뀜
6) 얼굴 근처의 털(머리카락·눈썹)은 음모로 치지 않음
"""
import numpy as np
import cv2

# ---------- 조절 값 ----------
SCALE = 1.15        # 감싸는 타원을 이만큼 키움
PAD_PX = 12         # 추가 여유 (픽셀)
FEATHER_W = 0.5     # 가장자리 투명해지는 폭 (타원 짧은 반지름 대비)
FEATHER_MAX = 150   # 번짐 폭 최대 (픽셀)
BLUR_FRAC = 0.20    # 블러 세기 (타원 짧은 지름 대비)
HAIR = True         # 음모 찾아서 같이 덮기
FAST_SAVE = False   # 블러 없는 구간은 원본 그대로 복사(시험): ProRes·DNxHD 원본만
FAST_SCAN = False   # 빠른 훑기(시험): 4K 등 큰 영상은 1080p 크기로 줄여 훑고, 털 판단 통계(중간값 등)는 그래픽카드로
HAIR_REACH = 0.8    # 음모 인정 거리 (원본 모자이크 짧은 변 대비)
REACH_MIN, REACH_MAX = 40, 250   # 음모 인정 거리 최소/최대 (픽셀)
HAIR_TEX = 2.2      # 음모: 가닥 결이 맨살의 이 배수보다 큼
HAIR_MEM = 8        # 음모를 놓쳐도 이 프레임 수만큼 기억
EXPOSE = True       # 노출 탐지(항문·성기·얼굴) 사용
EXPOSE_EVERY = 1    # 노출 탐지는 매 프레임 (1프레임만 나오는 노출도 있음)
EXPOSE_SCORE = 0.35
HOLD = 45           # Lada가 놓쳐도 이 프레임 수(1.5초)만큼 마지막 자리 유지
PRE = 5             # 처음 나타나기 이 프레임 수 전부터 미리 덮기 (그 앞에 페이드 3프레임이 더 붙음)
TAIL_STABLE = 20    # 오래 안정적으로 잡히던 모자이크가 사라지면 이만큼 더 유지 (Lada가 잠깐 놓친 것일 수 있어서)
GAP_FILL = 20       # 중간에 Lada가 놓친 빈틈은 이 프레임 수까지만 메움
TAIL_SHORT = 4      # 잠깐 잡혔던 모자이크·화면 밖으로 나간 모자이크는 이만큼만 유지
CONFIRM = 2         # 이 횟수 이상 잡혀야 진짜 모자이크로 인정 (엉뚱한 한 번은 무시)
SHRINK_RATIO = 1.6  # 타원이 필요한 크기의 이 배를 넘는 상태가
SHRINK_AFTER = 15   # 이 프레임 수 넘게 이어지면 천천히 줄임
CHASE = 0.35        # (예전 방식용)
WIN_MAX = 60        # 키프레임 타원 하나가 앞으로 감싸 둘 최대 프레임 수 (2초)
WIN_GROW = 1.5      # 앞으로 올 움직임까지 감싸느라 커지는 한도 (지금 필요한 크기의 이 배까지)
GLIDE_MAX = 15      # 옮길 때 이 프레임 수(0.5초)에 걸쳐 미리 부드럽게 옮김
OVER_RATIO = 2.2    # 타원이 필요한 크기의 이 배를 넘는 상태가
OVER_AFTER = 30     # 이 프레임 수(1초) 넘게 이어지면 새 키프레임으로 줄임
FILL_RES = 120      # 블러 재료 계산 크기
FILL_KEEP = 0.65    # 블러 속 이전 프레임 비율 (클수록 속 색이 천천히 바뀜)
BLUR_V2 = False     # (시험) 새 블러: 가장자리로 갈수록 '점점 흐려지게'(흐린 그림을 겹치지 않음) + 넓은 번짐 + 블러 속에 바깥 어두움이 덜 묻어남
FEATHER_W2 = 1.0    # 새 블러의 번짐 폭 (타원 짧은 반지름 대비, 크기 제한 없음)
CUT_DIFF = 28       # 화면 전환 판단 (작은 화면 밝기 차이)


# ======================= 1. 원본 모자이크 추적 =======================
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_tracks(all_boxes, n, fw=None, fh=None):
    """all_boxes: {"프레임": [[x1,y1,x2,y2,확신도,윤곽], ...]} → 프레임마다 [(추적번호, 상자, 윤곽, 이번에잡힘)]"""
    tracks = []          # {"id", "box", "poly", "last", "hits": {프레임: (상자, 윤곽)}}
    active = []
    for i in range(n):
        dets = []
        for d in (all_boxes.get(str(i)) or []):
            if d is None:
                continue
            dets.append((np.array(d[:4], float), d[5] if len(d) > 5 else None))
        used = set()
        pairs = []
        for ti, t in enumerate(active):
            tb = t["box"]
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            for di, (b, _) in enumerate(dets):
                iou = _iou(tb, b)
                dist = np.hypot((tb[0] + tb[2] - b[0] - b[2]) / 2, (tb[1] + tb[3] - b[1] - b[3]) / 2)
                if iou > 0.05 or dist < 0.6 * max(tw, th, b[2] - b[0], b[3] - b[1]):
                    pairs.append((iou - dist * 1e-4, ti, di))
        pairs.sort(reverse=True)
        took = set()
        for _, ti, di in pairs:
            if ti in took or di in used:
                continue
            took.add(ti); used.add(di)
            t = active[ti]
            t["box"], t["poly"], t["last"] = dets[di][0], dets[di][1], i
            t["hits"][i] = dets[di]
        for di, (b, p) in enumerate(dets):
            if di not in used:
                t = {"id": len(tracks), "box": b, "poly": p, "last": i, "hits": {i: (b, p)}}
                tracks.append(t); active.append(t)
        active = [t for t in active if i - t["last"] <= HOLD]
    per = [[] for _ in range(n)]
    for t in tracks:
        if len(t["hits"]) < CONFIRM:          # 한두 번만 잡힌 건 엉뚱한 것으로 봄
            continue
        fr = sorted(t["hits"])
        first, last = fr[0], fr[-1]
        # 사라진 뒤 얼마나 유지할지: 오래 안정적으로 잡히던 것만 조금 길게, 화면 밖으로 나갔으면 바로 끝
        recent = sum(1 for f in fr if last - 30 <= f <= last)
        tail = TAIL_STABLE if recent >= 12 else TAIL_SHORT
        lb = t["hits"][last][0]
        if fw and fh and (lb[0] <= 4 or lb[1] <= 4 or lb[2] >= fw - 4 or lb[3] >= fh - 4):
            tail = TAIL_SHORT
        pre = PRE if len(fr) >= 6 else 0
        k = 0
        for i in range(max(0, first - pre), min(n, last + tail + 1)):
            while k + 1 < len(fr) and fr[k + 1] <= i:
                k += 1
            lh = fr[k] if fr[k] <= i else first          # 마지막으로 잡힌 프레임 (처음 전이면 첫 프레임)
            if i > first and i - lh > (GAP_FILL if i <= last else tail):   # 중간에 잠깐 놓친 건 메움, 오래 빈 건 안 메움
                continue
            b, p = t["hits"][lh]
            per[i].append((t["id"], b, p, i in t["hits"]))   # 놓친 동안엔 마지막 자리 그대로 (미끄러지지 않음)
    for i in range(n):
        live = [e[1] for e in per[i] if e[3]]
        if live:
            def near(e, b):          # 기억 자리 근처에서 다른 모자이크가 실제로 잡히면 = 같은 게 움직인 것 → 기억 자리는 버림
                eb = e[1]
                size = max(eb[2] - eb[0], eb[3] - eb[1], b[2] - b[0], b[3] - b[1])
                d = np.hypot((eb[0] + eb[2] - b[0] - b[2]) / 2, (eb[1] + eb[3] - b[1] - b[3]) / 2)
                return _iou(eb, b) > 0.1 or d < 2.5 * size
            per[i] = [e for e in per[i] if e[3] or not any(near(e, b) for b in live)]
    return per


# ======================= 2. 타원 계산 =======================
def mvee(pts, tol=1e-3, it=200):
    """점들을 모두 감싸는 가장 작은 타원 → (중심, A)  [(p-c)^T A (p-c) <= 1]"""
    P = np.asarray(pts, float)
    if len(P) < 3 or np.linalg.matrix_rank(P - P.mean(0)) < 2:
        c = P.mean(0) if len(P) else np.zeros(2)
        r = max(5.0, np.abs(P - c).max() if len(P) else 5.0)
        return c, np.eye(2) / r ** 2
    N, d = P.shape
    Q = np.vstack([P.T, np.ones(N)])
    u = np.ones(N) / N
    for _ in range(it):
        X = Q @ np.diag(u) @ Q.T
        M = np.einsum("ij,ji->i", Q.T @ np.linalg.inv(X), Q)
        j = int(np.argmax(M))
        step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        nu = (1 - step) * u
        nu[j] += step
        if np.linalg.norm(nu - u) < tol:
            u = nu
            break
        u = nu
    c = P.T @ u
    A = np.linalg.inv(P.T @ np.diag(u) @ P - np.outer(c, c)) / d
    return c, A


def _axes(A):
    w, v = np.linalg.eigh(A)
    return 1 / np.sqrt(np.maximum(w, 1e-9)), v      # 반지름들, 방향


def grow(e, s, pad):
    c, A = e
    r, v = _axes(A)
    r = r * s + pad
    return c, v @ np.diag(1 / r ** 2) @ v.T


def boundary(e, k=32):
    c, A = e
    r, v = _axes(A)
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return c + (v @ (np.stack([np.cos(t) * r[0], np.sin(t) * r[1]]))).T


def contains(e, pts, slack=1.0):
    c, A = e
    d = pts - c
    return bool(np.all(np.einsum("ij,jk,ik->i", d, A, d) <= slack))


def area(e):
    return np.pi / np.sqrt(max(np.linalg.det(e[1]), 1e-18))


def lerp(e1, e2, t):
    M = (1 - t) * np.linalg.inv(e1[1]) + t * np.linalg.inv(e2[1])
    return (1 - t) * e1[0] + t * e2[0], np.linalg.inv(M)


# ======================= 3. 음모 찾기 =======================
_GPU = {"dev": None, "tried": False}


def _gpu():
    """빠른 훑기에서 쓸 그래픽카드 (없거나 못 쓰면 None → 원래대로 CPU)"""
    if not _GPU["tried"]:
        _GPU["tried"] = True
        try:
            import torch
            if torch.cuda.is_available():
                _GPU["dev"] = torch.device("cuda")
        except Exception:
            _GPU["dev"] = None
    return _GPU["dev"]


class _Stats:
    """털 판단에 쓰는 중간값·백분위를 한 번에 계산 (그래픽카드가 있으면 거기서, 없으면 numpy 와 같은 계산).
    numpy 의 percentile·median(직선 보간)과 같은 방식"""
    def __init__(self, dev, **arrs):
        self.dev = dev
        if dev is not None:
            import torch
            self.t = torch
            self.a = {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True) for k, v in arrs.items()}
        else:
            self.a = arrs

    def q(self, name, mask, p):
        if self.dev is None:
            return float(np.percentile(self.a[name][self.a[mask] if isinstance(mask, str) else mask], p))
        t = self.t
        m = self.a[mask] if isinstance(mask, str) else t.from_numpy(np.ascontiguousarray(mask)).to(self.dev)
        v = self.a[name][m].double()
        return float(t.quantile(v, p / 100.0)) if v.numel() < 16_000_000 else float(np.percentile(v.cpu().numpy(), p))

    def mean_lt(self, name, mask, thr):
        if self.dev is None:
            return float((self.a[name][self.a[mask] if isinstance(mask, str) else mask] < thr).mean())
        m = self.a[mask] if isinstance(mask, str) else self.t.from_numpy(np.ascontiguousarray(mask)).to(self.dev)
        return float((self.a[name][m] < thr).double().mean())


def hair_mask(frame, box, near, faces=()):
    """원본 모자이크 주변에서 음모(가닥 결이 많고 살색이 빠진 덩어리)만 찾음. 얼굴 근처는 제외"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - bw - near)), max(0, int(y1 - bh - near))
    X2, Y2 = min(W, int(x2 + bw + near)), min(H, int(y2 + bh + near))
    if X2 - X1 < 16 or Y2 - Y1 < 16:
        return None, (0, 0)
    lab = cv2.cvtColor(frame[Y1:Y2, X1:X2], cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    valid = cv2.erode((L > 25).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    if valid.sum() < 100:
        return None, (0, 0)
    dev = _gpu() if FAST_SCAN else None
    S0 = _Stats(dev, L=L, valid=valid) if dev is not None else None
    skinL = S0.q("L", "valid", 80) if S0 else np.percentile(L[valid], 80)
    skin = valid & (L > skinL * 0.85)
    if skin.sum() < 50:
        return None, (0, 0)
    hp = L - cv2.GaussianBlur(L, (0, 0), 2)
    sd = np.sqrt(np.maximum(cv2.blur(hp * hp, (9, 9)), 0))
    S = _Stats(dev, L=L, C=C, sd=sd, skin=skin)
    skinC = S.q("C", "skin", 50)
    base = S.q("sd", "skin", 50) + 1e-3
    m = (valid & (L < skinL * 0.8) & (C < skinC * 0.9)).astype(np.uint8)
    m = (cv2.blur(m.astype(np.float32), (15, 15)) > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    inner = np.zeros_like(m)                     # 원본 모자이크 안쪽은 빼고 봄
    ix1, iy1, ix2, iy2 = int(x1 - X1 + bw * 0.1), int(y1 - Y1 + bh * 0.1), int(x2 - X1 - bw * 0.1), int(y2 - Y1 - bh * 0.1)
    if ix2 > ix1 and iy2 > iy1:
        cv2.rectangle(inner, (ix1, iy1), (ix2, iy2), 1, -1)
    m &= (inner == 0)
    for cls, (fx1, fy1, fx2, fy2) in faces:      # 얼굴(+머리카락)·가슴(브라)·겨드랑이·발 근처는 음모가 아님
        fw, fh = fx2 - fx1, fy2 - fy1
        if cls.startswith("FACE"):
            ex = (fw * 0.4, fh * 0.7, fw * 0.4, fh * 0.3)
        else:
            ex = (fw * 0.1, fh * 0.1, fw * 0.1, fh * 0.1)
        cv2.rectangle(m, (int(fx1 - ex[0] - X1), int(fy1 - ex[1] - Y1)),
                      (int(fx2 + ex[2] - X1), int(fy2 + ex[3] - Y1)), 0, -1)
    touch = np.zeros_like(m)
    cv2.rectangle(touch, (int(x1 - X1 - near), int(y1 - Y1 - near)), (int(x2 - X1 + near), int(y2 - Y1 + near)), 1, -1)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    hit = np.unique(lab_[(touch > 0) & (m > 0)])
    min_area = max(120, bw * bh * 0.004)         # 작은 털도 장면 크기에 비례해서 인정
    keep = []
    for k in hit:
        if k == 0 or st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lab_ == k
        tx = S.q("sd", comp, 50) / base                 # 가닥 결 (맨살 대비)
        cr = S.q("C", comp, 50) / skinC                 # 색 진하기 (살색 대비)
        lr = S.q("L", comp, 50) / skinL                 # 밝기 (살 대비)
        dk = S.mean_lt("L", comp, skinL * 0.5)          # 확실히 검은 가닥 비율
        # 음모 = 가닥 결이 많고 + 살색이 빠지고 + 검은 가닥이 실제로 있음
        # (레이스·땀·피부결은 결은 있어도 밝고 검은 가닥이 없음 / 다리털은 살색이 남음 / 그림자는 결이 없음)
        # 검은 옷(니트 등)은 결·어두움은 비슷해도 색이 거의 없음(살색의 절반 미만) → 제외
        if ((tx > HAIR_TEX and cr < 0.8) or (tx > 3.0 and cr < 0.86)) and (lr <= 0.62 or dk >= 0.15) and cr >= 0.5:
            keep.append(k)
    if not keep:
        return None, (0, 0)
    out = np.isin(lab_, keep).astype(np.uint8) & touch           # 인정 거리 바깥 부분은 잘라냄
    if not out.any():
        return None, (0, 0)
    return out, (X1, Y1)


# ======================= 4. 블러 재료 =======================
def _small_size(w, h):
    f = max(1.0, min(h, w) / FILL_RES)
    return f, max(8, int(w / f)), max(8, int(h / f))


def natural_fill(sub, need, sigma, cov):
    h, w = need.shape
    f, sw, sh = _small_size(w, h)
    small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
    nd = cv2.resize(need * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
    cv_ = cv2.resize((cov > 0.05).astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    return cv2.resize(fill_small(small, nd, cv_, sigma, f), (w, h), interpolation=cv2.INTER_CUBIC)


def fill_small(small, nd, cv_, sigma, f):
    """작은 크기(짧은 변 FILL_RES)에서 블러 재료 계산: 털·검은 곳 메우고 흐리게"""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    Cc = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ok = (L > 25) & (Cc > np.percentile(Cc, 60))
    skinL = np.median(L[ok]) if ok.sum() > 20 else 170
    skinC = np.median(Cc[ok]) if ok.sum() > 20 else 25
    notskin = (L < skinL * 0.75) | ((L > skinL * 1.12) & (Cc < skinC * 0.7)) | (Cc < skinC * 0.45)
    msk = ((nd > 0) | (notskin & cv_)).astype(np.uint8) * 255
    msk = cv2.dilate(msk, np.ones((5, 5), np.uint8))
    if msk.any():
        small = cv2.inpaint(small, msk, 3, cv2.INPAINT_TELEA)
    s2 = max(1.0, sigma / f)
    k = int(s2 * 3) * 2 + 1
    if hasattr(cv2, "stackBlur") and k > 15:          # 아주 센 흐림은 빠른 방식으로 (모양은 가우시안과 거의 같음)
        k2 = int(s2 * 2.5) * 2 + 1
        small = cv2.stackBlur(small, (k2, k2))
    else:
        small = cv2.GaussianBlur(small, (k, k), s2)
    return small


def fill_small_v2(small, nd, alpha_s, sigma, f):
    """새 블러 재료: 흐림 세기를 4단계로 만들어 두고, 가장자리(투명도 낮음) → 약하게, 안쪽 → 세게 이어 붙임.
    털·모자이크 메우기는 타원 안쪽만. 세게 흐릴 때는 타원 안쪽 내용을 주로 써서 바깥의 검은 배경·머리카락이 안쪽까지 번지지 않게"""
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    Cc = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    core = alpha_s >= 0.99
    ref = core if core.sum() > 20 else alpha_s > 0.05
    Lr, Cr = L[ref], Cc[ref]
    ok = (Lr > 25) & (Cr > np.percentile(Cr, 60)) if Cr.size else np.zeros(0, bool)
    skinL = np.median(Lr[ok]) if ok.sum() > 20 else 170
    skinC = np.median(Cr[ok]) if ok.sum() > 20 else 25
    notskin = (L < skinL * 0.75) | ((L > skinL * 1.12) & (Cc < skinC * 0.7)) | (Cc < skinC * 0.45)
    msk = ((nd > 0) | (notskin & core)).astype(np.uint8) * 255
    msk = cv2.dilate(msk, np.ones((5, 5), np.uint8))
    img = small.astype(np.float32)
    if msk.any():                                     # 메운 곳과 안 메운 곳 사이에 선이 생기지 않게 부드럽게 섞음
        inp = cv2.inpaint(small, msk, 3, cv2.INPAINT_TELEA).astype(np.float32)
        ms = cv2.GaussianBlur(msk.astype(np.float32) / 255, (0, 0), 2.0)[..., None]
        img = img + (inp - img) * ms
    w = (0.2 + 0.8 * np.clip(alpha_s, 0, 1) ** 2).astype(np.float32)   # 안쪽 내용 위주로 (끊김 없이 이어지는 비율)
    iw = img * w[..., None]
    s2 = max(1.0, sigma / f)
    lv = []
    for t in (0.125, 0.25, 0.5, 1.0):
        sg = max(0.8, s2 * t)
        num = cv2.GaussianBlur(iw, (0, 0), sg)
        den = cv2.GaussianBlur(w, (0, 0), sg)
        lv.append(num / np.maximum(den, 1e-4)[..., None])
    p = np.clip(alpha_s * 4.0, 0, 4.0)
    idx = np.clip(np.floor(p), 1, 3).astype(np.int32)
    fr = np.clip(p - idx, 0, 1)[..., None]
    M = np.zeros_like(lv[0])
    for k in (1, 2, 3):
        m = (idx == k)[..., None]
        M = np.where(m, lv[k - 1] * (1 - fr) + lv[k] * fr, M)
    return M.astype(np.float32)


# ======================= 5. 프레임 처리 =======================
_detector = None


def exposures(frame):
    """노출 탐지: 반환 (덮을 곳 상자들, 얼굴 상자들)"""
    global _detector
    if _detector is None:
        from nudenet import NudeDetector
        _detector = NudeDetector()
    must, faces = [], []
    for d in _detector.detect(frame):
        if d["score"] < EXPOSE_SCORE:
            continue
        x, y, w, h = d["box"]
        b = [x, y, x + w, y + h]
        if d["class"] in ("ANUS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"):
            must.append(b)
        elif d["class"] in ("FACE_FEMALE", "FACE_MALE", "FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED",
                            "ARMPITS_EXPOSED", "ARMPITS_COVERED", "FEET_EXPOSED", "FEET_COVERED"):
            faces.append((d["class"], b))
    return must, faces


MUST_CLS = ("ANUS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED")
FACE_CLS = ("FACE_FEMALE", "FACE_MALE", "FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED",
            "ARMPITS_EXPOSED", "ARMPITS_COVERED", "FEET_EXPOSED", "FEET_COVERED")


class NudeBatch:
    """노출 탐지(NudeNet)를 여러 프레임 한꺼번에. 그래픽카드가 있으면 그래픽카드로 (같은 모델을 torch로 옮겨 실행)"""
    def __init__(self, size=320):
        import os, nudenet
        mod = nudenet.nudenet
        self.labels = getattr(mod, "__labels")
        path = os.path.join(os.path.dirname(mod.__file__), "320n.onnx")
        self.size, self.how, self.torch, self.sess = size, "CPU", None, None
        import onnxruntime
        self.sess = onnxruntime.InferenceSession(path)
        self.inp = self.sess.get_inputs()[0].name
        try:
            import torch, onnx, onnx2torch
            if torch.cuda.is_available():
                m = onnx2torch.convert(onnx.load(path)).eval().cuda()
                # 그래픽카드 결과가 원래(CPU) 결과와 같은지 한 번 확인 — 다르면 그래픽카드 안 씀 (조용히 놓치는 일 방지)
                rng = np.random.default_rng(0)
                x = rng.random((2, 3, size, size), dtype=np.float32)
                ref = self.sess.run(None, {self.inp: x})[0]
                with torch.no_grad():
                    got = m(torch.from_numpy(x).cuda()).float().cpu().numpy()
                if got.shape == ref.shape and float(np.abs(got[:, 4:] - ref[:, 4:]).max()) < 0.02:
                    self.torch = m
                    self.how = "그래픽카드"
        except Exception:
            self.torch = None

    def detect(self, frames):
        blobs, meta = [], []
        for f in frames:
            h, w = f.shape[:2]
            m = max(h, w)
            pad = cv2.copyMakeBorder(f, 0, m - h, 0, m - w, cv2.BORDER_CONSTANT)
            blobs.append(cv2.dnn.blobFromImage(pad, 1 / 255.0, (self.size, self.size), (0, 0, 0), swapRB=False, crop=False))   # 원래 NudeNet과 같은 색 순서
            meta.append((w, h, m))
        x = np.vstack(blobs)
        if self.torch is not None:
            import torch
            with torch.no_grad():
                out = self.torch(torch.from_numpy(x).cuda()).float().cpu().numpy()
        else:
            out = self.sess.run(None, {self.inp: x})[0]
        res = []
        for o, (w, h, m) in zip(out, meta):
            o = o.T                                   # (2100, 4+클래스)
            sc = o[:, 4:]; cid = sc.argmax(1); ms = sc.max(1)
            keep = ms >= 0.2
            o, cid, ms = o[keep], cid[keep], ms[keep]
            k = m / self.size
            bx = np.stack([(o[:, 0] - o[:, 2] / 2) * k, (o[:, 1] - o[:, 3] / 2) * k, o[:, 2] * k, o[:, 3] * k], 1)
            bx[:, 0] = bx[:, 0].clip(0, w); bx[:, 1] = bx[:, 1].clip(0, h)
            bx[:, 2] = np.minimum(bx[:, 2], w - bx[:, 0]); bx[:, 3] = np.minimum(bx[:, 3], h - bx[:, 1])
            idx = cv2.dnn.NMSBoxes(bx.tolist(), ms.tolist(), 0.25, 0.45) if len(bx) else []
            must, faces = [], []
            for j in np.array(idx).reshape(-1):
                if ms[j] < EXPOSE_SCORE:
                    continue
                x0, y0, ww, hh = [float(v) for v in bx[j]]
                b = [x0, y0, x0 + ww, y0 + hh]
                c = self.labels[int(cid[j])]
                if c in MUST_CLS:
                    must.append(b)
                elif c in FACE_CLS:
                    faces.append((c, b))
            res.append({"must": must, "faces": faces})
        return res


class Renderer:
    def __init__(self):
        self.E = {}          # 추적번호 → 지금 그리고 있는 타원
        self.over = {}
        self.hair = {}       # 추적번호 → 최근 음모 점들 [(프레임, 점들)]
        self.fill = None     # (자른 범위, 블러 재료)
        self.prev_small = None
        self.exp = ([], [])
        self.exp_i = -999
        self.must_mem = []   # [(프레임, 상자)] 노출 탐지 기억
        self.last = None
        self.pbox = {}
        self.jump = set()
        self.chase = set()   # 새 자리로 옮겨 가는 중인 추적번호
        self.goal = {}       # 옮겨 갈 목표 타원

    def process(self, frame, i, entries):
        """반환: (결과 프레임, 덮이지 않은 음모 픽셀 수)"""
        H, W = frame.shape[:2]
        entries = [tuple(e[:4]) + ((e[4] if len(e) > 4 else None),) for e in entries]
        planned = {e[0]: e[4] for e in entries}
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = self.prev_small is not None and np.abs(small - self.prev_small).mean() > CUT_DIFF
        self.prev_small = small
        if cut:                                   # 화면이 바뀌면 이전 자리 기억은 버림
            self.E.clear(); self.over.clear(); self.hair.clear(); self.chase.clear(); self.fill = None; self.must_mem = []
            entries = [e for e in entries if e[3]]
        if EXPOSE and (i - self.exp_i >= EXPOSE_EVERY or cut):
            try:
                self.exp = exposures(frame)
            except Exception:
                self.exp = ([], [])
            self.exp_i = i
            self.must_mem = [(j, b) for j, b in self.must_mem if i - j <= 6] + [(i, b) for b in self.exp[0]]
        faces = self.exp[1]
        must = [b for _, b in self.must_mem]

        self.jump = set()
        for tid, box, poly, det, _ in entries:
            if det:
                pb = self.pbox.get(tid)
                if pb is not None and _iou(pb, box) < 0.3:
                    self.jump.add(tid)
                self.pbox[tid] = box
        alive = set()
        need = np.zeros((H, W), np.uint8)
        hair_px = []
        targets = []
        for tid, box, poly, det, _ in entries:
            alive.add(tid)
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            short = max(8.0, min(bw, bh))
            reach = float(np.clip(short * HAIR_REACH, REACH_MIN, REACH_MAX))
            if poly and len(poly) >= 3:
                pts = np.asarray(poly, float)
                cv2.fillPoly(need, [np.asarray(poly, np.int32)], 1)
            else:
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
                cv2.rectangle(need, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
            allp = [pts]
            # 음모
            mem = [(j, p) for j, p in self.hair.get(tid, []) if i - j <= HAIR_MEM]
            if HAIR:
                hm, (ox, oy) = hair_mask(frame, box, reach, faces)
                if hm is not None:
                    hp_ = cv2.findNonZero(hm)
                    hull = cv2.convexHull(hp_).reshape(-1, 2).astype(float) + (ox, oy)
                    mem.append((i, hull))
                    sub = need[oy:oy + hm.shape[0], ox:ox + hm.shape[1]]
                    sub |= hm
                    hair_px.append((hm, ox, oy))
            self.hair[tid] = mem
            for _, p in mem:
                allp.append(p)
            # 노출 탐지(항문·성기): 이 모자이크 근처면 같이 감쌈
            for mb in must:
                mx, my = (mb[0] + mb[2]) / 2, (mb[1] + mb[3]) / 2
                if x1 - reach - bw * 0.5 <= mx <= x2 + reach + bw * 0.5 and y1 - reach - bh * 0.5 <= my <= y2 + reach + bh * 0.5:
                    allp.append(np.array([[mb[0], mb[1]], [mb[2], mb[1]], [mb[2], mb[3]], [mb[0], mb[3]]], float))
                    cv2.rectangle(need, (int(mb[0]), int(mb[1])), (int(mb[2]), int(mb[3])), 1, -1)   # 얼굴과 겹쳐도 지워지지 않게
            P = np.vstack(allp)
            core = mvee(cv2.convexHull(P.astype(np.float32)).reshape(-1, 2))   # 반드시 덮을 곳을 딱 감싸는 타원
            targets.append((tid, grow(core, SCALE, PAD_PX), core))
        # 어느 모자이크에도 안 붙은 노출(항문 등)도 따로 덮음
        used_must = set()
        for k, mb in enumerate(must):
            for tid, T, _ in targets:
                if contains(T, np.array([[(mb[0] + mb[2]) / 2, (mb[1] + mb[3]) / 2]])):
                    used_must.add(k)
        for k, mb in enumerate(must):
            if k in used_must:
                continue
            P = np.array([[mb[0], mb[1]], [mb[2], mb[1]], [mb[2], mb[3]], [mb[0], mb[3]]], float)
            tid = ("x", int(mb[0] // 80), int(mb[1] // 80))
            alive.add(tid)
            core = mvee(P)
            targets.append((tid, grow(core, SCALE * 1.1, PAD_PX), core))
            cv2.rectangle(need, (int(mb[0]), int(mb[1])), (int(mb[2]), int(mb[3])), 1, -1)

        # 타원 움직이기: 덮을 곳이 안에 있으면 가만히 / 벗어나면 바로 넓힘 / 한동안 너무 크면 천천히 줄임
        shown = []
        for tid, T, core in targets:
            Tb = boundary(T)
            need_b = boundary(grow(core, 1.04, 4))              # 반드시 덮을 곳(+약간) — 이게 벗어날 때만 움직임
            P_ = planned.get(tid)
            if P_ is not None:                                   # 키프레임 계획이 있으면 그대로 따름
                E = P_
                if not contains(E, need_b):                      # 음모·노출이 계획 밖으로 나오면 이 프레임만 넓혀 감쌈
                    E = mvee(np.vstack([boundary(E), need_b]))
                shown.append(E)
                continue
            E = self.E.get(tid)
            jump = tid in self.jump                              # 원본 모자이크가 순간이동(장면 전환·자세 급변)
            if E is None or jump or not contains(E, core[0][None]):
                E = T                                            # 처음이거나 완전히 다른 곳으로 이동
                self.over[tid] = 0
                self.chase.discard(tid)
            elif tid in self.chase or not contains(E, need_b):
                # 삐져나오려 함 → 그 순간의 새 자리(목표)를 정하고 몇 프레임에 걸쳐 부드럽게 옮겨 감
                # (목표는 옮기는 동안 고정 = 키프레임 두 개 사이를 잇는 것과 같음. 매 프레임 흔들리는 탐지값을 쫓지 않음)
                if tid not in self.chase:
                    self.goal[tid] = T
                    self.chase.add(tid)
                G = self.goal[tid]
                E2 = lerp(E, G, CHASE)
                rG, _ = _axes(G[1])
                if np.hypot(*(E2[0] - G[0])) < 0.05 * min(rG) and abs(area(E2) / area(G) - 1) < 0.1:
                    E2 = G
                    self.chase.discard(tid)                      # 목표 도착 → 다시 가만히
                if not contains(E2, need_b):
                    E2 = mvee(np.vstack([boundary(E2), need_b]))  # 옮기는 중에도 덮을 곳은 항상 안에
                E = E2
                self.over[tid] = 0
            else:                                                # 안에 있으면 가만히
                if area(E) > area(T) * SHRINK_RATIO:
                    self.over[tid] = self.over.get(tid, 0) + 1
                else:
                    self.over[tid] = 0
                if self.over[tid] > SHRINK_AFTER:                # 한동안 너무 컸으면 천천히 줄임
                    E2 = lerp(E, T, 0.08)
                    if contains(E2, need_b):
                        E = E2
            self.E[tid] = E
            shown.append(E)
        for tid in list(self.E):
            if tid not in alive:
                self.E.pop(tid); self.over.pop(tid, None); self.hair.pop(tid, None); self.chase.discard(tid)
        if not shown:
            self.fill = None
            self.last = None
            return frame, 0

        # 투명도: 타원 안 100%, 밖으로 갈수록 부드럽게 0
        boxes = []
        for c, A in shown:
            r, _ = _axes(A)
            F = float(np.clip(min(r) * FEATHER_W, 20, FEATHER_MAX))
            R = max(r) + F
            boxes.append((c, A, r, F, R))
        X0 = int(max(0, min(c[0] - R for c, _, _, _, R in boxes)))
        Y0 = int(max(0, min(c[1] - R for c, _, _, _, R in boxes)))
        X1 = int(min(W, max(c[0] + R for c, _, _, _, R in boxes) + 1))
        Y1 = int(min(H, max(c[1] + R for c, _, _, _, R in boxes) + 1))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            return frame, 0
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        alpha = np.zeros((Y1 - Y0, X1 - X0), np.float32)
        for c, A, r, F, R in boxes:
            dx, dy = xx - c[0], yy - c[1]
            q = A[0, 0] * dx * dx + 2 * A[0, 1] * dx * dy + A[1, 1] * dy * dy
            rr = np.sqrt(np.maximum(q, 0))
            dist = (rr - 1) * np.sqrt(r[0] * r[1])
            t = np.clip(1 - dist / F, 0, 1)
            alpha = np.maximum(alpha, np.where(rr <= 1, 1.0, t * t * (3 - 2 * t)))
        # 얼굴은 덮지 않음: 반드시 덮을 곳(원본 모자이크·음모·항문)이 아니면 얼굴 위 블러는 걷어냄
        nd = need[Y0:Y1, X0:X1]
        for cls, (fx1, fy1, fx2, fy2) in faces:
            if not cls.startswith("FACE"):
                continue
            fm = np.zeros_like(alpha)
            cv2.rectangle(fm, (int(fx1 - X0), int(fy1 - Y0)), (int(fx2 - X0), int(fy2 - Y0)), 1.0, -1)
            k = max(3, int((fx2 - fx1) * 0.3) | 1)
            fm = cv2.GaussianBlur(fm, (k, k), 0)
            alpha = np.where(nd > 0, alpha, alpha * (1 - fm))
        sub = frame[Y0:Y1, X0:X1]
        minr = min(min(r) for _, _, r, _, _ in boxes)
        soft = natural_fill(sub, need[Y0:Y1, X0:X1], max(15.0, 2 * minr * BLUR_FRAC), alpha).astype(np.float32)
        if self.fill is not None and self.fill[0] == (X0, Y0, X1, Y1):
            soft = FILL_KEEP * self.fill[1] + (1 - FILL_KEEP) * soft   # 블러 속은 천천히만 바뀜
        self.fill = ((X0, Y0, X1, Y1), soft)
        self.last = (X0, Y0, alpha, shown)
        self.need_crop = nd
        a = alpha[..., None]
        o = (sub.astype(np.float32) * (1 - a) + soft * a).clip(0, 255).astype(np.uint8)
        out = frame.copy()
        out[Y0:Y1, X0:X1] = o
        left = 0
        for hm, ox, oy in hair_px:                              # 점검: 덮이지 않은 음모
            h_, w_ = hm.shape
            ax0, ay0 = max(ox, X0), max(oy, Y0)
            ax1, ay1 = min(ox + w_, X1), min(oy + h_, Y1)
            total = int(hm.sum())
            inside = 0
            if ax1 > ax0 and ay1 > ay0:
                hs = hm[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox].astype(bool)
                inside = int((hs & (alpha[ay0 - Y0:ay1 - Y0, ax0 - X0:ax1 - X0] > 0.95)).sum())
            left += total - inside
        return out, left


# ======================= 6. 새 방식: 먼저 훑기 → 계획 → 그리기 =======================
# 프레임마다 반응하지 않음. 영상 전체의 '덮을 곳'을 먼저 모은 뒤 키프레임을 정하고, 그리기는 계획대로만.

class Analyzer:
    """1단계(훑기): 프레임마다 모자이크별 '반드시 덮을 점들'과, 어느 모자이크에도 안 붙은 노출 상자를 모음"""
    def __init__(self):
        self.hair = {}
        self.exp = ([], [])
        self.exp_i = -999
        self.must_mem = []
        self.prev_small = None

    def analyze(self, frame, i, entries, exp=None):
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = self.prev_small is not None and np.abs(small - self.prev_small).mean() > CUT_DIFF
        self.prev_small = small
        if cut:
            self.hair.clear(); self.must_mem = []
        if exp is not None:                       # 위치 찾기 단계에서 그래픽카드로 매 프레임 미리 찾아 둔 결과
            self.exp = (exp["must"], [tuple(f) for f in exp["faces"]])
            self.must_mem = [(j, b) for j, b in self.must_mem if i - j <= 6] + [(i, b) for b in self.exp[0]]
        elif EXPOSE and (i - self.exp_i >= EXPOSE_EVERY or cut):
            try:
                self.exp = exposures(frame)
            except Exception:
                self.exp = ([], [])
            self.exp_i = i
            self.must_mem = [(j, b) for j, b in self.must_mem if i - j <= 6] + [(i, b) for b in self.exp[0]]
        faces = self.exp[1]
        must = [b for _, b in self.must_mem]
        req = {}
        used = set()
        for e in entries:
            tid, box, poly, det = e[:4]
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            reach = float(np.clip(max(8.0, min(bw, bh)) * HAIR_REACH, REACH_MIN, REACH_MAX))
            if poly and len(poly) >= 3:
                pts = [np.asarray(poly, float)]
            else:
                pts = [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)]
            mem = [(j, p) for j, p in self.hair.get(tid, []) if i - j <= HAIR_MEM]
            if HAIR:
                hm, (ox, oy) = hair_mask(frame, box, reach, faces)
                if hm is not None:
                    hull = cv2.convexHull(cv2.findNonZero(hm)).reshape(-1, 2).astype(float) + (ox, oy)
                    mem.append((i, hull))
            self.hair[tid] = mem
            pts += [p for _, p in mem]
            for k, mb in enumerate(must):
                mx, my = (mb[0] + mb[2]) / 2, (mb[1] + mb[3]) / 2
                if x1 - reach - bw * 0.5 <= mx <= x2 + reach + bw * 0.5 and y1 - reach - bh * 0.5 <= my <= y2 + reach + bh * 0.5:
                    pts.append(np.array([[mb[0], mb[1]], [mb[2], mb[1]], [mb[2], mb[3]], [mb[0], mb[3]]], float))
                    used.add(k)
            P = np.vstack(pts)
            req[tid] = cv2.convexHull(P.astype(np.float32)).reshape(-1, 2)
        loose = [list(map(float, mb)) for k, mb in enumerate(must) if k not in used]
        return {"req": {str(k): v.tolist() for k, v in req.items()}, "loose": loose, "cut": bool(cut)}


SWITCH_COST = 15    # 블러를 한 번 바꾸는 비용 (화면 전체를 이 프레임 수만큼 덮는 것과 같은 무게) — 클수록 덜 바꿈
SEG_MAX = 900       # 고정 블러 한 구간 최대 길이 (프레임, 30초)


def _segments(bx, forced, fw, fh):
    """덮는 면적×시간 + 바꾸는 횟수×SWITCH_COST 가 가장 작게 구간을 나눔. bx: 프레임마다 덮을 곳 상자"""
    n = len(bx)
    best = np.full(n + 1, np.inf); best[0] = 0.0
    arg = np.zeros(n + 1, int)
    fa = float(fw * fh)
    for j in range(1, n + 1):
        lo = max(0, j - SEG_MAX)
        f = [k for k in forced if lo < k < j]
        if f:
            lo = max(f)                                  # 화면 전환 등은 반드시 끊음
        seg = bx[lo:j][::-1]
        x1 = np.minimum.accumulate(seg[:, 0]); y1 = np.minimum.accumulate(seg[:, 1])
        x2 = np.maximum.accumulate(seg[:, 2]); y2 = np.maximum.accumulate(seg[:, 3])
        L = np.arange(1, len(seg) + 1)
        cost = (x2 - x1) * (y2 - y1) / fa * L + SWITCH_COST
        starts = j - L
        tot = best[starts] + cost
        k = int(np.argmin(tot))
        best[j], arg[j] = tot[k], starts[k]
    out, j = [], n
    while j > 0:
        out.append((int(arg[j]), j))
        j = int(arg[j])
    return out[::-1]


SIZE_K = 0.8        # 블러 지름 = 장면 대표 크기(가려야 할 곳의 가로·세로 폭)의 이 배
SIZE_PCT = 80       # 장면 대표 크기 = 장면 안 프레임별 크기 중 이 백분위 (큰 쪽)
MIN_R = 50          # 블러 최소 반지름 (픽셀)
ASPECT_MAX = 1.6    # 블러 타원의 길쭉함 한도 (1 = 원)
DEAD = 0.25         # 가려야 할 곳 중심이 블러 중심에서 (블러 짧은 반지름 × 이 값) 안에서 움직이면 블러는 가만히
POS_SMOOTH = 10     # 위치 다듬기 (프레임)
SIZE_GLIDE = 15     # (예전)
FADE = (0.2, 0.5, 0.8)   # 나타날 때 불투명도 (0% 다음 프레임부터 20→50→80→100%), 사라질 땐 반대로
HAIR_BAND = 0.3     # 모자이크 가장자리에서 (모자이크 짧은 변 × 이 값) 안의 털은 무조건 덮음
HAIR_BAND_MAX = 90  # 그 거리 최대 (픽셀)


BRIDGE = 60         # 같은 자리 모자이크가 이 프레임 수(2초) 안에 다시 잡히면 블러를 끊지 않고 이어 감


def _parts(per, info, n, cutf):
    """원본 모자이크 추적 조각들을 '부위'로 묶음 → {추적번호: 부위번호}
    같은 자리에 겹쳐 잡히거나, 잠깐(BRIDGE 안) 끊겼다 같은 자리에서 다시 잡히면 같은 부위 (화면 전환을 넘어서는 안 묶음)"""
    def box_of(i, e):
        pts = info[i]["req"].get(str(e[0])) if info[i] else None
        if pts is not None and len(pts):
            P = np.asarray(pts, float)
            return np.array([P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()])
        return np.asarray(e[1], float)
    def near(a, b):
        if _iou(a, b) > 0:
            return True
        s_ = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
        return np.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2) < 0.6 * s_
    part, parts = {}, []                                  # parts[k] = [마지막 프레임, 마지막 상자]
    for i in range(n):
        if cutf[i]:
            for p in parts:
                p[0] = -10 ** 9                            # 화면이 바뀌면 이전 부위와는 안 이음
        now = {}
        for e in per[i]:
            b = box_of(i, e)
            tid = e[0]
            if tid not in part:
                best, bd = None, None
                for k, p in enumerate(parts):
                    if i - p[0] > BRIDGE or not near(p[1], b):
                        continue
                    dd = np.hypot((p[1][0] + p[1][2] - b[0] - b[2]) / 2, (p[1][1] + p[1][3] - b[1] - b[3]) / 2)
                    if bd is None or dd < bd:
                        best, bd = k, dd
                if best is None:
                    parts.append([i, b]); best = len(parts) - 1
                part[tid] = best
            k = part[tid]
            if k in now:
                o = now[k]
                now[k] = np.array([min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3])])
            else:
                now[k] = b
        for k, b in now.items():
            parts[k][0], parts[k][1] = i, b
    return part


def _ell(c, r1, r2, th):
    v = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.asarray(c, float), v @ np.diag([1 / r1 ** 2, 1 / r2 ** 2]) @ v.T


def plan(per, info, n, fw=1920, fh=1080):
    """2단계(계획): 포그팀 방식
    - 장면(화면 전환 사이, 크게 줌이 바뀌면 그 안에서도 나눔)마다 블러 크기·모양을 '한 번' 정해 고정
      크기 = 그 장면에서 가려야 할 곳(원본 모자이크+음모+노출) 크기의 큰 쪽 대표값 × SIZE_K, 모양은 둥근 타원
    - 위치는 가려야 할 곳의 중심을 부드럽게 다듬고, 조금 움직이는 건 무시(가만히), 크게 움직일 때만 천천히 따라감
    반환: (프레임마다 [타원, ...], 블러 밖으로 나간 가려야 할 곳이 있었던 프레임 목록)"""
    from scipy.ndimage import gaussian_filter1d
    # 원본 모자이크(+털·바로 옆 노출)만 덮음. 모자이크와 떨어진 노출만 보고 만드는 블러는 만들지 않음
    cutf = [bool(info[i]["cut"]) if info[i] else False for i in range(n)]
    part = _parts(per, info, n, cutf)
    acc = {}                                              # 부위 → 프레임 → (모양 점들, 무조건 덮을 점들)
    for i in range(n):
        if not info[i]:
            continue
        for e in per[i]:
            pts = info[i]["req"].get(str(e[0]))
            if pts is None:
                continue
            bb = e[1]
            mp = np.asarray(e[2], float) if e[2] and len(e[2]) >= 3 else \
                np.array([[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]], float)
            band = float(np.clip(min(bb[2] - bb[0], bb[3] - bb[1]) * HAIR_BAND, 30, HAIR_BAND_MAX))
            rq = np.asarray(pts, float).copy()                 # 모자이크 가까이 붙은 털·노출까지만 '무조건 덮을 곳'
            rq[:, 0] = np.clip(rq[:, 0], bb[0] - band, bb[2] + band)
            rq[:, 1] = np.clip(rq[:, 1], bb[1] - band, bb[3] + band)
            d = acc.setdefault(part[e[0]], {})
            if i in d:                                     # 같은 부위의 모자이크가 겹쳐 잡히면 하나로 합침
                d[i] = (np.vstack([d[i][0], np.asarray(pts, float)]), np.vstack([d[i][1], mp, rq]))
            else:
                d[i] = (np.asarray(pts, float), np.vstack([mp, rq]))
    seq = {}
    for pid, d in acc.items():
        fs = sorted(d)
        items = []
        for k, i in enumerate(fs):
            if k and 1 < i - fs[k - 1] <= BRIDGE and not any(cutf[fs[k - 1] + 1:i + 1]):
                pv = d[fs[k - 1]]                          # 잠깐 못 잡힌 동안은 마지막 자리 그대로 이어 감
                for g in range(fs[k - 1] + 1, i):
                    items.append((g, pv[0], False, pv[1]))
            items.append((i, d[i][0], cutf[i], d[i][1]))
        seq[("t", pid)] = items
    out = [[] for _ in range(n)]
    leak = set()
    segtab = {}
    for key, items in seq.items():
        frames = [it[0] for it in items]
        C, R1, R2, TH = [], [], [], []                      # 빠른 계산: 가장 작게 감싸는 회전 사각형으로 크기·방향
        for it in items:
            (cx, cy), (w, h), a_ = cv2.minAreaRect(it[1].astype(np.float32))
            if w < h:
                w, h, a_ = h, w, a_ + 90
            C.append((cx, cy)); R1.append(w / np.sqrt(2)); R2.append(max(h, 1.0) / np.sqrt(2)); TH.append(np.deg2rad(a_))
        C, R1, R2, TH = np.array(C, float), np.array(R1), np.array(R2), np.array(TH)
        bx = np.array([[it[3][:, 0].min(), it[3][:, 1].min(), it[3][:, 0].max(), it[3][:, 1].max()] for it in items], float)
        forced = [k for k in range(1, len(items)) if items[k][2] or frames[k] != frames[k - 1] + 1]
        segs = _segments(bx, forced, fw, fh)
        # 자세가 확 바뀌는 순간: 앞 구간 크기로 못 덮는 끝 프레임들은 다음(큰) 구간으로 넘겨 미리 바꿈
        def _ctr(P):
            return (P.min(0) + P.max(0)) / 2
        def _need(k, U):
            cc = _ctr(items[k][3])
            d = items[k][3] - cc
            return float(np.sqrt(np.einsum("ij,jk,ik->i", d, U, d).max()))
        def _shape(a, b):
            q1 = np.percentile(R1[a:b], SIZE_PCT); q2 = np.percentile(R2[a:b], SIZE_PCT)
            ratio = float(np.clip(q1 / max(q2, 1.0), 1.0, ASPECT_MAX))
            ang = 0.5 * np.arctan2(np.median(np.sin(2 * TH[a:b])), np.median(np.cos(2 * TH[a:b])))
            base = max(q1 / np.sqrt(2) * SIZE_K / ratio, q2 / np.sqrt(2) * SIZE_K)
            return ratio, ang, base
        fset = set(forced)
        for j in range(len(segs) - 1):
            a, b = segs[j]
            if b in fset:
                continue
            ratio, ang, base = _shape(a, b)
            U = _ell((0, 0), ratio, 1.0, ang)[1]
            nd = [_need(k, U) for k in range(a, b)]
            usual = max(base, float(np.percentile(nd, 90)))   # 이 구간에서 평소 필요한 크기
            moved = 0
            # 끝부분에 '평소보다 확 큰' 프레임만 다음 구간으로 넘김 (최대 0.5초). 구간을 통째로 넘기지 않음
            while b - a > 1 and moved < 15 and nd[b - 1 - a] > usual * 1.15:
                b -= 1; moved += 1
            segs[j] = (a, b); segs[j + 1] = (b, segs[j + 1][1])
        prev_r, prev_end = None, -9
        done_segs = []
        for a, b in segs:
            # 모양(길쭉함·기울기)과 크기: 이 구간에서 한 번만.
            # 구간 내내 '무조건 덮을 곳'이 들어가는 타원 중 가장 작은 모양을 고름 (각도·길쭉함 후보 중에서)
            q1 = np.percentile(R1[a:b], SIZE_PCT); q2 = np.percentile(R2[a:b], SIZE_PCT)
            mc = [_ctr(items[k][3]) for k in range(a, b)]
            ds = [items[k][3] - mc[k - a] for k in range(a, b)]
            best = None
            for ratio in (1.0, 1.2, 1.4, ASPECT_MAX):
                for ang in np.linspace(0, np.pi, 12, endpoint=False):
                    U_ = _ell((0, 0), ratio, 1.0, ang)[1]
                    f_ = max(float(np.sqrt(np.einsum("ij,jk,ik->i", d, U_, d).max())) for d in ds)
                    base_ = max(q1 / np.sqrt(2) * SIZE_K / ratio, q2 / np.sqrt(2) * SIZE_K)
                    r2_ = max(base_, f_ * 1.05)
                    ar = r2_ * r2_ * ratio
                    if best is None or ar < best[0] - 1e-6:
                        best = (ar, ratio, ang, U_, base_, f_)
            _, ratio, ang, U, base, fit = best
            if max(q1, q2) < 120 and ratio > 1.25:          # 작은 블러는 원에 가깝게
                ratio = 1.25
                U = _ell((0, 0), ratio, 1.0, ang)[1]
                fit = max(float(np.sqrt(np.einsum("ij,jk,ik->i", d, U, d).max())) for d in ds)
            r2 = max(base, fit * 1.05, MIN_R) + PAD_PX
            r1 = r2 * ratio
            A = U / r2 ** 2
            # 위치: 다듬은 중심을 '끈'처럼 따라감 — 조금 움직이면 가만히, 많이 움직일 때만 따라감
            cs = C[a:b].copy()
            if b - a > 3:
                cs = np.stack([gaussian_filter1d(cs[:, 0], POS_SMOOTH, mode="nearest"),
                               gaussian_filter1d(cs[:, 1], POS_SMOOTH, mode="nearest")], 1)
            dead = DEAD * r2
            cur = cs[0].copy()
            seg_e = []
            sid = len(segtab)
            segtab[sid] = None
            for k in range(a, b):
                d = cs[k - a] - cur
                L = float(np.hypot(*d))
                if L > dead:
                    cur = cur + d * (L - dead) / L
                # 원본 모자이크가 블러 밖으로 나가려 하면, 들어올 만큼만 옮김 (최우선)
                mpk = items[k][3]
                if not contains((cur, A), mpk):
                    tgt = mc[k - a]
                    lo, hi = 0.0, 1.0
                    for _ in range(12):
                        mid = (lo + hi) / 2
                        if contains((cur + (tgt - cur) * mid, A), mpk):
                            hi = mid
                        else:
                            lo = mid
                    cur = cur + (tgt - cur) * hi
                E = (cur.copy(), A)
                if not contains(E, mpk):
                    leak.add(frames[k])
                el = (E[0].tolist(), E[1].tolist(), 1.0, sid)
                out[frames[k]].append(el)
                seg_e.append(el)
            done_segs.append((frames[a], frames[b - 1], seg_e[0], seg_e[-1], bool(items[a][2])))
            segtab[sid] = {"start": frames[a], "end": frames[b - 1], "track": f"{key[0]}{key[1]}"}
        # 나타날 때·사라질 때·크기가 바뀔 때 부드럽게 (방송본부 방식: 0→20→50→80→100%, 1프레임씩)
        for j, (f0, f1, e0, e1, cut0) in enumerate(done_segs):
            joined = j > 0 and done_segs[j - 1][1] == f0 - 1 and not cut0
            if not cut0:                                   # 화면 전환과 동시에 시작하면 그냥 나타남
                for m, w in zip((f0 - 3, f0 - 2, f0 - 1), FADE):
                    if m >= 0 and not any(cutf[m + 1:f0 + 1]):   # 앞 화면으로 넘어가서까지 나타나진 않음
                        out[m].append((e0[0], e0[1], w, e0[3]))
            if joined:                                     # 크기가 바뀌는 곳: 옛 블러는 바뀐 뒤 서서히 사라짐
                pe = done_segs[j - 1][3]
                for m, w in zip((f0, f0 + 1, f0 + 2), FADE[::-1]):
                    if m < n:
                        out[m].append((pe[0], pe[1], w, pe[3]))
            last = j == len(done_segs) - 1 or done_segs[j + 1][0] != f1 + 1 or done_segs[j + 1][4]
            if last:
                for m, w in zip((f1 + 1, f1 + 2, f1 + 3), FADE[::-1]):
                    if m >= n or cutf[m]:                  # 화면이 바뀌면 거기서 끝
                        break
                    out[m].append((e1[0], e1[1], w, e1[3]))
    for i in range(n):                                     # 다른 블러 안에 완전히 들어가는(그리고 더 진하지 않은) 블러는 뺌
        if len(out[i]) > 1:
            order = sorted(range(len(out[i])), key=lambda k: (-out[i][k][2], -area((np.asarray(out[i][k][0]), np.asarray(out[i][k][1])))))
            keep = []
            for k in order:
                ek = (np.asarray(out[i][k][0]), np.asarray(out[i][k][1]))
                if not any(contains((np.asarray(q[0]), np.asarray(q[1])), boundary(ek, 24)) for q in keep):
                    keep.append(out[i][k])
            out[i] = keep
    return out, sorted(leak), segtab


class Painter:
    """3단계(그리기): 계획된 타원대로만 블러를 그림. 프레임마다 다시 판단하지 않음"""
    def __init__(self):
        self.fill = None
        self.prev_small = None
        self.akey = None
        self.alpha = None
        self.a3 = None
        self.fill_s = None
        self.alpha_c = self.alpha_s = None

    def cut_check(self, small):
        """화면이 바뀌면 블러 속 색을 이어 쓰지 않음 (small: 64x36 흑백)"""
        if self.prev_small is not None and np.abs(small - self.prev_small).mean() > CUT_DIFF:
            self.fill = "cut" if getattr(self, "subs", None) is not None else None
            self.fill_s = None
        self.prev_small = small

    def core(self, W, H, ells, polys, get_sub, align=None):
        """블러 영역(X0,Y0,X1,Y1), 그 영역의 원본(sub, BGR), 블러 결과(soft), 투명도(alpha) 계산"""
        if not ells:
            self.fill = None
            return None
        boxes, ws = [], []
        for e in ells:
            c, A = np.asarray(e[0], float), np.asarray(e[1], float)
            ws.append(float(e[2]) if len(e) > 2 else 1.0)
            r, _ = _axes(A)
            fe = float(e[5]) if len(e) > 5 and e[5] else 1.0       # 검수 화면에서 고친 가장자리 번짐 (배수)
            F = float(np.clip(min(r) * FEATHER_W * fe, 20, FEATHER_MAX * fe))
            boxes.append((c, A, r, F, max(r) + F))
        X0 = int(max(0, min(c[0] - R for c, _, _, _, R in boxes)))
        Y0 = int(max(0, min(c[1] - R for c, _, _, _, R in boxes)))
        X1 = int(min(W, max(c[0] + R for c, _, _, _, R in boxes) + 1))
        Y1 = int(min(H, max(c[1] + R for c, _, _, _, R in boxes) + 1))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            return None
        if align is not None:
            X0, Y0, X1, Y1 = align(X0, Y0, X1, Y1)
        key = (X0, Y0, X1, Y1, tuple(np.round(np.concatenate([np.ravel(c) for c, A, _, _, _ in boxes] +
                                                             [np.ravel(A) * 1e6 for c, A, _, _, _ in boxes] + [np.array(ws), np.array([F for _, _, _, F, _ in boxes])]), 3)))
        if self.akey == key:                         # 블러가 가만히 있으면 투명도 계산을 다시 하지 않음 (빠르게)
            alpha = self.alpha
        else:
            yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
            alpha = np.zeros((Y1 - Y0, X1 - X0), np.float32)
            for (c, A, r, F, R), wt in zip(boxes, ws):
                dx, dy = xx - c[0], yy - c[1]
                q = A[0, 0] * dx * dx + 2 * A[0, 1] * dx * dy + A[1, 1] * dy * dy
                rr = np.sqrt(np.maximum(q, 0))
                t = np.clip(1 - (rr - 1) * np.sqrt(r[0] * r[1]) / F, 0, 1)
                alpha = np.maximum(alpha, wt * np.where(rr <= 1, 1.0, t * t * (3 - 2 * t)))
            alpha = alpha.astype(np.float32)
            self.akey, self.alpha = key, alpha
            self.a3 = cv2.merge([alpha, alpha, alpha])
        need = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        for p in polys:
            if p is not None and len(p) >= 3:
                cv2.fillPoly(need, [(np.asarray(p, np.float32) - (X0, Y0)).astype(np.int32)], 1)
        sub = get_sub(X0, Y0, X1, Y1)
        minr = min(min(r) for _, _, r, _, _ in boxes)
        kmax = max([float(e[4]) if len(e) > 4 and e[4] else 1.0 for e in ells] or [1.0])   # 검수 화면에서 고친 흐림 세기 (배수)
        soft = natural_fill(sub, need, max(15.0, 2 * minr * BLUR_FRAC) * kmax, alpha).astype(np.float32)
        if self.fill is not None:                    # 블러 속은 이전 프레임과 섞어 천천히만 바뀜 (타원이 옮겨 가는 중에도)
            (pX0, pY0, pX1, pY1), ps = self.fill
            ox0, oy0, ox1, oy1 = max(X0, pX0), max(Y0, pY0), min(X1, pX1), min(Y1, pY1)
            if ox1 > ox0 and oy1 > oy0:
                cur = soft[oy0 - Y0:oy1 - Y0, ox0 - X0:ox1 - X0]
                soft[oy0 - Y0:oy1 - Y0, ox0 - X0:ox1 - X0] = FILL_KEEP * ps[oy0 - pY0:oy1 - pY0, ox0 - pX0:ox1 - pX0] + (1 - FILL_KEEP) * cur
        self.fill = ((X0, Y0, X1, Y1), soft)
        return X0, Y0, X1, Y1, sub, soft, alpha

    def paint(self, frame, ells, polys):
        """블러마다 따로 그려서 겹침 (여러 블러를 한 덩어리로 계산하면 흐림 세기가 가장 작은 블러에 맞춰져 큰 블러가 묽어짐)"""
        if len(ells) <= 1:
            return self._paint1(frame, ells, polys)
        out, last = frame, None
        for e in ells:
            out, r = Painter()._paint1(out, [e], polys)
            last = r or last
        return out, last

    def _paint1(self, frame, ells, polys):
        H, W = frame.shape[:2]
        self.cut_check(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32))
        r = self.core(W, H, ells, polys, lambda X0, Y0, X1, Y1: frame[Y0:Y1, X0:X1])
        if r is None:
            return frame, None
        X0, Y0, X1, Y1, sub, soft, alpha = r
        out = frame.copy()
        sf = sub.astype(np.float32)
        out[Y0:Y1, X0:X1] = np.clip(cv2.add(sf, cv2.multiply(cv2.subtract(soft, sf), self.a3)), 0, 255).astype(np.uint8)
        return out, (X0, Y0, alpha)

    def paint_yuv_old(self, yf, ells, polys):
        """(예전 방식: 4K면 블러 영역 전체를 큰 크기로 계산 — 비교 시험용)"""
        self.cut_check(yf.small_gray())
        r = self.core(yf.W, yf.H, ells, polys, yf.bgr, align=yf.align)
        if r is None:
            return False
        X0, Y0, X1, Y1, sub, soft, alpha = r
        yf.blend(X0, Y0, soft, alpha)
        return True

    def paint_yuv(self, yf, ells, polys):
        """블러마다 따로 그려서 겹침 — 여러 블러를 한 덩어리로 계산하면 흐림 세기가 가장 작은 블러에 맞춰져서,
        다른 블러가 같이 나오는 동안 큰 블러가 묽어졌음. 블러마다 이전 프레임 기억도 따로 둠"""
        self.cut_check(yf.small_gray())
        subs = getattr(self, "subs", None)
        if subs is None or self.fill == "cut":
            subs = {}
        self.fill = None
        keep, seen = {}, {}
        changed = False
        for e in ells:
            sid = e[3] if len(e) > 3 else None
            k = (sid, seen.get(sid, 0)); seen[sid] = k[1] + 1
            sp = subs.get(k) or Painter()
            changed |= sp._paint_yuv1(yf, [e], polys)
            keep[k] = sp
        self.subs = keep
        return changed

    def _paint_yuv1(self, yf, ells, polys):
        """원본 형식(YUV) 프레임에 바로 블러를 섞음. 블러 밖 픽셀은 원본 그대로.
        블러 재료는 원래도 작은 크기(짧은 변 120)에서 계산하므로, 큰 크기로 키우는 건 맨 마지막 섞을 때 한 번만 함
        (예전엔 4K 블러 영역 전체를 BGR로 바꾸고·키우고·이전 프레임과 섞고·다시 YUV로 바꾸느라 한 장에 0.1초씩 걸렸음)"""
        if not ells:
            self.fill = self.fill_s = None
            return False
        W, H = yf.W, yf.H
        boxes, ws = [], []
        for e in ells:
            c, A = np.asarray(e[0], float), np.asarray(e[1], float)
            ws.append(float(e[2]) if len(e) > 2 else 1.0)
            r, _ = _axes(A)
            fe = float(e[5]) if len(e) > 5 and e[5] else 1.0
            F = float(np.clip(min(r) * FEATHER_W2 * fe, 20, 2000)) if BLUR_V2 else float(np.clip(min(r) * FEATHER_W * fe, 20, FEATHER_MAX * fe))
            boxes.append((c, A, r, F, max(r) + F))
        X0 = int(max(0, min(c[0] - R for c, _, _, _, R in boxes)))
        Y0 = int(max(0, min(c[1] - R for c, _, _, _, R in boxes)))
        X1 = int(min(W, max(c[0] + R for c, _, _, _, R in boxes) + 1))
        Y1 = int(min(H, max(c[1] + R for c, _, _, _, R in boxes) + 1))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            self.fill_s = None
            return False
        X0, Y0, X1, Y1 = yf.align(X0, Y0, X1, Y1)
        w, h = X1 - X0, Y1 - Y0
        sx, sy = yf.sx, yf.sy
        wc, hc = w // sx, h // sy
        f, sw, sh = _small_size(w, h)
        key = (X0, Y0, X1, Y1, tuple(np.round(np.concatenate([np.ravel(c) for c, A, _, _, _ in boxes] +
                                                             [np.ravel(A) * 1e6 for c, A, _, _, _ in boxes] + [np.array(ws), np.array([F for _, _, _, F, _ in boxes])]), 3)))
        if self.akey != key:
            # 투명도: 가장자리 번짐(최소 20픽셀)이 8칸 이상 되는 범위에서 줄인 크기로 계산 → 부드럽게 키움
            q = float(np.clip(min(F for _, _, _, F, _ in boxes) / 8.0, 1.0, 4.0))
            aw, ah = max(2, int(np.ceil(w / q))), max(2, int(np.ceil(h / q)))
            xs = (X0 + (np.arange(aw, dtype=np.float32) + 0.5) * (w / aw) - 0.5).astype(np.float32)
            ys = (Y0 + (np.arange(ah, dtype=np.float32) + 0.5) * (h / ah) - 0.5).astype(np.float32)
            xx, yy = np.meshgrid(xs, ys)
            a_ = np.zeros((ah, aw), np.float32)
            for (c, A, r, F, R), wt in zip(boxes, ws):
                dx, dy = xx - np.float32(c[0]), yy - np.float32(c[1])
                qq = A[0, 0] * dx * dx + 2 * A[0, 1] * dx * dy + A[1, 1] * dy * dy
                rr = np.sqrt(np.maximum(qq, 0))
                t = np.clip(1 - (rr - 1) * np.sqrt(r[0] * r[1]) / F, 0, 1)
                a_ = np.maximum(a_, wt * np.where(rr <= 1, 1.0, t * t * (3 - 2 * t)))
            a_ = a_.astype(np.float32)
            self.akey = key
            self.alpha = cv2.resize(a_, (w, h), interpolation=cv2.INTER_LINEAR) if (aw, ah) != (w, h) else a_
            self.alpha_c = self.alpha if (sx, sy) == (1, 1) else cv2.resize(a_, (wc, hc), interpolation=cv2.INTER_LINEAR)
            self.alpha_s = cv2.resize(a_, (sw, sh), interpolation=cv2.INTER_AREA)
        alpha, alpha_c = self.alpha, self.alpha_c
        # 원본 블러 영역을 바로 작은 크기로 (Y·U·V 각각 줄인 뒤 BGR로)
        Ys = cv2.resize(yf.Y[Y0:Y1, X0:X1].astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        Us = cv2.resize(yf.U[Y0 // sy:Y0 // sy + hc, X0 // sx:X0 // sx + wc].astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        Vs = cv2.resize(yf.V[Y0 // sy:Y0 // sy + hc, X0 // sx:X0 // sx + wc].astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
        fwd, back = yf._mats()
        small = np.clip(cv2.transform(cv2.merge([Ys, Us, Vs]), back) + 0.5, 0, 255).astype(np.uint8)
        nd = np.zeros((sh, sw), np.uint8)
        for p in polys:
            if p is not None and len(p) >= 3:
                pp = (np.asarray(p, np.float32) - (X0, Y0)) * (sw / w, sh / h)
                cv2.fillPoly(nd, [np.round(pp).astype(np.int32)], 255)
        cv_ = self.alpha_s > 0.05
        minr = min(min(r) for _, _, r, _, _ in boxes)
        kmax = max([float(e[4]) if len(e) > 4 and e[4] else 1.0 for e in ells] or [1.0])
        sig = max(15.0, 2 * minr * BLUR_FRAC) * kmax
        soft = (fill_small_v2(small, nd, self.alpha_s, sig, f) if BLUR_V2 else fill_small(small, nd, cv_, sig, f)).astype(np.float32)
        fs = getattr(self, "fill_s", None)
        if fs is not None:                           # 블러 속은 이전 프레임과 섞어 천천히만 바뀜 (작은 크기에서, 위치를 맞춰서)
            (pX0, pY0, pw, ph), ps = fs
            psh, psw = ps.shape[:2]
            ax, ay = (w / sw) * (psw / pw), (h / sh) * (psh / ph)
            bx = (X0 - pX0 + 0.5 * w / sw) * (psw / pw) - 0.5
            by = (Y0 - pY0 + 0.5 * h / sh) * (psh / ph) - 0.5
            M = np.float32([[ax, 0, bx], [0, ay, by]])
            prev = cv2.warpAffine(ps, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ok = cv2.warpAffine(np.ones((psh, psw), np.float32), M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
            k3 = (ok * FILL_KEEP)[..., None]
            soft = soft + (prev - soft) * k3
        self.fill_s = ((X0, Y0, w, h), soft)
        yuv = cv2.transform(soft, fwd)
        ys_, us_, vs_ = cv2.split(yuv)
        def mix(P, src, a):
            o = P.astype(np.float32)
            r_ = cv2.add(o, cv2.multiply(cv2.subtract(src, o), a))
            np.clip(r_, 0, yf.maxv, out=r_)
            r_ += 0.5
            P[...] = r_.astype(P.dtype)
        mix(yf.Y[Y0:Y1, X0:X1], cv2.resize(ys_, (w, h), interpolation=cv2.INTER_CUBIC), alpha)
        mix(yf.U[Y0 // sy:Y0 // sy + hc, X0 // sx:X0 // sx + wc], cv2.resize(us_, (wc, hc), interpolation=cv2.INTER_CUBIC), alpha_c)
        mix(yf.V[Y0 // sy:Y0 // sy + hc, X0 // sx:X0 // sx + wc], cv2.resize(vs_, (wc, hc), interpolation=cv2.INTER_CUBIC), alpha_c)
        return True


# ======================= 7. 검수 화면에서 고친 내용 적용 =======================
def _ell_from(cx, cy, rx, ry, ang):
    return _ell((cx, cy), max(rx, 5.0), max(ry, 5.0), ang)


def edit_offset(ed, f):
    """그 프레임에서의 이동·크기·회전(라디안)·길이 배율(sx)·폭 배율(sy). 키프레임이 있으면 키프레임 사이를 직선으로 부드럽게 잇고, 처음·끝 바깥은 가까운 키프레임 그대로"""
    ed = ed or {}
    keys = sorted(ed.get("keys") or [], key=lambda k: k["f"])
    g = lambda k: (float(k.get("dx", 0.0)), float(k.get("dy", 0.0)), float(k.get("scale", 1.0)), float(k.get("rot", 0.0)),
                   float(k.get("sx", 1.0)), float(k.get("sy", 1.0)))
    if not keys:
        return g(ed)
    if f <= keys[0]["f"]:
        return g(keys[0])
    if f >= keys[-1]["f"]:
        return g(keys[-1])
    for k0, k1 in zip(keys[:-1], keys[1:]):
        if k0["f"] <= f <= k1["f"]:
            t = (f - k0["f"]) / max(1, k1["f"] - k0["f"])
            a, b = g(k0), g(k1)
            return tuple(a[q] + (b[q] - a[q]) * t for q in range(6))
    return g(keys[-1])


def _axesA(A):
    """타원 행렬 → (긴반지름, 짧은반지름, 긴축 각도). 검수 화면(axesOf)과 같은 식"""
    a, b, c = float(A[0][0]), float(A[0][1]), float(A[1][1])
    m = (a + c) / 2
    d = np.sqrt(((a - c) / 2) ** 2 + b * b)
    lmin, lmax = m - d, m + d
    ang = (0.0 if a <= c else np.pi / 2) if abs(b) < 1e-12 else np.arctan2(lmin - a, b)
    return 1 / np.sqrt(lmin), 1 / np.sqrt(lmax), ang


def _makeA(rx, ry, ang):
    c, s_ = np.cos(ang), np.sin(ang)
    p, q = 1 / (rx * rx), 1 / (ry * ry)
    return np.array([[p * c * c + q * s_ * s_, (p - q) * c * s_], [(p - q) * c * s_, p * s_ * s_ + q * c * c]])


def _stretchA(A, sx, sy):
    """타원의 긴축 방향으로 sx배, 짧은축 방향으로 sy배"""
    if sx == 1 and sy == 1:
        return np.asarray(A, float)
    rx, ry, ang = _axesA(A)
    return _makeA(rx * sx, ry * sy, ang)


def _rotA(A, th):
    if not th:
        return A
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, -s_], [s_, c]])
    return R @ A @ R.T


def apply_edits(ells, segtab, edits, n):
    """검수 화면에서 고친 내용(구간별 이동·크기·삭제·기간·흐림 세기·키프레임, 새로 추가한 블러)을 블러 계획에 적용.
    edits = {"segs": {구간번호: {"dx","dy","scale" 또는 "keys":[{"f","dx","dy","scale","rot","sx","sy"}], "del","start","end","blur","fea"}},
             "added": [{"start","end","cx","cy","rx","ry","ang","blur","fea", "keys"}]}
    blur = 흐림 세기 배수, fea = 가장자리 번짐 폭 배수"""
    se = {int(k): v for k, v in (edits or {}).get("segs", {}).items()}
    out = [[] for _ in range(n)]
    base = {}
    first = {}                       # 구간마다 처음·마지막 프레임의 타원 (기간을 늘릴 때 씀)
    for i in range(n):
        for e in ells[i]:
            sid = e[3] if len(e) > 3 else None
            if sid is None or sid not in segtab or segtab[sid] is None:
                continue
            st = segtab[sid]
            if st["start"] <= i <= st["end"] and e[2] >= 1:
                base.setdefault(sid, {})[i] = e
                first.setdefault(sid, [e, e])
                if i == st["start"]:
                    first[sid][0] = e
                if i == st["end"]:
                    first[sid][1] = e

    def tf(e, ed, f):
        dx, dy, sc, rot, sx, sy = edit_offset(ed, f)
        c = np.asarray(e[0], float) + (dx, dy)
        A = _rotA(_stretchA(e[1], sx, sy) / (sc ** 2), rot)
        return [c.tolist(), A.tolist()]

    for i in range(n):
        for e in ells[i]:
            sid = e[3] if len(e) > 3 else None
            ed = se.get(sid) if sid is not None else None
            if ed is None:
                out[i].append(tuple(e))
                continue
            if ed.get("del"):
                continue
            st = segtab[sid]
            a, b = int(ed.get("start", st["start"])), int(ed.get("end", st["end"]))
            changed_range = (a, b) != (st["start"], st["end"])
            inside = st["start"] <= i <= st["end"]
            if not inside and changed_range:
                continue                                  # 기간을 바꿨으면 원래 페이드는 버리고 새로 붙임
            if inside and not (a <= i <= b):
                continue
            c, A = tf(e, ed, i)
            out[i].append((c, A, e[2], sid, float(ed.get("blur", 1.0)), float(ed.get("fea", 1.0))))
    for sid, ed in se.items():
        if ed.get("del") or sid not in segtab or segtab[sid] is None or sid not in first:
            continue
        st = segtab[sid]
        a, b = int(ed.get("start", st["start"])), int(ed.get("end", st["end"]))
        if (a, b) == (st["start"], st["end"]):
            continue
        bs = base.get(sid, {})
        es_ = bs.get(a, first[sid][0]) if a >= st["start"] else first[sid][0]
        ee_ = bs.get(b, first[sid][1]) if b <= st["end"] else first[sid][1]
        kb = float(ed.get("blur", 1.0))
        kf = float(ed.get("fea", 1.0))
        for i in range(max(0, a), min(n, b + 1)):
            if i < st["start"]:
                c, A = tf(first[sid][0], ed, i)
                out[i].append((c, A, 1.0, sid, kb, kf))
            elif i > st["end"]:
                c, A = tf(first[sid][1], ed, i)
                out[i].append((c, A, 1.0, sid, kb, kf))
        for m, w in zip((a - 3, a - 2, a - 1), FADE):
            if 0 <= m < n:
                c, A = tf(es_, ed, m)
                out[m].append((c, A, w, sid, kb, kf))
        for m, w in zip((b + 1, b + 2, b + 3), FADE[::-1]):
            if 0 <= m < n:
                c, A = tf(ee_, ed, m)
                out[m].append((c, A, w, sid, kb, kf))
    for k, ad in enumerate((edits or {}).get("added", [])):
        if ad.get("del"):
            continue
        a, b = int(ad["start"]), int(ad["end"])
        kb = float(ad.get("blur", 1.0))
        kf = float(ad.get("fea", 1.0))

        def el_at(f):
            dx, dy, sc, rot, sx, sy = edit_offset(ad if ad.get("keys") else {}, f)
            c, A = _ell_from(ad["cx"] + dx, ad["cy"] + dy, ad["rx"] * sc * sx, ad["ry"] * sc * sy, ad.get("ang", 0.0) + rot)
            return c.tolist(), A.tolist()
        for i in range(max(0, a), min(n, b + 1)):
            c, A = el_at(i)
            out[i].append((c, A, 1.0, f"a{k}", kb, kf))
        for m, w in list(zip((a - 3, a - 2, a - 1), FADE)) + list(zip((b + 1, b + 2, b + 3), FADE[::-1])):
            if 0 <= m < n:
                c, A = el_at(m)
                out[m].append((c, A, w, f"a{k}", kb, kf))
    return out
