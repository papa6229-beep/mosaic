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
FEATHER_W = 0.4     # 가장자리 투명해지는 폭 (타원 짧은 반지름 대비)
FEATHER_MAX = 90    # 번짐 폭 최대 (픽셀)
BLUR_FRAC = 0.30    # 블러 세기 (타원 짧은 지름 대비)
HAIR = True         # 음모 찾아서 같이 덮기
HAIR_REACH = 0.8    # 음모 인정 거리 (원본 모자이크 짧은 변 대비)
REACH_MIN, REACH_MAX = 40, 250   # 음모 인정 거리 최소/최대 (픽셀)
HAIR_TEX = 2.2      # 음모: 가닥 결이 맨살의 이 배수보다 큼
HAIR_MEM = 8        # 음모를 놓쳐도 이 프레임 수만큼 기억
EXPOSE = True       # 노출 탐지(항문·성기·얼굴) 사용
EXPOSE_EVERY = 2    # 노출 탐지는 이 프레임마다 한 번 (사이 프레임은 직전 결과 사용)
EXPOSE_SCORE = 0.35
HOLD = 45           # Lada가 놓쳐도 이 프레임 수(1.5초)만큼 마지막 자리 유지
PRE = 15            # 처음 나타나기 이 프레임 수(0.5초) 전부터 미리 덮기
CONFIRM = 3         # 한 자리에서 이 횟수 이상 잡혀야 진짜 모자이크로 인정 (엉뚱한 한두 번 무시)
SHRINK_RATIO = 1.6  # 타원이 필요한 크기의 이 배를 넘는 상태가
SHRINK_AFTER = 15   # 이 프레임 수 넘게 이어지면 천천히 줄임
FILL_RES = 120      # 블러 재료 계산 크기
FILL_KEEP = 0.65    # 블러 속 이전 프레임 비율 (클수록 속 색이 천천히 바뀜)
CUT_DIFF = 28       # 화면 전환 판단 (작은 화면 밝기 차이)


# ======================= 1. 원본 모자이크 추적 =======================
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_tracks(all_boxes, n):
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
        k = 0
        for i in range(max(0, first - PRE), min(n, last + HOLD + 1)):
            while k + 1 < len(fr) and fr[k + 1] <= i:
                k += 1
            lh = fr[k] if fr[k] <= i else first          # 마지막으로 잡힌 프레임 (처음 전이면 첫 프레임)
            if i > first and i - lh > HOLD:
                continue
            b, p = t["hits"][lh]
            per[i].append((t["id"], b, p, i in t["hits"]))   # 놓친 동안엔 마지막 자리 그대로 (미끄러지지 않음)
    for i in range(n):
        live = [e[1] for e in per[i] if e[3]]
        if live:
            per[i] = [e for e in per[i] if e[3] or not any(_iou(e[1], b) > 0.1 for b in live)]
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
    skinL = np.percentile(L[valid], 80)
    skin = valid & (L > skinL * 0.85)
    if skin.sum() < 50:
        return None, (0, 0)
    skinC = np.median(C[skin])
    hp = L - cv2.GaussianBlur(L, (0, 0), 2)
    sd = np.sqrt(np.maximum(cv2.blur(hp * hp, (9, 9)), 0))
    base = np.median(sd[skin]) + 1e-3
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
        tx = np.median(sd[comp]) / base                 # 가닥 결 (맨살 대비)
        cr = np.median(C[comp]) / skinC                 # 색 진하기 (살색 대비)
        lr = np.median(L[comp]) / skinL                 # 밝기 (살 대비)
        dk = float((L[comp] < skinL * 0.5).mean())       # 확실히 검은 가닥 비율
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
def natural_fill(sub, need, sigma, cov):
    h, w = need.shape
    f = max(1.0, min(h, w) / FILL_RES)
    sw, sh = max(8, int(w / f)), max(8, int(h / f))
    small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
    nd = cv2.resize(need * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
    cv_ = cv2.resize((cov > 0.05).astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
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
    small = cv2.GaussianBlur(small, (k, k), s2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


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

    def process(self, frame, i, entries):
        """반환: (결과 프레임, 덮이지 않은 음모 픽셀 수)"""
        H, W = frame.shape[:2]
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = self.prev_small is not None and np.abs(small - self.prev_small).mean() > CUT_DIFF
        self.prev_small = small
        if cut:                                   # 화면이 바뀌면 이전 자리 기억은 버림
            self.E.clear(); self.over.clear(); self.hair.clear(); self.fill = None; self.must_mem = []
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
        for tid, box, poly, det in entries:
            if det:
                pb = self.pbox.get(tid)
                if pb is not None and _iou(pb, box) < 0.3:
                    self.jump.add(tid)
                self.pbox[tid] = box
        alive = set()
        need = np.zeros((H, W), np.uint8)
        hair_px = []
        targets = []
        for tid, box, poly, det in entries:
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
            E = self.E.get(tid)
            jump = tid in self.jump                              # 원본 모자이크가 순간이동(장면 전환·자세 급변)
            if E is None or jump or not contains(E, core[0][None]):
                E = T                                            # 처음이거나 완전히 다른 곳으로 이동
                self.over[tid] = 0
            elif not contains(E, need_b):
                E = mvee(np.vstack([boundary(E), Tb]))           # 삐져나오려 함 → 지금 자리와 새 자리를 같이 감싸게 넓힘
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
                self.E.pop(tid); self.over.pop(tid, None); self.hair.pop(tid, None)
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
