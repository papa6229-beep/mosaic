"""
원본 모자이크 위치(프레임별 상자) -> 시간순으로 다듬기 -> 자연스러운 블러로 덮기

사용법
  1) 위치 저장:  boxes.json  = {"0": [x1,y1,x2,y2], "1": null, "2": [..], ...}
     (Lada 탐지 결과에서 프레임마다 모자이크 영역의 바깥 상자만 뽑아 저장. 못 찾은 프레임은 null)
  2) python cover.py in.mp4 boxes.json out.mp4

핵심: 영상 전체의 상자를 먼저 모은 뒤, 앞뒤 프레임을 같이 보고 다듬는다.
"""
import sys, json
import numpy as np
import cv2
from scipy.ndimage import maximum_filter1d, gaussian_filter1d

# ---------- 조절 값 (여기만 만지면 됨) ----------
GAP_FILL   = 45     # 이 프레임 수 이하로 끊긴 구간은 앞뒤를 이어서 채움
HOLD_PAD   = 6      # 나타나기 전/사라진 후 몇 프레임 더 덮을지 (깜빡임 방지)
SIZE_WIN   = 9      # 크기는 앞뒤 이 프레임 중 '가장 큰 값'으로 (삐져나옴 방지)
SMOOTH_POS = 3.0    # 위치 부드럽게 (클수록 천천히 움직임)
SMOOTH_SIZE= 4.0    # 크기 부드럽게 (클수록 크기가 덜 흔들림)
SCALE      = 1.25   # 원본 모자이크보다 몇 배 크게 덮을지
PAD_PX     = 12     # 추가로 늘릴 픽셀
DARK_GAP   = 35     # 주변 피부보다 이만큼 이상 어두우면 '털 비침'으로 셈
FEATHER    = 0.35   # (이전 방식용)   # 가장자리 옅어지는 폭 (상자 짧은 변 대비 비율)
BLUR_FRAC  = 0.30   # 안쪽 블러 세기 (상자 짧은 변 대비 비율)
HAIR       = True   # 원본 모자이크에 붙어 있는 털까지 찾아서 같이 덮기
HAIR_PAD   = 15     # 찾은 털 덩어리 주변으로 더 덮을 픽셀
HAIR_REACH = 0.3    # 원본 모자이크에서 얼마나 떨어진 털까지 잡을지 (상자 크기 대비)
# -----------------------------------------------


def load_boxes(path, n):
    d = json.load(open(path))
    arr = np.full((n, 4), np.nan)
    for k, v in d.items():
        k = int(k)
        if v is not None and 0 <= k < n:
            arr[k] = v
    return arr


def to_cxcywh(b):
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    return cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]


def smooth_track(boxes):
    """상자 목록(n,4, 없으면 nan) -> 다듬어진 (cx,cy,w,h, on) 반환"""
    n = len(boxes)
    cx, cy, w, h = to_cxcywh(boxes)
    have = ~np.isnan(cx)
    idx = np.arange(n)
    if have.sum() == 0:
        return None

    # 1) 짧게 끊긴 구간은 앞뒤를 직선으로 이음, 긴 구간은 '없음' 유지
    on = have.copy()
    h_idx = idx[have]
    for a, b in zip(h_idx[:-1], h_idx[1:]):
        if 1 < b - a <= GAP_FILL + 1:
            on[a:b + 1] = True
    full = []
    for arr in (cx, cy, w, h):
        full.append(np.interp(idx, h_idx, arr[have]))
    cx, cy, w, h = full

    # 2) 켜짐 구간을 앞뒤로 조금 더 늘림 (깜빡임 방지)
    on = maximum_filter1d(on.astype(np.uint8), size=2 * HOLD_PAD + 1).astype(bool)

    # 3) 크기: 앞뒤 중 최대값 -> 부드럽게  (작아져서 삐져나오는 것 방지)
    w = gaussian_filter1d(maximum_filter1d(w, size=SIZE_WIN), SMOOTH_SIZE)
    h = gaussian_filter1d(maximum_filter1d(h, size=SIZE_WIN), SMOOTH_SIZE)
    # 4) 위치: 부드럽게
    cx = gaussian_filter1d(cx, SMOOTH_POS)
    cy = gaussian_filter1d(cy, SMOOTH_POS)
    return cx, cy, w, h, on


def skin_color(frame, mask_inner, mask_outer):
    """덮을 영역 바로 바깥 둘레의 중간 색(= 주변 피부색)"""
    ring = (mask_outer > 0) & (mask_inner == 0)
    px = frame[ring]
    if len(px) < 20:
        return frame.reshape(-1, 3).mean(0)
    return np.median(px, axis=0)


def big_blur(img, sigma):
    """큰 블러를 작게 줄여서 걸고 다시 키움 (결과는 거의 같고 훨씬 빠름)"""
    f = max(1.0, sigma / 4.0)
    if f <= 1.0:
        n = int(sigma * 3) * 2 + 1
        return cv2.GaussianBlur(img, (n, n), sigma)
    h, w = img.shape[:2]
    sw, sh = max(1, int(w / f)), max(1, int(h / f))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    s2 = sigma / f
    n = int(s2 * 3) * 2 + 1
    small = cv2.GaussianBlur(small, (n, n), s2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def hair_mask(frame, box, search=1.0, near=40):
    """원본 모자이크 주변에서 털 색(살색보다 어둡고 + 붉은기·노란기가 빠진 거무스름한 색)을 찾아,
    모자이크에 붙어 있는 덩어리만 돌려줌. 그림자는 어두워도 살색이 그대로라 안 잡힘."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - bw * search)), max(0, int(y1 - bh * search))
    X2, Y2 = min(W, int(x2 + bw * search)), min(H, int(y2 + bh * search))
    full = np.zeros((H, W), np.uint8)
    if X2 - X1 < 16 or Y2 - Y1 < 16:
        return full
    lab = cv2.cvtColor(frame[Y1:Y2, X1:X2], cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)          # 색 진하기 (살색일수록 큼)
    valid = cv2.erode((L > 25).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0   # 검은 띠 제외
    if valid.sum() < 100:
        return full
    skinL = np.percentile(L[valid], 80)                          # 이 근처 살 밝기
    skinC = np.median(C[valid & (L > skinL * 0.85)])             # 이 근처 살색 진하기
    hp = L - cv2.GaussianBlur(L, (0, 0), 2)
    sd = np.sqrt(np.maximum(cv2.blur(hp * hp, (9, 9)), 0))
    base = np.median(sd[valid & (L > skinL * 0.85)]) + 1e-3     # 맨살의 잔결
    m = (valid & (L < skinL * 0.8) & (C < skinC * 0.9)).astype(np.uint8)
    m = (cv2.blur(m.astype(np.float32), (15, 15)) > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    # 원본 모자이크 안쪽은 빼고 봄 (모자이크 속 검은 칸을 타고 남자 다리털·그림자가 이어지는 것 방지)
    inner = np.zeros_like(m)
    ix1, iy1, ix2, iy2 = int(x1 - X1 + bw * 0.1), int(y1 - Y1 + bh * 0.1), int(x2 - X1 - bw * 0.1), int(y2 - Y1 - bh * 0.1)
    if ix2 > ix1 and iy2 > iy1:
        cv2.rectangle(inner, (ix1, iy1), (ix2, iy2), 1, -1)
    m = m & (inner == 0)
    touch = np.zeros_like(m)
    cv2.rectangle(touch, (int(x1 - X1 - near), int(y1 - Y1 - near)), (int(x2 - X1 + near), int(y2 - Y1 + near)), 1, -1)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    hit = np.unique(lab_[(touch > 0) & (m > 0)])
    keep_ids = []
    for k in hit:
        if k == 0 or st[k, cv2.CC_STAT_AREA] <= max(1500, bw * bh * 0.01):   # 작은 부스러기(다리털 몇 가닥 등)는 무시
            continue
        comp = lab_ == k
        # 음모: 꼬불꼬불한 가닥이 빽빽해 잔결이 맨살보다 훨씬 많음 / 남자 다리털·그림자: 잔결이 적음
        # (아주 빽빽해서 색이 거의 빠진 덩어리는 잔결과 상관없이 음모로 인정)
        tx = np.median(sd[comp]) / base            # 가닥 결 (맨살 대비)
        cr = np.median(C[comp]) / skinC            # 색 진하기 (살색 대비)
        # 음모 = 가닥 결이 많고 + 살색이 빠진 덩어리. 결만 있고 살색이 남으면 다리털, 색만 빠지고 결이 없으면 그림자
        if (tx > HAIR_TEX and cr < 0.8) or (tx > 3.0 and cr < 0.86):
            keep_ids.append(k)
    keep = np.isin(lab_, keep_ids)
    full[Y1:Y2, X1:X2] = keep.astype(np.uint8)
    return full


def big_blur_mask(m, sigma):
    return big_blur(m.astype(np.float32), sigma)


BAND       = 0.6    # 털은 원본 모자이크에서 이 거리(상자 짧은 변 대비) 안쪽만 인정
FEATHER_W  = 0.4    # 가장자리가 투명해지는 폭 (덮는 크기 대비) — 넓을수록 자연스럽게 사라짐
FEATHER_MAX = 90    # 번짐 폭 최대 (픽셀) — 가까운 장면에서 블러가 과하게 커지는 것 방지
BAND_MAX   = 9999    # 털 인정 거리 최대 (픽셀)
NEAR_MAX   = 40     # 털 덩어리는 원본 모자이크에서 이 거리(픽셀) 안에 닿아 있어야 인정
HAIR_TEX   = 2.2    # 털 덩어리의 잔결이 맨살의 이 배수보다 크면 음모로 인정
HAIR_GRAY  = 0.7     # 털 덩어리의 색 진하기가 살색의 이 비율보다 낮아야 음모로 인정
HAIR_MEM   = 0.95    # 털을 놓쳐도 몇 프레임 기억 (깜빡임 방지)
GHOST      = 0.8    # 자리를 옮길 때 예전 자리가 사라지는 속도 (작을수록 빨리 사라짐)
BRIDGE     = 100    # 이 거리(픽셀) 안의 털 덩어리는 모자이크와 이어 붙임
FILL_RES   = 120    # 블러 재료를 만들 때 줄여서 계산하는 크기 (작을수록 빠름, 어차피 강하게 흐려서 화질 차이 거의 없음)
SHRINK_AFTER = 15   # 필요 이상으로 큰 상태가 이 프레임 수 넘게 이어지면 천천히 줄임
SHRINK_RATIO = 1.6  # 지금 블러가 필요한 크기의 몇 배를 넘으면 '너무 큼'으로 봄


def natural_fill(sub, need, sigma, cover):
    """블러 칠할 재료: 덮는 범위 안에서 살색이 아닌 것(털·검은 부분·흰 배경 등)은 먼저 주변 피부로 메운 뒤 강하게 흐림
    → 블러가 통째로 살색이 되고, 피부 명암만 자연스럽게 남음"""
    h, w = need.shape
    f = max(1.0, min(h, w) / FILL_RES)
    sw, sh = max(8, int(w / f)), max(8, int(h / f))
    small = cv2.resize(sub, (sw, sh), interpolation=cv2.INTER_AREA)
    nd = cv2.resize(need * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
    cv_ = cv2.resize((cover > 0.05).astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    Cc = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ok = (L > 25) & (Cc > np.percentile(Cc, 60))        # 색이 있는(살색 쪽) 픽셀로 피부 기준을 잡음
    skinL = np.median(L[ok]) if ok.sum() > 20 else 170
    skinC = np.median(Cc[ok]) if ok.sum() > 20 else 25
    notskin = (L < skinL * 0.75) | ((L > skinL * 1.12) & (Cc < skinC * 0.7)) | (Cc < skinC * 0.45)
    msk = ((nd > 0) | (notskin & cv_)).astype(np.uint8) * 255
    msk = cv2.dilate(msk, np.ones((5, 5), np.uint8))
    if msk.any():
        small = cv2.inpaint(small, msk, 3, cv2.INPAINT_TELEA)
    s2 = max(1.0, sigma / f)
    n = int(s2 * 3) * 2 + 1
    small = cv2.GaussianBlur(small, (n, n), s2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def cover_frame(frame, cx, cy, w, h, prev_color, prev_state=None):
    """반환: (결과, None, 덮이지 않은 털 픽셀 수, 상태)"""
    H, W = frame.shape[:2]
    st = prev_state or {"C": None, "hair": None, "over": 0}
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    short = max(8.0, min(w, h))

    # 1) 반드시 덮어야 할 곳 R = 원본 모자이크(타원) + 바로 옆에 붙은 털
    R = np.zeros((H, W), np.uint8)
    cv2.ellipse(R, (int(cx), int(cy)), (int(w * SCALE / 2 + PAD_PX), int(h * SCALE / 2 + PAD_PX)), 0, 0, 360, 1, -1)
    hb = None
    if HAIR:
        hm = hair_mask(frame, [x1, y1, x2, y2], near=min(NEAR_MAX, max(20, short * HAIR_REACH))).astype(np.float32)
        if st["hair"] is not None:
            hm = np.maximum(hm, st["hair"] * HAIR_MEM)
        st["hair"] = hm
        hb = (hm > 0.3).astype(np.uint8)
        band = np.zeros((H, W), np.uint8)                         # 모자이크 주변 일정 거리까지만
        b = int(min(short * BAND, BAND_MAX))
        cv2.rectangle(band, (int(x1) - b, int(y1) - b), (int(x2) + b, int(y2) + b), 1, -1)
        hb &= band
        if hb.any():
            kk = HAIR_PAD * 2 + 1
            ell = R.copy()
            R |= cv2.dilate(hb, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
            # 모자이크 가까이 있는데 살짝 떨어진 털 덩어리는 모자이크와 이어 붙임 (따로 떨어진 블러 없이 한 덩어리로)
            q = 4
            small = cv2.resize(R, (W // q, H // q), interpolation=cv2.INTER_NEAREST)
            kb = max(3, int(BRIDGE / q) | 1)
            small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
            R |= cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
            n_, lab_ = cv2.connectedComponents(R)                       # 원본 모자이크와 이어진 덩어리만 남김
            keep = np.unique(lab_[ell > 0]); keep = keep[keep > 0]
            R = np.isin(lab_, keep).astype(np.uint8)

    # 2) 목표 블러: R 안은 100%, 바깥으로 갈수록 점점 투명 (부드러운 곡선)
    F = min(FEATHER_MAX, max(20.0, short * SCALE * FEATHER_W))
    rx, ry, rw, rh = cv2.boundingRect(R)
    m = int(F + 4)
    X0, Y0, X1, Y1 = max(0, rx - m), max(0, ry - m), min(W, rx + rw + m), min(H, ry + rh + m)
    d = cv2.distanceTransform((R[Y0:Y1, X0:X1] == 0).astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(1.0 - d / F, 0, 1)
    A = np.zeros((H, W), np.float32)
    A[Y0:Y1, X0:X1] = t * t * (3 - 2 * t)

    # 3) 작은 움직임엔 가만히: 덮어야 할 곳이 이미 덮여 있으면 그대로 둠
    C = st["C"]
    if C is None:
        C = A
    else:
        miss = (R.astype(bool) & (C < 0.97)).sum()
        if miss > 0.002 * R.sum():          # 삐져나오려 함 → 바로 새 자리로 (예전 자리는 서서히 사라짐)
            C = np.maximum(A, C * GHOST)
            st["over"] = 0
        else:
            big = (C > 0.5).sum() / max(1, (A > 0.5).sum())
            st["over"] = st["over"] + 1 if big > SHRINK_RATIO else 0
            if st["over"] > SHRINK_AFTER:   # 한동안 너무 컸음 → 천천히 줄임
                C = np.maximum(A, C - 0.08)
    st["C"] = C

    # 4) 블러 칠하기
    cx0, cy0, cw, ch = cv2.boundingRect((C > 0.01).astype(np.uint8))
    if cw == 0:
        return frame, None, 0, st
    pad = int(short * BLUR_FRAC) + 4
    X0, Y0 = max(0, cx0 - pad), max(0, cy0 - pad)
    X1, Y1 = min(W, cx0 + cw + pad), min(H, cy0 + ch + pad)
    sub = frame[Y0:Y1, X0:X1]
    soft = natural_fill(sub, R[Y0:Y1, X0:X1], max(15.0, short * BLUR_FRAC), C[Y0:Y1, X0:X1])
    a = C[Y0:Y1, X0:X1][..., None]
    o = (sub.astype(np.float32) * (1 - a) + soft.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    out = frame.copy()
    out[Y0:Y1, X0:X1] = o
    left = int(((hb > 0) & (C < 0.9)).sum()) if hb is not None else 0
    return out, None, left, st


def main(src, boxes_json, dst):
    cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = load_boxes(boxes_json, n)
    tr = smooth_track(boxes)
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    col = None
    hs = None
    bad = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if tr is not None and tr[4][i]:
            f, col, d, hs = cover_frame(f, tr[0][i], tr[1][i], tr[2][i], tr[3][i], col, hs)
            if d > 0.005:
                bad.append((i, round(d, 4)))
        else:
            col = None
            hs = None
        vw.write(f)
        i += 1
    vw.release()
    print("done", i, "frames")
    print("털 비침 의심 프레임:", len(bad), "개", bad[:20])
    miss = int(np.isnan(boxes[:, 0]).sum())
    print("탐지가 비어 있던 프레임:", miss, "개 / 최종 덮은 프레임:", int(tr[4].sum()) if tr else 0)


if __name__ == "__main__":
    main(*sys.argv[1:4])
