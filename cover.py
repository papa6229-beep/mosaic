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
FEATHER    = 0.35   # 가장자리 옅어지는 폭 (상자 짧은 변 대비 비율)
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
    m = (valid & (L < skinL * 0.8) & (C < skinC * 0.9)).astype(np.uint8)
    m = (cv2.blur(m.astype(np.float32), (15, 15)) > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    touch = np.zeros_like(m)
    cv2.rectangle(touch, (int(x1 - X1 - near), int(y1 - Y1 - near)), (int(x2 - X1 + near), int(y2 - Y1 + near)), 1, -1)
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m)
    hit = np.unique(lab_[(touch > 0) & (m > 0)])
    keep = np.isin(lab_, [k for k in hit if k > 0 and st[k, cv2.CC_STAT_AREA] > 150])
    full[Y1:Y2, X1:X2] = keep.astype(np.uint8)
    return full


def big_blur_mask(m, sigma):
    return big_blur(m.astype(np.float32), sigma)


def cover_frame(frame, cx, cy, w, h, prev_color, prev_state=None):
    """원본 모자이크 + 붙어 있는 털을 바깥 테두리를 따라 감싸고(고무줄처럼), 여유를 두고 블러로 덮음.
    반환: (결과, 피부색, 털비침비율, 상태)"""
    H, W = frame.shape[:2]
    prev_hair, prev_region = prev_state if prev_state else (None, None)
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    # 1) 원본 모자이크 영역 + 털
    u = np.zeros((H, W), np.uint8)
    cv2.rectangle(u, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    hair_state = None
    if HAIR:
        hm = hair_mask(frame, [x1, y1, x2, y2], near=max(20, min(w, h) * HAIR_REACH)).astype(np.float32)
        if prev_hair is not None:                       # 앞 프레임과 섞어서 깜빡임 방지
            hm = np.maximum(hm, prev_hair * 0.7)
        hair_state = hm
        u |= (hm > 0.3).astype(np.uint8)

    # 2) 바깥 테두리를 따라 감싸기 + 여유
    hull = cv2.convexHull(cv2.findNonZero(u))
    hx, hy, hw, hh = cv2.boundingRect(hull)
    short = max(8.0, min(hw, hh))
    margin = short * max(0.0, SCALE - 1.0) + PAD_PX
    k = int(short * FEATHER) | 1
    bk = int(short * BLUR_FRAC) | 1
    m = int(margin + 2 * k + 2 * bk + 8)
    x0, y0 = max(0, hx - m), max(0, hy - m)
    x1c, y1c = min(W, hx + hw + m), min(H, hy + hh + m)
    hmask = np.zeros((y1c - y0, x1c - x0), np.uint8)
    cv2.drawContours(hmask, [hull - np.array([x0, y0])], -1, 1, -1)
    d = cv2.distanceTransform((hmask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    core = (d <= margin).astype(np.float32)

    # 3) 모양 출렁임 줄이기: 앞 프레임 영역과 섞되, 이번 영역은 반드시 포함
    full = np.zeros((H, W), np.float32)
    full[y0:y1c, x0:x1c] = core
    if prev_region is not None:
        full = np.maximum(full, 0.6 * prev_region + 0.4 * full)
    region_state = full
    core = (full[y0:y1c, x0:x1c] > 0.5).astype(np.uint8)

    sub = frame[y0:y1c, x0:x1c]
    dist = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 3)
    outer = (dist <= short * 0.25).astype(np.uint8)
    alpha = (big_blur_mask(core * 255, k / 2.0) / 255.0).clip(0, 1)   # 처음 버전과 같은 가장자리

    col = skin_color(sub, core, outer)
    col = col if prev_color is None else 0.8 * prev_color + 0.2 * col
    filled = sub.copy()
    filled[core > 0] = col.astype(np.uint8)
    soft = big_blur(filled, bk / 2.0)

    a = alpha[..., None]
    o = (sub.astype(np.float32) * (1 - a) + soft.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
    out = frame.copy()
    out[y0:y1c, x0:x1c] = o
    inner = alpha > 0.95
    dark = 0.0
    if inner.any():
        lum = o[inner].astype(np.float32).mean(1)
        dark = float((lum < col.mean() - DARK_GAP).mean())
    return out, col, dark, (hair_state, region_state)


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
