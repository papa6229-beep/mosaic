"""블러 비교 (몇 분): 작업한 영상에서 고른 프레임만 원본에서 꺼내, 지금 블러와 새 블러(시험)를 나란히 그린 그림을 만듦.
영상 전체를 다시 돌리지 않음. 결과: 작업 폴더 안 '블러비교' 폴더
사용: 블러비교.bat 실행 → 작업 고르기 → 프레임 번호 입력 (예: 3665 11576-11831 23151-24050)"""
import os, sys, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WARM = 8          # 앞 프레임 몇 장을 먼저 그려서 실제 저장 때처럼 블러 속 색이 자리 잡게 함
PER_RANGE = 4     # 구간(예: 11576-11831)에서 고르는 장 수


def pick_frames(text, n):
    fs = []
    for tok in text.replace(",", " ").split():
        if "-" in tok:
            a, b = (int(x) for x in tok.split("-", 1))
            a, b = min(a, b), max(a, b)
            fs += [round(a + (b - a) * k / (PER_RANGE - 1)) for k in range(PER_RANGE)]
        elif tok.isdigit():
            fs.append(int(tok))
    return sorted({f for f in fs if 0 <= f < n})


def main():
    import numpy as np, cv2
    import app, media, cover2
    jobs = sorted(glob.glob(os.path.join(ROOT, "jobs", "*", "review.json")), key=os.path.getmtime, reverse=True)
    jobs = [j for j in jobs if os.path.exists(os.path.join(os.path.dirname(j), "blur_plan.json"))][:9]
    if not jobs:
        print("블러 작업이 없어요.")
        return 1
    print("어느 작업으로 비교할까요?")
    for k, j in enumerate(jobs):
        r = json.load(open(j, encoding="utf-8"))
        print(f"  {k + 1}. {os.path.basename(r['src'])}")
    s = input("번호 (그냥 엔터 = 1번): ").strip() or "1"
    d = os.path.dirname(jobs[int(s) - 1])
    r = json.load(open(os.path.join(d, "review.json"), encoding="utf-8"))
    n, W, H, fps = r["n"], r["W"], r["H"], r["fps"]
    fs = pick_frames(input("비교할 프레임 번호 (띄어 쓰기로 여러 개, 구간은 23151-24050 처럼): "), n)
    if not fs:
        print("프레임 번호가 없어요.")
        return 1
    for k, v in r.get("params", {}).items():
        if k in ("BLUR_FRAC", "FEATHER_W", "FEATHER_MAX", "FILL_KEEP"):
            setattr(cover2, k, v)
    ells = json.load(open(os.path.join(d, "blur_plan.json")))
    segtab = {int(k): v for k, v in json.load(open(os.path.join(d, "segments.json"))).items()}
    ep = os.path.join(d, "edits.json")
    ells = cover2.apply_edits(ells, segtab, json.load(open(ep)) if os.path.exists(ep) else {}, n)
    polys = json.load(open(os.path.join(d, "polys.json")))
    info = media.probe(r["src"])
    fmt = media.compose_fmt(info)
    fsz = media.frame_bytes(fmt, W, H)
    table = app._load_table(d)
    app._set_hw(r["src"])
    outd = os.path.join(d, "블러비교")
    os.makedirs(outd, exist_ok=True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for f in fs:
        if not ells[f]:
            print(f"  {f}: 이 프레임엔 블러가 없어요 — 건너뜀")
            continue
        s0 = max(0, f - WARM)
        dec = app.Dec(r["src"], s0, f - s0 + 1, W, H, fps, fmt, fsz, t0=info.get("t0", 0.0), table=table)
        P = {False: cover2.Painter(), True: cover2.Painter()}
        res = {}
        while True:
            g = dec.read()
            if g is None:
                break
            i = g[0]
            for v2 in (False, True):
                cover2.BLUR_V2 = v2
                yf = media.YuvFrame(g[1], fmt, W, H, info["matrix"], info["range"])
                P[v2].paint_yuv(yf, [tuple(e) for e in ells[i]], polys[i] if i < len(polys) else [])
                if i == f:
                    yuv = np.concatenate([yf.Y.ravel(), yf.U.ravel(), yf.V.ravel()])
                    res[v2] = yuv
        dec.close()
        cover2.BLUR_V2 = False
        if len(res) < 2:
            print(f"  {f}: 프레임을 못 읽었어요 — 건너뜀")
            continue
        # 블러 주변만 잘라서 (새 블러 번짐까지 들어가게)
        xs, ys = [], []
        for e in ells[f]:
            c, A = np.asarray(e[0], float), np.asarray(e[1], float)
            Ai = np.linalg.inv(A)
            rr = 1 / np.sqrt(np.linalg.eigvalsh(A))
            m = 1 + cover2.FEATHER_W2 * rr.min() / rr.max() + 0.25
            ex, ey = np.sqrt(Ai[0, 0]) * m, np.sqrt(Ai[1, 1]) * m
            xs += [c[0] - ex, c[0] + ex]; ys += [c[1] - ey, c[1] + ey]
        x0, x1 = int(max(0, min(xs))), int(min(W, max(xs)))
        y0, y1 = int(max(0, min(ys))), int(min(H, max(ys)))
        tiles = []
        for v2, lab in ((False, "now"), (True, "new")):
            bgr = media_to_bgr(res[v2], fmt, W, H, info)
            t = bgr[y0:y1, x0:x1]
            k = 900 / max(1, t.shape[0])
            t = cv2.resize(t, (max(1, int(t.shape[1] * k)), 900), interpolation=cv2.INTER_AREA)
            cv2.putText(t, lab, (16, 52), font, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(t, lab, (16, 52), font, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            tiles.append(t)
        gap = np.full((900, 12, 3), 255, np.uint8)
        out = np.hstack([tiles[0], gap, tiles[1]])
        p = os.path.join(outd, f"f{f:06d}.jpg")
        cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(p)
        print(f"  {f}: 저장 → {p}")
    print("\n끝. 왼쪽 now = 지금 블러, 오른쪽 new = 새 블러(시험). 블러비교 폴더의 그림을 보내 주세요.")
    try:
        os.startfile(outd)
    except Exception:
        pass
    return 0


def media_to_bgr(yuv, fmt, W, H, info):
    """YUV 한 장 → 화면용 8비트 BGR"""
    import numpy as np, cv2, media
    yf = media.YuvFrame(yuv.tobytes(), fmt, W, H, info["matrix"], info["range"])
    return yf.bgr(0, 0, W, H)


if __name__ == "__main__":
    try:
        r = main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("오류가 났어요. 이 창을 캡처해서 보내 주세요.")
        r = 1
    if os.name == "nt":
        input("\n엔터를 누르면 닫혀요")
    sys.exit(r)
