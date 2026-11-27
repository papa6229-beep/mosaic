"""전체 속도 점검 (4~6분): 영상 전체에서 고르게 5초씩 8군데를 떼어 이어 붙여서, 실제 생성기와 똑같이
위치 찾기 → 훑기 → 계획 → 저장을 처음부터 끝까지 돌려 단계별로 1초에 몇 장인지 재고,
그 속도로 영상 전체를 돌리면 단계별로 몇 분 걸릴지 계산함. 결과 영상은 지우고 숫자만 남김.
→ 영상 전체(몇 시간)를 돌리지 않고도 이 PC에서 실제로 얼마나 걸릴지 알 수 있음.
사용: 전체속도점검.bat 위에 원본 영상을 끌어다 놓기 (첫 화면 설정은 '빠른 훑기 켬' 기준)"""
import os, sys, time, shutil, subprocess, json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
PIECES, PSEC = 8, 5      # 영상 전체에 고르게 흩어진 5초짜리 8조각 (모두 40초)
lines = []


LOGF = os.path.join(ROOT, "전체속도점검결과.txt")


def out(s=""):
    print(s, flush=True)
    lines.append(s)
    try:                                   # 한 줄씩 바로 파일에도 (창이 갑자기 닫혀도 어디까지 됐는지 남게)
        open(LOGF, "w", encoding="utf-8").write("\n".join(lines))
    except Exception:
        pass


def mmss(x):
    x = int(round(x))
    return f"{x // 60}분 {x % 60:02d}초"


def main():
    from cut import _find_path
    src = _find_path(sys.argv[1:])
    while not src:
        print("점검할 원본 영상을 이 창에 끌어다 놓은 뒤 엔터를 눌러 주세요.")
        try:
            line = input("> ").strip()
        except Exception:
            return 1
        if line.startswith("& "):
            line = line[2:]
        src = _find_path([line.strip().strip("'")])
    import cv2, media, app, cover2
    info = media.probe(src)
    W, H = info["W"], info["H"]
    cap = cv2.VideoCapture(src); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 3000; cap.release()
    fps = float(info["fps"])
    work = os.path.join(ROOT, "tmp", "bench_all")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    clip = os.path.join(work, "clip" + os.path.splitext(src)[1])
    out("===== 전체 속도 점검 =====")
    out(f"파일: {os.path.basename(src)} ({info.get('codec')} {W}x{H}, 1초에 {fps:.2f}장, 전체 {total}장 = {mmss(total / fps)})")
    out(f"영상 전체에서 고르게 {PSEC}초씩 {PIECES}군데를 떼어 이어 붙인 뒤, 실제와 똑같이 처음부터 끝까지 만들어 봐요.")
    out("다른 프로그램은 쓰지 말고 기다려 주세요 (4~6분)...")
    dur = total / fps
    pieces = PIECES if dur >= PIECES * PSEC * 3 else 1      # 짧은 영상은 통째로
    lst = os.path.join(work, "pieces.txt")
    with open(lst, "w", encoding="utf-8") as fl:
        for k in range(pieces):
            at = dur * (k + 0.5) / PIECES - PSEC / 2 if pieces > 1 else 0
            pp = os.path.join(work, f"p{k}" + os.path.splitext(src)[1])
            subprocess.run([media.ffmpeg_exe(), "-y", "-v", "error", "-ss", f"{max(0, at):.2f}", "-i", src, "-t", str(PSEC if pieces > 1 else dur + 1),
                            "-map", "0", "-c", "copy", pp], capture_output=True)
            fl.write(f"file '{pp.replace(os.sep, '/')}'\n")
    subprocess.run([media.ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-map", "0", "-c", "copy", clip],
                   capture_output=True)
    cap = cv2.VideoCapture(clip); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if n <= 0:
        out("영상을 자르지 못했어요.")
        return 1
    # 첫 화면 기본 설정과 같게 (빠른 훑기 켬, 털까지 덮기 켬)
    cover2.FAST_SCAN = True
    cover2.HAIR = True
    job = {}
    app.ensure_model(job)
    app._set_hw(clip, job)
    out("· 위치 찾는 중…")
    t = time.time()
    table = []
    boxes, all_boxes, exps, n2 = app.detect_all(clip, job, n, None, table)
    t_det = time.time() - t
    n = n2 if n2 > 0 else n
    json.dump(table, open(os.path.join(work, "pts.json"), "w"))
    bj = os.path.join(work, "boxes.json")
    json.dump(boxes, open(bj, "w"))
    json.dump(all_boxes, open(os.path.join(work, "boxes_all.json"), "w"))
    if exps is not None:
        json.dump(exps, open(os.path.join(work, "exposures.json"), "w"))
    out(f"· 위치 찾기 끝 ({time.time() - t:.1f}초). 훑기·계획·저장 중…")
    times = {}
    rep = app.render(clip, bj, os.path.join(work, "x_cover" + os.path.splitext(src)[1]), job, n, times)
    tm = rep.get("times", times)
    st = [("위치 찾기", t_det), ("훑기", tm.get("scan", 0)), ("계획", tm.get("plan", 0)), ("저장 (이어 붙이기 포함)", tm.get("paint", 0) + tm.get("mux", 0))]
    out("")
    blur_frames = rep.get("covered", 0)
    if blur_frames == 0:
        out("")
        out("※ 떼어 온 장면에서 원본 모자이크를 하나도 못 찾았어요. 이미 블러를 입힌 결과 영상(_cover·_fix)을 넣으셨다면")
        out("  모자이크가 가려져 있어서 그래요 — 이 숫자는 블러 작업 시간이 빠진 값이라 맞지 않아요. 원본 영상으로 다시 해 주세요.")
    out(f"떼어 온 {n}장 중 블러가 들어간 장: {blur_frames}장 ({blur_frames * 100 // max(1, n)}%) — 영상 전체와 비율이 비슷할수록 계산이 정확해요")
    out(f"{'단계':<14}{'떼어 온 것':>10}{'1초에':>10}   영상 전체({mmss(total / fps)})를 돌리면")
    whole = 0
    for name, sec in st:
        rate = n / sec if sec > 0 else 0
        full = sec * total / n
        whole += full
        out(f"{name:<14}{sec:>8.1f}초{rate:>8.1f}장   약 {mmss(full)}")
    out(f"{'합계':<14}{sum(x for _, x in st):>8.1f}초{'':>10}   약 {mmss(whole)}  (영상 길이의 {whole / (total / fps):.1f}배)")
    out("")
    out(f"저장 방식: {tm.get('save_mode', '?')}  ·  찾기: {job.get('det_how', '?')}")
    out("※ 첫 화면 기본 설정 기준. 처음 켜서 모델을 받는 시간 등은 빠져 있어요.")
    out("끝. 이 창을 캡처하거나, 생성기 폴더의 전체속도점검결과.txt를 보내 주세요.")
    try:
        open(os.path.join(ROOT, "전체속도점검결과.txt"), "w", encoding="utf-8").write("\n".join(lines))
    except Exception:
        pass
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    import faulthandler
    try:                                   # 프로그램이 통째로 죽어도 어디서 죽었는지 파일에 남김
        _fh = open(os.path.join(ROOT, "전체속도점검_오류.txt"), "w", encoding="utf-8")
        faulthandler.enable(_fh, all_threads=True)
    except Exception:
        pass
    try:
        r = main()
    except Exception:
        import traceback
        traceback.print_exc()
        out("점검 중 오류가 났어요. 이 창을 캡처해서 보내 주세요.")
        r = 1
    if os.name == "nt":
        input("\n엔터를 누르면 닫혀요")
    sys.exit(r)
