"""목소리 높낮이로 남자/여자 말소리 구분 (자막 하이픈용)

말소리의 기본 높이(초당 떨림 수)를 10ms 마다 잼: 남자 보통 85~160, 여자 보통 170~300.
→ 자막 한 줄 안에서 문장마다 남/여를 정해서, 화자가 바뀌는 곳을 번역기에 알려 줌.
"""
import os, re, wave
import numpy as np

SR = 16000
HOP = 160          # 10ms
WIN = 640          # 40ms
FMIN, FMAX = 70, 400
SPLIT_HZ = 165     # 이보다 낮으면 남자, 높으면 여자


def pitch_track(wav, cache=None):
    """→ (f0 배열, 10ms 간격). 말소리가 아닌 곳은 0"""
    if cache and os.path.exists(cache):
        return np.load(cache)
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = max(0, (len(x) - WIN) // HOP)
    f0 = np.zeros(n, np.float32)
    lo, hi = SR // FMAX, SR // FMIN
    nfft = 2048
    win = np.hanning(WIN).astype(np.float32)
    for s in range(0, n, 4000):                      # 40초씩 나눠서 계산 (메모리 아끼려고)
        e = min(n, s + 4000)
        idx = (np.arange(s, e) * HOP)[:, None] + np.arange(WIN)[None, :]
        fr = x[idx]
        fr = fr - fr.mean(1, keepdims=True)
        en = (fr ** 2).mean(1)
        F = np.fft.rfft(fr * win, nfft)
        ac = np.fft.irfft(np.abs(F) ** 2, nfft)[:, :hi + 1]
        ac = ac / np.maximum(ac[:, :1], 1e-9)
        k = lo + np.argmax(ac[:, lo:hi + 1], 1)
        peak = ac[np.arange(len(k)), k]
        thr = max(1e-6, np.percentile(en, 95) * 0.01)       # 아주 작은 소리(숨소리·잡음)는 빼기
        ok = (peak > 0.45) & (en > thr)
        f0[s:e] = np.where(ok, SR / k, 0)
    if cache:
        np.save(cache, f0)
    return f0


def _sex(f0, a, b):
    seg = f0[int(a * 100):max(int(a * 100) + 1, int(b * 100))]
    v = seg[seg > 0]
    if len(v) < 8:                                   # 말소리가 너무 적으면 모름
        return None
    return "F" if np.median(v) >= SPLIT_HZ else "M"


_SENT = re.compile(r".+?(?:[。！？!?…]+|$)")


def turns(text, start, end, f0):
    """자막 한 줄의 원문을 문장으로 나누고(글자 수만큼 시간 배분), 문장마다 남/여를 정해 화자별로 묶음
    → [(성별, 원문)] (화자가 한 명이면 길이 1)"""
    sents = [m.group(0).strip() for m in _SENT.finditer(text) if m.group(0).strip()]
    if len(sents) < 2 or f0 is None:
        return [(None, text)]
    tot = sum(len(s) for s in sents)
    t, out = start, []
    for s in sents:
        e = t + (end - start) * len(s) / tot
        out.append([_sex(f0, t, e), s])
        t = e
    for i in range(len(out)):                        # 모르는 문장은 앞(없으면 뒤) 문장 사람으로
        if out[i][0] is None:
            out[i][0] = out[i - 1][0] if i else next((o[0] for o in out if o[0]), None)
    merged = []
    for sx, s in out:
        if merged and merged[-1][0] == sx:
            merged[-1][1] += s
        else:
            merged.append([sx, s])
    return [tuple(m) for m in merged]
