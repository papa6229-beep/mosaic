"""검수자가 고친 자막 모으기
1) 번역 기억: 고친 한국어(일본어 → 고친 한국어)를 모아 두고, 다음 번역 때 비슷한 일본어 줄의 '고친 정답'을 예시로 줌
2) 받아쓰기 훈련 자료: 검수자가 일본어를 고친 줄의 소리 조각 + 고친 일본어 (나중에 받아쓰기 모델을 우리 자료로 다시 훈련할 때 씀)"""
import os, json, glob, wave

ROOT = os.path.dirname(os.path.abspath(__file__))
TM_DIR = os.path.join(ROOT, "settings", "tm")
ASR_DIR = os.path.join(ROOT, "settings", "train_asr")


def _ov(a, b):
    return min(a["end"], b["end"]) - max(a["start"], b["start"]) > 0.5 * min(a["end"] - a["start"], b["end"] - b["start"], 10)


def record(jid, job_dir, orig, lines):
    """저장할 때마다: 처음 만든 자막(orig)과 비교해서 고친 줄을 모음"""
    if not orig:
        return
    tmp, asr = [], []
    for l in lines:
        o = next((x for x in orig if _ov(x, l)), None)
        ja, ko = (l.get("src") or "").strip(), (l.get("ko") or "").strip()
        if not ja:
            continue
        if ko and (o is None or ko != (o.get("ko") or "").strip()):
            tmp.append({"ja": ja, "ko": ko})
        if o is not None and ja != (o.get("src") or "").strip() and l["end"] - l["start"] < 30:
            asr.append({"start": round(l["start"], 2), "end": round(l["end"], 2), "ja": ja})
    os.makedirs(TM_DIR, exist_ok=True)
    json.dump(tmp, open(os.path.join(TM_DIR, jid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    wav = os.path.join(job_dir, "audio16k.wav")
    if asr and not os.path.exists(wav):                   # 소리 파일이 없으면 원본에서 꺼냄 (16kHz 한 채널)
        try:
            import subprocess
            src = json.load(open(os.path.join(job_dir, "review.json"), encoding="utf-8"))["src"]
            try:
                import imageio_ffmpeg
                ff = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ff = "ffmpeg"
            subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
                           timeout=1800)
        except Exception:
            pass
    if asr and os.path.exists(wav):
        cd = os.path.join(ASR_DIR, "clips")
        os.makedirs(cd, exist_ok=True)
        meta = []
        with wave.open(wav) as w:
            sr, n = w.getframerate(), w.getnframes()
            for a in asr:
                s0, s1 = max(0, int((a["start"] - 0.2) * sr)), min(n, int((a["end"] + 0.2) * sr))
                fn = f"{jid}_{int(a['start'] * 100):07d}.wav"
                w.setpos(s0)
                with wave.open(os.path.join(cd, fn), "wb") as o:
                    o.setnchannels(w.getnchannels()); o.setsampwidth(w.getsampwidth()); o.setframerate(sr)
                    o.writeframes(w.readframes(s1 - s0))
                meta.append({"audio": "clips/" + fn, "text": a["ja"]})
        json.dump(meta, open(os.path.join(ASR_DIR, jid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)


def _all():
    out = []
    for f in glob.glob(os.path.join(TM_DIR, "*.json")):
        try:
            out += json.load(open(f, encoding="utf-8"))
        except Exception:
            pass
    return out


def _grams(t):
    t = "".join(ch for ch in t if not ch.isspace() and ch not in "。、…！？!?「」")
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t}


def similar(texts, k=6):
    """이번에 번역할 일본어 줄들과 비슷한, 예전에 고친 줄 (최대 k개)"""
    mem = _all()
    if not mem:
        return []
    scored = []
    for m in mem:
        g = _grams(m["ja"])
        best = max((len(g & _grams(t)) / max(1, len(g | _grams(t))) for t in texts), default=0)
        if best >= 0.34:
            scored.append((best, m))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, m in scored:
        if m["ja"] not in seen:
            seen.add(m["ja"]); out.append(m)
        if len(out) >= k:
            break
    return out


def prompt_for(texts):
    s = similar(texts)
    if not s:
        return ""
    return "\nPAST CORRECTIONS by this team for similar lines (Japanese → their final Korean; follow their choices): " + \
        " | ".join(f"{m['ja']} → {m['ko']}" for m in s)


def stats():
    asr = 0
    for f in glob.glob(os.path.join(ASR_DIR, "*.json")):
        try:
            asr += len(json.load(open(f, encoding="utf-8")))
        except Exception:
            pass
    return {"tm": len(_all()), "asr": asr, "asr_dir": ASR_DIR}
