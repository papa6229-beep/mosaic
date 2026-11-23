"""자막 컨셉(스타일): 팀이 정해 두는 번역 기준 — 단어 사전(일본어→한국어), 바꿔 쓰기(한국어→한국어, 100% 적용),
말투 메모, 받아쓰기 힌트 단어, 실제 자막 예시(SRT를 올려 뽑음). 번역할 때 번역 AI 옆에 펼쳐 놓고 보게 함"""
import os, re, json, glob, hashlib, shutil, collections, random

ROOT = os.path.dirname(os.path.abspath(__file__))
PDIR = os.path.join(ROOT, "settings", "sub_profiles")
PRESETS = os.path.join(ROOT, "presets")
DEFAULT_ID = "hq"
FIELDS = ("name", "glossary", "replace", "memo", "asr_words", "examples", "stats")


def _p(pid):
    return os.path.join(PDIR, re.sub(r"[^0-9A-Za-z_-]", "", pid) + ".json")


def _ensure():
    os.makedirs(PDIR, exist_ok=True)
    for f in glob.glob(os.path.join(PRESETS, "*.json")):          # 처음 한 번: 기본 컨셉들을 설정 폴더로 복사 (이후엔 팀이 고친 게 그대로)
        dst = os.path.join(PDIR, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy(f, dst)


def list_profiles():
    _ensure()
    out = []
    for f in sorted(glob.glob(os.path.join(PDIR, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
            pid = os.path.splitext(os.path.basename(f))[0]
            out.append({"id": pid, "name": d.get("name", pid), "default": pid == DEFAULT_ID})
        except Exception:
            pass
    out.sort(key=lambda x: (not x["default"], x["name"]))
    return out


def get(pid=None):
    _ensure()
    for k in (pid, DEFAULT_ID):
        if k and os.path.exists(_p(k)):
            d = json.load(open(_p(k), encoding="utf-8"))
            d["id"] = k
            for f in FIELDS:
                d.setdefault(f, [] if f in ("glossary", "replace", "asr_words", "examples") else ({} if f == "stats" else ""))
            return d
    return {"id": "", "name": "기본", "glossary": [], "replace": [], "memo": "", "asr_words": [], "examples": [], "stats": {}}


def save(pid, data):
    _ensure()
    cur = get(pid) if os.path.exists(_p(pid)) else {}
    cur.update({k: data[k] for k in FIELDS if k in data})
    cur.pop("id", None)
    json.dump(cur, open(_p(pid), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return get(pid)


def create(name, copy_from=None):
    _ensure()
    base = get(copy_from) if copy_from else get(None)
    pid = "p" + hashlib.md5((name + str(os.urandom(4))).encode()).hexdigest()[:8]
    base = dict(base, name=name)
    if not copy_from:
        base.update(glossary=[], replace=[], memo="", examples=[], stats={})
    return save(pid, base)


def delete(pid):
    if pid == DEFAULT_ID:
        return False
    try:
        os.remove(_p(pid))
        return True
    except FileNotFoundError:
        return False


def version(prof):
    """번역에 영향을 주는 내용이 바뀌었는지 알아보는 값"""
    keep = {k: prof.get(k) for k in ("glossary", "replace", "memo", "examples")}
    return hashlib.md5(json.dumps(keep, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]


# ---------------- 번역에 쓰기 ----------------
def prompt_part(prof):
    """번역 AI에게 같이 주는 팀 기준 (영어 지시 + 한국어 내용)"""
    s = ""
    g = [(a, b) for a, b in prof.get("glossary", []) if a and b]
    if g:
        s += "\nTEAM DICTIONARY (always use these Korean renderings): " + "; ".join(f"{a} → {b}" for a, b in g) + "."
    if (prof.get("memo") or "").strip():
        s += "\nTEAM STYLE NOTES (follow them): " + prof["memo"].strip().replace("\n", " / ")
    ex = prof.get("examples", [])[:40]
    if ex:
        s += "\nEXAMPLES of this team's real Korean subtitles (match this tone, length and wording; do not copy them): " + " | ".join(ex)
    return s


def apply_replace(text, prof):
    """바꿔 쓰기: 번역이 끝난 뒤 글자를 그대로 바꿈 (항상 적용)"""
    for a, b in prof.get("replace", []):
        if a:
            text = text.replace(a, b or "")
    return text


# ---------------- SRT로 배우기 ----------------
def _srt_lines(text):
    out = []
    for bl in re.split(r"\n\s*\n", text.replace("\r", "").strip()):
        ls = bl.split("\n")
        i = [k for k, l in enumerate(ls) if "-->" in l]
        if not i:
            continue
        try:
            a, b = ls[i[0]].split("-->")
            tt = lambda x: sum(float(v) * m for v, m in zip(x.strip().split()[0].replace(",", ".").split(":"), (3600, 60, 1)))
            dur = tt(b) - tt(a)
        except Exception:
            dur = 0
        out.append((dur, [re.sub(r"<[^>]+>", "", l).strip() for l in ls[i[0] + 1:] if l.strip()]))
    return out


def learn(pid, files):
    """올린 SRT들(파일 내용 bytes 목록)에서 이 팀의 자막 예시와 특징을 뽑아 컨셉에 넣음"""
    cues = []
    for b in files:
        for enc in ("utf-8-sig", "cp949", "utf-16"):
            try:
                cues += _srt_lines(b.decode(enc))
                break
            except Exception:
                continue
    lines = [l.lstrip("- ").strip() for _, ls in cues for l in ls]
    lines = [l for l in lines if l]
    if not lines:
        raise ValueError("SRT에서 자막을 찾지 못했어요")
    cnt = collections.Counter(lines)
    common = [l for l, _ in cnt.most_common(30) if len(l) >= 2]                    # 자주 쓰는 짧은 표현
    rnd = random.Random(0)
    longer = sorted({l for l in lines if 6 <= len(l) <= 18 and l not in common})
    rnd.shuffle(longer)
    prof = get(pid)
    ex = list(dict.fromkeys(common[:20] + longer[:25] + prof.get("examples", [])))[:60]
    lens = sorted(len(l) for l in lines)
    durs = sorted(d for d, _ in cues if d > 0)
    stats = {"cues": len(cues), "line_len_median": lens[len(lens) // 2], "line_len_p90": lens[int(len(lens) * 0.9)],
             "two_line": round(sum(len(ls) == 2 for _, ls in cues) / max(1, len(cues)), 2),
             "hyphen": round(sum(any(l.startswith("-") for l in ls) for _, ls in cues) / max(1, len(cues)), 2),
             "dur_median": round(durs[len(durs) // 2], 2) if durs else 0,
             "top": [f"{l} ({n})" for l, n in cnt.most_common(15)]}
    return save(pid, {"examples": ex, "stats": stats})
