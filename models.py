"""자막 모델 고르기 (팀 전체 설정): 번역 모델·받아쓰기 모델·목소리만 걸러내기. 고른 모델은 처음 쓸 때 자동으로 받음.
직접 받은 파일(GGUF)이나 우리 자료로 훈련한 받아쓰기 모델 폴더도 쓸 수 있게"""
import os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "settings", "models.json")

TR_MODELS = [
    {"id": "q9", "label": "Qwen3.5 9B 무검열 — 기본 · 그래픽카드에 다 들어가서 빠름 · 약 5GB",
     "repo": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive", "file": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
     "mb": 5300, "gpu_all": True},
    {"id": "q35a3", "label": "Qwen3.5 35B-A3B 무검열 — 더 정확 · 그래픽카드+메모리 나눠 씀 · 약 20GB (메모리 32GB 이상)",
     "repo": "HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive", "file": "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
     "mb": 21200, "gpu_all": False},
    {"id": "q27", "label": "Qwen3.5 27B 무검열 — 가장 정확 · 느림 · 약 16GB",
     "repo": "HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive", "file": "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
     "mb": 16500, "gpu_all": False},
    {"id": "file", "label": "직접 받은 GGUF 파일 쓰기 (아래에 파일 위치)", "gpu_all": False},
]
ASR_MODELS = [
    {"id": "galgame", "label": "Qwen3-ASR JA Anime-Galgame — 숨소리·속삭임 섞인 목소리용 (지금 기본)", "model_id": "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame"},
    {"id": "base", "label": "Qwen3-ASR 1.7B — 일반 모델", "model_id": "Qwen/Qwen3-ASR-1.7B"},
    {"id": "neosophie", "label": "Qwen3-ASR 1.7B JA (neosophie) — 이름·고유명사에 강함", "model_id": "neosophie/Qwen3-ASR-1.7B-JA"},
    {"id": "custom", "label": "직접 고른 모델 (허깅페이스 이름이나 우리 자료로 훈련한 모델 폴더)"},
]
ENHANCERS = [
    {"id": "none", "label": "끄기 (지금)"},
    {"id": "mel-band-roformer", "label": "목소리만 걸러내기 — Mel-Band RoFormer (시험)"},
    {"id": "htdemucs", "label": "목소리만 걸러내기 — Demucs (시험)"},
]
DEFAULT = {"tr": "q9", "tr_file": "", "asr": "galgame", "asr_custom": "", "enhancer": "none"}


def load():
    d = dict(DEFAULT)
    try:
        d.update(json.load(open(PATH, encoding="utf-8")))
    except Exception:
        pass
    return d


def save(new):
    d = load()
    d.update({k: v for k, v in new.items() if k in DEFAULT})
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    json.dump(d, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d


def tr_model():
    d = load()
    m = next((x for x in TR_MODELS if x["id"] == d["tr"]), TR_MODELS[0])
    if m["id"] == "file":
        if not d.get("tr_file") or not os.path.exists(d["tr_file"]):
            return TR_MODELS[0]
        return dict(m, path=d["tr_file"])
    return m


def asr_model_id():
    d = load()
    if d["asr"] == "custom" and d.get("asr_custom"):
        return d["asr_custom"].strip()
    return next((x["model_id"] for x in ASR_MODELS if x["id"] == d["asr"] and "model_id" in x), ASR_MODELS[0]["model_id"])


def enhancer():
    e = load().get("enhancer", "none")
    return e if any(x["id"] == e for x in ENHANCERS) else "none"


def tr_key():
    m = tr_model()
    return m["id"] + (":" + os.path.basename(m.get("path", "")) if m["id"] == "file" else "")


def asr_key():
    return asr_model_id() + "|" + enhancer()
