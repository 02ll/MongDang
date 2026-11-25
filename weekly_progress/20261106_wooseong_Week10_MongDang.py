"""
몽당 - 손글씨 수학 풀이를 읽고 어디서 막혔는지 짚어 주는 웹앱 (Flask + Gemini)

파일 순서
  1. 설정          모델, API 키, 기록 파일 위치
  2. AI 호출       서버가 붐비면 모델을 바꿔 가며 다시 시도
  3. 프롬프트      풀이 분석, 연습 문제
  4. AI 응답 정리  개념 정렬, JSON 읽기
  5. 요청 값 정리  사진, 판독 정정
  6. 작업          분석은 오래 걸려서 백그라운드로 돌리고, 브라우저는 /status 로 결과를 확인
  7. 기록 저장     SQLite (analyses, mistakes 표)
  8. 주소          /analyze, /practice, /status, /history, /mistake-note
  9. 화면          HTML / CSS / JS

실행: GEMINI_API_KEY 를 설정한 터미널에서  python 이_파일.py  →  http://127.0.0.1:5000
"""
import os
import io
import json
import random
import re
import sqlite3
import time
import threading
import uuid
from contextlib import closing, contextmanager
from flask import Flask, request, jsonify, Response
from google import genai
from google.genai import types
from PIL import Image, ImageOps

# =========================
# 설정
# =========================
MODEL_NAME = os.environ.get("MONGDANG_MODEL", "gemini-3.6-flash")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90000))
app = Flask(__name__)

# 주 모델이 붐비면 순서대로 넘어갈 예비 모델들
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "MONGDANG_FALLBACKS", "gemini-3.8-flash").split(",") if m.strip()]
MODEL_CHAIN = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
BUSY_WORDS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded", "high demand", "500", "INTERNAL", "DEADLINE", "timed out", "Timeout", "timeout", "ReadTimeout", "ConnectError", "disconnected")

# 작업 하나에 쓸 수 있는 전체 시간 (프론트 대기 한도 260초보다 짧게)
JOB_TIME_BUDGET = 220

# AI가 답하기 전에 "생각"할 수 있는 분량.
# 1024로 제한했을 때 어려운 문제에서 AI가 문제를 끝까지 풀지 못하고 진단이 매번 달라졌다 (측정 결과 9회 중 약 3회만 정확).
THINKING_BUDGET = int(os.environ.get("MONGDANG_THINKING", "4096"))
SEED = 7  # 같은 입력에는 최대한 같은 답이 나오도록 고정

# 분석 기록을 저장할 SQLite 파일 (앱 파일과 같은 폴더). 브라우저 캐시를 지워도 기록이 남는다.
DB_PATH = os.environ.get("MONGDANG_DB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "mongdang.db")


# =========================
# AI 호출
# =========================
def make_config(with_thinking=True, max_tokens=12000):
    kw = dict(response_mime_type="application/json", temperature=0, max_output_tokens=max_tokens)
    if with_thinking:
        try:
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        except Exception:
            pass
    try:
        return types.GenerateContentConfig(seed=SEED, **kw)
    except Exception:
        # seed를 지원하지 않는 라이브러리 버전이면 seed 없이
        return types.GenerateContentConfig(**kw)


_discovered = None


def model_chain():
    """설정한 모델(주 모델 + 예비 모델) 뒤에, 이 API 키로 실제 쓸 수 있는 다른 Flash 모델을 붙인다.
    모델 이름을 짐작해서 적는 대신, 서버에 목록을 물어봐서 확인된 것만 쓴다. (처음 한 번만 조회)"""
    global _discovered
    if _discovered is None:
        _discovered = []
        try:
            for m in client.models.list():
                name = (m.name or "").split("/")[-1]
                actions = getattr(m, "supported_actions", None) or []
                if ("flash" in name and "generateContent" in actions
                        and not any(w in name for w in ("lite", "image", "tts", "live", "audio", "embedding"))):
                    _discovered.append(name)
            _discovered.sort(reverse=True)  # 버전 숫자가 높은 것부터
            print(f"[몽당] 사용 가능한 Flash 모델: {_discovered}", flush=True)
        except Exception as e:
            print(f"[몽당] 모델 목록 조회 실패 (설정한 모델만 사용): {str(e)[:120]}", flush=True)
    return MODEL_CHAIN + [m for m in _discovered if m not in MODEL_CHAIN]


def generate_with_retry(contents, max_tokens=12000, deadline=None):
    """모델 호출. 서버가 붐비면(503 등) 모델을 번갈아 바꿔 가며, 한 바퀴 돌 때마다 점점 길게 쉬었다가 다시 시도한다.
    - 한 모델이 붐비는 동안 같은 모델만 계속 두드리지 않고 다른 모델로 바로 넘어간다.
    - 대기 시간은 2초, 4초, 8초, 16초... 로 늘리고(최대 20초), 여러 요청이 동시에 몰리지 않게 약간의 무작위 시간을 더한다.
    - deadline(시각)이 지나면 멈춘다."""
    if deadline is None:
        deadline = time.time() + 170
    chain = model_chain()
    no_thinking = set()   # 생각 설정을 거부한 모델
    dead = set()          # 이 키로는 없는 모델 (NOT_FOUND)
    last_err = None
    rnd = 0
    while time.time() < deadline:
        alive = [m for m in chain if m not in dead]
        if not alive:
            break
        for model in alive:
            if time.time() >= deadline:
                break
            with_thinking = model not in no_thinking
            try:
                print(f"[몽당] {model} 시도 (라운드 {rnd+1}, thinking={with_thinking}, max={max_tokens})", flush=True)
                t0 = time.time()
                r = client.models.generate_content(
                    model=model, contents=contents,
                    config=make_config(with_thinking=with_thinking, max_tokens=max_tokens),
                )
                print(f"[몽당] {model} 성공 {time.time()-t0:.1f}초", flush=True)
                return r
            except Exception as e:
                last_err = e
                msg = str(e)
                print(f"[몽당] {model} 실패: {msg[:200]}", flush=True)
                if with_thinking and ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
                    no_thinking.add(model)   # 다음 라운드에선 생각 설정 없이
                    continue
                if "NOT_FOUND" in msg or "404" in msg:
                    dead.add(model)
                    continue
                if not any(w in msg for w in BUSY_WORDS):
                    raise
        # 모든 모델이 한 바퀴 다 실패 → 점점 길게 쉬고 다시
        wait = min(20, 2 * 2 ** rnd) + random.uniform(0, 1)
        if time.time() + wait >= deadline:
            break
        print(f"[몽당] 모든 모델이 붐빔. {wait:.1f}초 쉬고 다시 시도", flush=True)
        time.sleep(wait)
        rnd += 1
    raise last_err or TimeoutError("분석 시간이 너무 오래 걸렸어요")


# =========================
# 프롬프트
# =========================
IMAGES_TWO = "두 장의 이미지가 주어진다. 첫 번째 = 수학 문제, 두 번째 = 학생이 직접 쓴 풀이."
IMAGES_ONE = ("한 장의 이미지가 주어진다. 이 이미지 안에 수학 문제와 학생이 직접 쓴 풀이가 함께 있다. "
              "인쇄된 글씨(문제 영역)는 문제로, 그 위나 옆에 쓴 손글씨는 학생의 풀이로 구분해서 읽는다. "
              "문제에 원래 있던 표시와 학생이 쓴 흔적이 헷갈리면 확정하지 말고 \"판독이 불확실함\"이라고 쓴다.")

PROMPT = """
너는 수학 학습 분석 AI인 몽당이다.
__IMAGES__
정답과 해설을 주는 것이 목적이 아니다. 학생의 실제 풀이 흔적을 보고 어디서 왜 막혔는지 진단한다.

원칙:
- 학생의 풀이를 실제로 분석한 결과만 말한다. 정확하면 정확하다고 인정한다.
- 손글씨가 불분명하면 확정하지 말고 "판독이 불확실함"이라고 쓴다.
- 오류가 명확하지 않으면 추측해서 단정하지 않는다.
- 학생에게 보여 줄 문장은 짧고 쉽게 쓴다. 같은 내용을 여러 칸에 되풀이하지 않는다.
- steps 의 student_wrote 에는 학생이 쓴 내용을 해석하지 말고 읽은 그대로 옮긴다. (학생이 판독 결과를 확인하고 고칠 수 있도록)

[진단 순서] 반드시 이 순서로 판단하고, JSON도 이 순서대로 채운다.
1. reference: 학생 풀이를 평가하기 전에 먼저 문제를 직접 끝까지 푼다.
   - 그림이나 그래프의 모양만 보고 함수 사이의 관계(역함수, 대칭, 평행이동, 접함 등)를 짐작하지 않는다. 반드시 식을 직접 변형하거나 대입해서 확인하고, 그 확인 과정을 key_checks 에 쓴다.
   - 정답까지 구한 뒤, 검산한다.
2. steps, errors: 학생 풀이의 각 단계를 reference 와 하나씩 비교해 판단한다. reference 와 다르게 판단할 근거가 없으면 학생의 단계를 맞다고 하지 않는다.
3. 나머지 칸(verdict, one_line, concepts)은 errors 에서 내린 판단을 요약만 한다. 새로운 판단을 하지 않는다.
   - 한 칸에서 틀렸다고 한 내용을 다른 칸에서 맞았다고 하지 않는다. (예: errors 에서 어떤 관계를 잘못 판단했다고 했다면, one_line 에서 그 관계를 "잘 활용했다"고 하지 않는다.)
4. kind 구분: 그 단계의 근거가 된 수학적 사실이나 관계를 잘못 알았으면 "개념 오류", 방법과 근거는 맞는데 계산만 틀렸으면 "계산 실수".

[자주 하는 실수]
- 오류마다 학생이 "자주 하는 실수"를 찾아 errors.mistake 에 쓴다.
  자주 하는 실수 = 학생이 맞다고 믿고 일관되게 적용하는 잘못된 방식. 개념 이름이 아니라, 학생이 실제로 하는 행동을 "~한다" 형태의 한 문장으로 쓴다.
  예: "$(a+b)^2$ 를 $a^2+b^2$ 로 전개한다", "부등식 양변에 음수를 곱해도 부등호 방향을 그대로 둔다", "$a^m a^n$ 을 $a^{mn}$ 으로 계산한다"
- 다른 문제에도 그대로 적용될 수 있도록 이 문제의 숫자가 아니라 일반적인 형태로 쓴다.
- 숫자를 잘못 옮겨 적거나 덧셈을 한 번 틀린 것처럼 반복될 만한 실수가 아니라 우연한 실수면 mistake 는 빈 문자열로 둔다. 억지로 만들지 않는다.
__KNOWN_RULES__
[지우고 버린 흔적]
- 지우개 자국, 줄 그어 지운 부분, 쓰다 만 식, 여백의 계산처럼 최종 풀이에 쓰이지 않은 흔적을 찾아 abandoned 에 쓴다.
- 이미지에 실제로 보이는 것만 쓴다. 없으면 빈 배열로 둔다. 흐려서 내용을 읽을 수 없으면 what 에 "판독이 불확실함"이라고 쓴다.
- 각 흔적의 방향이 맞았는지(direction) 판단한다. comment 에는 어떤 접근이었는지 쓰고, 버린 시도가 사실 맞는 방향이었다면 반드시 그렇게 말하고 왜 버렸을지(추정이면 추정이라고 표시)와 계속했다면 어떻게 됐을지를 쓴다.

[개념]
- 학생의 오류나 풀이 습관과 직접 관련된 개념만 최대 3개 고른다. 문제에 쓰인 개념을 전부 나열하지 않는다.
- 학생이 잘못 알고 있던 수학적 사실이 속한 가장 기초 단원은 반드시 포함한다.
- 이름은 아래 [표준 개념 목록]에서 가장 가까운 것을 그대로 고른다. 목록에 정말 없을 때만 교과서 단원 이름 형태로 짧게 새로 쓴다.

[표준 개념 목록]
다항식의 연산, 인수분해, 나머지정리와 인수정리, 복소수, 이차방정식의 근과 계수의 관계, 판별식, 이차함수와 직선의 위치 관계, 이차함수의 최대최소, 여러 가지 방정식, 부등식, 절댓값 방정식과 부등식, 집합의 연산, 명제와 조건, 함수의 정의와 그래프, 합성함수, 역함수, 유리함수와 무리함수, 평행이동과 대칭이동, 경우의 수, 순열과 조합, 확률의 기본 성질, 조건부확률, 통계, 지수와 로그의 계산, 지수함수, 로그함수, 지수함수와 로그함수의 그래프, 지수방정식과 로그방정식, 삼각함수의 정의, 삼각함수의 그래프, 삼각방정식과 부등식, 사인법칙과 코사인법칙, 등차수열, 등비수열, 수열의 합, 점화식, 수학적 귀납법, 함수의 극한, 함수의 연속, 미분계수와 도함수, 접선의 방정식, 함수의 증감과 극값, 부정적분, 정적분, 정적분과 넓이, 이차곡선, 벡터, 공간도형, 기타 계산 실수
- 모든 수식은 LaTeX. 인라인은 $...$ 만 사용한다. ($$ 는 쓰지 않는다.)
- JSON 문자열 안의 백슬래시는 반드시 두 번 쓴다. 예: "$\\\\frac{1}{2}$"

반드시 아래 JSON 형식으로만 답한다. 다른 글은 쓰지 않는다.
{
  "problem": "문제 내용 정리 (조건, 선택지 포함)",
  "reference": {
    "key_checks": ["문제를 푸는 데 결정적인 관계·조건을 식으로 직접 확인한 결과 (한 항목에 한 가지)"],
    "approach": "올바른 풀이의 핵심 흐름 2~4문장",
    "answer": "정답 (구할 수 없으면 판단 불가)"
  },
  "steps": [
    {"title": "단계 이름", "student_wrote": "학생이 쓴 내용", "status": "ok" | "error" | "unclear", "comment": "짧은 코멘트"}
  ],
  "errors": [
    {"step": "오류 단계", "kind": "계산 실수" | "개념 오류", "wrote": "학생이 쓴 내용", "why_wrong": "무엇이 왜 잘못됐나", "correct": "이렇게 고치면 된다", "mistake": "자주 하는 실수 한 문장 또는 빈 문자열"}
  ],
  "abandoned": [
    {"where": "흔적의 위치 (예: 오른쪽 여백, 셋째 줄 위 줄 그은 부분)", "what": "남아 있는 내용을 읽은 그대로", "direction": "맞는 방향" | "틀린 방향" | "판단 불가", "comment": "어떤 접근이었고, 왜 버렸을지, 계속했다면 어땠을지"}
  ],
  "verdict": "정답" | "오답" | "판단 불가능",
  "one_line": "학생에게 건네는 핵심 피드백 1~2문장 (errors 의 판단과 일치해야 함)",
  "concepts": ["개념 이름"]
}
errors, abandoned, concepts 는 해당 없으면 빈 배열로 둔다.
"""

KNOWN_RULES_NOTE = """- 아래는 이 학생에게서 이전에 발견된 자주 하는 실수다. 이번 오류가 이 중 하나와 같은 실수면 mistake 에 그 문장을 글자 하나 바꾸지 말고 그대로 쓴다. (같은 실수가 몇 번 반복되는지 세기 위해서다.) 다른 실수면 새로 쓴다. 억지로 맞추지 않는다.
"""

CORRECTION_NOTE = """
[학생의 판독 정정]
이전 분석에서 AI가 손글씨를 잘못 읽은 부분을 학생이 직접 고쳤다.
아래 정정 내용을 학생이 실제로 쓴 것으로 간주하고 처음부터 다시 분석한다. 이미지와 달라 보여도 정정 내용을 우선한다.
"""

# 연습 문제: 결과 화면과 실수 노트가 함께 쓴다
PRACTICE_PROMPT = """
너는 수학 학습 도우미 몽당이다. 학생이 아래 문제를 풀었다.

[원래 문제]
__PROBLEM__

[관련 개념]
__CONCEPTS__

[학생이 자주 하는 실수 (학생이 맞다고 믿고 쓰는 잘못된 방식)]
__MISTAKES__

이 학생을 위한 연습 문제 2개를 만든다.
- 자주 하는 실수가 있으면, 그 실수대로 풀면 반드시 틀린 답이 나오고 올바르게 풀어야만 정답이 나오도록 만든다.
  실수대로 해도 우연히 정답이 나오는 문제는 만들지 않는다. trap 에는 그 실수대로 풀면 나오는 오답과 이유를 쓴다.
- 자주 하는 실수가 없으면 trap 은 빈 문자열로 두고, 같은 개념을 다른 각도에서 묻는다. 관련 개념도 없으면(정답을 맞힌 경우) 같은 단원에서 한 단계 어렵게 만든다.
- 원래 문제에서 숫자만 바꾼 문제는 만들지 않는다. 난이도는 원래 문제와 비슷하게 한다.
- 정답은 하나로 확정되어야 한다. 직접 풀어서 정답이 맞는지, 그리고 그 실수대로 풀면 정말 다른 답이 나오는지 반드시 검산한다.
- 모든 수식은 LaTeX. 인라인은 $...$ 만 사용한다.
- JSON 문자열 안의 백슬래시는 반드시 두 번 쓴다. 예: "$\\\\frac{1}{2}$"

반드시 아래 JSON 형식으로만 답한다. 다른 글은 쓰지 않는다.
{
  "problems": [
    {"question": "문제", "hint": "막힐 때 볼 짧은 힌트", "answer": "정답", "trap": "그 실수대로 풀면 나오는 오답과 이유, 또는 빈 문자열", "solution": "핵심만 담은 짧은 풀이"}
  ]
}
"""


def build_prompt(single, corrections, known_rules=None):
    p = PROMPT.replace("__IMAGES__", IMAGES_ONE if single else IMAGES_TWO)
    block = KNOWN_RULES_NOTE + "".join(f"  - {r}\n" for r in known_rules) if known_rules else ""
    p = p.replace("__KNOWN_RULES__\n", block)
    if corrections:
        lines = "\n".join(
            f'- 단계 "{c["title"]}": AI가 "{c["read"]}"(으)로 읽음 → 학생 정정: "{c["wrote"]}"'
            for c in corrections)
        p += CORRECTION_NOTE + lines + "\n위 JSON 형식 규칙은 그대로 지킨다.\n"
    return p


# =========================
# AI 응답 정리
# =========================
# 프롬프트의 [표준 개념 목록]을 코드에서도 그대로 쓴다 (목록 순서 = 교육과정 순서)
STANDARD_CONCEPTS = [c.strip() for c in PROMPT.split("[표준 개념 목록]\n", 1)[1].split("\n", 1)[0].split(",")]
CONCEPT_ORDER = {name: i for i, name in enumerate(STANDARD_CONCEPTS)}


def normalize_analysis(data):
    """개념을 교육과정 순서로 정렬하고, 맨 앞(가장 기초) 개념을 핵심 개념으로 정한다.
    AI가 함께 고르는 개념은 매번 조금씩 달라질 수 있지만, 핵심 개념은 이 규칙으로 정해져서 통계가 흔들리지 않는다."""
    names = []
    for c in data.get("concepts") or []:
        name = str(c.get("name", "") if isinstance(c, dict) else c).strip()  # 10주차 초기 기록은 {"name": ...} 형태
        if name and name not in names:
            names.append(name)
    names.sort(key=lambda n: CONCEPT_ORDER.get(n, len(CONCEPT_ORDER)))  # 목록에 없는 이름은 뒤로
    data["concepts"] = names
    data["primary_concept"] = names[0] if names else ""
    return data


# JSON 문자열 안의 LaTeX 백슬래시 고치기
# - 이미 두 번 쓴 백슬래시(\\)는 그대로 둔다
# - \log, \frac 처럼 뒤에 영문자가 2개 이상 오면 LaTeX 명령이므로 두 번 쓴 것으로 바꾼다
# - 그 밖에 JSON에서 허용되지 않는 \l, \( 같은 것도 두 번 쓴 것으로 바꾼다
_LATEX_BACKSLASH = re.compile(r'\\\\|\\(?=[A-Za-z]{2,})|\\(?![\\"/bfnrtu])')


def repair_backslashes(t):
    return _LATEX_BACKSLASH.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "\\\\", t)


def parse_json(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        # AI가 LaTeX 백슬래시를 한 번만 쓰면 "Invalid \escape" 오류가 난다.
        # seed가 고정이라 다시 불러도 같은 응답이 오므로, 재시도 대신 여기서 고쳐서 읽는다.
        data = json.loads(repair_backslashes(t))
    if not isinstance(data, dict):
        raise ValueError("JSON 객체가 아님")
    return data


# =========================
# 요청 값 정리
# =========================
def parse_corrections(raw):
    """판독 정정 목록 (브라우저가 JSON 문자열로 보냄)"""
    try:
        items = json.loads(raw or "[]")
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    return [{k: str(c.get(k, ""))[:500] for k in ("title", "read", "wrote")}
            for c in items[:20] if isinstance(c, dict) and str(c.get("wrote", "")).strip()]


def clean_list(raw, limit, max_len):
    """문자열 목록을 빈 값·중복 없이 정리"""
    out = []
    for x in raw if isinstance(raw, list) else []:
        s = str(x or "").strip().replace("\n", " ")[:max_len]
        if s and s not in out:
            out.append(s)
    return out[:limit]


def load_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read()))
    # 폰 사진은 EXIF 회전 정보로만 세워져 있는 경우가 많아서, 실제 픽셀을 바로 세워 준다
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    return img


def make_thumb(img, width=360):
    """기록에 함께 저장할 작은 사진 (JPEG 바이트)"""
    t = img.copy()
    t.thumbnail((width, width * 4))
    buf = io.BytesIO()
    t.save(buf, "JPEG", quality=70)
    return buf.getvalue()


# =========================
# 작업 (분석은 오래 걸려서 백그라운드 스레드로 돌리고, 브라우저는 /status 로 결과를 확인)
# =========================
JOBS = {}


def friendly_error(e):
    msg = str(e)
    if any(w in msg for w in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand")):
        return "지금 AI 서버가 많이 붐벼요. 잠시 후 다시 눌러 주세요."
    if "timed out" in msg.lower() or "timeout" in msg.lower() or "deadline" in msg.lower():
        return "AI 응답이 너무 늦어요. 잠시 후 다시 시도해 주세요."
    return msg


def run_json_job(job_id, contents, token_steps, post=None):
    """모델을 불러 JSON을 받는다. 응답이 잘려 파싱이 안 되면 출력 한도를 늘려 한 번 더 시도.
    post 가 있으면 결과를 넘기기 전에 한 번 거친다 (분석 결과 정리·저장)."""
    job = JOBS[job_id]
    deadline = time.time() + JOB_TIME_BUDGET
    try:
        for i, max_tokens in enumerate(token_steps):
            if time.time() >= deadline:
                break
            response = generate_with_retry(contents, max_tokens=max_tokens, deadline=deadline)
            text = response.text or ""
            try:
                data = parse_json(text)
            except (json.JSONDecodeError, ValueError):
                print(f"[몽당] JSON 파싱 실패 ({i+1}/{len(token_steps)}). 응답 끝부분:", text[-200:], flush=True)
                continue
            job.update(status="done", data=post(data) if post else data)
            return
        job.update(status="error", error="AI 응답을 읽지 못했어요. 다시 한 번 시도해 주세요.")
    except Exception as e:
        print("[몽당] 작업 실패:", repr(e)[:300], flush=True)
        job.update(status="error", error=friendly_error(e))


def start_job(contents, token_steps, post=None):
    """작업을 시작하고 작업 번호를 바로 돌려준다"""
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v["t"] > 1800]:  # 30분 지난 작업은 정리
        JOBS.pop(k, None)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "t": now}
    threading.Thread(target=run_json_job, args=(job_id, contents, token_steps, post), daemon=True).start()
    return jsonify({"job_id": job_id})


# =========================
# 기록 저장 (SQLite)
# =========================
# analyses : 분석 한 번 = 한 줄. AI 결과 전체(JSON)와 작은 사진을 같이 저장한다.
# mistakes : 그 분석에서 나온 자주 하는 실수. 실수 횟수와 실수 노트는 이 표에서 SQL로 바로 꺼낸다.
SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      INTEGER NOT NULL,   -- 저장한 시각 (밀리초, 자바스크립트 Date와 같은 단위)
    verdict         TEXT,
    one_line        TEXT,
    problem         TEXT,
    primary_concept TEXT,
    data_json       TEXT NOT NULL,
    problem_thumb   BLOB,
    solution_thumb  BLOB
);
CREATE TABLE IF NOT EXISTS mistakes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL REFERENCES analyses(id),
    created_at  INTEGER NOT NULL,
    mistake     TEXT NOT NULL,
    step        TEXT,
    wrote       TEXT,
    why_wrong   TEXT,
    correct     TEXT
);
"""


def init_db():
    """기록 파일과 표를 만든다. 앱 폴더에 쓸 수 없으면 (맥 바탕화면 권한 등) 홈 폴더에 만든다."""
    global DB_PATH
    for path in (DB_PATH, os.path.join(os.path.expanduser("~"), "mongdang.db")):
        try:
            with closing(sqlite3.connect(path)) as con:
                con.executescript(SCHEMA)
            DB_PATH = path
            print("[몽당] 기록 파일:", path, flush=True)
            return
        except sqlite3.Error as e:
            print(f"[몽당] {path} 에 기록 파일을 만들 수 없어요 ({e})", flush=True)
    raise SystemExit("기록 파일(mongdang.db)을 만들 수 없습니다.")


@contextmanager
def db():
    """요청마다 새로 연결한다. 분석 결과는 다른 스레드에서 저장되기 때문에 연결을 함께 쓰지 않는다."""
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row  # 결과를 row["열 이름"] 으로 읽을 수 있게
    try:
        with con:  # 블록이 정상으로 끝나면 저장(commit), 중간에 오류가 나면 되돌림(rollback)
            yield con
    finally:
        con.close()


def save_record(con, data, thumbs):
    """분석 결과 하나를 저장하고 기록 번호를 돌려준다"""
    now = int(time.time() * 1000)
    cur = con.execute(
        "INSERT INTO analyses (created_at, verdict, one_line, problem, primary_concept, data_json, problem_thumb, solution_thumb)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (now, str(data.get("verdict", "")), str(data.get("one_line", "")), str(data.get("problem", "")),
         data["primary_concept"], json.dumps(data, ensure_ascii=False), thumbs["problem"], thumbs["solution"]))
    saved = set()
    for e in data.get("errors") or []:
        mistake = str(e.get("mistake") or "").strip()
        if mistake and mistake not in saved:  # 한 분석 안에서 같은 실수가 두 번 나와도 한 번만 센다
            saved.add(mistake)
            con.execute(
                "INSERT INTO mistakes (analysis_id, created_at, mistake, step, wrote, why_wrong, correct)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cur.lastrowid, now, mistake, *(str(e.get(k) or "") for k in ("step", "wrote", "why_wrong", "correct"))))
    return cur.lastrowid


def delete_record(con, analysis_id):
    con.execute("DELETE FROM mistakes WHERE analysis_id = ?", (analysis_id,))
    con.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))


def mistake_counts(con, data):
    """이 분석에 나온 실수가 지금까지 전체 기록에서 몇 번 나왔는지 {실수: 횟수}"""
    counts = {}
    for e in data.get("errors") or []:
        mistake = str(e.get("mistake") or "").strip()
        if mistake:
            counts[mistake] = con.execute("SELECT COUNT(*) FROM mistakes WHERE mistake = ?", (mistake,)).fetchone()[0]
    return counts


def known_mistakes():
    """AI에게 알려 줄 이전 실수 목록 (많이 나온 순, 최대 30개)"""
    with db() as con:
        rows = con.execute("SELECT mistake FROM mistakes GROUP BY mistake"
                           " ORDER BY COUNT(*) DESC, MAX(created_at) DESC LIMIT 30").fetchall()
    return [r["mistake"] for r in rows]


def save_after_analysis(data, thumbs, replace_id):
    """분석이 끝나면 바로 저장한다. 판독을 고쳐 다시 분석한 것이면 원래 기록은 지운다 (같은 풀이가 두 번 세어지지 않게).
    저장에 실패해도 분석 결과는 그대로 보여 준다."""
    data = normalize_analysis(data)
    try:
        with db() as con:
            if replace_id:
                delete_record(con, replace_id)
            history_id = save_record(con, data, thumbs)
            counts = mistake_counts(con, data)
        data.update(history_id=history_id, mistake_counts=counts)
    except Exception as e:
        print("[몽당] 기록 저장 실패:", repr(e)[:200], flush=True)
    return data


# =========================
# 주소 (라우트)
# =========================
@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/analyze", methods=["POST"])
def analyze():
    single = request.form.get("single") == "1"
    photos = [request.files.get("problem")] if single else [request.files.get("problem"), request.files.get("solution")]
    if not all(photos):
        return jsonify({"error": "문제 사진과 풀이 사진을 모두 올려주세요."}), 400
    try:
        imgs = [load_image(f) for f in photos]
    except Exception:
        return jsonify({"error": "이미지를 읽을 수 없어요. 다른 사진으로 시도해 주세요."}), 400
    corrections = parse_corrections(request.form.get("corrections"))
    replace_id = request.form.get("replace", type=int)  # 판독을 고쳐 다시 분석할 때, 바꿀 원래 기록 번호
    thumbs = {"problem": make_thumb(imgs[0]), "solution": make_thumb(imgs[1]) if len(imgs) > 1 else None}
    # 전에 찾은 자주 하는 실수를 같이 보내서, 같은 실수면 같은 문장으로 쓰게 한다 (반복 횟수를 세기 위해)
    prompt = build_prompt(single, corrections, known_mistakes())
    return start_job([prompt] + imgs, (12000, 20000), post=lambda data: save_after_analysis(data, thumbs, replace_id))


@app.route("/practice", methods=["POST"])
def practice():
    b = request.get_json(silent=True) or {}
    problem = str(b.get("problem") or "").strip()[:3000]
    if not problem:
        return jsonify({"error": "원래 문제 정보가 없어요."}), 400
    concepts = clean_list(b.get("concepts"), 3, 50)
    mistakes = clean_list(b.get("mistakes"), 3, 200)
    prompt = (PRACTICE_PROMPT
              .replace("__CONCEPTS__", ", ".join(concepts) or "(없음)")
              .replace("__MISTAKES__", "\n".join("- " + m for m in mistakes) or "(없음)")
              .replace("__PROBLEM__", problem))  # 문제 글은 맨 마지막에 넣는다 (문제 안의 글자가 다른 칸으로 바뀌지 않게)
    return start_job([prompt], (6000, 12000))


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "작업을 찾을 수 없어요. 다시 시도해 주세요."})
    return jsonify({k: v for k, v in job.items() if k != "t"})


@app.route("/history")
def history():
    """내 흔적 화면에 필요한 것을 한 번에: 통계, 자주 하는 실수, 핵심 개념 순위, 지난 분석 목록"""
    with db() as con:
        total, right, wrong = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(verdict = '정답'), 0), COALESCE(SUM(verdict = '오답'), 0) FROM analyses").fetchone()
        mistakes = con.execute(
            "SELECT mistake, COUNT(*) AS count, MAX(created_at) AS last FROM mistakes"
            " GROUP BY mistake ORDER BY count DESC, last DESC LIMIT 10").fetchall()
        # 핵심 개념만 센다 (함께 고른 관련 개념은 매번 조금씩 달라질 수 있어서 통계에서 뺌)
        concepts = con.execute(
            "SELECT primary_concept AS name, COUNT(*) AS count FROM analyses WHERE primary_concept != ''"
            " GROUP BY primary_concept ORDER BY count DESC LIMIT 6").fetchall()
        items = con.execute(
            "SELECT id, created_at, verdict, one_line FROM analyses ORDER BY created_at DESC LIMIT 100").fetchall()
    return jsonify({"total": total, "right": right, "wrong": wrong,
                    "mistakes": [dict(r) for r in mistakes],
                    "concepts": [dict(r) for r in concepts],
                    "items": [dict(r) for r in items]})


@app.route("/history/<int:hid>")
def history_one(hid):
    with db() as con:
        row = con.execute("SELECT data_json, solution_thumb IS NOT NULL AS has_solution FROM analyses WHERE id = ?",
                          (hid,)).fetchone()
        if not row:
            return jsonify({"error": "기록을 찾을 수 없어요."}), 404
        data = normalize_analysis(json.loads(row["data_json"]))
        data.update(history_id=hid, mistake_counts=mistake_counts(con, data))
    thumbs = {"problem": f"/thumb/{hid}/problem", "solution": f"/thumb/{hid}/solution" if row["has_solution"] else None}
    return jsonify({"data": data, "thumbs": thumbs})


@app.route("/history/<int:hid>", methods=["DELETE"])
def history_delete(hid):
    with db() as con:
        delete_record(con, hid)
    return jsonify({"ok": True})


@app.route("/thumb/<int:hid>/<kind>")
def thumb(hid, kind):
    if kind not in ("problem", "solution"):
        return "", 404
    with db() as con:
        row = con.execute(f"SELECT {kind}_thumb FROM analyses WHERE id = ?", (hid,)).fetchone()
    if not row or not row[0]:
        return "", 404
    return Response(row[0], mimetype="image/jpeg")


@app.route("/mistake-note")
def mistake_note():
    """자주 하는 실수 하나가 나온 문제를 최근 순으로 모두 모은다"""
    text = request.args.get("text", "").strip()
    with db() as con:
        rows = con.execute(
            "SELECT m.analysis_id AS id, m.created_at, m.step, m.wrote, m.why_wrong, m.correct, a.verdict, a.problem"
            " FROM mistakes m JOIN analyses a ON a.id = m.analysis_id"
            " WHERE m.mistake = ? ORDER BY m.created_at DESC", (text,)).fetchall()
    if not rows:
        return jsonify({"error": "이 실수가 나온 기록이 없어요."}), 404
    return jsonify({"text": text, "items": [dict(r) for r in rows]})


# =========================
# 화면 (HTML / CSS / JS)
# =========================
PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>몽당</title>
<script>
MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }, options: { skipHtmlTags: ['script','noscript','style','textarea','pre'] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css">
<style>
  :root {
    --bg:#F5F3FF; --card:#fff; --primary:#7C5CFF; --pink:#FF6FAE;
    --text:#26224A; --sub:#8A86A8; --line:#ECE8FB; --soft:#F1EDFF;
    --ok:#22B07D; --ok-bg:#E6F8F1; --bad:#F0506E; --bad-bg:#FFEBEF; --warn:#F5A524; --warn-bg:#FFF5E0;
    --shadow:0 4px 18px rgba(124,92,255,.06);
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:'Pretendard','Malgun Gothic','Apple SD Gothic Neo',sans-serif; background:var(--bg); color:var(--text); line-height:1.7; font-size:16px; }
  button { font:inherit; cursor:pointer; }
  .wrap { max-width:960px; margin:0 auto; padding:0 20px 80px; }
  .hidden { display:none !important; }

  /* 상단 바 */
  .nav { display:flex; align-items:center; justify-content:space-between; padding:22px 0; }
  .logo { font-size:26px; font-weight:800; letter-spacing:-0.5px; background:linear-gradient(135deg,var(--primary),var(--pink)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .tabs { display:flex; gap:6px; background:#fff; padding:5px; border-radius:999px; box-shadow:0 2px 10px rgba(124,92,255,.08); }
  .tab { border:0; background:transparent; padding:9px 20px; border-radius:999px; font-weight:600; color:var(--sub); transition:.2s; }
  .tab.on { background:var(--primary); color:#fff; box-shadow:0 4px 12px rgba(124,92,255,.35); }

  .hero { text-align:center; padding:26px 0 30px; }
  .hero h1 { margin:0; font-size:40px; font-weight:800; letter-spacing:-1px; line-height:1.25; }
  .hero h1 em { font-style:normal; background:linear-gradient(135deg,var(--primary),var(--pink)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .hero p { margin:10px 0 0; color:var(--sub); font-size:17px; }

  /* 업로드 */
  .mode { display:flex; justify-content:center; margin:0 0 18px; }
  .switch { display:inline-flex; align-items:center; gap:10px; background:#fff; padding:8px 18px 8px 8px; border-radius:999px; box-shadow:0 2px 10px rgba(124,92,255,.08); cursor:pointer; font-weight:600; font-size:15px; user-select:none; }
  .switch input { display:none; }
  .switch .knob { width:42px; height:24px; border-radius:999px; background:#DCD5F8; position:relative; transition:.2s; flex:0 0 auto; }
  .switch .knob::after { content:''; position:absolute; top:3px; left:3px; width:18px; height:18px; border-radius:50%; background:#fff; transition:.2s; }
  .switch input:checked + .knob { background:var(--primary); }
  .switch input:checked + .knob::after { transform:translateX(18px); }
  .drops { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .drops.one { grid-template-columns:1fr; max-width:520px; margin:0 auto; }
  .drop { position:relative; background:var(--card); border:2px dashed #CFC6F7; border-radius:24px; min-height:260px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:22px; cursor:pointer; transition:.2s; overflow:hidden; box-shadow:0 6px 24px rgba(124,92,255,.07); }
  .drop:hover { border-color:var(--primary); transform:translateY(-2px); }
  .drop.over { border-color:var(--pink); background:#FFF5FA; transform:scale(1.02); }
  .drop.filled { border-style:solid; border-color:transparent; padding:0; }
  .drop input { display:none; }
  .empty-state { display:flex; flex-direction:column; align-items:center; gap:14px; }
  .empty-state .badge { font-size:12px; font-weight:700; color:#fff; padding:4px 12px; border-radius:999px; background:var(--primary); }
  .drop.sol .badge { background:var(--pink); }
  .empty-state .plus { width:60px; height:60px; border-radius:50%; background:var(--soft); color:var(--primary); font-size:32px; font-weight:300; display:flex; align-items:center; justify-content:center; padding-bottom:3px; transition:.2s; }
  .drop:hover .plus { background:var(--primary); color:#fff; transform:scale(1.08); }
  .empty-state .big { font-size:19px; font-weight:700; line-height:1.3; }
  .empty-state .small { color:var(--sub); font-size:14px; margin-top:-10px; }
  .drop img { width:100%; height:100%; max-height:300px; object-fit:contain; background:#faf9ff; }
  .drop .fbar { position:absolute; left:0; right:0; bottom:0; display:flex; justify-content:space-between; align-items:center; gap:8px; padding:10px 14px; background:rgba(255,255,255,.92); font-size:13px; }
  .drop .fbar span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--sub); }
  .drop .fbar button { border:0; background:var(--soft); color:var(--primary); font-weight:700; border-radius:999px; padding:5px 12px; font-size:13px; }
  .go-area { text-align:center; margin-top:28px; }
  .go { border:0; font-size:18px; font-weight:700; color:#fff; padding:16px 54px; border-radius:999px; background:linear-gradient(135deg,var(--primary),var(--pink)); box-shadow:0 10px 28px rgba(124,92,255,.35); transition:.2s; }
  .go:hover:not(:disabled) { transform:translateY(-2px); }
  .go:disabled { background:#D8D2F3; box-shadow:none; cursor:not-allowed; }
  .go.small { font-size:15px; padding:11px 26px; box-shadow:0 6px 18px rgba(124,92,255,.28); display:inline-flex; align-items:center; gap:8px; }
  .tip { color:var(--sub); font-size:14px; margin-top:12px; }
  kbd { background:#fff; border:1px solid var(--line); border-radius:6px; padding:1px 7px; font-size:12px; }

  /* 로딩 */
  .loading { text-align:center; padding:90px 0; }
  .spin { width:64px; height:64px; margin:0 auto 26px; border-radius:50%; border:6px solid var(--soft); border-top-color:var(--primary); border-right-color:var(--pink); animation:sp 1s linear infinite; }
  .mini { width:16px; height:16px; border-radius:50%; border:3px solid rgba(255,255,255,.4); border-top-color:#fff; animation:sp 1s linear infinite; display:inline-block; }
  @keyframes sp { to { transform:rotate(360deg); } }
  .loading h2 { margin:0 0 6px; font-size:22px; }
  .loading p { color:var(--sub); margin:0; }
  .dots { display:flex; justify-content:center; gap:8px; margin-top:22px; }
  .dots i { width:10px; height:10px; border-radius:50%; background:#DCD5F8; transition:.3s; }
  .dots i.on { background:var(--primary); transform:scale(1.3); }

  /* 결과 */
  .back { border:0; background:#fff; color:var(--primary); font-weight:700; padding:9px 18px; border-radius:999px; box-shadow:0 2px 10px rgba(124,92,255,.1); margin-bottom:18px; }
  .summary { border-radius:26px; padding:30px 34px; color:#fff; box-shadow:0 14px 36px rgba(124,92,255,.25); }
  .summary.ok { background:linear-gradient(135deg,#22B07D,#4CC9A0); }
  .summary.bad { background:linear-gradient(135deg,#F0506E,#FF8A6F); }
  .summary.unk { background:linear-gradient(135deg,#7C5CFF,#A58BFF); }
  .summary .v { display:inline-block; background:rgba(255,255,255,.25); padding:5px 16px; border-radius:999px; font-weight:800; font-size:14px; }
  .summary .line { font-size:22px; font-weight:700; line-height:1.5; margin-top:12px; }
  .summary .cc { display:flex; flex-wrap:wrap; gap:6px; margin-top:14px; }
  .summary .cc span { background:rgba(255,255,255,.22); padding:3px 12px; border-radius:999px; font-size:13px; font-weight:700; }
  .thumbs { display:flex; gap:12px; margin-top:18px; }
  .thumbs img { height:70px; border-radius:10px; background:#fff; border:2px solid rgba(255,255,255,.6); }

  .sec { background:var(--card); border-radius:22px; padding:8px 26px; margin-top:16px; box-shadow:var(--shadow); }
  .sec > summary { list-style:none; cursor:pointer; padding:18px 0; font-size:18px; font-weight:800; display:flex; justify-content:space-between; }
  .sec > summary::-webkit-details-marker { display:none; }
  .sec > summary::after { content:'+'; color:var(--primary); font-size:24px; font-weight:400; }
  .sec[open] > summary::after { content:'–'; }
  .sec .body { padding:0 0 22px; }
  .prob { background:var(--soft); border-radius:12px; padding:8px 14px; margin-bottom:8px; font-size:14px; }
  .prob summary { cursor:pointer; color:var(--primary); font-weight:700; }

  .step { display:flex; gap:14px; padding:14px 0; border-top:1px solid var(--line); }
  .step:first-of-type { border-top:0; }
  .step .sb { flex:1; min-width:0; }
  .dot { flex:0 0 32px; height:32px; border-radius:50%; color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; font-size:14px; }
  .dot.ok { background:var(--ok); } .dot.error { background:var(--bad); } .dot.unclear { background:var(--warn); }
  .step .t { font-weight:700; }
  .step .w { color:var(--sub); font-size:15px; }

  /* 판독 수정 */
  .fixhint { background:var(--warn-bg); color:#9A6400; border-radius:12px; padding:10px 14px; font-size:14px; }
  .fixbtn { border:0; background:var(--soft); color:var(--primary); font-size:13px; font-weight:700; border-radius:999px; padding:4px 12px; margin-top:8px; }
  .fixbtn.on { background:var(--warn-bg); color:#9A6400; }
  .fix { display:block; width:100%; margin-top:8px; border:2px solid var(--warn); border-radius:12px; padding:10px 12px; font:inherit; font-size:15px; resize:vertical; background:#FFFDF7; }
  .refix { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-top:14px; padding:14px 16px; border-radius:16px; background:var(--soft); font-weight:600; font-size:15px; }

  /* 틀린 곳 */
  .err { border-radius:16px; padding:18px; background:var(--bad-bg); margin-top:12px; }
  .err:first-child { margin-top:0; }
  .k { display:inline-block; font-size:12px; font-weight:800; padding:2px 10px; border-radius:999px; background:var(--bad); color:#fff; margin-right:6px; }
  .k.calc { background:var(--warn); }
  .rule { background:#fff; border:2px dashed var(--bad); border-radius:14px; padding:12px 14px; margin:10px 0 4px; }
  .rule .rl { font-size:13px; font-weight:800; color:var(--bad); display:flex; flex-wrap:wrap; align-items:center; gap:6px; }
  .rule .rt { font-size:17px; font-weight:700; margin-top:4px; }
  .notebtn { margin-left:auto; border:0; background:var(--bad-bg); color:var(--bad); font-size:12px; font-weight:800; border-radius:999px; padding:3px 12px; }
  .notebtn:hover { background:var(--bad); color:#fff; }
  .tag { display:inline-block; font-size:11px; font-weight:800; padding:2px 9px; border-radius:999px; background:var(--soft); color:var(--sub); }
  .tag.rep { background:var(--bad); color:#fff; }
  dl { margin:0; }
  dt { font-weight:700; margin-top:10px; font-size:13px; color:var(--bad); }
  dd { margin:2px 0 0; }
  .good { background:var(--ok-bg); border-radius:16px; padding:16px 18px; color:#137A56; font-weight:600; }

  /* 지우고 버린 흔적 */
  .gem { background:var(--ok-bg); color:#137A56; border-radius:14px; padding:12px 16px; font-weight:700; }
  .ab { border:1px solid var(--line); border-left:6px solid var(--warn); border-radius:14px; padding:14px 16px; margin-top:12px; }
  .ab .k { background:var(--warn); }
  .ab.right { border-left-color:var(--ok); } .ab.right .k { background:var(--ok); }
  .ab.wrong { border-left-color:var(--bad); } .ab.wrong .k { background:var(--bad); }
  .ab dt { color:var(--sub); }

  /* 연습 문제 */
  .pwrap { margin-top:16px; text-align:center; }
  .pbody { text-align:left; }
  .sim { background:#fff; border-radius:22px; padding:22px 24px; margin-top:14px; box-shadow:var(--shadow); }
  .sim .sn { color:var(--sub); font-size:14px; }
  .sim .q { font-size:17px; margin-top:6px; }
  .sim details { margin-top:10px; border-top:1px solid var(--line); padding-top:10px; }
  .sim summary { cursor:pointer; font-weight:700; color:var(--primary); }
  .sim .trap { margin-top:8px; color:var(--bad); }
  .sim .sol { color:var(--sub); margin-top:6px; }

  /* 내 흔적 */
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  .stat { background:#fff; border-radius:20px; padding:20px; text-align:center; box-shadow:var(--shadow); }
  .stat .n { font-size:32px; font-weight:800; color:var(--primary); }
  .stat .l { color:var(--sub); font-size:14px; }
  .h2 { font-size:20px; font-weight:800; margin:30px 0 12px; }
  .h2 small { font-size:13px; font-weight:600; color:var(--sub); margin-left:6px; }
  .box { background:#fff; border-radius:20px; padding:8px 22px; box-shadow:var(--shadow); }
  .rr { display:flex; align-items:center; gap:12px; padding:12px 0; border-top:1px solid var(--line); cursor:pointer; }
  .rr:first-child { border-top:0; }
  .rr:hover .rx { color:var(--primary); }
  .rr .rc { min-width:44px; text-align:center; font-weight:800; font-size:14px; padding:3px 10px; border-radius:999px; background:var(--soft); color:var(--sub); }
  .rr.hot .rc { background:var(--bad); color:#fff; }
  .rr .rx { flex:1; min-width:0; font-weight:600; }
  .rr .rd { font-size:12px; color:var(--sub); font-weight:400; }
  .rr .arr { color:var(--sub); font-size:22px; }
  .rk { display:flex; align-items:center; gap:12px; padding:10px 0; }
  .rk .nm { flex:0 0 38%; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .rk .bar { flex:1; height:12px; background:var(--soft); border-radius:999px; overflow:hidden; }
  .rk .bar i { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--primary),var(--pink)); }
  .rk .ct { flex:0 0 34px; text-align:right; color:var(--sub); font-weight:700; }
  .muted { color:var(--sub); padding:12px 0; }
  .hist { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:14px; }
  .hc { background:#fff; border-radius:20px; padding:16px; cursor:pointer; transition:.2s; box-shadow:var(--shadow); position:relative; }
  .hc:hover { transform:translateY(-3px); }
  .hc img { width:100%; height:110px; object-fit:cover; border-radius:12px; background:#faf9ff; }
  .vd { display:inline-block; font-size:12px; font-weight:800; color:#fff; padding:2px 10px; border-radius:999px; }
  .vd.ok { background:var(--ok); } .vd.bad { background:var(--bad); } .vd.unk { background:var(--primary); }
  .hc .vd { margin-top:10px; }
  .hc .ln { margin-top:6px; font-size:14px; line-height:1.5; max-height:66px; overflow:hidden; }
  .hc .dt { color:var(--sub); font-size:12px; margin-top:6px; }
  .hc .x { position:absolute; top:10px; right:12px; border:0; background:rgba(255,255,255,.9); border-radius:50%; width:26px; height:26px; color:var(--sub); }
  .empty { text-align:center; color:var(--sub); padding:50px 0; background:#fff; border-radius:20px; }

  /* 실수 노트 */
  .notehead { background:#fff; border:2px dashed var(--bad); border-radius:22px; padding:22px 24px; }
  .notehead .rl { font-size:13px; font-weight:800; color:var(--bad); }
  .notehead .rt { font-size:22px; font-weight:800; line-height:1.5; margin-top:4px; }
  .notehead .meta { color:var(--sub); font-size:14px; margin-top:8px; }
  .occ { display:flex; gap:16px; background:#fff; border-radius:20px; padding:16px; margin-top:12px; box-shadow:var(--shadow); }
  .occ img { flex:0 0 130px; width:130px; height:130px; object-fit:cover; border-radius:12px; background:#faf9ff; }
  .occ .ob { flex:1; min-width:0; }
  .occ .oh { font-size:13px; color:var(--sub); }
  .occ .os { margin-top:6px; font-weight:700; }
  .occ .open { margin-top:12px; border:0; background:var(--soft); color:var(--primary); font-size:13px; font-weight:700; border-radius:999px; padding:5px 14px; }

  .toast { position:fixed; left:50%; bottom:30px; transform:translateX(-50%) translateY(calc(100% + 40px)); opacity:0; background:var(--text); color:#fff; padding:12px 22px; border-radius:999px; transition:.3s; z-index:9; max-width:90vw; }
  .toast.show { transform:translateX(-50%) translateY(0); opacity:1; }

  @media (max-width:700px) {
    .drops { grid-template-columns:1fr; }
    .drop { min-height:200px; }
    .tip.pc { display:none; }
    .hero h1 { font-size:30px; }
    .tab { padding:8px 14px; }
    .summary { padding:24px; } .summary .line { font-size:19px; }
    .sec { padding:4px 18px; }
    .sim { padding:18px; }
    .stats { gap:8px; } .stat .n { font-size:24px; }
    .occ { flex-direction:column; } .occ img { flex-basis:auto; width:100%; height:150px; }
    .notehead .rt { font-size:19px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="nav">
    <div class="logo">몽당</div>
    <div class="tabs">
      <button class="tab on" data-tab="home">분석하기</button>
      <button class="tab" data-tab="trace">내 흔적</button>
    </div>
  </div>

  <!-- 분석하기 -->
  <section id="home">
    <div id="upload">
      <div class="hero">
        <h1>완벽한 답보다,<br><em>남겨진 흔적</em>을</h1>
        <p>내 풀이를 읽고, 어디서 막혔는지 짚어줘요</p>
      </div>
      <div class="mode">
        <label class="switch"><input type="checkbox" id="single"><span class="knob"></span>문제와 풀이가 한 장에 있어요</label>
      </div>
      <div class="drops">
        <label class="drop" id="dz-problem"><input type="file" accept="image/*"></label>
        <label class="drop sol" id="dz-solution"><input type="file" accept="image/*"></label>
      </div>
      <div class="go-area">
        <button class="go" id="go" disabled>분석 시작하기</button>
        <div class="tip pc">사진을 복사했다면 <kbd>Ctrl</kbd> + <kbd>V</kbd> 로 바로 붙여넣을 수도 있어요</div>
      </div>
    </div>

    <div id="loading" class="loading hidden">
      <div class="spin"></div>
      <h2 id="lt">풀이를 읽는 중...</h2>
      <p id="lp">잠시만 기다려 주세요 (보통 10~40초)</p>
      <div class="dots"><i class="on"></i><i></i><i></i></div>
    </div>

    <div id="result" class="hidden"></div>
  </section>

  <!-- 내 흔적 / 실수 노트 -->
  <section id="trace" class="hidden"></section>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const files = { problem: null, solution: null };
let single = false;

/* ---------- 공통 ---------- */
function esc(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}
function typeset(el) { if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([el]); }
function verdictClass(v) { return v === '정답' ? 'ok' : v === '오답' ? 'bad' : 'unk'; }
const fmtDate = ms => new Date(ms).toLocaleDateString('ko-KR');
const fmtTime = ms => new Date(ms).toLocaleString('ko-KR');

// 서버에 요청하고 JSON으로 받는다. 실패하면 서버가 보낸 오류 문장으로 예외를 던진다
async function getJSON(url, init) {
  const r = await fetch(url, init);
  let j;
  try { j = await r.json(); } catch { throw new Error('서버 응답을 읽지 못했어요'); }
  if (!r.ok || j.error) throw new Error(j.error || '요청에 실패했어요');
  return j;
}

// AI 작업을 맡기고, 끝날 때까지 2초마다 /status 로 확인한다
async function runJob(url, init, onTick) {
  const { job_id } = await getJSON(url, init);
  const t0 = Date.now();
  while (true) {
    await new Promise(r => setTimeout(r, 2000));
    if (Date.now() - t0 > 260000) throw new Error('시간이 너무 오래 걸려요. 다시 시도해 주세요.');
    const j = await getJSON('/status/' + job_id);
    if (j.status === 'done') return j.data;
    if (onTick) onTick(Math.round((Date.now() - t0) / 1000));
  }
}

/* ---------- 탭 ---------- */
$$('.tab').forEach(b => b.onclick = () => {
  // 지난 기록을 보다가 '분석하기' 탭을 누르면 업로드 화면으로 돌아간다
  if (b.dataset.tab === 'home' && $('#result').dataset.past) {
    $('#result').classList.add('hidden');
    $('#upload').classList.remove('hidden');
  }
  showTab(b.dataset.tab);
});
function showTab(name, render = true) {
  $$('.tab').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  $('#home').classList.toggle('hidden', name !== 'home');
  $('#trace').classList.toggle('hidden', name !== 'trace');
  if (name === 'trace' && render) renderTrace();
  window.scrollTo({ top: 0 });
}

/* ---------- 업로드 ---------- */
function renderEmpty(kind) {
  if (files[kind]) return;
  const isP = kind === 'problem';
  const badge = isP ? (single ? '사진 한 장' : 'STEP 1') : 'STEP 2';
  const big = isP ? (single ? '문제 + 풀이 사진' : '문제 사진') : '내 풀이 사진';
  const small = isP && single ? '문제지에 바로 푼 사진 한 장이면 돼요' : '끌어다 놓거나 클릭해서 선택';
  const dz = $('#dz-' + kind);
  dz.querySelector('.empty-state')?.remove();
  dz.insertAdjacentHTML('beforeend', `<div class="empty-state"><span class="badge">${badge}</span>
    <div class="plus">+</div><div class="big">${big}</div><div class="small">${small}</div></div>`);
}
function updateGo() { $('#go').disabled = !(files.problem && (single || files.solution)); }

function setFile(kind, file) {
  if (!file || !file.type.startsWith('image/')) { toast('이미지 파일만 올릴 수 있어요'); return; }
  files[kind] = file;
  const dz = $('#dz-' + kind);
  dz.classList.add('filled');
  dz.querySelectorAll('.empty-state, img, .fbar').forEach(el => el.remove());
  dz.insertAdjacentHTML('beforeend', `<img src="${URL.createObjectURL(file)}" alt="">
    <div class="fbar"><span>${esc(file.name)}</span><button type="button">다시 고르기</button></div>`);
  dz.querySelector('.fbar button').onclick = e => { e.preventDefault(); e.stopPropagation(); dz.querySelector('input').click(); };
  updateGo();
}

['problem', 'solution'].forEach(kind => {
  const dz = $('#dz-' + kind), input = dz.querySelector('input');
  renderEmpty(kind);
  input.onchange = () => { if (input.files[0]) setFile(kind, input.files[0]); input.value = ''; };
  ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', e => { if (e.dataTransfer.files[0]) setFile(kind, e.dataTransfer.files[0]); });
});
// 창 아무 데나 떨어뜨렸을 때 브라우저가 파일을 열어버리는 것 방지
['dragover', 'drop'].forEach(ev => window.addEventListener(ev, e => e.preventDefault()));

// 한 장 모드
$('#single').onchange = e => {
  single = e.target.checked;
  $('#dz-solution').classList.toggle('hidden', single);
  $('.drops').classList.toggle('one', single);
  renderEmpty('problem');
  updateGo();
};

// Ctrl+V 붙여넣기: 비어 있는 칸부터 채움
window.addEventListener('paste', e => {
  if ($('#upload').classList.contains('hidden')) return;
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (!item) return;
  const f = item.getAsFile();
  const kind = single || !files.problem ? 'problem' : 'solution';
  setFile(kind, new File([f], 'pasted-' + kind + '.png', { type: f.type }));
  toast(single ? '사진을 붙여넣었어요' : (kind === 'problem' ? '문제' : '풀이') + ' 사진으로 붙여넣었어요');
});

function resetUpload() {
  files.problem = files.solution = null;
  ['problem', 'solution'].forEach(kind => {
    const dz = $('#dz-' + kind);
    dz.classList.remove('filled');
    dz.querySelectorAll('img, .fbar').forEach(el => el.remove());
    renderEmpty(kind);
  });
  updateGo();
  $('#result').classList.add('hidden');
  $('#upload').classList.remove('hidden');
}

/* ---------- 분석 ---------- */
$('#go').onclick = () => startAnalysis();

// corrections: 판독 정정 목록, replaceId: 다시 분석할 때 바꿀 원래 기록 번호
async function startAnalysis(corrections, replaceId) {
  const fromResult = !$('#result').classList.contains('hidden');
  $('#upload').classList.add('hidden');
  $('#result').classList.add('hidden');
  $('#loading').classList.remove('hidden');
  window.scrollTo({ top: 0 });
  const texts = ['풀이를 읽는 중...', '오류를 찾는 중...', '진단을 정리하는 중...'];
  const dots = $$('.dots i');
  let i = 0;
  dots.forEach((d, k) => d.classList.toggle('on', k === 0));
  $('#lt').textContent = corrections ? '고친 내용으로 다시 보는 중...' : texts[0];
  $('#lp').textContent = '잠시만 기다려 주세요 (보통 10~40초)';
  const timer = setInterval(() => {
    i = Math.min(i + 1, 2);
    $('#lt').textContent = texts[i];
    dots.forEach((d, k) => d.classList.toggle('on', k <= i));
  }, 7000);

  try {
    const fd = new FormData();
    fd.append('problem', files.problem);
    if (single) fd.append('single', '1'); else fd.append('solution', files.solution);
    if (corrections) fd.append('corrections', JSON.stringify(corrections));
    if (replaceId) fd.append('replace', replaceId);
    // 저장은 서버가 분석이 끝나자마자 한다
    const data = await runJob('/analyze', { method: 'POST', body: fd },
      s => { $('#lp').textContent = '잠시만 기다려 주세요 (' + s + '초 경과)'; });
    renderResult(data, {
      problem: URL.createObjectURL(files.problem),
      solution: single ? null : URL.createObjectURL(files.solution),
    });
  } catch (err) {
    toast(err.message);
    $(corrections && fromResult ? '#result' : '#upload').classList.remove('hidden');
  } finally {
    clearInterval(timer);
    $('#loading').classList.add('hidden');
  }
}

/* ---------- 결과 화면 ---------- */
// back: 지난 기록을 열 때 돌아갈 곳 { label, go }. 방금 분석한 결과면 없음
function renderResult(d, thumbs, back) {
  // 원본 사진이 있을 때만 판독 수정 후 다시 분석할 수 있다 (지난 기록은 원본이 없음)
  const canFix = !back && !!files.problem && (single || !!files.solution);
  const concepts = d.concepts || [], errors = d.errors || [], abandoned = d.abandoned || [];
  const counts = d.mistake_counts || {};  // 서버가 센 횟수 (이번 분석 포함)

  const stepsHTML = (d.steps || []).map((s, n) => `
    <div class="step" data-n="${n}">
      <div class="dot ${esc(s.status)}">${s.status === 'ok' ? '✓' : s.status === 'error' ? '!' : '?'}</div>
      <div class="sb"><div class="t">${esc(s.title)}</div>
        <div class="w">${esc(s.student_wrote)}</div>
        ${canFix ? '<textarea class="fix hidden" rows="2"></textarea>' : ''}
        <div>${esc(s.comment)}</div>
        ${canFix ? '<button type="button" class="fixbtn">잘못 읽었어요</button>' : ''}
      </div>
    </div>`).join('');

  const mistakeBox = (e, i) => {
    const m = String(e.mistake || '').trim();
    if (!m) return '';
    const n = counts[m] || 1;
    return `<div class="rule">
      <div class="rl">자주 하는 실수
        <span class="tag ${n > 1 ? 'rep' : ''}">${n > 1 ? `기록에서 ${n}번째` : '처음 발견'}</span>
        ${d.history_id ? `<button type="button" class="notebtn" data-i="${i}">실수 노트 →</button>` : ''}</div>
      <div class="rt">${esc(m)}</div></div>`;
  };
  const errorsHTML = errors.length ? errors.map((e, i) => `
    <div class="err">
      <span class="k ${e.kind === '계산 실수' ? 'calc' : ''}">${esc(e.kind)}</span><b>${esc(e.step)}</b>
      ${mistakeBox(e, i)}
      <dl><dt>내가 쓴 것</dt><dd>${esc(e.wrote)}</dd>
        <dt>왜 틀렸나</dt><dd>${esc(e.why_wrong)}</dd>
        <dt>이렇게 고치기</dt><dd>${esc(e.correct)}</dd></dl>
    </div>`).join('') : '<div class="good">틀린 곳이 없어요. 풀이가 정확해요.</div>';

  const dirClass = v => v === '맞는 방향' ? 'right' : v === '틀린 방향' ? 'wrong' : '';
  const abandonedHTML = abandoned.length ? `
    <details class="sec" open><summary>지우고 버린 흔적</summary><div class="body">
      ${abandoned.some(a => a.direction === '맞는 방향') ? '<div class="gem">버린 시도 중에 사실 맞는 방향이 있었어요. 초록색 흔적을 확인해 보세요.</div>' : ''}
      ${abandoned.map(a => `
        <div class="ab ${dirClass(a.direction)}">
          <span class="k">${esc(a.direction || '판단 불가')}</span><b>${esc(a.where)}</b>
          <dl><dt>남아 있던 흔적</dt><dd>${esc(a.what)}</dd>
            <dt>몽당의 생각</dt><dd>${esc(a.comment)}</dd></dl>
        </div>`).join('')}
    </div></details>` : '';

  $('#result').innerHTML = `
    <button class="back" id="back">${back ? back.label : '← 새 문제 분석하기'}</button>
    <div class="summary ${verdictClass(d.verdict)}">
      <span class="v">${esc(d.verdict)}</span>
      <div class="line">${esc(d.one_line)}</div>
      ${concepts.length ? `<div class="cc">${concepts.map((c, i) => `<span>${i === 0 ? '핵심 개념 · ' : ''}${esc(c)}</span>`).join('')}</div>` : ''}
      <div class="thumbs"><img src="${thumbs.problem}" alt="문제">${thumbs.solution ? `<img src="${thumbs.solution}" alt="풀이">` : ''}</div>
    </div>
    <details class="sec" open><summary>내 풀이</summary><div class="body">
      ${d.problem ? `<details class="prob"><summary>AI가 읽은 문제 보기</summary><div>${esc(d.problem)}</div></details>` : ''}
      ${canFix ? `<div class="fixhint">AI가 손글씨를 잘못 읽었다면 그 단계의 '잘못 읽었어요'를 눌러 고쳐 주세요.</div>` : ''}
      ${stepsHTML}
      ${canFix ? `<div class="refix hidden">고친 내용을 실제 풀이로 보고 다시 분석해요<button type="button" class="go small" id="refixBtn">다시 분석하기</button></div>` : ''}
    </div></details>
    <details class="sec" open><summary>틀린 곳</summary><div class="body">${errorsHTML}</div></details>
    ${abandonedHTML}
    <div class="pwrap"><button type="button" class="go small" id="practiceBtn">연습 문제 만들기</button><div class="pbody"></div></div>`;

  $('#result').dataset.past = back ? '1' : '';
  $('#result').classList.remove('hidden');
  $('#upload').classList.add('hidden');
  if (back) showTab('home');
  $('#back').onclick = () => back ? back.go() : resetUpload();
  if (canFix) wireFix(d);
  $('#practiceBtn').onclick = () => makePractice($('#practiceBtn'), $('#result .pbody'), {
    problem: d.problem, concepts, mistakes: errors.map(e => e.mistake),
  });
  $$('#result .notebtn').forEach(b => b.onclick = () => openNote(errors[+b.dataset.i].mistake.trim()));
  typeset($('#result'));
  window.scrollTo({ top: 0 });
}

/* ---------- 판독 수정 ---------- */
function wireFix(d) {
  const openFixes = () => [...$$('#result .fix')].filter(t => !t.classList.contains('hidden'));
  $$('#result .step').forEach(stepEl => {
    const s = d.steps[+stepEl.dataset.n];
    const ta = stepEl.querySelector('.fix'), btn = stepEl.querySelector('.fixbtn');
    btn.onclick = () => {
      const open = ta.classList.contains('hidden');  // 지금 닫혀 있으면 연다
      ta.classList.toggle('hidden', !open);
      btn.classList.toggle('on', open);
      btn.textContent = open ? '수정 취소' : '잘못 읽었어요';
      if (open) { if (!ta.value) ta.value = s.student_wrote || ''; ta.focus(); }
      $('#result .refix').classList.toggle('hidden', !openFixes().length);
    };
  });
  $('#refixBtn').onclick = () => {
    const corr = [];
    $$('#result .step').forEach(stepEl => {
      const ta = stepEl.querySelector('.fix'), s = d.steps[+stepEl.dataset.n];
      const v = ta.value.trim();
      if (!ta.classList.contains('hidden') && v && v !== String(s.student_wrote || '').trim())
        corr.push({ title: s.title, read: s.student_wrote, wrote: v });
    });
    if (!corr.length) { toast('고친 내용이 없어요'); return; }
    startAnalysis(corr, d.history_id);
  };
}

/* ---------- 연습 문제 (결과 화면, 실수 노트 공통) ---------- */
async function makePractice(btn, box, payload) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.innerHTML = '<span class="mini"></span>문제 만드는 중...';
  try {
    const data = await runJob('/practice', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    });
    const ps = data.problems || [];
    if (!ps.length) throw new Error('문제를 만들지 못했어요. 다시 눌러 주세요.');
    if (!btn.isConnected) return;  // 기다리는 동안 다른 화면으로 넘어감
    btn.remove();
    box.innerHTML = ps.map((p, i) => `
      <div class="sim">
        <b class="sn">연습 ${i + 1}</b>
        <div class="q">${esc(p.question)}</div>
        ${p.hint ? `<details><summary>힌트 보기</summary><div>${esc(p.hint)}</div></details>` : ''}
        <details><summary>정답 보기</summary>
          <div><b>정답</b> ${esc(p.answer)}</div>
          ${p.trap ? `<div class="trap"><b>함정</b> ${esc(p.trap)}</div>` : ''}
          <div class="sol">${esc(p.solution)}</div></details>
      </div>`).join('');
    typeset(box);
  } catch (err) {
    if (!btn.isConnected) return;
    toast(err.message);
    btn.disabled = false;
    btn.textContent = label;
  }
}

/* ---------- 내 흔적 ---------- */
async function renderTrace() {
  const el = $('#trace');
  const hero = sub => `<div class="hero"><h1>내 <em>흔적</em></h1><p>${sub}</p></div>`;
  let h;
  try { h = await getJSON('/history'); }
  catch (err) { el.innerHTML = hero('') + `<div class="empty">기록을 불러오지 못했어요.<br>${esc(err.message)}</div>`; return; }
  if (!h.total) {
    el.innerHTML = hero('분석할 때마다 여기에 쌓여요') + '<div class="empty">아직 기록이 없어요.<br>문제를 하나 분석해 보세요!</div>';
    return;
  }
  const max = h.concepts.length ? h.concepts[0].count : 1;

  el.innerHTML = hero('쌓인 풀이로 내 약점을 한눈에') + `
    <div class="stats">
      <div class="stat"><div class="n">${h.total}</div><div class="l">분석한 문제</div></div>
      <div class="stat"><div class="n">${h.right}</div><div class="l">정답</div></div>
      <div class="stat"><div class="n">${h.wrong}</div><div class="l">오답</div></div>
    </div>
    <div class="h2">자주 하는 실수${h.mistakes.length ? '<small>누르면 실수 노트가 열려요</small>' : ''}</div>
    <div class="box">${h.mistakes.length ? h.mistakes.map((r, i) => `
      <div class="rr ${r.count > 1 ? 'hot' : ''}" data-i="${i}"><div class="rc">${r.count}회</div>
        <div class="rx">${esc(r.mistake)}<div class="rd">마지막 ${fmtDate(r.last)}</div></div><div class="arr">›</div></div>`).join('')
      : '<div class="muted">아직 찾은 실수가 없어요. 오답을 분석하면 여기에 쌓여요</div>'}</div>
    <div class="h2">자주 걸리는 핵심 개념</div>
    <div class="box">${h.concepts.length ? h.concepts.map(c => `
      <div class="rk"><div class="nm">${esc(c.name)}</div><div class="bar"><i style="width:${c.count / max * 100}%"></i></div><div class="ct">${c.count}회</div></div>`).join('')
      : '<div class="muted">아직 집계할 개념이 없어요</div>'}</div>
    <div class="h2">지난 분석</div>
    <div class="hist">${h.items.map(x => `
      <div class="hc" data-id="${x.id}">
        <button class="x" data-del="${x.id}" title="삭제">×</button>
        <img src="/thumb/${x.id}/problem" alt="">
        <span class="vd ${verdictClass(x.verdict)}">${esc(x.verdict)}</span>
        <div class="ln">${esc(x.one_line)}</div>
        <div class="dt">${fmtTime(x.created_at)}</div>
      </div>`).join('')}</div>`;

  el.querySelectorAll('.rr').forEach(r => r.onclick = () => openNote(h.mistakes[+r.dataset.i].mistake));
  el.querySelectorAll('.hc').forEach(c => c.onclick = () =>
    openRecord(c.dataset.id, { label: '← 내 흔적으로', go: () => showTab('trace') }));
  el.querySelectorAll('.x').forEach(b => b.onclick = async e => {
    e.stopPropagation();  // 카드가 열리지 않게
    try { await getJSON('/history/' + b.dataset.del, { method: 'DELETE' }); } catch (err) { toast(err.message); }
    renderTrace();
  });
  typeset(el);
}

// 지난 기록 하나를 서버에서 받아 결과 화면으로 보여 준다
async function openRecord(id, back) {
  try {
    const r = await getJSON('/history/' + id);
    renderResult(r.data, r.thumbs, back);
  } catch (err) { toast(err.message); }
}

/* ---------- 실수 노트: 실수 하나가 나온 문제를 모아 보기 ---------- */
async function openNote(text) {
  let n;
  try { n = await getJSON('/mistake-note?text=' + encodeURIComponent(text)); }
  catch (err) { toast(err.message); return; }
  const items = n.items, latest = items[0], first = items[items.length - 1];
  showTab('trace', false);
  const el = $('#trace');
  el.innerHTML = `
    <button class="back" id="noteBack">← 내 흔적으로</button>
    <div class="notehead">
      <div class="rl">실수 노트</div>
      <div class="rt">${esc(n.text)}</div>
      <div class="meta">${items.length}번 나옴 · 처음 ${fmtDate(first.created_at)} · 마지막 ${fmtDate(latest.created_at)}</div>
    </div>
    <div class="pwrap"><button type="button" class="go small" id="notePractice">이 실수로 연습 문제 만들기</button><div class="pbody"></div></div>
    <div class="h2">이 실수가 나온 문제</div>
    ${items.map(x => `
      <div class="occ">
        <img src="/thumb/${x.id}/problem" alt="">
        <div class="ob">
          <div class="oh"><span class="vd ${verdictClass(x.verdict)}">${esc(x.verdict)}</span> ${fmtTime(x.created_at)}</div>
          <div class="os">${esc(x.step)}</div>
          <dl><dt>내가 쓴 것</dt><dd>${esc(x.wrote)}</dd>
            <dt>왜 틀렸나</dt><dd>${esc(x.why_wrong)}</dd>
            <dt>이렇게 고치기</dt><dd>${esc(x.correct)}</dd></dl>
          <button type="button" class="open" data-id="${x.id}">이 풀이 전체 보기</button>
        </div>
      </div>`).join('')}`;

  $('#noteBack').onclick = () => showTab('trace');
  // 가장 최근에 이 실수가 나온 문제를 바탕으로, 이 실수만 노리는 문제를 만든다
  $('#notePractice').onclick = () => makePractice($('#notePractice'), $('#trace .pbody'),
    { problem: latest.problem, mistakes: [n.text] });
  el.querySelectorAll('.open').forEach(b => b.onclick = () =>
    openRecord(b.dataset.id, { label: '← 실수 노트로', go: () => openNote(n.text) }));
  typeset(el);
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    # host="0.0.0.0" 으로 바꾸면 같은 와이파이의 폰에서도 접속 가능
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
