import os
import io
import json
from flask import Flask, request, jsonify, Response
from google import genai
from google.genai import types
from PIL import Image

# =========================
# 설정
# =========================
MODEL_NAME = os.environ.get("MONGDANG_MODEL", "gemini-3.6-flash")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90000))
app = Flask(__name__)

import time

# 주 모델이 붐비면 순서대로 넘어갈 예비 모델들
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "MONGDANG_FALLBACKS", "gemini-3.8-flash").split(",") if m.strip()]
MODEL_CHAIN = [MODEL_NAME] + [m for m in FALLBACK_MODELS if m != MODEL_NAME]
BUSY_WORDS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded", "high demand", "500", "INTERNAL", "DEADLINE", "timed out", "Timeout", "timeout", "ReadTimeout", "ConnectError", "disconnected")


def make_config(with_thinking=True):
    kw = dict(response_mime_type="application/json", temperature=0.2, max_output_tokens=6000)
    if with_thinking:
        try:
            # 오래 "생각"하느라 멈추는 것을 막기 위해 생각 분량을 작게 제한
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=1024)
        except Exception:
            pass
    return types.GenerateContentConfig(**kw)


def generate_with_retry(contents):
    """모델 호출. 붐비면 재시도, 안 되면 예비 모델. 전체 시간 예산 170초."""
    deadline = time.time() + 170
    last_err = None
    for model in MODEL_CHAIN:
        with_thinking = True
        attempt = 0
        while attempt < 2 and time.time() < deadline:
            try:
                print(f"[몽당] {model} 시도 {attempt+1}/2 (thinking={with_thinking})", flush=True)
                t0 = time.time()
                r = client.models.generate_content(
                    model=model, contents=contents,
                    config=make_config(with_thinking=with_thinking),
                )
                print(f"[몽당] {model} 성공 {time.time()-t0:.1f}초", flush=True)
                return r
            except Exception as e:
                last_err = e
                msg = str(e)
                print(f"[몽당] {model} 실패: {msg[:200]}", flush=True)
                if with_thinking and ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
                    with_thinking = False
                    continue
                if "NOT_FOUND" in msg or "404" in msg:
                    break
                if not any(w in msg for w in BUSY_WORDS):
                    raise
                attempt += 1
                time.sleep(2 * attempt)
    raise last_err or TimeoutError("분석 시간이 너무 오래 걸렸어요")


PROMPT = """
너는 수학 학습 분석 AI인 몽당이다.
두 장의 이미지가 주어진다. 첫 번째 = 수학 문제, 두 번째 = 학생이 직접 쓴 풀이.
정답과 해설을 주는 것이 목적이 아니다. 학생의 실제 풀이 흔적을 분석해 무엇을 알고 무엇을 어려워하는지 진단한다.

원칙:
- 학생의 풀이를 실제로 분석한 결과만 말한다. 정확하면 정확하다고 인정한다.
- 손글씨가 불분명하면 확정하지 말고 "판독이 불확실함"이라고 쓴다.
- 오류가 명확하지 않으면 추측해서 단정하지 않는다.
- 단순 계산 실수와 개념 오류를 구분한다.
- 정답이어도 실제로 개선할 점이 있을 때만 효율성을 지적한다. 없으면 "현재 풀이에서 특별히 개선할 부분이 발견되지 않았습니다." 라고 쓴다. 억지로 단점을 만들지 않는다.
- 부족한 개념은 학생의 실제 오류나 풀이 습관과 관련된 것만 고른다. 문제에 쓰인 개념을 전부 나열하지 않는다.
- 개념 이름(concepts.name)은 아래 [표준 개념 목록]에서 가장 가까운 것을 그대로 골라 쓴다. 목록에 정말 없을 때만 새 이름을 쓰되, "~구하기" 같은 문제 풀이 행위가 아니라 교과서 단원/개념 이름 형태로 짧게 쓴다.
- 개념은 최대 3개, 학생의 오류나 풀이 습관과 직접 관련된 것만 고른다. 같은 입력이면 같은 개념을 고르도록 일관되게 판단한다.

[표준 개념 목록]
다항식의 연산, 인수분해, 나머지정리와 인수정리, 복소수, 이차방정식의 근과 계수의 관계, 판별식, 이차함수와 직선의 위치 관계, 이차함수의 최대최소, 여러 가지 방정식, 부등식, 절댓값 방정식과 부등식, 집합의 연산, 명제와 조건, 함수의 정의와 그래프, 합성함수, 역함수, 유리함수와 무리함수, 평행이동과 대칭이동, 경우의 수, 순열과 조합, 확률의 기본 성질, 조건부확률, 통계, 지수와 로그의 계산, 지수함수, 로그함수, 지수함수와 로그함수의 그래프, 지수방정식과 로그방정식, 삼각함수의 정의, 삼각함수의 그래프, 삼각방정식과 부등식, 사인법칙과 코사인법칙, 등차수열, 등비수열, 수열의 합, 점화식, 수학적 귀납법, 함수의 극한, 함수의 연속, 미분계수와 도함수, 접선의 방정식, 함수의 증감과 극값, 부정적분, 정적분, 정적분과 넓이, 이차곡선, 벡터, 공간도형, 기타 계산 실수
- 모든 수식은 LaTeX. 인라인은 $...$ 만 사용한다. ($$ 는 쓰지 않는다.)
- JSON 문자열 안의 백슬래시는 반드시 두 번 쓴다. 예: "$\\\\frac{1}{2}$"

반드시 아래 JSON 형식으로만 답한다. 다른 글은 쓰지 않는다.
{
  "verdict": "정답" | "오답" | "판단 불가능",
  "one_line": "학생에게 건네는 핵심 한 줄 피드백",
  "problem": "문제 내용 정리 (조건, 선택지 포함)",
  "steps": [
    {"title": "단계 이름", "student_wrote": "학생이 쓴 내용", "status": "ok" | "error" | "unclear", "comment": "짧은 코멘트"}
  ],
  "errors": [
    {"step": "오류 단계", "kind": "계산 실수" | "개념 오류", "wrote": "학생이 쓴 내용", "why_wrong": "무엇이 잘못되었나", "correct": "올바른 방법", "cause": "이런 오류가 난 이유(추정이면 추정이라고 표시)"}
  ],
  "efficiency": "개선점 설명 또는 개선할 부분이 없다는 문장",
  "concepts": [
    {"name": "개념 이름", "why": "왜 공부해야 하나", "how": "이 문제에서 어떻게 쓰이나"}
  ],
  "roadmap": ["우선순위 순서로 다음에 공부할 것"],
  "final_feedback": "학생에게 직접 설명하듯 짧고 명확한 최종 피드백"
}
errors 는 오류가 없으면 빈 배열, concepts 는 해당 없으면 빈 배열로 둔다.
"""


def load_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read()))
    img = img.convert("RGB")
    if max(img.size) > 1600:
        img.thumbnail((1600, 1600))
    return img


import threading, uuid

JOBS = {}


def friendly_error(e):
    msg = str(e)
    if any(w in msg for w in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "high demand")):
        return "지금 AI 서버가 많이 붐벼요. 잠시 후 다시 눌러 주세요."
    if "timed out" in msg.lower() or "timeout" in msg.lower() or "deadline" in msg.lower():
        return "AI 응답이 너무 늦어요. 잠시 후 다시 시도해 주세요."
    return msg


def run_job(job_id, imgs):
    job = JOBS[job_id]
    try:
        response = generate_with_retry([PROMPT] + imgs)
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            print("[몽당] JSON 파싱 실패. 응답 앞부분:", text[:300], flush=True)
            job.update(status="error", error="AI 응답을 읽지 못했어요. 다시 한 번 시도해 주세요.")
            return
        job.update(status="done", data=data)
    except Exception as e:
        print("[몽당] 작업 실패:", repr(e)[:300], flush=True)
        job.update(status="error", error=friendly_error(e))


@app.route("/analyze", methods=["POST"])
def analyze():
    p = request.files.get("problem")
    s_ = request.files.get("solution")
    if not p or not s_:
        return jsonify({"error": "문제 사진과 풀이 사진을 모두 올려주세요."}), 400
    try:
        imgs = [load_image(p), load_image(s_)]
    except Exception:
        return jsonify({"error": "이미지를 읽을 수 없어요. 다른 사진으로 시도해 주세요."}), 400
    now = time.time()
    for k in [k for k, v in JOBS.items() if now - v["t"] > 1800]:
        JOBS.pop(k, None)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "t": now}
    threading.Thread(target=run_job, args=(job_id, imgs), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "error", "error": "작업을 찾을 수 없어요. 다시 시도해 주세요."})
    return jsonify({k: v for k, v in job.items() if k != "t"})


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


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
    --bg:#F5F3FF; --card:#fff; --primary:#7C5CFF; --primary-d:#6545F0; --pink:#FF6FAE;
    --text:#26224A; --sub:#8A86A8; --line:#ECE8FB; --soft:#F1EDFF;
    --ok:#22B07D; --ok-bg:#E6F8F1; --bad:#F0506E; --bad-bg:#FFEBEF; --warn:#F5A524; --warn-bg:#FFF5E0;
  }
  * { box-sizing: border-box; }
  body {
    margin:0; font-family:'Pretendard','Malgun Gothic','Apple SD Gothic Neo',sans-serif;
    background:var(--bg); color:var(--text); line-height:1.7; font-size:16px;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 0 20px 80px; }

  /* 상단 바 */
  .nav { display:flex; align-items:center; justify-content:space-between; padding:22px 0; }
  .logo { font-size:26px; font-weight:800; letter-spacing:-0.5px;
    background:linear-gradient(135deg,var(--primary),var(--pink)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .tabs { display:flex; gap:6px; background:#fff; padding:5px; border-radius:999px; box-shadow:0 2px 10px rgba(124,92,255,.08); }
  .tab { border:0; background:transparent; padding:9px 20px; border-radius:999px; font:inherit; font-weight:600; color:var(--sub); cursor:pointer; transition:.2s; }
  .tab.on { background:var(--primary); color:#fff; box-shadow:0 4px 12px rgba(124,92,255,.35); }

  /* 히어로 */
  .hero { text-align:center; padding:26px 0 30px; }
  .hero h1 { margin:0; font-size:40px; font-weight:800; letter-spacing:-1px; line-height:1.25; }
  .hero h1 em { font-style:normal; background:linear-gradient(135deg,var(--primary),var(--pink)); -webkit-background-clip:text; background-clip:text; color:transparent; }
  .hero p { margin:10px 0 0; color:var(--sub); font-size:17px; }

  /* 업로드 */
  .drops { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .drop {
    position:relative; background:var(--card); border:2px dashed #CFC6F7; border-radius:24px;
    min-height:260px; display:flex; flex-direction:column; align-items:center; justify-content:center;
    text-align:center; padding:22px; cursor:pointer; transition:.2s; overflow:hidden;
    box-shadow:0 6px 24px rgba(124,92,255,.07);
  }
  .drop:hover { border-color:var(--primary); transform:translateY(-2px); }
  .drop.over { border-color:var(--pink); background:#FFF5FA; transform:scale(1.02); }
  .drop.filled { border-style:solid; border-color:transparent; padding:0; }
  .drop .badge { display:inline-block; font-size:12px; font-weight:700; color:#fff; padding:4px 12px; border-radius:999px; background:var(--primary); margin-bottom:12px; }
  .drop.sol .badge { background:var(--pink); }
  .drop .big { font-size:19px; font-weight:700; }
  .drop .small { color:var(--sub); font-size:14px; margin-top:6px; }
  .drop .empty-state { display:flex; flex-direction:column; align-items:center; justify-content:center; width:100%; }
  .drop .empty-state .badge { order:1; margin-bottom:16px; }
  .drop .empty-state .plus { order:2; margin-bottom:16px; }
  .drop .empty-state .big { order:3; }
  .drop .empty-state .small { order:4; }
  .drop .plus { width:60px; height:60px; border-radius:50%; background:var(--soft); color:var(--primary); font-size:32px; font-weight:300; display:flex; align-items:center; justify-content:center; padding-bottom:3px; transition:.2s; }
  .drop:hover .plus { background:var(--primary); color:#fff; transform:scale(1.08); }
  .drop img { width:100%; height:100%; max-height:300px; object-fit:contain; background:#faf9ff; }
  .drop .fbar { position:absolute; left:0; right:0; bottom:0; display:flex; justify-content:space-between; align-items:center; gap:8px;
    padding:10px 14px; background:rgba(255,255,255,.92); backdrop-filter:blur(6px); font-size:13px; }
  .drop .fbar span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--sub); }
  .drop .fbar button { border:0; background:var(--soft); color:var(--primary); font-weight:700; border-radius:999px; padding:5px 12px; cursor:pointer; font:inherit; font-size:13px; }
  .drop input { display:none; }

  .go-area { text-align:center; margin-top:28px; }
  .go { border:0; font:inherit; font-size:18px; font-weight:700; color:#fff; padding:16px 54px; border-radius:999px; cursor:pointer;
    background:linear-gradient(135deg,var(--primary),var(--pink)); box-shadow:0 10px 28px rgba(124,92,255,.35); transition:.2s; }
  .go:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 14px 32px rgba(124,92,255,.45); }
  .go:disabled { background:#D8D2F3; box-shadow:none; cursor:not-allowed; }
  .tip { color:var(--sub); font-size:14px; margin-top:12px; }
  kbd { background:#fff; border:1px solid var(--line); border-radius:6px; padding:1px 7px; font-size:12px; }

  /* 로딩 */
  .loading { text-align:center; padding:90px 0; }
  .spin { width:64px; height:64px; margin:0 auto 26px; border-radius:50%; border:6px solid var(--soft); border-top-color:var(--primary); border-right-color:var(--pink); animation:sp 1s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }
  .loading h2 { margin:0 0 6px; font-size:22px; }
  .loading p { color:var(--sub); margin:0; }
  .dots { display:flex; justify-content:center; gap:8px; margin-top:22px; }
  .dots i { width:10px; height:10px; border-radius:50%; background:#DCD5F8; transition:.3s; }
  .dots i.on { background:var(--primary); transform:scale(1.3); }

  /* 결과 */
  .back { border:0; background:#fff; color:var(--primary); font:inherit; font-weight:700; padding:9px 18px; border-radius:999px; cursor:pointer; box-shadow:0 2px 10px rgba(124,92,255,.1); margin-bottom:18px; }
  .summary { border-radius:26px; padding:34px; color:#fff; box-shadow:0 14px 36px rgba(124,92,255,.25); }
  .summary.ok { background:linear-gradient(135deg,#22B07D,#4CC9A0); }
  .summary.bad { background:linear-gradient(135deg,#F0506E,#FF8A6F); }
  .summary.unk { background:linear-gradient(135deg,#7C5CFF,#A58BFF); }
  .summary .v { display:inline-block; background:rgba(255,255,255,.25); padding:5px 16px; border-radius:999px; font-weight:800; font-size:14px; letter-spacing:.5px; }
  .summary .line { font-size:24px; font-weight:700; line-height:1.5; margin-top:14px; letter-spacing:-.3px; }
  .thumbs { display:flex; gap:12px; margin-top:20px; }
  .thumbs img { height:70px; border-radius:10px; background:#fff; border:2px solid rgba(255,255,255,.6); }

  .sec { background:var(--card); border-radius:22px; padding:8px 26px; margin-top:16px; box-shadow:0 4px 18px rgba(124,92,255,.06); }
  .sec > summary { list-style:none; cursor:pointer; padding:18px 0; font-size:18px; font-weight:800; display:flex; align-items:center; justify-content:space-between; }
  .sec > summary::-webkit-details-marker { display:none; }
  .sec > summary::after { content:'+'; color:var(--primary); font-size:24px; font-weight:400; transition:.2s; }
  .sec[open] > summary::after { content:'–'; }
  .sec .body { padding:0 0 22px; }

  .step { display:flex; gap:14px; padding:14px 0; border-top:1px solid var(--line); }
  .step:first-child { border-top:0; }
  .dot { flex:0 0 32px; height:32px; border-radius:50%; color:#fff; font-weight:800; display:flex; align-items:center; justify-content:center; font-size:14px; }
  .dot.ok { background:var(--ok); } .dot.error { background:var(--bad); } .dot.unclear { background:var(--warn); }
  .step .t { font-weight:700; }
  .step .w { color:var(--sub); font-size:15px; }
  .step .c { margin-top:2px; }

  .err { border-radius:16px; padding:18px; background:var(--bad-bg); margin-top:12px; }
  .err .k { display:inline-block; font-size:12px; font-weight:800; padding:3px 11px; border-radius:999px; background:var(--bad); color:#fff; margin-bottom:8px; }
  .err .k.calc { background:var(--warn); }
  .err dt { font-weight:700; margin-top:10px; font-size:14px; color:var(--bad); }
  .err dd { margin:2px 0 0; }
  .good { background:var(--ok-bg); border-radius:16px; padding:16px 18px; color:#137A56; font-weight:600; }
  .note { background:var(--soft); border-radius:16px; padding:16px 18px; }

  .chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:6px; }
  .chip { background:var(--soft); color:var(--primary); font-weight:700; font-size:14px; padding:6px 14px; border-radius:999px; }
  .concept { border:1px solid var(--line); border-radius:16px; padding:16px 18px; margin-top:12px; }
  .concept h4 { margin:0 0 6px; color:var(--primary); font-size:17px; }
  .concept b { font-size:13px; color:var(--sub); }
  .road { counter-reset:r; padding:0; list-style:none; margin:0; }
  .road li { counter-increment:r; display:flex; gap:12px; padding:10px 0; }
  .road li::before { content:counter(r); flex:0 0 28px; height:28px; border-radius:50%; background:var(--primary); color:#fff; font-weight:800; font-size:13px; display:flex; align-items:center; justify-content:center; }
  .final { margin-top:16px; border-radius:22px; padding:26px; background:#fff; border-left:8px solid var(--pink); box-shadow:0 4px 18px rgba(124,92,255,.06); }
  .final h3 { margin:0 0 8px; color:var(--pink); font-size:15px; }

  /* 내 흔적 */
  .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
  .stat { background:#fff; border-radius:20px; padding:20px; text-align:center; box-shadow:0 4px 18px rgba(124,92,255,.06); }
  .stat .n { font-size:32px; font-weight:800; color:var(--primary); }
  .stat .l { color:var(--sub); font-size:14px; }
  .h2 { font-size:20px; font-weight:800; margin:30px 0 12px; }
  .rank { background:#fff; border-radius:20px; padding:18px 22px; box-shadow:0 4px 18px rgba(124,92,255,.06); }
  .rk { display:flex; align-items:center; gap:12px; padding:8px 0; }
  .rk .nm { flex:0 0 38%; font-weight:700; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .rk .bar { flex:1; height:12px; background:var(--soft); border-radius:999px; overflow:hidden; }
  .rk .bar i { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--primary),var(--pink)); }
  .rk .ct { flex:0 0 34px; text-align:right; color:var(--sub); font-weight:700; }
  .hist { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:14px; }
  .hc { background:#fff; border-radius:20px; padding:16px; cursor:pointer; transition:.2s; box-shadow:0 4px 18px rgba(124,92,255,.06); position:relative; }
  .hc:hover { transform:translateY(-3px); }
  .hc img { width:100%; height:110px; object-fit:cover; border-radius:12px; background:#faf9ff; }
  .hc .vd { display:inline-block; font-size:12px; font-weight:800; color:#fff; padding:2px 10px; border-radius:999px; margin-top:10px; }
  .vd.ok { background:var(--ok); } .vd.bad { background:var(--bad); } .vd.unk { background:var(--primary); }
  .hc .ln { margin-top:6px; font-size:14px; line-height:1.5; max-height:66px; overflow:hidden; }
  .hc .dt { color:var(--sub); font-size:12px; margin-top:6px; }
  .hc .x { position:absolute; top:10px; right:12px; border:0; background:rgba(255,255,255,.9); border-radius:50%; width:26px; height:26px; cursor:pointer; color:var(--sub); }
  .empty { text-align:center; color:var(--sub); padding:50px 0; background:#fff; border-radius:20px; }

  .toast { position:fixed; left:50%; bottom:30px; transform:translateX(-50%) translateY(80px); background:var(--text); color:#fff; padding:12px 22px; border-radius:999px; transition:.3s; z-index:9; max-width:90vw; }
  .toast.show { transform:translateX(-50%) translateY(0); }
  .hidden { display:none !important; }

  @media (max-width:700px) {
    .drops { grid-template-columns:1fr; }
    .hero h1 { font-size:30px; }
    .summary { padding:24px; } .summary .line { font-size:20px; }
    .stats { grid-template-columns:1fr 1fr 1fr; gap:8px; } .stat .n { font-size:24px; }
    .sec { padding:4px 18px; }
    .tab { padding:8px 14px; }
    .drop { min-height:200px; }
    .tip.pc { display:none; }
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
      <div class="drops">
        <label class="drop" id="dz-problem" data-kind="problem">
          <input type="file" accept="image/*">
          <div class="empty-state">
            <div class="plus">+</div>
            <span class="badge">STEP 1</span>
            <div class="big">문제 사진</div>
            <div class="small">끌어다 놓거나 클릭해서 선택</div>
          </div>
        </label>
        <label class="drop sol" id="dz-solution" data-kind="solution">
          <input type="file" accept="image/*">
          <div class="empty-state">
            <div class="plus">+</div>
            <span class="badge">STEP 2</span>
            <div class="big">내 풀이 사진</div>
            <div class="small">끌어다 놓거나 클릭해서 선택</div>
          </div>
        </label>
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

  <!-- 내 흔적 -->
  <section id="trace" class="hidden"></section>
</div>
<div class="toast" id="toast"></div>

<script>
const $ = (s, r=document) => r.querySelector(s);
const files = { problem:null, solution:null };
const STEP_TEXT = ['풀이를 읽는 중...', '오류를 찾는 중...', '진단을 정리하는 중...'];
let loadTimer = null;

/* ---------- 유틸 ---------- */
function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}
function typeset(el) { if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([el]); }
function verdictClass(v) { return v === '정답' ? 'ok' : v === '오답' ? 'bad' : 'unk'; }

/* ---------- 탭 ---------- */
document.querySelectorAll('.tab').forEach(b => b.onclick = () => showTab(b.dataset.tab));
function showTab(name) {
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('on', b.dataset.tab === name));
  $('#home').classList.toggle('hidden', name !== 'home');
  $('#trace').classList.toggle('hidden', name !== 'trace');
  if (name === 'trace') renderTrace();
  window.scrollTo({ top:0 });
}

/* ---------- 업로드 ---------- */
function setFile(kind, file) {
  if (!file || !file.type.startsWith('image/')) { toast('이미지 파일만 올릴 수 있어요'); return; }
  files[kind] = file;
  const dz = $('#dz-' + kind);
  const url = URL.createObjectURL(file);
  dz.classList.add('filled');
  dz.querySelector('.empty-state')?.remove();
  dz.querySelector('img')?.remove();
  dz.querySelector('.fbar')?.remove();
  dz.insertAdjacentHTML('beforeend',
    `<img src="${url}" alt=""><div class="fbar"><span>${esc(file.name)}</span><button type="button">다시 고르기</button></div>`);
  dz.querySelector('.fbar button').onclick = e => { e.preventDefault(); e.stopPropagation(); dz.querySelector('input').click(); };
  $('#go').disabled = !(files.problem && files.solution);
}

['problem','solution'].forEach(kind => {
  const dz = $('#dz-' + kind), input = dz.querySelector('input');
  input.onchange = () => { if (input.files[0]) setFile(kind, input.files[0]); input.value = ''; };
  ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave','drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', e => { if (e.dataTransfer.files[0]) setFile(kind, e.dataTransfer.files[0]); });
});
// 창 아무 데나 떨어뜨렸을 때 브라우저가 파일을 열어버리는 것 방지
['dragover','drop'].forEach(ev => window.addEventListener(ev, e => e.preventDefault()));

// Ctrl+V 붙여넣기: 비어 있는 칸부터 채움
window.addEventListener('paste', e => {
  if ($('#upload').classList.contains('hidden')) return;
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (!item) return;
  const f = item.getAsFile();
  const kind = !files.problem ? 'problem' : !files.solution ? 'solution' : 'solution';
  setFile(kind, new File([f], 'pasted-' + kind + '.png', { type: f.type }));
  toast((kind === 'problem' ? '문제' : '풀이') + ' 사진으로 붙여넣었어요');
});

/* ---------- 분석 ---------- */
$('#go').onclick = async () => {
  $('#upload').classList.add('hidden');
  $('#result').classList.add('hidden');
  $('#loading').classList.remove('hidden');
  let i = 0;
  const dots = document.querySelectorAll('.dots i');
  $('#lt').textContent = STEP_TEXT[0];
  loadTimer = setInterval(() => {
    i = Math.min(i + 1, 2);
    $('#lt').textContent = STEP_TEXT[i];
    dots.forEach((d, k) => d.classList.toggle('on', k <= i));
  }, 7000);

  try {
    const fd = new FormData();
    fd.append('problem', files.problem);
    fd.append('solution', files.solution);
    const start = await fetch('/analyze', { method:'POST', body:fd });
    const startData = await start.json();
    if (!start.ok || startData.error) throw new Error(startData.error || '분석을 시작하지 못했어요');
    const t0 = Date.now();
    let data = null;
    while (true) {
      await new Promise(r => setTimeout(r, 2000));
      if (Date.now() - t0 > 240000) throw new Error('시간이 너무 오래 걸려요. 다시 시도해 주세요.');
      const r = await fetch('/status/' + startData.job_id);
      const j = await r.json();
      if (j.status === 'done') { data = j.data; break; }
      if (j.status === 'error') throw new Error(j.error || '분석에 실패했어요');
      $('#lp').textContent = '잠시만 기다려 주세요 (' + Math.round((Date.now() - t0) / 1000) + '초 경과)';
    }
    const thumbs = { problem: await makeThumb(files.problem, 360), solution: await makeThumb(files.solution, 360) };
    saveHistory(data, thumbs);
    renderResult(data, thumbs);
  } catch (err) {
    toast(err.message);
    $('#upload').classList.remove('hidden');
  } finally {
    clearInterval(loadTimer);
    $('#loading').classList.add('hidden');
  }
};

/* ---------- 결과 화면 ---------- */
function renderResult(d, thumbs, fromTrace=false) {
  const vc = verdictClass(d.verdict);
  const steps = (d.steps || []).map((s, n) => `
    <div class="step">
      <div class="dot ${esc(s.status)}">${s.status === 'ok' ? '✓' : s.status === 'error' ? '!' : '?'}</div>
      <div><div class="t">${esc(s.title)}</div>
        <div class="w">${esc(s.student_wrote)}</div>
        <div class="c">${esc(s.comment)}</div></div>
    </div>`).join('');

  const errors = (d.errors || []).length ? d.errors.map(e => `
    <div class="err">
      <span class="k ${e.kind === '계산 실수' ? 'calc' : ''}">${esc(e.kind)}</span> <b>${esc(e.step)}</b>
      <dl>
        <dt>내가 쓴 내용</dt><dd>${esc(e.wrote)}</dd>
        <dt>무엇이 잘못됐나</dt><dd>${esc(e.why_wrong)}</dd>
        <dt>올바른 방법</dt><dd>${esc(e.correct)}</dd>
        <dt>이런 실수가 나온 이유</dt><dd>${esc(e.cause)}</dd>
      </dl>
    </div>`).join('') : `<div class="good">발견된 오류가 없어요. 풀이가 정확해요.</div>`;

  const concepts = (d.concepts || []).length ? `
    <div class="chips">${d.concepts.map(c => `<span class="chip">${esc(c.name)}</span>`).join('')}</div>` +
    d.concepts.map(c => `
    <div class="concept"><h4>${esc(c.name)}</h4>
      <b>왜 필요한가요?</b><div>${esc(c.why)}</div>
      <b style="display:block;margin-top:8px">이 문제에서는</b><div>${esc(c.how)}</div>
    </div>`).join('') : `<div class="good">이번 풀이에서 보충이 필요한 개념은 보이지 않아요.</div>`;

  const road = (d.roadmap || []).length ? `<ol class="road">${d.roadmap.map(r => `<li><div>${esc(r)}</div></li>`).join('')}</ol>` : '';

  $('#result').innerHTML = `
    <button class="back" id="back">${fromTrace ? '← 내 흔적으로' : '← 새 문제 분석하기'}</button>
    <div class="summary ${vc}">
      <span class="v">${esc(d.verdict)}</span>
      <div class="line">${esc(d.one_line)}</div>
      ${thumbs ? `<div class="thumbs"><img src="${thumbs.problem}" alt="문제"><img src="${thumbs.solution}" alt="풀이"></div>` : ''}
    </div>
    <details class="sec" open><summary>문제 내용</summary><div class="body">${esc(d.problem)}</div></details>
    <details class="sec" open><summary>내 풀이 단계별 보기</summary><div class="body">${steps}</div></details>
    <details class="sec" open><summary>풀이 과정 분석</summary><div class="body">${errors}
      ${d.efficiency ? `<div class="note" style="margin-top:12px"><b>효율성</b><br>${esc(d.efficiency)}</div>` : ''}</div></details>
    <details class="sec" open><summary>보충하면 좋은 개념</summary><div class="body">${concepts}</div></details>
    ${road ? `<details class="sec" open><summary>추천 학습 방향</summary><div class="body">${road}</div></details>` : ''}
    <div class="final"><h3>몽당의 한마디</h3>${esc(d.final_feedback)}</div>`;
  $('#result').classList.remove('hidden');
  $('#upload').classList.add('hidden');
  $('#back').onclick = () => fromTrace ? showTab('trace') : resetUpload();
  if (fromTrace) showTab('home');
  typeset($('#result'));
  window.scrollTo({ top:0 });
}

function resetUpload() {
  files.problem = files.solution = null;
  ['problem','solution'].forEach(kind => {
    const dz = $('#dz-' + kind);
    dz.classList.remove('filled');
    dz.querySelector('img')?.remove();
    dz.querySelector('.fbar')?.remove();
    if (!dz.querySelector('.empty-state')) {
      dz.insertAdjacentHTML('beforeend', `<div class="empty-state"><div class="plus">+</div>
        <span class="badge">STEP ${kind === 'problem' ? 1 : 2}</span>
        <div class="big">${kind === 'problem' ? '문제 사진' : '내 풀이 사진'}</div>
        <div class="small">끌어다 놓거나 클릭해서 선택</div></div>`);
    }
  });
  $('#go').disabled = true;
  $('#result').classList.add('hidden');
  $('#upload').classList.remove('hidden');
}

/* ---------- 기록 (브라우저 localStorage) ---------- */
const KEY = 'mongdang_history_v1';
function loadHistory() { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } }
function saveHistory(data, thumbs) {
  const h = loadHistory();
  h.unshift({ id: Date.now(), data, thumbs });
  try { localStorage.setItem(KEY, JSON.stringify(h.slice(0, 40))); }
  catch { localStorage.setItem(KEY, JSON.stringify(h.slice(0, 10))); }
}
function makeThumb(file, w) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const r = Math.min(1, w / img.width), c = document.createElement('canvas');
      c.width = img.width * r; c.height = img.height * r;
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL('image/jpeg', 0.7));
    };
    img.src = URL.createObjectURL(file);
  });
}

function renderTrace() {
  const h = loadHistory(), el = $('#trace');
  if (!h.length) {
    el.innerHTML = `<div class="hero"><h1>내 <em>흔적</em></h1><p>분석할 때마다 여기에 쌓여요</p></div>
      <div class="empty">아직 기록이 없어요.<br>문제를 하나 분석해 보세요!</div>`;
    return;
  }
  const wrong = h.filter(x => x.data.verdict === '오답').length;
  const right = h.filter(x => x.data.verdict === '정답').length;
  const cnt = {};
  h.forEach(x => (x.data.concepts || []).forEach(c => { cnt[c.name] = (cnt[c.name] || 0) + 1; }));
  const top = Object.entries(cnt).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const max = top.length ? top[0][1] : 1;

  el.innerHTML = `
    <div class="hero"><h1>내 <em>흔적</em></h1><p>쌓인 풀이로 내 약점을 한눈에</p></div>
    <div class="stats">
      <div class="stat"><div class="n">${h.length}</div><div class="l">분석한 문제</div></div>
      <div class="stat"><div class="n">${right}</div><div class="l">정답</div></div>
      <div class="stat"><div class="n">${wrong}</div><div class="l">오답</div></div>
    </div>
    <div class="h2">자주 걸리는 개념 TOP</div>
    <div class="rank">${top.length ? top.map(([n, c]) => `
      <div class="rk"><div class="nm">${esc(n)}</div><div class="bar"><i style="width:${c / max * 100}%"></i></div><div class="ct">${c}회</div></div>`).join('')
      : '<div style="color:var(--sub)">아직 집계할 개념이 없어요</div>'}</div>
    <div class="h2">지난 분석</div>
    <div class="hist">${h.map(x => `
      <div class="hc" data-id="${x.id}">
        <button class="x" data-del="${x.id}" title="삭제">×</button>
        <img src="${x.thumbs?.problem || ''}" alt="">
        <span class="vd ${verdictClass(x.data.verdict)}">${esc(x.data.verdict)}</span>
        <div class="ln">${esc(x.data.one_line)}</div>
        <div class="dt">${new Date(x.id).toLocaleString('ko-KR')}</div>
      </div>`).join('')}</div>`;

  el.querySelectorAll('.hc').forEach(c => c.onclick = e => {
    if (e.target.dataset.del) return;
    const item = loadHistory().find(x => String(x.id) === c.dataset.id);
    if (item) renderResult(item.data, item.thumbs, true);
  });
  el.querySelectorAll('[data-del]').forEach(b => b.onclick = e => {
    e.stopPropagation();
    localStorage.setItem(KEY, JSON.stringify(loadHistory().filter(x => String(x.id) !== b.dataset.del)));
    renderTrace();
  });
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    # host="0.0.0.0" 으로 바꾸면 같은 와이파이의 폰에서도 접속 가능
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
