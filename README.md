# 몽당 ✏️

> 완벽한 답보다, 남겨진 흔적을

손글씨 수학 풀이 사진을 올리면 AI가 **어느 단계에서, 왜 막혔는지** 짚어 주고, 반복되는 **실수 습관**을 모아 주는 웹앱입니다.
정답을 알려 주는 앱이 아니라, 내가 직접 쓴 풀이를 돌아보는 **복습 도구**입니다.

## 주요 기능

| 기능 | 설명 |
|---|---|
| 풀이 진단 | 문제·풀이 사진(또는 한 장)을 올리면 단계별로 맞음 / 틀림 / 판독 불확실을 표시하고, 틀린 이유와 고치는 법을 알려 줍니다. 지우개 자국·여백 계산처럼 버린 흔적도 분석합니다. |
| 판독 수정 | AI가 손글씨를 잘못 읽은 단계는 직접 고쳐서 다시 분석할 수 있습니다. |
| 학교급 자동 판별 | 초등·중등·고등·대학을 스스로 판단해 말투와 개념, 연습 문제 수준을 맞춥니다. |
| 자주 하는 실수 · 실수 노트 | 같은 실수가 몇 번 반복됐는지 세고, 그 실수가 나온 문제를 한곳에 모아 봅니다. |
| 내 흔적 | 정답률, 자주 하는 실수, 핵심 개념 통계와 지난 분석 기록을 보여 줍니다. |
| 연습 문제 | 내 실수를 노리는 **실수 문제**와, 겉보기는 같지만 그 방식이 통하는 **짝 문제**를 한 쌍으로 만듭니다. 필요하면 그래프·도형·표도 함께 나옵니다. |

## 어떻게 진단하나요

1. AI가 학생 풀이를 보기 전에 **문제를 먼저 끝까지 풀고 검산**합니다.
2. 학생 풀이의 각 단계를 그 풀이와 하나씩 비교합니다.
3. 반복될 만한 실수는 `"부등식 양변에 음수를 곱해도 부등호 방향을 그대로 둔다"`처럼 **행동 한 문장**으로 기록하고, 다음 분석 때 같은 문장으로 맞춰 횟수를 셉니다.
4. 개념은 2022 개정 교육과정 기준 **137개 표준 개념 목록**(초등 → 중등 → 고등 → 대학 순서)에서 고르고, 가장 기초가 되는 개념을 핵심 개념으로 정합니다.
   고등 문제라도 분수 계산을 틀렸다면 핵심 개념은 초등의 "분수의 덧셈과 뺄셈"이 됩니다.

### 연습 문제 그림

AI에게 그림을 직접 그리게 하면 좌표가 틀리기 쉬워서, **AI는 설계도(함수식·좌표·표)만 JSON으로 주고 그림은 화면에서 코드가 계산해 그립니다.**
서버는 설계도의 점이 곡선 위에 정확히 있는지 다시 계산해, 어긋나면 문제를 새로 만듭니다.

## 기술 스택

Python · Flask · Google Gemini API (`google-genai`) · SQLite · Pillow · JavaScript · MathJax · math.js

## 실행 방법

1. Python 3.9 이상을 준비합니다.
2. 라이브러리를 설치합니다.
   ```bash
   pip install flask google-genai pillow
   ```
3. [Google AI Studio](https://aistudio.google.com/)에서 Gemini API 키를 발급받아 환경변수로 설정합니다.
   ```bash
   # macOS / Linux
   export GEMINI_API_KEY="발급받은_키"
   # Windows PowerShell
   $env:GEMINI_API_KEY="발급받은_키"
   ```
4. 실행한 뒤 브라우저에서 http://127.0.0.1:5000 을 엽니다.
   ```bash
   python mongdang.py
   ```

분석 기록은 같은 폴더의 `mongdang.db`(SQLite)에 저장되며, 처음 실행할 때 자동으로 만들어집니다.

## 폴더 구조

```
mongdang.py              서버 (Flask): AI 호출, 프롬프트, 결과 정리, 기록 저장
templates/index.html     화면 뼈대
templates/login.html     비밀번호 화면 (MONGDANG_PASSWORD 를 설정했을 때만)
static/app.js            화면 동작: 업로드, 결과, 내 흔적, 실수 노트, 연습 문제
static/figure.js         연습 문제 그림: 설계도로 그래프·도형·표를 그림
static/style.css         화면 꾸밈
measure_consistency.py   진단 일관성 측정 스크립트
check_practice.py        연습 문제 품질 확인 스크립트
pythonanywhere_wsgi.py   인터넷 서버(PythonAnywhere)용 설정 예시
```

## 설정 (환경변수)

| 이름 | 필수 | 설명 |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Gemini API 키 |
| `MONGDANG_MODEL` | | 사용할 모델 (기본 `gemini-3.6-flash`) |
| `MONGDANG_FALLBACKS` | | 주 모델이 붐빌 때 넘어갈 예비 모델 (쉼표로 구분) |
| `MONGDANG_THINKING` | | AI가 생각할 수 있는 분량 (기본 4096) |
| `MONGDANG_DB` | | 기록 파일 위치 (기본: 앱 폴더의 `mongdang.db`) |
| `MONGDANG_PASSWORD` | | 설정하면 비밀번호를 아는 사람만 들어올 수 있음 (인터넷에 올릴 때) |
| `MONGDANG_SYNC` | | `1`이면 분석을 요청 안에서 바로 처리 (백그라운드 스레드를 못 쓰는 무료 서버용) |

## 보조 스크립트

- `measure_consistency.py` : 같은 풀이를 여러 번 분석해 판정·핵심 개념이 얼마나 흔들리는지 측정합니다.
  ```bash
  python measure_consistency.py 문제사진 풀이사진 --runs 5
  ```
- `check_practice.py` : 실제 Gemini로 연습 문제를 여러 번 만들어 보고, 정답·그림 설계도를 확인할 수 있게 `practice_check.json`으로 저장합니다.
  ```bash
  python check_practice.py 문제사진 풀이사진 --runs 3
  ```

## 만든 배경

수원대학교 「소프트웨어적 사고」 AI 활용 Python 프로젝트 (2026년 2학기)
