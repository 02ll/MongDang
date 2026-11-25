import os
import webbrowser
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

from google import genai
from PIL import Image
import markdown


# =========================
# Gemini 설정
# =========================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    messagebox.showerror(
        "DeepTrack 오류",
        "Gemini API Key가 설정되지 않았습니다."
    )
    exit()

client = genai.Client(api_key=api_key)


# =========================
# 문제 + 학생 풀이 분석
# =========================

def analyze_problem(problem_path, solution_path):

    problem_image = Image.open(problem_path)
    solution_image = Image.open(solution_path)

    prompt = """
너는 수학 학습 분석 AI인 몽당이다.

너에게는 두 장의 이미지가 주어진다.

첫 번째 이미지 = 수학 문제
두 번째 이미지 = 학생이 직접 작성한 풀이

단순히 문제의 정답과 해설을 제공하는 것이 목적이 아니다.

학생의 실제 풀이를 분석하여
학생이 무엇을 알고 있고 무엇을 어려워하는지를 진단해야 한다.


# 1. 문제 내용

문제의 내용을 정확하게 읽고 정리한다.

조건과 선택지가 있다면 빠뜨리지 않는다.


# 2. 학생 풀이 판독

학생이 작성한 풀이를 가능한 한 정확하게 읽는다.

풀이가 여러 단계라면 각 단계를 순서대로 정리한다.

손글씨가 불분명한 부분은 억지로 확정하지 말고
"판독이 불확실함"이라고 표시한다.


# 3. 정답 여부

학생의 최종 답이 문제의 정답과 일치하는지 판단한다.

다음 중 하나로 명확하게 표시한다.

- 정답
- 오답
- 판단 불가능

오답이라면 학생이 어느 단계에서 잘못되었는지 찾아낸다.


# 4. 풀이 과정 분석

학생의 풀이를 단계별로 분석한다.

오류가 있다면 다음을 설명한다.

- 오류가 발생한 단계
- 학생이 작성한 내용
- 무엇이 잘못되었는지
- 올바른 방법은 무엇인지
- 왜 그런 오류가 발생했을 가능성이 있는지

단순 계산 실수와 개념적인 오류를 구분한다.


# 5. 정답이더라도 풀이 분석

학생의 답이 맞았다고 해서 분석을 끝내지 않는다.

다음 사항을 확인한다.

- 불필요하게 긴 풀이를 했는가?
- 계산을 지나치게 많이 했는가?
- 문제의 핵심 조건을 제대로 활용하지 않았는가?
- 더 간단한 개념이나 정리를 이용할 수 있었는가?
- 다른 풀이보다 비효율적인 접근을 했는가?

개선할 부분이 실제로 있을 때만 설명한다.

개선할 부분이 없다면
"현재 풀이에서 특별히 개선할 부분이 발견되지 않았습니다."
라고 솔직하게 말한다.

절대로 억지로 단점을 만들어내지 않는다.


# 6. 부족한 개념 진단

학생의 풀이를 근거로,
학생에게 부족하거나 다시 공부하면 도움이 될 가능성이 높은 수학 개념을 찾아낸다.

단순히 이 문제에 사용된 모든 개념을 나열하지 않는다.

학생의 실제 오류나 풀이 습관과 관련된 개념만 추천한다.

각 개념에 대해 다음을 설명한다.

- 개념 이름
- 왜 이 개념을 공부해야 하는지
- 이 문제에서 어떻게 사용되는지


# 7. 추천 학습 방향

학생이 다음에 공부하면 좋을 내용을 우선순위에 따라 제시한다.

다만 학생의 풀이에서 충분히 이해하고 있다고 판단되는 개념은
불필요하게 추천하지 않는다.


# 8. 최종 피드백

마지막에는 학생에게 직접 설명하듯이 짧고 명확하게 정리한다.

예를 들어:

"정답은 맞았지만, 이 문제에서는 판별식을 이용하면 훨씬 짧게 풀 수 있습니다."

또는

"정답이 틀린 이유는 이차방정식의 근과 계수의 관계를 적용하는 과정에서
부호를 잘못 처리했기 때문입니다."

처럼 작성한다.


중요한 원칙:

학생의 풀이를 실제로 분석한 결과만 이야기한다.

문제의 정답과 해설만 제공하지 않는다.

학생의 풀이가 정확하다면 정확하다고 인정한다.

학생의 풀이가 비효율적이라면 구체적으로 어떤 부분을 개선할 수 있는지 설명한다.

학생의 오류가 명확하지 않다면 추측해서 단정하지 않는다.

모든 수학식은 반드시 LaTeX 형식으로 작성한다.

인라인 수식:
$...$

독립된 수식:
$$...$$
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            prompt,
            problem_image,
            solution_image
        ]
    )

    return response.text


# =========================
# 결과를 브라우저로 표시
# =========================

def show_result(result):

    html_content = markdown.markdown(
        result,
        extensions=["extra"]
    )

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

    <meta charset="UTF-8">

    <h1>✏️ 몽당</h1>

    <script>
    MathJax = {{
        tex: {{
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']]
        }}
    }};
    </script>

    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>

    body {{
        font-family: Arial, sans-serif;
        background: #f5f5f5;
        margin: 0;
        padding: 40px;
        line-height: 1.8;
        font-size: 17px;
    }}

    .container {{
        max-width: 1000px;
        margin: auto;
        background: white;
        padding: 45px;
        border-radius: 18px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.12);
    }}

    h1 {{
        text-align: center;
        margin-bottom: 40px;
    }}

    h2 {{
        margin-top: 35px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 8px;
    }}

    h3 {{
        margin-top: 25px;
    }}

    </style>

    </head>

    <body>

    <div class="container">

    <h1>🧠 DeepTrack</h1>

    {html_content}

    </div>

    </body>

    </html>
    """

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".html",
        mode="w",
        encoding="utf-8"
    )

    temp_file.write(html)
    temp_file.close()

    webbrowser.open(
        "file://" + temp_file.name
    )


# =========================
# 문제 사진 선택
# =========================

def select_problem():

    file_path = filedialog.askopenfilename(
        title="수학 문제 사진 선택",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.webp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    selected_problem_label.config(
        text=f"문제: {os.path.basename(file_path)}"
    )

    global selected_problem
    selected_problem = file_path

    update_analyze_button()


# =========================
# 풀이 사진 선택
# =========================

def select_solution():

    file_path = filedialog.askopenfilename(
        title="내 풀이 사진 선택",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.webp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    selected_solution_label.config(
        text=f"풀이: {os.path.basename(file_path)}"
    )

    global selected_solution
    selected_solution = file_path

    update_analyze_button()


# =========================
# 분석 버튼 상태 업데이트
# =========================

def update_analyze_button():

    if selected_problem and selected_solution:

        analyze_button.config(
            state="normal"
        )

    else:

        analyze_button.config(
            state="disabled"
        )


# =========================
# 분석 시작
# =========================

def start_analysis():

    if not selected_problem or not selected_solution:
        return

    analyze_button.config(
        state="disabled",
        text="분석 중..."
    )

    root.update()

    try:

        result = analyze_problem(
            selected_problem,
            selected_solution
        )

        show_result(result)

    except Exception as e:

        messagebox.showerror(
            "DeepTrack 오류",
            str(e)
        )

    finally:

        analyze_button.config(
            state="normal",
            text="🔍 문제 분석하기"
        )


# =========================
# GUI
# =========================

selected_problem = None
selected_solution = None

root = tk.Tk()

root.title("몽당")

root.geometry("600x600")

root.resizable(False, False)


# 제목

title = tk.Label(
    root,
    text="✏️ 몽당",
    font=("Arial", 28, "bold")
)

title.pack(
    pady=(50, 10)
)


# 부제목

subtitle = tk.Label(
    root,
    text="완벽한 답보다, 남겨진 흔적을",
    font=("Arial", 14)
)

subtitle.pack(
    pady=(0, 35)
)


# 문제 사진 버튼

problem_button = tk.Button(
    root,
    text="📷 문제 사진 선택",
    font=("Arial", 14, "bold"),
    width=25,
    height=2,
    command=select_problem
)

problem_button.pack()


# 문제 파일 이름

selected_problem_label = tk.Label(
    root,
    text="문제: 선택되지 않음",
    font=("Arial", 11),
    fg="gray"
)

selected_problem_label.pack(
    pady=(10, 25)
)


# 풀이 사진 버튼

solution_button = tk.Button(
    root,
    text="✏️ 내 풀이 사진 선택",
    font=("Arial", 14, "bold"),
    width=25,
    height=2,
    command=select_solution
)

solution_button.pack()


# 풀이 파일 이름

selected_solution_label = tk.Label(
    root,
    text="풀이: 선택되지 않음",
    font=("Arial", 11),
    fg="gray"
)

selected_solution_label.pack(
    pady=(10, 30)
)


# 분석 버튼

analyze_button = tk.Button(
    root,
    text="🔍 문제 분석하기",
    font=("Arial", 15, "bold"),
    width=25,
    height=2,
    state="disabled",
    command=start_analysis
)

analyze_button.pack()


# 프로그램 실행

root.mainloop()