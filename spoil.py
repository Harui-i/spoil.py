import argparse
from enum import Enum
import logging
import requests
import re
import subprocess
import http.cookiejar as cookielib
from bs4 import BeautifulSoup
import os
from time import sleep, time, strftime
import asyncio
import aiohttp

# cookie.jarファイルのパス
OJ_COOKIE_JAR_PATH = os.getenv("OJ_COOKIE_JAR_PATH")
# Open AI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Cloud API key ( for Gemini )
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Google Cloud Project ID
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')


class MODEL(Enum):
    GPT4O = "gpt-4o"
    GEMINIPRO = "gemini-1.5-pro"


# Create a custom logger
logger = logging.getLogger(__name__)

# Set level of logger
logger.setLevel(logging.DEBUG)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(filename=f'spoilpy-{strftime("%Y-%m-%d_%H-%M-%S")}.log')

# Set level of handlers
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Create formatters and add it to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


def get_solution_file_name(model: MODEL) -> str:
    return f"solution_by_{model}.py"


def load_cookies_from_lwp(cookie_jar_path):
    # LWP形式のcookie.jarを読み込む
    cookie_jar = cookielib.LWPCookieJar()
    cookie_jar.load(cookie_jar_path, ignore_discard=True, ignore_expires=True)
    return cookie_jar


def get_session_with_cookies(cookie_jar_path):
    session = requests.Session()
    cookies = load_cookies_from_lwp(cookie_jar_path)
    session.cookies.update(cookies)
    return session


def get_contest_problem_html(session, contest_id, problem_id):
    problem_url = f"https://atcoder.jp/contests/{contest_id}/tasks/{problem_id}"
    response = session.get(problem_url)
    return response.text


def extract_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    english_parts = soup.select("span.lang-en")
    problem_title = soup.title.string

    html_content = f"<h1>{problem_title}</h1>\n\n"

    for part in english_parts:
        clone = part

        # Remove "Copy" buttons
        for button_class in ["btn-copy", "btn-pre", "div-btn-copy"]:
            for button in clone.select(f".{button_class}"):
                button.decompose()

        # Remove Katex elements
        for katex in clone.select(".katex"):
            tex = katex.select_one(".katex-mathml annotation")
            if tex:
                text_node = soup.new_string(tex.text)
                katex.replace_with(text_node)

        html_content += str(clone) + "\n\n"

    return html_content


def html_to_markdown(html):
    # Simple conversion rules
    rules = [
        (r"<h3>(.*?)<\/h3>", r"\n### \1\n"),
        (r"<h2>(.*?)<\/h2>", r"\n## \1\n"),
        (r"<h1>(.*?)<\/h1>", r"\n# \1\n"),
        (r"<p>(.*?)<\/p>", r"\1\n"),
        (r"<ul>(.*?)<\/ul>", r"\1"),
        (r"<li>(.*?)<\/li>", r"- \1"),
        (r"<pre.*?>(.*?)<\/pre>", r"\n\n```\n\1\n```"),
        (r"<var>(.*?)<\/var>", r"`\1`"),
        (r"<div.*?>(.*?)<\/div>", r"\1"),
        (r"<span.*?>(.*?)<\/span>", r"\1"),
        (r"<section.*?>(.*?)<\/section>", r"\1"),
        (r"<hr>", r"---"),
        (r"<br>", r"\n"),
    ]

    markdown = html
    for regex, replacement in rules:
        markdown = re.sub(regex, replacement, markdown, flags=re.DOTALL)

    # Remove any remaining HTML tags
    markdown = re.sub(r"</?[^>]+(>|$)", "", markdown)

    return markdown.strip()


# extract python code from chatbot's response
def parse_code_from_response(response_message: str, prompt: str, model_name : str) -> str:
    lines = response_message.split("\n")

    solution_code = f'#This code is automatically generated using {model_name}. \
            github.com/Harui-i/spoil.py \n"""PROMPT: {prompt} \n Response : \n""" '  # Pythonの文字列結合は遅いけど競プロのコードの解答くらいなら問題ないだろ

    state: int = 0  # 0: before the code, 1: in the code, 2: after the code

    for line in lines:
        if state == 0:
            if line.startswith("```python"):  # 解答が始まったらコメントアウトを辞める
                state = 1
            line = "#" + line

        elif state == 1 and line.startswith("```"):
            line = "#" + line
            state = 2

        elif state == 2:
            line = "#" + line

        solution_code += line + "\n"

    return solution_code


async def make_problem_md_and_html(session, contest_id, problem_id, contest_dir) -> [str, str]:
    logger.info(f"Making problem statements' HTML and md files for {problem_id} ...")
    problem_suffix = problem_id.rsplit("_")[-1]
    problem_dir = os.path.join(contest_dir, problem_suffix)
    problem_url: str = f"https://atcoder.jp/contests/{contest_id}/tasks/{problem_id}"
    md_path = os.path.join(problem_dir, f"{problem_id}_en.md")
    html_path = os.path.join(problem_dir, f"{problem_id}_en.html")

    # make directory for the problem
    os.makedirs(problem_dir, exist_ok=True)

    if os.path.exists(md_path) and os.path.getsize(md_path) >= 100:
        logger.info(f"{md_path} already exists.")
        with open(md_path, "r", encoding="utf-8") as f:
            markdown = f.read()
    else:
        html = get_contest_problem_html(session, contest_id, problem_id)
        processed_html = extract_html(html)
        markdown = html_to_markdown(processed_html)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info(f"Markdown file :  {md_path} has been created.")

    if os.path.exists(html_path) and os.path.getsize(html_path) >= 100:
        logger.info(f"{html_path} already exists.")

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
    else:
        html = get_contest_problem_html(session, contest_id, problem_id)
        processed_html = extract_html(html)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(processed_html)
        logger.info(f"HTML file {html_path} has been created.")

    return markdown, html


async def fetch_solution(markdown_content, problem_id, contest_dir, model: MODEL, trickey=False):
    start_time = time()
    logger.info(f"fetching solution for {problem_id} ..., trickey={trickey}")

    prompt = f"Solve the following competition programming problem using Python.\n\
##Problem Statement: \n\n{markdown_content} \n \
Your response should follow the following 3-part-separeted format.\n \
1. A brief description of algorithm\n\
2. Code(Do NOT hard-code sample test cases in the code) \n\
3. PRAY(recommended phrase: May the code be AC!) \n\
"

    # 一度間違えた問題である場合はそう伝える。これが効果あるのかはわからないが、直感的には効果があると思われる。
    # 1.　一度間違えた問題は、同じ様なプロンプトを与えても間違える可能性が高そう。
    # 2.  間違えた解答を出したセッションで間違えたことを伝えても、誤答に引っ張られてしまいそう。
    # 3.  始めからトリッキーであると伝えると、不要な高速化などをして間違えてしまいそうだが一度間違えてから注意するように伝えると良い解答が出せるんじゃないか？

    if trickey:
        prompt += f"\n###Advice\nIn the past session, you have failed to solve this problem. So this Problem may be a little bit trickey. But I certainly believe you can solve this problem. Dont be nervous. Be Confident! \n\n"

    if model == MODEL.GPT4O:
        async with aiohttp.ClientSession() as session:
            apipost_time = time()
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant with a strong background in Python and competitive programming.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            ) as response:
                response_json = await response.json()
                logger.debug(f"{response_json=}")
                logger.info(
                    f"Open AI API has responded in {time() - apipost_time} seconds for {problem_id}"
                )

                response_message = response_json["choices"][0]["message"]["content"]
                solution_code = parse_code_from_response(response_message, prompt, 'gpt-4o')

                logger.debug(f"response against {problem_id} was {solution_code}")
                problem_dir = os.path.join(contest_dir, problem_id.rsplit("_")[-1])
                with open(os.path.join(problem_dir, get_solution_file_name(model)),"w",encoding="utf-8") as f:
                    f.write(solution_code)

            logger.info(f"Solution code by {model} for {problem_id} has been created.")

    elif model == MODEL.GEMINIPRO:
        async with aiohttp.ClientSession() as session:
            apipost_time = time()
            async with session.post(
                f"https://us-central1-aiplatform.googleapis.com/v1/projects/{GCP_PROJECT_ID}/locations/us-central1/publishers/google/models/gemini-1.5-pro:streamGenerateContent",
                headers={
                    "Authorization" : f"Bearer {GEMINI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json = {
                    "contents" : {
                        "role" : "user",
                        "parts" : {
                            "text" : prompt
                        }
                    },
                    "safety_settings" : {
                        "category" : "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold" : "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    "generation_config" : {
                        "temperature" : 1,
                        "top_p" : 0.95,
                        "top_k" : 64,
                        "max_output_tokens" : 8192,
                        "response_mime_type" : "text/plain"
                    } 
                },
                ) as response:
                    response_json = await response.json()
                    logger.debug(f"Gemini response : {response_json=}")
                    logger.info(
                        f"Gemini AI API has responded in {time() - apipost_time} seconds for {problem_id}"
                    )

                    response_message = response_json["candidates"][0]["content"]
                    solution_code = parse_code_from_response(response_message, prompt, "gemini1.5-pro")

                    logger.debug(f"response against {problem_id} by {MODEL} was {solution_code}")
                    problem_dir = os.path.join(contest_dir, problem_id.rsplit("_")[-1])
                    with open(
                        os.path.join(problem_dir, get_solution_file_name(model)),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(solution_code)

        logger.info(f"Solution code by {model} for {problem_id} has been created.")
    else:
        raise NotImplementedError(f"model {model} not supported")

async def process_problem(session, contest_id, problem_id, contest_dir, model : MODEL):
    logger.info(f"Processing {problem_id} ...")
    problem_suffix = problem_id.rsplit("_")[-1]
    problem_dir = os.path.join(contest_dir, problem_suffix)
    problem_url: str = f"https://atcoder.jp/contests/{contest_id}/tasks/{problem_id}"

    # make directory for the problem
    os.makedirs(problem_dir, exist_ok=True)

    # make markdown and html files
    markdown, html = await make_problem_md_and_html(session, contest_id, problem_id, contest_dir)

    # download test cases using online-judge tools
    result = subprocess.run(
        [f"oj", "d", problem_url], cwd=problem_dir, capture_output=True, text=True
    )
    logger.debug(f"the result of oj d for {problem_id} was : {result}")

    # test using online-judge tools
    test_success = False
    for attempt in range(2):  # Try up to 2 times (initial + 1 retry)

        # generate solution
        await fetch_solution(markdown, problem_id, contest_dir, model, trickey=attempt > 0)

        test_result = subprocess.run(
            [f"oj", "t", "-c", f"python3 {get_solution_file_name(model)}"],
            cwd=problem_dir,
            capture_output=True,
            text=True,
        )
        logger.debug(f"test result for {problem_id} by {model} was {test_result}")

        if test_result.returncode == 0:
            logger.info(f"Test for {problem_id} by {model} has PASSED!.")
            test_success = True
            break
        else:
            logger.warning(
                f"Test for {problem_id} by {model} has failed. Retrying... (Attempt {attempt + 1}/2)"
            )

    if test_success:
        # if submit process hasn't finished within 10 seconds, retry
        submit_success = False
        for attempt in range(3):  # Retry up to 3 times
            try:
                # -l 5078 : PyPy
                # -l 5055 : CPython
                # -y      : skip the confirmation
                logger.info(
                    f"Submitting {problem_dir}/{get_solution_file_name(model)} for {problem_id} ..."
                )
                
                
                submit_result = await asyncio.wait_for(
                    asyncio.to_thread(subprocess.run, ['oj', 'submit', f'{get_solution_file_name(model)}', '-l', '5078', '-y'], cwd=problem_dir, capture_output=True, text=True),
                    timeout=10
                ) 

                submit_result = "NOW TESTING"
                logger.info(f"Submission for {problem_id} has been made.")
                logger.debug(f"Submission result for {problem_id} by {model} was {submit_result}")
                submit_success = True
                break
            except asyncio.TimeoutError:
                logger.warning(
                    f"Submission for {problem_id} by {model} timed out. Retrying... (Attempt {attempt + 1}/3)"
                )

        if not submit_success:
            logger.error(f"Submission for {problem_id} failed after 3 attempts.")
    else:
        logger.warning(f"Test for {problem_id} has failed after 2 attempts.")

    return


async def main(contest_dir: str):
    contest_id = os.path.basename(contest_dir)
    session = get_session_with_cookies(OJ_COOKIE_JAR_PATH)

    # 問題IDのリストを生成
    problem_ids = [
        f"{contest_id}_a",
        f"{contest_id}_b",
        f"{contest_id}_c",
        f"{contest_id}_d",
        f"{contest_id}_e",
        f"{contest_id}_f",
        f"{contest_id}_g",
    ]

    tasks = []
    for problem_id in problem_ids:
        # process_problemをタスクとしてスケジュール

        task1 = asyncio.create_task(
            process_problem(session, contest_id, problem_id, contest_dir, MODEL.GPT4O)
        )

        await asyncio.sleep(0.75) # sleep for atcoder 

#        task2 = asyncio.create_task(
#            process_problem(session, contest_id, problem_id, contest_dir, MODEL.GPT4O)
#        )

        tasks.append(task1)
#        tasks.append(task2)

        logger.info(f"Task for {problem_id} has been scheduled.")
        # 0.75秒待機
        

    # 全てのタスクが完了するのを待つ
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    print(f"Start the program.")
    logger.info("Start the program.")
    parser = argparse.ArgumentParser(
        description="Fetch and convert AtCoder problem statement to Markdown."
    )
    parser.add_argument(
        "contest_dir", type=str, help="The directory path of the contest"
    )

    args = parser.parse_args()

    asyncio.run(main(args.contest_dir))
    # asyncio.run(main("abc355")) # for debugging
