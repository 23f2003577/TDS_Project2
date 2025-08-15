import re
import traceback
from google import genai
import os
from dotenv import load_dotenv
import json
import subprocess
import sys
import tempfile
from app_trial.llm_controller import infer_expected_format  # <-- import for fallback

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def detect_required_files(code: str):
    # Detect file usage patterns
    file_patterns = re.findall(r'read_csv\(\s*[\'"](.+?\.csv)[\'"]', code)
    file_patterns += re.findall(r'open\(\s*[\'"](.+?)[\'"]', code)
    return set(file_patterns)


def sanitize_result(result):
    try:
        json.dumps(result)  # Try serializing
        return result
    except TypeError:
        return str(result)  # Fallback: convert to string


async def generate_code(task: list, notes: str) -> str:
    prompt = f"""You are a Python data analyst. Generate Python code to perform the given task. 
    Add inline dependencies for any imports that are required.
    Do not add explanations. Output only Python code.
    All files and data you need to read are in the 'uploads' directory. Use them efficiently.
    If you scrape data, save it to the 'uploads' directory with a meaningful name.
    Use the given additional notes and instructions for context.
    Instructions:
    - Use Playwright (async) for any web scraping tasks. Do not use requests or BeautifulSoup.
    - You can use libraries like pandas, numpy, duckdb, matplotlib, Playwright, etc.
    - You MUST assign the final answer to a variable called 'result_data'.
    - Do not print or return anything.

    Examples:
    result_data = 42
    result_data = "Maharashtra"
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=[prompt, task, notes]
    )

    code_block = response.text.strip()
    code_match = re.sub(r"^```python\s*|\s*```$", "", code_block, flags=re.DOTALL)
    if not code_match:
        raise ValueError("No valid Python code found in response")
    return code_match


def execute_code(code: str) -> (bool, dict): # type: ignore
    try:
        required_files = detect_required_files(code)
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return False, f"Missing required file(s): {missing_files}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(
                code +
                "\n\n"
                "import json, sys, numpy, json\n"
                "def convert_np(obj):\n"
                "    if isinstance(obj, np.integer):\n"
                "        return int(obj)\n"
                "    elif isinstance(obj, np.floating):\n"
                "        return float(obj)\n"
                "    elif isinstance(obj, np.ndarray):\n"
                "        return obj.tolist()\n"
                "    raise TypeError(f\"Type {type(obj)} not serializable\")\n"
                "with open(sys.argv[1], 'w') as f:\n"
                "    json.dumps(result_data, default=convert_np, f)\n"
            )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_file:
            out_path = out_file.name

        result = subprocess.run(
            [sys.executable, tmp_path, out_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return False, f"STDERR:\n{result.stderr}"

        with open(out_path, "r") as f:
            try:
                result_data = json.load(f)
            except json.JSONDecodeError:
                result_data = None

        os.remove(tmp_path)
        os.remove(out_path)

        if result_data is not None:
            return True, result_data
        else:
            return False, {"message": "result_data not defined."}

    except Exception as e:
        return False, traceback.format_exc()


async def fix_code(task: str, faulty_code: str, error_message: str) -> str:
    prompt = f"""The following Python code was generated to perform the task: {task}
Code:
{faulty_code}

But it failed with the following error: {error_message}

Debug the code to make sure it runs successfully and return only the corrected Python code.
Add inline dependencies for any imports that are required.
Use the uploads directory for any files you need to read or write.
Do not add explanations or comments.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt]
        )
        code_block = response.text.strip()
        code_match = re.sub(r"^```python\s*|\s*```$", "", code_block, flags=re.DOTALL)
        if not code_match:
            raise ValueError("No valid Python code found in response")
        return code_match
    except Exception as e:
        print(f"Gemini Fixing Error: {e}")
        return faulty_code


async def process_task(task: list, notes: list, max_retries=2) -> dict:
    """
    Runs the full cycle of generating, executing, retrying, and falling back to expected format.
    """
    print(f"Processing task: {task}")
    try:
        code = await generate_code(task, notes)
    except Exception as e:
        return {"task": task, "error": f"Code generation failed: {str(e)}"}

    print(f"Generated Code:\n{code}")

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1} executing...")
        success, result = execute_code(code)

        if success:
            print(f"Execution successful: {result}")
            result = sanitize_result(result)
            expected_format = await infer_expected_format(task, result)
            return expected_format if expected_format else result

        print(f"Execution failed: {result}")
        print(f"Attempt {attempt + 1} failed. Trying to fix...")
        code = await fix_code(task, code, result)

    print("Task failed after multiple attempts. Returning fallback format...")

    # Fallback: infer expected format and return placeholders
    expected_format = await infer_expected_format(task)
    return expected_format if expected_format else {"error": "Failed to produce result"}