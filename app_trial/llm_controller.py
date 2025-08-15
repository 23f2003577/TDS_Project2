import os
from google import genai
from dotenv import load_dotenv
import re
import json

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_json_from_response(response_text: str):
    """Extract JSON from LLM text output, removing code fences if present."""
    try:
        json_str = re.sub(r"^```json\s*|\s*```$", "", response_text.strip(), flags=re.DOTALL)
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def infer_final_answer_steps(steps):
    """Infer which steps are final answer steps if LLM misses or mislabels them."""
    final_keywords = {"answer", "calculate", "return", "output", "predict", "determine", "find", "result", "identify", "generate"}
    non_final_keywords = {"scrape", "load", "preprocess", "clean", "fetch", "download", "prepare"}

    inferred = []
    for step in steps:
        title = step.get("title", "").lower()
        details_text = " ".join(step.get("details", [])).lower()

        if any(word in title or word in details_text for word in final_keywords) and not any(
            word in title or word in details_text for word in non_final_keywords
        ):
            inferred.append(step.get("step_number"))

    return inferred

async def infer_expected_format(question: str, result=None):
    """
    Reads questions.txt content and returns a JSON 'template' 
    describing the expected output format.
    """
    prompt = f"""
Read the following task description and figure out the structure of the expected output.
If the result variable is not None, use it's value to infer the expected format.
Rules:
- Output ONLY valid JSON that serves as a TEMPLATE for the expected format.
- Do NOT include commentary, markdown fences, or explanations.
- The JSON should be shaped exactly as the final answer is expected to be, 
  but you can use placeholder values (e.g., "string", 123, [], {{}}).
- Keep it minimal but structurally correct.

Task:
{question}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt]
        )
        return extract_json_from_response(response.text)
    except Exception as e:
        print(f"Expected format inference failed: {e}")
        return {}
    
async def breakdown_question(question: str):
    prompt = '''You are a data analyst agent. Your job is to break down the given task into smaller, clear, programmable steps.

Rules:
1. Output only valid JSON. Do not include Markdown code fences, commentary, or any text outside the JSON object.  
2. Do not solve the task, only break it down into steps.
3. Retain all important details such as dataset names, URLs, images, column names, and exact variables.
4. If the task involves web scraping, always recommend Playwright (async) instead of requests or synchronous BeautifulSoup.
5. The "steps" array must contain each step with:
   - step_number: integer starting from 1
   - title: short descriptive title
   - details: array of specific detailed actions
6. Add a field "final_answer_steps": an array of step_number values that correspond to final user-facing answers only — exclude intermediate scraping, preprocessing, or setup steps.
7. Use this exact JSON structure:

{
  "task": [
    "Short bullet points summarizing the full approach in order."
  ],
  "steps": [
    {
      "step_number": 1,
      "title": "Short descriptive title",
      "details": [
        "Specific, detailed action 1",
        "Specific, detailed action 2"
      ]
    }
  ],
  "notes": [
    "Extra guidance or warnings if necessary."
  ],
  "final_answer_steps": [1, 3]
}

Now, break down the following task:
[TASK GOES HERE]'''

    try:
        # Send prompt and question to LLM
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt, question]
        )
        content = response.text

        # Try parsing JSON
        parsed = extract_json_from_response(content)
        if not parsed:
            raise ValueError("No valid JSON found in LLM response.")

        # Safety net: Ensure final_answer_steps is present & valid
        if "final_answer_steps" not in parsed or not parsed["final_answer_steps"]:
            parsed["final_answer_steps"] = infer_final_answer_steps(parsed.get("steps", []))

        with open("task_broken.json", "w") as f:
            json.dump(parsed, f, indent=2)

        return parsed

    except Exception as e:
        print(f"API Error: {e}")
        return {
            "task": [],
            "steps": [],
            "notes": [],
            "final_answer_steps": []
        }
    
async def get_dummy_guess(task_details: str):
    """
    Ask the LLM to make a quick guess for a failed step without explanations.
    Returns only the guessed value.
    """
    prompt = f'''This task failed to execute successfully:
    {task_details}

    Make your best guess of the answer.
    Rules:
    - Do not explain your reasoning.
    - Return only the guessed value.
    - If the answer is numeric, return it as a number, not as a string.
    - If the answer is a string, return it as a plain string.
    - Do not add any extra quotes or backslashes or Markdown of any kind.

    Examples:
    42
    "Maharashtra"
    3.14
    True
    '''

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[prompt]
        )
        guess_text = response.text.strip()

        # Try to parse if it's valid JSON, else return raw text
        try:
            parsed_guess = json.loads(guess_text)
            return parsed_guess
        except json.JSONDecodeError:
            return guess_text

    except Exception as e:
        print(f"Dummy guess generation error: {e}")
        return None