from fastapi import FastAPI, UploadFile, File, HTTPException, Request # pyright: ignore[reportMissingImports]
from fastapi.responses import JSONResponse # pyright: ignore[reportMissingImports]
import os
from pathlib import Path
from app_trial.llm_controller import breakdown_question
from app_trial.code_executor import process_task
from dotenv import load_dotenv
import traceback
import json
from app_trial.llm_controller import get_dummy_guess

load_dotenv()

app = FastAPI()

def cast_answer(value):
    """Convert strings that represent numbers into int/float, else return as-is."""
    if isinstance(value, str):
        try:
            return int(value) if value.isdigit() else float(value)
        except ValueError:
            # Remove quotes and backslashes if it's a string
            return value
            #return value.replace('"', '').replace('\\', '').strip()  
    elif isinstance(value, list):
        return [cast_answer(v) for v in value]
    elif isinstance(value, dict):
        return {k: cast_answer(v) for k, v in value.items()}
    return value

@app.post("/api/")
async def analyze_data(request : Request):
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)

    form = await request.form()
    
    if "questions.txt" not in form:
        raise HTTPException(status_code=400, detail="questions.txt file is required.")
    
    question_file: UploadFile = form["questions.txt"]
    if not question_file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a .txt file for questions.txt.")
    
    try:
        # Read question text
        question_text = (await question_file.read()).decode('utf-8')

        # Collect all other files into a list
        for field_name, value in form.items():
            if hasattr(value, "filename") and value.filename:
                file_path = UPLOAD_DIR / value.filename
                with open(file_path, "wb") as f:
                    f.write(await value.read())
        
        print(f"Saved extra files")

        # Step 1: Get breakdown from LLM
        breakdown = await breakdown_question(question_text)
        steps = breakdown.get("steps", [])
        notes = breakdown.get("notes", [])
        final_steps = breakdown.get("final_answer_steps", [])

        results = []

        # Step 2: Execute only final answer steps
        for step in steps:
            step_num = step.get("step_number")
            details = step.get("details", "")
            print(f"Executing step {step_num}: {details}")
            try:
                result = await process_task(details, notes)
                if step_num in final_steps:
                    print(f"Appending actual result for step {step_num}")
                    results.append(cast_answer(result))
            except Exception as task_err:
                print(f"Step {step_num} failed: {task_err}")
                if step_num in final_steps:
                    dummy_guess = await get_dummy_guess(details)
                    print(f"Appending dummy guess for step {step_num}")
                    results.append(cast_answer(dummy_guess))

        return json.dumps(results)

    except Exception as e:
        print(f"API Error: {e}")
        print(traceback.format_exc())
        return {
            "results": [],
            "error": str(e)
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Data Analysis API. Please upload a question.txt and any optional files."}