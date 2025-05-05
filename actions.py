import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Union

from unstructured.partition.api import partition_via_api
from unstructured.chunking.title import chunk_by_title

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === TOOLS ===

def process_pdf(filename: str) -> List[str]:
    """Extract text chunks from PDF using Unstructured API"""
    try:
        api_key = os.getenv("UNSTRUCTURED_API_KEY")
        elements = partition_via_api(
            filename=filename,
            api_key=api_key,
            strategy="hi_res",
            ocr_language=['eng'],
            extract_image_block_types=["Table"]
        )
        chunks = chunk_by_title(elements)
        return [str(chunk) for chunk in chunks]
    except Exception as e:
        print(f"PDF Processing Error: {e}")
        return []

def generate_mcqs(text_chunk: str) -> Union[Dict, None]:
    """Generate a single MCQ from a text chunk"""
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"""
Generate 1 multiple choice question from this text: {text_chunk}
Format the output in JSON with: question, options (list of 4), correct_answer, and explanation.
"""}
            ],
            temperature=0.3
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"MCQ Generation Error: {e}")
        return None

# === AGENT LOOP ===

def run_agent(filename: str, max_turns: int = 5) -> List[Dict]:
    messages = [
        {
            "role": "system",
            "content": """You are a ReAct-style AI assistant. Follow this loop:

1. Thought: Decide what action to take.
2. Action: Use a tool from the available ones.
3. PAUSE: After every action, wait for Action_Response.
4. Action_Response: Receive the result of the tool.
Repeat this until you're ready to produce the final Answer.

Your tools:
- process_pdf(filename: str): extracts chunks from a PDF
- generate_mcqs(text_chunk: str): generates an MCQ from a text chunk
"""
        },
        {
            "role": "user",
            "content": f"Generate MCQs from the PDF file: {filename}"
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "process_pdf",
                "description": "Extracts text chunks from a PDF.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"}
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_mcqs",
                "description": "Generates MCQs from a given text chunk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text_chunk": {"type": "string"}
                    },
                    "required": ["text_chunk"]
                }
            }
        }
    ]

    all_mcqs = []
    text_chunks = []

    for _ in range(max_turns):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response = completion.choices[0].message
        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if tool_name == "process_pdf":
                    text_chunks = process_pdf(**args)
                    tool_output = text_chunks

                elif tool_name == "generate_mcqs":
                    tool_output = generate_mcqs(**args)
                    if tool_output:
                        all_mcqs.append(tool_output)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(tool_output)
                })

        else:
            # Final answer (not a tool call)
            if "Answer:" in response.content:
                try:
                    answer_json = response.content.replace("Answer:", "").strip()
                    final = json.loads(answer_json)
                    return final if isinstance(final, list) else [final]
                except json.JSONDecodeError:
                    return all_mcqs

    return all_mcqs


