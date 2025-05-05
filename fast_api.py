from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
import uuid
import random
from actions import run_agent
from db_handler import SupabaseHandler 

app = FastAPI()
db = SupabaseHandler()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files - this should come after API routes
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    with open("frontend/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF file upload"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate unique filename
        material_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{material_id}.pdf")
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "material_id": material_id,
            "file_path": file_path
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_mcqs/{material_id}")
async def generate_mcqs_endpoint(material_id: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{material_id}.pdf")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Use ReAct agent to generate MCQs from the PDF
        all_mcqs = run_agent(file_path)

        if not all_mcqs:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate any valid MCQs"
            )

        for mcq in all_mcqs:
            db.store_mcqs(mcq, material_id)

        return {
            "status": "success",
            "material_id": material_id,
            "num_questions": len(all_mcqs),
            "mcqs": all_mcqs
        }

    except Exception as e:
        print(f"Error generating MCQs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit_answer")
async def submit_answer(
    material_id: str,
    question_id: int,
    user_id: str,
    answer: str
):
    try:
        # Get MCQs for this material
        mcqs = db.get_mcqs(material_id)
        if not mcqs:
            raise HTTPException(404, "MCQs not found")
            
        # Validate answer
        question = mcqs[question_id]
        is_correct = answer == question["correct_answer"]
        
        # Store user response
        db.store_user_response(user_id, question_id, answer, is_correct)
        
        return {
            "correct": is_correct,
            "explanation": question["explanation"],
            "correct_answer": question["correct_answer"]
        }
    
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/user_stats/{user_id}")
async def get_user_stats(user_id: str):
    try:
        stats = db.get_user_stats(user_id)
        return stats
    
    except Exception as e:
        raise HTTPException(500, str(e))
