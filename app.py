# app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from transformers import pipeline
import torch
import os
import tempfile
import jiwer
import re # Import the regular expression library for cleaning

# 1. Initialize the application
app = FastAPI()

# 2. Load your fine-tuned ASR model
print("Loading your fine-tuned SpeakEase model...")
pipe = pipeline(
    "automatic-speech-recognition",
    model="./final-svarah-whisper-model",  # This path points to your local model
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
print("Your fine-tuned model was loaded successfully!")


# --- NEW HELPER FUNCTION ---
def clean_text(text: str) -> str:
    """
    Cleans text by lowercasing, removing all punctuation and special characters,
    and normalizing whitespace.
    """
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word/non-space characters
    text = re.sub(r'\s+', ' ', text)       # Collapse multiple spaces into one
    return text
# -------------------------


# 3. Create an endpoint to serve the HTML webpage
@app.get("/")
async def serve_page():
    """Serves the main index.html file."""
    return FileResponse('index.html')


# 4. Create the endpoint to handle the full analysis
@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...), correct_text: str = Form(...)):
    """
    Accepts an audio file and the correct text, performs a full analysis,
    and returns a detailed report.
    """
    # Use a temporary file to save the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_path:
        temp_file_name = temp_audio_path.name
        temp_audio_path.write(await file.read())

    try:
        # Run the transcription pipeline to get detailed output
        output = pipe(temp_file_name, return_timestamps="word")
        word_chunks = output.get("chunks", [])
        recognized_text = output.get("text", "").strip()

        # --- Calculate Detailed Scores ---
        
        # --- UPDATED PART ---
        # Create cleaned versions of the text specifically for scoring
        correct_text_clean = clean_text(correct_text)
        recognized_text_clean = clean_text(recognized_text)
        # --------------------

        # Use the CLEANED text for accurate Word Error Rate (WER) calculation
        wer_output = jiwer.process_words(correct_text_clean, recognized_text_clean)
        wer = wer_output.wer
        accuracy_score = (1 - wer) * 100

        # Calculate Words Per Minute (WPM) based on the ORIGINAL correct text
        duration_seconds = word_chunks[-1]['timestamp'][1] if word_chunks and word_chunks[-1].get('timestamp') and word_chunks[-1]['timestamp'][1] is not None else 0
        num_words = len(correct_text.split())
        wpm = (num_words / duration_seconds) * 60 if duration_seconds > 0 else 0

        # Assemble the scores into a dictionary
        scores = {
            "accuracy_score": f"{accuracy_score:.2f}%",
            "word_error_rate": f"{wer:.2f}",
            "words_per_minute": f"{wpm:.2f}"
        }

        # Return the complete report, including the ORIGINAL transcription
        return {
            "metrics": scores,
            "full_transcription": recognized_text, # Return the original, uncleaned text for display
            "word_analysis": word_chunks
        }
    finally:
        # Clean up and delete the temporary file
        os.remove(temp_file_name)