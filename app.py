from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for local testing)
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Home route for serving the index.html file
@app.get("/")
async def get_home():
    return FileResponse("static/index.html")

# Define request model
class MoodRequest(BaseModel):
    mood: str

# Load recommendations from JSONL file
def load_mood_recommendations(file_path):
    mood_recommendations = {}
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return mood_recommendations

    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            mood_data = json.loads(line.strip())
            mood_recommendations[mood_data['mood'].lower()] = mood_data
    return mood_recommendations

# Update this path based on your system
jsonl_file_path = os.path.join(os.getcwd(), "C:/Users/Admin/Desktop/Research/Mood Chatbot/mood_recommendations.jsonl")

mood_recommendations = load_mood_recommendations(jsonl_file_path)

@app.post("/chatbot")
async def chatbot_response(request: MoodRequest):
    user_mood = request.mood.lower()

    # Fetch mood recommendations
    mood_data = mood_recommendations.get(user_mood, {
        "song_recommendations": [],
        "activity_ideas": [],
        "book_suggestions": [],
        "tv_show_recommendations": []
    })

    return {
        "mood": user_mood,
        "suggestions": {
            "song_recommendations": mood_data["song_recommendations"],
            "activity_ideas": mood_data["activity_ideas"],
            "book_suggestions": mood_data["book_suggestions"],
            "tv_show_recommendations": mood_data["tv_show_recommendations"]
        }
    }
