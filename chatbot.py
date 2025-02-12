from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import json

# Load trained model and tokenizer
model_path = r"C:/Users/Admin/Desktop/Research/Mood Chatbot/trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)


# Set model to evaluation mode
model.eval()
print("Model Loaded Successfully!")

# Load mood-based recommendations from JSONL file
recommendations_file = "C:/Users/Admin/Desktop/Research/Mood Chatbot/mood_recommendations.jsonl"
mood_data = {}

with open(recommendations_file, "r", encoding="utf-8") as file:
    for line in file:
        mood_entry = json.loads(line.strip())
        mood_data[mood_entry["mood"].lower()] = mood_entry

def get_predefined_recommendations(mood):
    """Fetch recommendations from JSONL file based on the user's mood."""
    mood = mood.lower()
    return mood_data.get(mood, None)

def chatbot_response(user_input):
    """Generate chatbot response using GPT-2 for mood-based recommendations."""
    predefined_recommendations = get_predefined_recommendations(user_input)
    
    if predefined_recommendations:
        return predefined_recommendations  # Return predefined recommendations if available

    # If no predefined recommendation, use GPT-2 to generate response
    input_text = f"Mood: {user_input} | Recommendations: "
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=150, 
            temperature=0.7,  # Adjust for randomness
            top_p=0.9,  # Sampling strategy
            repetition_penalty=1.2
        )

    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return {"mood": user_input, "generated_recommendations": response}

# Example usage:
if __name__ == "__main__":
    while True:
        user_mood = input("Enter your mood (or type 'exit' to quit): ").strip()
        if user_mood.lower() == "exit":
            break
        
        recommendations = chatbot_response(user_mood)
        print("\nRecommendations:", json.dumps(recommendations, indent=4))
