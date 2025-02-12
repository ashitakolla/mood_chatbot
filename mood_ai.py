import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset('json', data_files='C:/Users/Admin/Desktop/Research/Mood Chatbot/mood_recommendations.jsonl')

print("Dataset Loaded Successfully!")
print(dataset)

# Initialize GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')  # Use a medium-sized model
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# Set pad token as eos_token for GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Concatenating mood with all recommendation fields
    input_texts = [
        f"Mood: {m} | Songs: {s} | Activities: {a} | Books: {b} | TV Shows: {tv}"
        for m, s, a, b, tv in zip(
            examples['mood'], 
            examples['song_recommendations'], 
            examples['activity_ideas'], 
            examples['book_suggestions'], 
            examples['tv_show_recommendations']
        )
    ]
    
    tokenized_inputs = tokenizer(input_texts, padding="max_length", truncation=True, max_length=256)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # GPT-2 needs labels
    return tokenized_inputs


# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset (train 80%, test 20%)
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset, test_dataset = train_test_split["train"], train_test_split["test"]

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  # Reduce to avoid overfitting
    per_device_train_batch_size=16,  # Increase for efficiency
    per_device_eval_batch_size=16,
    learning_rate=3e-5,  # Slightly lower learning rate
    warmup_ratio=0.1,  # Dynamic warmup steps
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,  # Mixed precision for speed
    gradient_accumulation_steps=2,  # Helps with memory limits
    report_to="none"  # Avoids unnecessary logging to W&B
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Start training
trainer.train()

# Save final trained model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

# Re-save the trained model and tokenizer
model.save_pretrained("C:/Users/Admin/Desktop/Research/Mood Chatbot/trained_model")
tokenizer.save_pretrained("C:/Users/Admin/Desktop/Research/Mood Chatbot/trained_model")

# Evaluate model
eval_results = trainer.evaluate()

# Print evaluation results
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"{key}: {value}")

print("\nTraining Completed! Model Saved Successfully.")
