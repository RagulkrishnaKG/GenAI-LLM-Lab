from transformers import pipeline

text_generator = pipeline("text-generation", model="gpt2")
prompt = "Write a short description of a smart study assistant:"

result = text_generator(
    prompt,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    num_return_sequences=1,
)

print("=== Generated Content ===")
print(result[0]["generated_text"])
