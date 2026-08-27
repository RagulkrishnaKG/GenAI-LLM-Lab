from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Use a local image named sample.jpg in this folder.
image = Image.open("sample.jpg").convert("RGB")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = processor(images=image, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(output_ids[0], skip_special_tokens=True)

print("Generated Caption:", caption)

# Visual question answering with the same image.
question = "What is visible in the image?"
inputs = processor(images=image, text=question, return_tensors="pt")
output_ids = model.generate(**inputs, max_new_tokens=20)
answer = processor.decode(output_ids[0], skip_special_tokens=True)

print("Question:", question)
print("Answer:", answer)
