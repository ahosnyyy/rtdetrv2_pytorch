import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Load image from URL
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Load model and processor
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r18vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")

# Preprocess image
inputs = image_processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Post-process detections
results = image_processor.post_process_object_detection(
    outputs,
    target_sizes=torch.tensor([(image.height, image.width)]),
    threshold=0.5
)

# Draw results on the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for result in results:
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        label_text = f"{model.config.id2label[label]}: {score:.2f}"

        # Draw box and label
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), label_text, fill="red", font=font)

        print(label_text, box)

# Save or show the image
image.save("output.jpg")
print("Saved output image to output.jpg")
