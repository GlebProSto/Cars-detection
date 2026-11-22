from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torch import nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "convnext_cars_196.pth"
TEST_DIR = BASE_DIR / "cars for testing"


checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
class_names = checkpoint["class_names"]

model = models.convnext_tiny(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

correct = 0
tested = 0
images = sorted(TEST_DIR.iterdir())

for image_path in images:
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    tested += 1
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.softmax(output, dim=1)
        confidence, index = torch.max(probability, dim=1)

    predicted = class_names[index.item()]
    expected = image_path.stem
    is_correct = predicted == expected

    if is_correct:
        correct += 1

    print(f"Фото: {image_path.name}")
    print(f"Ожидалось: {expected}")
    print(f"Предсказано: {predicted}")
    print(f"Уверенность: {confidence.item() * 100:.2f}%")
    print("Верно" if is_correct else "Неверно")
    print()

print(f"Результат: {correct} из {tested} фото правильных")