import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ArabicCharNet(nn.Module):
    def __init__(self, num_classes):
        super(ArabicCharNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArabicCharNet(num_classes=28).to(device)
model.load_state_dict(torch.load('Arabic_OCR_PyTorch.pth', map_location=device))
model.eval()

arabic_chars = 'أبتثجحخدذرزسشصضطظعغفقكلمنهوي'
arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                    'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                    'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ImageData(BaseModel):
    image: str

@app.post("/classify")
async def classify_image(image_data: ImageData):
    try:
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data.image.split(',')[1])
        
        # Log the image size
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Received image size: {image.size}")
        
        # Save the received image for inspection
        image.save("received_image.png")

        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")
        print(f"Tensor min: {image_tensor.min()}, max: {image_tensor.max()}")

        # Perform the classification
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top3_prob, top3_catid = torch.topk(probabilities, 3)

        results = []
        for i in range(3):
            char_index = top3_catid[0][i].item()
            results.append({
                "arabic_char": arabic_chars[char_index],
                "english_name": arabic_characters[char_index],
                "confidence": f"{top3_prob[0][i].item()*100:.2f}%"
            })

        return results
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)