import base64
import io
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
logger.info(f"Using device: {device}")

model = ArabicCharNet(num_classes=28).to(device)
model.load_state_dict(torch.load('Arabic_OCR_PyTorch.pth', map_location=device))
model.eval()

logger.info("Model loaded successfully")
logger.debug(f"Model architecture: {model}")

arabic_chars = 'أبتثجحخدذرزسشصضطظعغفقكلمنهوي'
arabic_characters = ['alef', 'beh', 'teh', 'theh', 'jeem', 'hah', 'khah', 'dal', 'thal',
                     'reh', 'zain', 'seen', 'sheen', 'sad', 'dad', 'tah', 'zah', 'ain',
                     'ghain', 'feh', 'qaf', 'kaf', 'lam', 'meem', 'noon', 'heh', 'waw', 'yeh']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
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
        
        # Open the image and convert to grayscale
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        logger.info(f"Received image size: {image.size}")
        
        # Save the received image for debugging
        image.save('received_image.png')

        # Resize image to 32x32
        resized_image = image.resize((32, 32))
        logger.info(f"Resized image size: {resized_image.size}")

        # Save the resized image for debugging
        resized_image.save('resized_image.png')

        # Check if the image is empty (all pixels are black or very close to black)
        bbox = ImageChops.difference(image, Image.new('L', image.size, 0)).getbbox()
        if bbox is None:
            logger.warning("Empty canvas detected")
            return []

        # Calculate the percentage of non-black pixels
        img_array = np.array(image)
        non_black_pixels = np.sum(img_array > 10)  # Threshold of 10 to account for slight variations
        total_pixels = img_array.size
        non_black_percentage = (non_black_pixels / total_pixels) * 100

        logger.info(f"Percentage of non-black pixels: {non_black_percentage:.2f}%")

        # If less than 1% of pixels are non-black, consider it an empty canvas
        if non_black_percentage < 1:
            logger.warning("Nearly empty canvas detected")
            return []

        # Preprocess the image
        image_tensor = transform(resized_image).unsqueeze(0).to(device)
        logger.info(f"Preprocessed tensor shape: {image_tensor.shape}")
        logger.info(f"Tensor min: {image_tensor.min().item()}, max: {image_tensor.max().item()}, mean: {image_tensor.mean().item()}")

        # Perform the classification
        with torch.no_grad():
            output = model(image_tensor)
            logger.debug(f"Raw model output: {output}")

            probabilities = torch.nn.functional.softmax(output, dim=1)
            logger.debug(f"Softmax probabilities: {probabilities}")

            top3_prob, top3_catid = torch.topk(probabilities, 3)
            logger.debug(f"Top 3 category IDs: {top3_catid}")
            logger.debug(f"Top 3 probabilities: {top3_prob}")

        results = []
        for i in range(3):
            char_index = top3_catid[0][i].item()
            results.append({
                "arabic_char": arabic_chars[char_index],
                "english_name": arabic_characters[char_index],
                "confidence": f"{top3_prob[0][i].item()*100:.2f}%"
            })

        logger.info(f"Classification results: {results}")
        return results
    except Exception as e:
        logger.exception(f"Error in classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)