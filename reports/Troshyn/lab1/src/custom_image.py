import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "best.pth"

print("Device:", DEVICE)

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

model = CNN().to(DEVICE)

# загрузка обученной модели

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded:", MODEL_PATH)

transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = image.copy()

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    # Отключаем вычисление градиентов
    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        probability = probabilities[
            0,
            predicted_class
        ].item()

    print()
    print("Результат классификации")

    print(
        "Изображение:",
        image_path
    )

    print(
        "Предсказанный класс:",
        classes[predicted_class]
    )

    print(
        "Вероятность:",
        f"{probability * 100:.2f}%"
    )


    plt.figure(figsize=(5, 5))

    plt.imshow(
        original_image
    )

    plt.title(
        f"Pred: {classes[predicted_class]}\n"
        f"{probability * 100:.2f}%"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()

predict_image("cat.jpg")
