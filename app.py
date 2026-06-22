import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")

CLASS_NAMES = ["cat", "dog"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    )
    
    state_dict = torch.load("resnet50_best_model.pth", map_location=device)
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "fc.1.weight":
            new_state_dict["fc.0.weight"] = value
        elif key == "fc.1.bias":
            new_state_dict["fc.0.bias"] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("Cat vs Dog Image Classification")
st.write("Upload an image and the model will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item() * 100

    predicted_class = CLASS_NAMES[predicted_index]

    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")