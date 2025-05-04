from django.shortcuts import render
from rest_framework.generics import ListCreateAPIView
from rest_framework.response import Response
from rest_framework import status
import os
import gdown
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model():
    model_dir = "models"
    if os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print("Model already exists. Skipping download.")
        return model_dir
    os.makedirs(model_dir, exist_ok=True)
    file_id = "1ZuOwayvVRt4Cp6tfbpNIVIWettaCge74"
    url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading model...")
    gdown.download(url, os.path.join(model_dir, "pytorch_model.bin"), quiet=False)
    print("Model downloaded successfully.")
    return model_dir

model_dir = download_model()

tokenizer = AutoTokenizer.from_pretrained("enoch10jason/spam-detector-model")
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class CheckSpam(ListCreateAPIView):
    def post(self, request, *args, **kwargs):
        message = request.data.get('message', '')
        if not message:
            return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=-1).item()

        if prediction == 0:
            return Response({'message': 'Spam'}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Not Spam'}, status=status.HTTP_200_OK)
