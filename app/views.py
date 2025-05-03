from django.shortcuts import render
from rest_framework.generics import ListCreateAPIView
from rest_framework.response import Response
from rest_framework import status

from django.conf import settings
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# model = BertForSequenceClassification.from_pretrained('app/my_model')
model_name = "enoch10jason/spam-detector-model"
# tokenizer = BertTokenizer.from_pretrained('app/my_model')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()



class CheckSpam(ListCreateAPIView):
    def post(self, request, *args, **kwargs):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        message = request.data.get('message', '')
        if not message:
            return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Tokenize the input message
        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Move input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get prediction
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        # Return result
        if prediction == 0:
            return Response({'message': 'Spam'}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Not Spam'}, status=status.HTTP_200_OK)
