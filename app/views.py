from django.shortcuts import render
from rest_framework.generics import ListCreateAPIView
from rest_framework.response import Response
from rest_framework import status
import torch
from transformers import BertForSequenceClassification, BertTokenizer


model = BertForSequenceClassification.from_pretrained('app/my_model')
tokenizer = BertTokenizer.from_pretrained('app/my_model')
model.eval()

class CheckSpam(ListCreateAPIView):
    def post(self, request, *args, **kwargs):
        message = request.data.get('message', '')
        
        if not message:
            return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

     
        inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True, max_length=512)

       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        model.to(device)

       
        with torch.no_grad():
            outputs = model(**inputs)
        
        prediction = torch.argmax(outputs.logits, dim=-1).item()

        if prediction == 0:
            return Response({'message': 'Spam'}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Not Spam'}, status=status.HTTP_200_OK)
