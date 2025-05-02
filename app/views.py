from django.shortcuts import render
from rest_framework.generics import ListCreateAPIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the trained model and fitted vectorizer
model = joblib.load('app/spam_model.pkl')
feature_extraction = joblib.load('app/feature_extraction.pkl')  # Assuming you saved it after fitting

class CheckSpam(ListCreateAPIView):
    def post(self, request, *args, **kwargs):
        message = request.data.get('message', '')
        
        if not message:
            return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure message is passed as a list
        input_data_features = feature_extraction.transform([message])
        
        prediction = model.predict(input_data_features)
        
        if prediction[0] == 0:
            return Response({'message': 'Spam'}, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'Not Spam'}, status=status.HTTP_200_OK)
