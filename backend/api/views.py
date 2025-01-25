from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load('./backend/models/svm_model.pkl')
vectorizer = joblib.load('./backend/models/vectorizer.pkl')

class EmotionPredictor(APIView):
    def post(self, request):
        try:
            text = request.data.get("text", "")
            if not text:
                return Response({"error": "Text is required"}, status=status.HTTP_400_BAD_REQUEST)

            # Transform the input text
            transformed_text = vectorizer.transform([text]).toarray()

            # Predict emotion
            prediction = model.predict(transformed_text)
            return Response({"emotion": prediction[0]}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)