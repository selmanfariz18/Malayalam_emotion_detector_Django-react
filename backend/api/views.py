import os
import json
import joblib
import numpy as np
import logging
from django.http import JsonResponse
from django.conf import settings
from rest_framework.views import APIView
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from django.contrib.auth.models import User
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate, login



class Signup(APIView):
    authentication_classes = [] 
    permission_classes = [AllowAny]  

    def post(self, request):
        data = request.data

        # Extract fields from the request
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')


        # Validate required fields
        required_fields = ['username', 'password', 'email']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return Response(
                {'error': f"Missing fields: {', '.join(missing_fields)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            user = User.objects.create_user(
                username=username,
                password=password,
                email=email
            )
            return Response({'message': 'Account created successfully'}, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            return Response({'error': f'Error creating user: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class Signin(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        data = request.data
        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return Response({'error': 'Both fields required'}, status=400)

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            # Explicitly save session to ensure cookie is set
            request.session.save()
            return Response({'message': 'Login successful'}, status=200)
        else:
            return Response({'error': 'Invalid credentials'}, status=401)

        


logger = logging.getLogger(__name__)

class PredictEmotion(APIView):
    bert_model = None
    bert_tokenizer = None
    bert_label_mapping = None

    def post(self, request):
        text = request.data.get('text', '').strip()
        model_type = request.data.get('model', '').lower()

        if not text or not model_type:
            return JsonResponse(
                {'error': 'Both text and model parameters are required'},
                status=400
            )

        try:
            model_handlers = {
                'countvectorizer+svm': self.svm_countvectorizer_predict,
                'tf-idf+svm': self.svm_tfidf_predict,
                'countvectorizer+naivebayes': self.nb_countvectorizer_predict,
                'tf-idf+naivebayes': self.nb_tfidf_predict,
                'bert': self.bert_predict,
                'bert+svm': self.bert_svm_predict,
                'bert+naivebayes': self.bert_naivebayes_predict
            }

            handler = model_handlers.get(model_type)
            if not handler:
                return JsonResponse(
                    {'error': 'Invalid model specified'},
                    status=400
                )
                
            return handler(text)

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return JsonResponse(
                {'error': f"Prediction failed: {str(e)}"},
                status=500
            )

    # Helper methods
    def _load_components(self, model_dir, model_file, vectorizer_file=None):
        model_path = os.path.join(settings.BASE_DIR, 'api', 'models', model_dir, model_file)
        components = {'model': joblib.load(model_path)}
        
        if vectorizer_file:
            vectorizer_path = os.path.join(settings.BASE_DIR, 'api', 'models', model_dir, vectorizer_file)
            components['vectorizer'] = joblib.load(vectorizer_path)
        
        return components

    def _get_confidence(self, model, features):
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            return np.max(proba)
        else:
            decision = model.decision_function(features)
            return self._normalize_decision_scores(decision)

    def _normalize_decision_scores(self, scores):
        exp_scores = np.exp(scores - np.max(scores))
        return np.max(exp_scores / exp_scores.sum(axis=1, keepdims=True))

    # Existing SVM Models
    def svm_countvectorizer_predict(self, text):
        components = self._load_components(
            'svm_countvectorizer',
            'svm_emotion_model.pkl',
            'count_vectorizer.pkl'
        )
        text_vec = components['vectorizer'].transform([text])
        return JsonResponse({
            'emotion': components['model'].predict(text_vec)[0],
            'confidence': float(self._get_confidence(components['model'], text_vec))
        })

    def svm_tfidf_predict(self, text):
        components = self._load_components(
            'svm_tfidf',
            'model.pkl',
            'vectorizer.pkl'
        )
        text_vec = components['vectorizer'].transform([text])
        return JsonResponse({
            'emotion': components['model'].predict(text_vec)[0],
            'confidence': float(self._get_confidence(components['model'], text_vec))
        })

    # New Naive Bayes Models
    def nb_countvectorizer_predict(self, text):
        components = self._load_components(
            'nb_countvectorizer',
            'nb_model.pkl',
            'count_vectorizer.pkl'
        )
        text_vec = components['vectorizer'].transform([text])
        return JsonResponse({
            'emotion': components['model'].predict(text_vec)[0],
            'confidence': float(self._get_confidence(components['model'], text_vec))
        })

    def nb_tfidf_predict(self, text):
        components = self._load_components(
            'nb_tfidf',
            'nb_model.pkl',
            'tfidf_vectorizer.pkl'
        )
        text_vec = components['vectorizer'].transform([text])
        return JsonResponse({
            'emotion': components['model'].predict(text_vec)[0],
            'confidence': float(self._get_confidence(components['model'], text_vec))
        })

    # BERT Base Model
    def _load_bert_components(self):
        if self.bert_model is None:
            model_dir = os.path.join(settings.BASE_DIR, 'api', 'models', 'bert_model')
            tokenizer_dir = os.path.join(model_dir, 'bert_tokenizer')
            
            # Load tokenizer from specific subdirectory
            self.bert_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
            
            # Load BERT model
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            
            # Load label mapping
            with open(os.path.join(model_dir, "label_mapping.json"), "r") as f:
                self.bert_label_mapping = json.load(f)

    def bert_predict(self, text):
        self._load_bert_components()
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        return JsonResponse({
            'emotion': self.bert_label_mapping[str(predicted_class.item())],
            'confidence': float(confidence.item())
        })

    # BERT Hybrid Models
    def _get_bert_embeddings(self, text):
        self._load_bert_components()
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1][:, 0, :].cpu().numpy()

    def bert_svm_predict(self, text):
        embeddings = self._get_bert_embeddings(text)
        components = self._load_components('bert_svm', 'svm_model.pkl')
        
        numerical_prediction = components['model'].predict(embeddings)[0]
        
        return JsonResponse({
            'emotion': self.bert_label_mapping[str(numerical_prediction)],
            'confidence': float(self._get_confidence(components['model'], embeddings))
        })

    def bert_naivebayes_predict(self, text):
        embeddings = self._get_bert_embeddings(text)
        model_path = os.path.join(
            settings.BASE_DIR, 'api', 'models', 'bert_nb', 
            'malayalam_text_classifier.pkl'
        )
        nb_model = joblib.load(model_path)
        
        string_prediction = nb_model.predict(embeddings)[0]
        
        # Create inverse mapping for verification
        inverse_mapping = {v.lower(): k for k, v in self.bert_label_mapping.items()}
        
        try:
            # Case-insensitive match and ensure proper capitalization
            normalized_prediction = string_prediction.strip().lower()
            numerical_key = inverse_mapping[normalized_prediction]
            proper_emotion = self.bert_label_mapping[numerical_key]
            
            return JsonResponse({
                'emotion': proper_emotion,
                'confidence': float(self._get_confidence(nb_model, embeddings))
            })
        except KeyError:
            logger.error(f"Label mismatch - Model output: '{string_prediction}', Allowed values: {list(inverse_mapping.keys())}")
            return JsonResponse(
                {'error': f"Prediction error: Unknown emotion '{string_prediction}'"},
                status=500
            )
