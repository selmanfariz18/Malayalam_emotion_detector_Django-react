from django.urls import path
from .views import PredictEmotion

urlpatterns = [
    path('predict/', PredictEmotion.as_view(), name='emotion-predict'),
]