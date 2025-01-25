from django.urls import path
from .views import EmotionPredictor

urlpatterns = [
    path('predict/', EmotionPredictor.as_view(), name='emotion-predict'),
]