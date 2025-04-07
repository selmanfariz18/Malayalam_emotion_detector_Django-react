from django.urls import path
from .views import PredictEmotion, Signup, Signin

urlpatterns = [
    path('predict/', PredictEmotion.as_view(), name='emotion-predict'),
    path('signup/', Signup.as_view(), name='signup'),
    path('signin/', Signin.as_view(), name='signin'),
]