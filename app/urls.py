from django.urls import path
from .views import CheckSpam

urlpatterns = [
    path('checkspam/', CheckSpam.as_view(), name='check_spam'),
]