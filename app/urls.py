from django.urls import path
from django.shortcuts import redirect
from .views import CheckSpam

def redirect_to_checkspam(request):
    return redirect('/checkspam/')


urlpatterns = [
    path('', redirect_to_checkspam, name='root_redirect'),
    path('checkspam/', CheckSpam.as_view(), name='check_spam'),
]