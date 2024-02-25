from django.urls import path
from .auth import login, register, forget_password
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('login/',login.Login_Form,name='login'),
    path('register/',register.RegisterView,name='register'),
    path('forget_password/',forget_password.ForgetPassword,name='forgetpassword'),

    path('logout/',LogoutView.as_view(),name='logout'),
]
