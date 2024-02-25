from django.shortcuts import render, redirect
from django.contrib.auth.views import auth_login
from django.contrib.auth import authenticate
from users.models import User


def Login_Form(request):
    context = {}
   

    if request.method == "POST" : 
        email = request.POST.get('email', None)
        password = request.POST.get('password', None)



        user = authenticate(email=email,password=password)

        if user is not None :
            auth_login(request,user)
            return redirect('Operation')
        else:
            context['error'] = "invalid email or password"

    return  render(request, 'Login Form.html',context=context)
