from django.shortcuts import render, redirect
from users.models import User
from django.contrib.auth.views import auth_login


def RegisterView (request) : 

    context = {}

    if request.method == "POST" : 
        full_name = request.POST['full_name']
        email = request.POST['email']
        password = request.POST['password']
        country = request.POST['country']

        if User.objects.filter(email=email).exists() is False: 

            user = User.objects.create_user(
                full_name = full_name,
                email = email,
                password = password,
                country = country,
            )

            user.save()
            auth_login(request,user)
            return redirect('Operation')
        context['error'] = {'error':"this email already exists"}

    return render(request, 'Create Account.html', context)
