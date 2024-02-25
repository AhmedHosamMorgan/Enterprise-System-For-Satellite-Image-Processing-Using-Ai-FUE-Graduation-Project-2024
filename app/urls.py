from django.urls import path
from . import views

urlpatterns = [
    path('',views.Welcome,name='Welcome'),
    path('Operation', views.Operation, name='Operation'),  
    path('contact/', views.Contact_US, name='Contact_US'),
    path('Meet_the_Team/', views.Meet_the_Team, name='Meet_the_Team'),
    path('Documentation/', views.Documentation, name='Documentation'),
    
    # path('Forget_Password/', views.Forget_Password, name='Forget_Password'),  
    # path('Create_Account/', views.Create_Account, name='Create_Account'), 
    # path('Create_Account/Login Form.html', views.Login_Form, name='login_form'), 
    # path('Create_Account/Operation.html', views.Operation, name='Operation'), 


    

]