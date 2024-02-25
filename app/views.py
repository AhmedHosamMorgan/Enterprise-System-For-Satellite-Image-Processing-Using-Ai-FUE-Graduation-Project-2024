from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest
import cv2
import matplotlib.pyplot as plt
from .models import ImageModel
from uuid import uuid4
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
import numpy as np

def Welcome (request) : 
    return render(request,'Welcome.html')
    #return render(request,'operation.html')


def Forget_Password(request):
    # Logic to render the login page
    return render(request, 'Forget Password.html')  # Replace 'login.html' with the actual name of your login HTML file

def Contact_US(request):
    # Your logic for the Contact Us page
    return render(request, 'Contact_US.html')  # Render the Contact Us template

def Meet_the_Team(request):
    # Your logic for the Contact Us page
    return render(request, 'Meet_the_Team.html')  # Render the Contact Us template

def Documentation(request):
    # Your logic for the Contact Us page
    return render(request, 'Documentation.html')  # Render the Contact Us template


@login_required
def Operation (request) : 
    return render(request,'Operation.html')



@csrf_exempt
def blur_img (request) :


    if request.method == "GET" :
        return HttpResponseBadRequest(request)
    
    uploaded_img = request.FILES['img']

    model = ImageModel.objects.create(image=uploaded_img)
    model.save()

    

    image = cv2.imread(f'media/images/{uploaded_img.name}')



    #Plot the original image
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(image)

    # Remove noise using a median filter
    filtered_image = cv2.medianBlur(image, 11)

    #Plot the blured image
    plt.subplot(1, 2, 2)
    plt.title("Median Blur")
    plt.imshow(filtered_image)
    
    saved_path = f'media/uploaded-images/{uuid4()}.png'
    plt.savefig(saved_path)
    plt.close()

    return HttpResponse(saved_path)


@csrf_exempt
def hostogrm (request) :


    if request.method == "GET" :
        return HttpResponseBadRequest(request)
    
    uploaded_img = request.FILES['img']

    model = ImageModel.objects.create(image=uploaded_img)
    model.save()

    


    # Load the input image with RGB channels
    img_sam = cv2.imread(f'media/images/{uploaded_img.name}', cv2.IMREAD_COLOR)

    # Split the channels into red, green, and blue
    rd, gn, bl = cv2.split(img_sam)

    # Plot the histograms for each channel
    plt.hist(rd.ravel(), bins=277, color='red', alpha=0.10)
    plt.hist(gn.ravel(), bins=277, color='green', alpha=0.10)
    plt.hist(bl.ravel(), bins=277, color='blue', alpha=0.10)

    # Stretch the contrast for each channel
    red_c_stretch = cv2.equalizeHist(rd)
    green_c_stretch = cv2.equalizeHist(gn)
    blue_c_stretch = cv2.equalizeHist(bl)

    # Merge the channels back together
    img_stretch = cv2.merge((red_c_stretch, green_c_stretch, blue_c_stretch))

    # Display the original and stretched images side by side
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img_sam, cv2.COLOR_BGR2RGB))
    plt.title('Original_Image')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_stretch, cv2.COLOR_BGR2RGB))
    plt.title('Stretched_Image')

    saved_path = f'media/uploaded-images/{uuid4()}.png'
    plt.savefig(saved_path)
    plt.close()

    return HttpResponse(saved_path)

@csrf_exempt
def soble_filter (request) :


    if request.method == "GET" :
        return HttpResponseBadRequest(request)
    
    uploaded_img = request.FILES['img']

    model = ImageModel.objects.create(image=uploaded_img)
    model.save()

    
    saved_path = f'media/uploaded-images/{uuid4()}.png'

    img0 = cv2.imread(f'media/images/{uploaded_img.name}',)

    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.savefig(saved_path)
    plt.close()

    return HttpResponse(saved_path)
