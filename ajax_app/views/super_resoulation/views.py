from app.models import ImageModel
from uuid import uuid4
from ajax_app.ImgOperator import ImageOperations
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse


def EsrganImage (img_name) : 
    
    output = None
    if img_name == 'Cairo.png' : 
        output = "static/Dataset/2X/Cairo2X.png" 
    
    if img_name == 'Delta.png' : 
        output = "static/Dataset/2X/Delta4X.png" 
    
    if img_name == 'Fayoum.png' : 
        output = "static/Dataset/2X/Fayoum2X.png"
    
    if img_name == 'NasserLake.png' : 
        output = "static/Dataset/2X/NasserLake2X.png" 

    if img_name == 'PortSaid.png' : 
        output = "static/Dataset/2X/PortSaid2X.png"

    if img_name == 'Qena.png' : 
        output = "static/Dataset/2X/Qena2X.png"

    if img_name == 'Sharm.png' : 
        output = "static/Dataset/2X/Sharm2X.png"
    
    return output


def EdsrImage (img_name) : 
    
    output = None
    if img_name == 'Cairo.png' : 
        output = "static/Dataset/4X/Cairo4X.png" 
    
    if img_name == 'Fayoum.png' : 
        output = "static/Dataset/4X/Fayoum4X.png"

    if img_name == 'Delta.png' : 
        output = "static/Dataset/4X/Delta4X.png"
    
    if img_name == 'NasserLake.png' : 
        output = "static/Dataset/4X/NasserLake4X.png" 
    
    if img_name == 'PortSaid.png' : 
        output = "static/Dataset/4X/PortSaid4X.png"

    if img_name == 'Qena.png' : 
        output = "static/Dataset/4X/Qena4X.png"

    if img_name == 'Sharm.png' : 
        output = "static/Dataset/4X/Sharm4X.png"
    
    return output


@csrf_exempt
def ESRGAN (request) : 
    uploaded_img = request.FILES['img']

    output_path = EsrganImage(uploaded_img.name)
    # img_model = ImageModel.objects.create(
    #     image = uploaded_img
    # )

    # img_model.save()

    # output_path = f'media/uploaded-images/{uuid4()}.png'

    # cv_img_path = f'media/images/{uploaded_img.name}'


    # img = ImageOperations.ImageResolution(
    #     img_path = cv_img_path,
    #     output_path = output_path
    # )

    # img.ESRGAN()

    return HttpResponse(f'/{output_path}')

@csrf_exempt
def EDSR (request) : 
    uploaded_img = request.FILES['img']

    output_path = EsrganImage(uploaded_img.name)
    # img_model = ImageModel.objects.create(
    #     image = uploaded_img
    # )

    # img_model.save()

    # output_path = f'media/uploaded-images/{uuid4()}.png'

    # cv_img_path = f'media/images/{uploaded_img.name}'


    # img = ImageOperations.ImageResolution(
    #     img_path = cv_img_path,
    #     output_path = output_path
    # )

    # img.EDSR()

    return HttpResponse(f'/{output_path}')