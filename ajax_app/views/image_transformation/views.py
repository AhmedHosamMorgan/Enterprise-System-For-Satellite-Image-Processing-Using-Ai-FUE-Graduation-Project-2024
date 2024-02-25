from app.models import ImageModel
from uuid import uuid4
from ajax_app.ImgOperator import ImageOperations
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse


@csrf_exempt
def PCA (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uuid4()}.png'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.ImageTransformattion(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.PCA()

    return HttpResponse(f'/{output_path}')

@csrf_exempt
def NDVI (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uuid4()}.png'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.ImageTransformattion(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.NDVI()

    return HttpResponse(f'/{output_path}')

@csrf_exempt
def ICA (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uploaded_img.name}'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.ImageTransformattion(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.ICA()

    return HttpResponse(output_path)



@csrf_exempt
def MNF (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uploaded_img.name}'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.ImageTransformattion(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.MNF()

    return HttpResponse(f'/{output_path}')