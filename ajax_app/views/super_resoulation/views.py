from app.models import ImageModel
from uuid import uuid4
from ajax_app.ImgOperator import ImageOperations
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse



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
