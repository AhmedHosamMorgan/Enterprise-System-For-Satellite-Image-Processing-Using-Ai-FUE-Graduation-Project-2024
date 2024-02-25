from app.models import ImageModel
from uuid import uuid4
from ajax_app.ImgOperator import ImageOperations
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse


@csrf_exempt
def histogram_equalization (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uuid4()}.png'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.HistogramStreatching(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.Histogram_Equalization()

    return HttpResponse(f'/{output_path}')


@csrf_exempt
def Minimum_Maximum (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uuid4()}.png'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.HistogramStreatching(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.Minimum_Maximum()

    return HttpResponse(f'/{output_path}')


@csrf_exempt
def Standard_Deviation (request) : 
    uploaded_img = request.FILES['img']

    img_model = ImageModel.objects.create(
        image = uploaded_img
    )

    img_model.save()

    output_path = f'media/uploaded-images/{uuid4()}.png'

    cv_img_path = f'media/images/{uploaded_img.name}'


    img = ImageOperations.HistogramStreatching(
        img_path = cv_img_path,
        output_path = output_path
    )

    img.Standard_Deviation()

    return HttpResponse(f'/{output_path}')


