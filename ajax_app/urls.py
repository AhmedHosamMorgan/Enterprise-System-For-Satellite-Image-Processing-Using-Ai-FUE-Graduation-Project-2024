from .views.histogram_streaming import views as his
from .views.image_filteration import views as filter_view
from .views.image_transformation import views as transforamtion_view
from .views.super_resoulation import views as super_view
from django.urls import path

urlpatterns = [
    # histogram_streaming
    path('Histogram_Stretching/histogram_equalization/',his.histogram_equalization,name='he'),
    path('Histogram_Stretching/minimum_maximum/',his.Minimum_Maximum,name='mm'),
    path('Histogram_Stretching/standard_deviation/',his.Standard_Deviation,name='sd'),

    # Image Filteration  Image Filteration
    path('Image_Filtration/Sobel_Filter/',filter_view.Sobel_Filter,name='cf'),
    path('Image_Filtration/high_pass_filter/',filter_view.High_Pass_Filter,name='hpf'),
    path('Image_Filtration/low_pass_filter/',filter_view.Low_Pass_Filter,name='lpf'),
    path('Image_Filtration/remove_noise/',filter_view.Remove_Noise,name='rn'),

    # Image Transforamtion
    path('image-transformation/PCA/',transforamtion_view.PCA,name='pca'),
    path('image-transformation/NDVI/',transforamtion_view.NDVI,name='ndvi'),
    path('image-transformation/ICA/',transforamtion_view.ICA,name='ica'),
    path('image-transformation/MNF/',transforamtion_view.MNF,name='mnf'),

    # Image Resoulation
    path('image-resoulation/ESRGAN/',super_view.ESRGAN,name='esrgan'),
    path('image-resoulation/EDSR/',super_view.EDSR,name='edsr'),

] 