from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),
    path('register', views.register, name='register'),
    path('activate/<uidb64>/<token>', views.activate, name='activate'),
    path('login',views.user_login, name='login'),
    path('logout', views.user_logout, name='logout'),
    path('predict', views.predict, name='predict'),
    path('appoint', views.appoint, name='appoint'),
    path('index/', views.index, name='index'),
    # path('dental_view/', views.dental_view, name='dental_view'),
    path('admin/', admin.site.urls),
    # path('dental/', views.dental_view, name='dental_view'),
    path('getprediction/', views.predict_stability, name='get_prediction'),
    path('predict_pneumonia/', views.predict_pneumonia, name='predict_pneumonia'),
    path('brain_mri_segmentation/', views.brain_mri_segmentation, name='brain_mri_segmentation'),
    path('predict_diabetes', views.predict_diabetes, name='predict_diabetes'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
