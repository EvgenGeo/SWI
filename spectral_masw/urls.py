from enum import verify

from django.urls import path
from . import views

app_name = 'spectral_masw'

urlpatterns = [
    # вкладка 1
    path('segy/', views.segy_list_view, name='segy_list'),
    path('segy/upload/', views.segy_upload_view, name='segy_upload'),
    path('segy/<int:pk>/process/', views.segy_process_view, name='segy_process'),
    path('segy/<int:pk>/delete/', views.segy_delete_view, name='segy_delete'),

    # вкладка 2
    path('spectra/', views.spectra_list_view, name='spectra_list'),
    path('spectra/<int:pk>/plot/', views.spectra_plot_view, name='spectra_plot'),
    path('spectra/<int:pk>/delete/', views.spectra_delete_view, name='spectra_delete'),
]
