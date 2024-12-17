from django.urls import path
from .views import get_cartridge_ids_view, upload_data_view, search_project_view, delete_project_view, example_view, analysis_view, run_nanoinsights


app_name = 'input_project'

urlpatterns = [
    path('upload/', upload_data_view, name='upload_data'),
    path('get_cartridge_ids/', get_cartridge_ids_view, name='get_cartridge_ids'),
    path('search/<str:project_id>/', search_project_view, name='search_project'),
    path('delete_project/', delete_project_view, name='delete_project'),
    path('example/', example_view, name='example'),
    path('analysis/<str:project_id>/', analysis_view, name='analysis'),
    path('run-nanoinsights/<str:project_id>/', run_nanoinsights, name='run_nanoinsights'),
]
