from django.contrib import admin
from django.urls import include, path
from pages.views import home_view, usage_view, video_view, example_view, citation_view
from contact.views import contact_view

from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('usage/', usage_view, name='usage'),
    path('video/', video_view, name='video'),
    path('example/', example_view, name='example'),
    path('contact/', contact_view, name='contact'),
    path('citation/', citation_view, name='citation'),
    path('input_project/', include('input_project.urls', namespace='input_project')),
]

urlpatterns += static('/uploads/', document_root=settings.INPUT_PROJECTS_ROOT)