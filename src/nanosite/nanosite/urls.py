from django.contrib import admin
from django.urls import path
from pages.views import home_view, usage_view, video_view, example_view, analysis_view, contact_view, citation_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('usage/', usage_view, name='usage'),
    path('video/', video_view, name='video'),
    path('example/', example_view, name='example'),
    path('analysis/', analysis_view, name='analysis'),
    path('contact/', contact_view, name='contact'),
    path('citation/', citation_view, name='citation'),
]
