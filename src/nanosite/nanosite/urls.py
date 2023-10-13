from django.contrib import admin
from django.urls import path
from pages.views import home_view, contact_view, citation_view, usage_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home_view, name='home'),
    path('usage/', usage_view, name='usage'),
    path('contact/', contact_view, name='contact'),
    path('citation/', citation_view, name='citation'),
]
