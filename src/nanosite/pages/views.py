from django.shortcuts import render


def home_view(request, *args, **kwargs):
    return render(request, "home.html", {})

def usage_view(request, *args, **kwargs):
    return render(request, "usage.html", {})

def contact_view(request, *args, **kwargs):
    return render(request, "contact.html", {})

def citation_view(request, *args, **kwargs):
    return render(request, "citation.html", {})