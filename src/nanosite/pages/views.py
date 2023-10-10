from django.shortcuts import render


def home_view(request, *args, **kwargs):
    return render(request, "base.html", {})




def my_analysis(request, *args, **kwargs):
    return HttpResponse("<h1>My Analysis</h1>")

def website_stats(request, *args, **kwargs):
    return HttpResponse("<h1>Website Stats</h1>")




def contact_us(request, *args, **kwargs):
    return render(request, "contact.html", {})

def cite_us(request, *args, **kwargs):
    return render(request, "citation.html", {})