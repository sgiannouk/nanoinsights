from django.http import JsonResponse
from django.shortcuts import render
from django.core.mail import send_mail
from django.conf import settings


def contact_view(request):
    context = {}  # Default empty context
    if request.method == "POST":
        message_name = request.POST.get('name')
        message_email = request.POST.get('email')
        message_subject = request.POST.get('subject')
        message = request.POST.get('message')

        # Validate input fields
        if not message_name or not message_email or not message_subject or not message:
            context['error'] = "Your message could not be sent. Please ensure all fields are filled out correctly and try again!"
            return render(request, "contact.html", context)

        try:
            # Send the email
            send_mail(
                f"Email from NanoInsights WebServer: {message_subject}",
                f"Name: {message_name}\nEmail: {message_email}\n\n{message}",
                settings.EMAIL_HOST_USER,
                ['StavrosGi@gmail.com'])
            
            context['success'] = "Thank you for reaching out! Your message has been sent successfully, and we will contact you as soon as possible :)"
        
        except Exception as e:
            context['error'] = f"We encountered an issue while sending your message: {str(e)}. Please try again later."

    return render(request, "contact.html", context)
