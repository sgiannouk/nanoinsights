import os
import shutil
import random
import string
from django.conf import settings
from django.shortcuts import render
from input_project.models import Project


def generate_14_id():
    """Generate a unique 14-character alphanumeric ID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=14))


def home_view(request, *args, **kwargs):
    # Cleanup orphaned directories
    for directory in os.listdir(settings.INPUT_PROJECTS_ROOT):
        dir_path = os.path.join(settings.INPUT_PROJECTS_ROOT, directory)
        if os.path.isdir(dir_path):
            config_path = os.path.join(dir_path, 'config.init')
            if not os.path.exists(config_path):
                shutil.rmtree(dir_path)

    # Handle the session and project ID
    if 'project_id' not in request.session:
        project = Project.objects.create(used_parameters={})
        request.session['project_id'] = project.project_id

    project_id = request.session['project_id']
    project_dir = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id)
    os.makedirs(project_dir, exist_ok=True)

    # Pass context to the template
    return render(request, 'home.html', {
        'project_id': project_id,
        'config_exists': os.path.exists(os.path.join(project_dir, 'config.init')),
    })

def usage_view(request, *args, **kwargs):
    return render(request, "usage.html", {})

def video_view(request, *args, **kwargs):
    return render(request, "video.html", {})

def example_view(request, *args, **kwargs):
    return render(request, "example.html", {})

def citation_view(request, *args, **kwargs):
    return render(request, "citation.html", {})