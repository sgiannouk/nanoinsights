import random
import string
import os
from django.db import models
from django.conf import settings


def generate_project_id():
    """Generate a unique 14-character alphanumeric ID for the project."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=14))


class Project(models.Model):
    project_id = models.CharField(max_length=14, unique=True, default=generate_project_id)
    project_dir = models.CharField(max_length=255, default=settings.INPUT_PROJECTS_ROOT)
    input_data = models.FileField(upload_to="input_data/", blank=False, null=False)
    used_parameters = models.JSONField(blank=False, null=False, default=dict)
    project_completed = models.BooleanField(default=False)
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)
    finish_time = models.DateTimeField(blank=True, null=True)
    run_duration = models.DurationField(blank=True, null=True)
    output_data = models.FileField(upload_to="output_data/", blank=True, null=True)

    def save(self, *args, **kwargs):
        """Custom save method to create the project directory."""
        project_path = os.path.join(settings.INPUT_PROJECTS_ROOT, self.project_id)
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        self.project_dir = project_path
        super().save(*args, **kwargs)

    def clean(self):
        """Ensure that the input_data path exists when provided."""
        if self.input_data and not os.path.exists(self.input_data.path):
            raise ValueError(f"Input data path {self.input_data.path} does not exist.")

    def __str__(self):
        return self.project_id
