# Generated by Django 4.2.6 on 2024-11-24 20:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('input_project', '0002_project_project_dir_alter_project_input_data_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='project',
            name='used_parameters',
            field=models.JSONField(default=dict),
        ),
    ]
