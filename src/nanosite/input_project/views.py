import re
import shutil
import os, sys
import tempfile
import subprocess
import configparser
import zipfile, tarfile
from django.shortcuts import render
from django.conf import settings
from django.http import JsonResponse
from datetime import datetime, timedelta
from django.shortcuts import get_object_or_404
from .models import Project



def populate_config_file(project_dir, user_inputs):
    config_template_path = os.path.join(settings.CONFIG_ROOT, 'config.init')
    config_dest_path = os.path.join(project_dir, 'config.init')

    # Read the template configuration file
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve case sensitivity for keys
    config.read(config_template_path)

    # Update the configuration with values from user_inputs
    for section in config.sections():
        for key in config[section]:
            if key in user_inputs:
                config[section][key] = str(user_inputs[key])

    # Write the updated configuration back to the destination
    with open(config_dest_path, 'w') as configfile:
        config.write(configfile)


# Updated functions for handling file uploads and extraction
def extract_compressed_file(uploaded_file, destination_dir):
    """
    Extracts compressed files (zip or tar.gz) to the specified destination directory.
    Checks if the files already exist before extracting to avoid duplication errors.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)

        # Save the uploaded file to a temp directory
        with open(temp_file_path, 'wb') as temp_file:
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)

        try:
            # Extract the compressed file
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(temp_file_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    # Check if all files already exist in the destination directory
                    if all(
                        os.path.exists(os.path.join(destination_dir, os.path.basename(f)))
                        for f in file_list
                    ):
                        return  # Skip extraction if files already exist
                    zip_ref.extractall(temp_dir)
            elif uploaded_file.name.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(temp_file_path, 'r:gz') as tar_ref:
                    file_list = tar_ref.getnames()
                    # Check if all files already exist in the destination directory
                    if all(
                        os.path.exists(os.path.join(destination_dir, os.path.basename(f)))
                        for f in file_list
                    ):
                        return  # Skip extraction if files already exist
                    tar_ref.extractall(temp_dir)
            else:
                raise ValueError(f"Unsupported compressed file format: {uploaded_file.name}")

            # Move `.RCC` and `.rcc` files to the destination directory
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(('.RCC', '.rcc')) and file[0].isalnum():
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(destination_dir, os.path.basename(file))
                        if not os.path.exists(dest_file):  # Avoid overwriting
                            shutil.move(src_file, dest_file)
        except Exception as e:
            raise RuntimeError(f"Failed to extract and process {uploaded_file.name}: {str(e)}")


def handle_file_upload(files, destination_dir):
    """
    Handles the upload of files to the specified destination directory.
    Extracts compressed files and moves `.RCC` or `.rcc` files to the directory.
    """
    os.makedirs(destination_dir, exist_ok=True)

    for uploaded_file in files:
        # Process uploaded files directly from the request
        if uploaded_file.name.endswith(('.zip', '.tar.gz', '.tgz')):
            # Handle compressed files
            extract_compressed_file(uploaded_file, destination_dir)
        elif uploaded_file.name.endswith(('.RCC', '.rcc')):
            # Save RCC/rcc files directly
            file_path = os.path.join(destination_dir, uploaded_file.name)
            if not os.path.exists(file_path):  # Avoid overwriting
                with open(file_path, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}. Only RCC, rcc, zip, and tar.gz files are allowed.")


def get_cartridge_ids_view(request):
    # Endpoint to extract CartridgeID values from RCC files in the training_set directory.
    if request.method == "GET":
        # Retrieve the project ID from the request
        project_id = request.GET.get('project_id')
        if not project_id:
            return JsonResponse({'status': 'error', 'message': 'Project ID not provided.'}, status=400)

        # Define the training_set directory
        training_set_dir = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id, 'training_set')

        if not os.path.exists(training_set_dir):
            return JsonResponse({'status': 'error', 'message': 'Training set directory does not exist.'}, status=404)

        # Find all RCC files in the directory
        cartridge_ids = set()
        for root, _, files in os.walk(training_set_dir):
            for file in files:
                if file.endswith('.RCC') or file.endswith('.rcc'):
                    file_path = os.path.join(root, file)

                    # Extract CartridgeID from the RCC file
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            match = re.search(r'CartridgeID,([^,\n\r]+)', content)
                            if match:
                                cartridge_ids.add(match.group(1).strip())
                    except Exception as e:
                        return JsonResponse({'status': 'error', 'message': f'Error reading file {file}: {str(e)}'}, status=500)

        return JsonResponse({'status': 'success', 'cartridge_ids': list(cartridge_ids)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)

    
def validate_uploaded_files(training_set_dir, clinical_file_path):
    errors = []

    # 1. Validate clinical file header
    try:
        with open(clinical_file_path, 'r') as clinical_file:
            header = clinical_file.readline().strip().split(',')
            if header[:2] != ['Filename', 'Condition']:
                errors.append("The clinical file must have 'Filename' and 'Condition' as the first two columns.")
    except Exception as e:
        errors.append(f"Error reading clinical file: {str(e)}")

    # 2. Check RCC file count and match with clinical file
    try:
        rcc_files = [f for f in os.listdir(training_set_dir) if f.endswith(('.RCC', '.rcc'))]
        filenames_in_clinical = set()
        with open(clinical_file_path, 'r') as clinical_file:
            next(clinical_file)  # Skip header
            for line in clinical_file:
                columns = line.strip().split(',')
                if len(columns) < 2:
                    continue
                filenames_in_clinical.add(columns[0])

        missing_in_rcc = filenames_in_clinical - set(rcc_files)
        missing_in_clinical = set(rcc_files) - filenames_in_clinical

        if missing_in_rcc or missing_in_clinical:
            errors.append(
                f"Discrepancies found:\n"
                f"Missing in RCC files: {', '.join(missing_in_rcc) if missing_in_rcc else 'None'}\n"
                f"Missing in clinical file: {', '.join(missing_in_clinical) if missing_in_clinical else 'None'}"
            )
    except Exception as e:
        errors.append(f"Error validating RCC files: {str(e)}")

    # 3. Check for missing values or NAs in first two columns
    try:
        with open(clinical_file_path, 'r') as clinical_file:
            next(clinical_file)  # Skip header
            for line in clinical_file:
                columns = line.strip().split(',')
                if len(columns) < 2 or not columns[0] or not columns[1] or 'NA' in columns[:2]:
                    errors.append("Missing or invalid values found in 'Filename' or 'Condition' columns.")
                    break
    except Exception as e:
        errors.append(f"Error checking for missing values: {str(e)}")

    return errors


def upload_data_view(request):
    if request.method == "POST":
        try:
            # Ensure session has a project ID
            if 'project_id' not in request.session:
                project = Project.objects.create(used_parameters={})
                request.session['project_id'] = project.project_id
            else:
                project_id = request.session['project_id']
                project = get_object_or_404(Project, project_id=project_id)

            # Create project directory
            project_dir = os.path.join(settings.INPUT_PROJECTS_ROOT, project.project_id)
            os.makedirs(project_dir, exist_ok=True)

            # Create subdirectory for training set
            training_set_dir = os.path.join(project_dir, 'training_set')
            os.makedirs(training_set_dir, exist_ok=True)

            # Check testingType for "extSet"
            validation_type = request.POST.get('testingType', '')
            test_set_dir = None
            ext_clinical_file_path = None
            if validation_type == 'extSet':
                test_set_dir = os.path.join(project_dir, 'test_set')
                os.makedirs(test_set_dir, exist_ok=True)

            # Handle raw training data upload
            if 'raw_files[]' in request.FILES:
                raw_files = request.FILES.getlist('raw_files[]')
                handle_file_upload(raw_files, training_set_dir)

            # Handle clinical training data upload
            clinical_file_path = None
            if 'clinical_file' in request.FILES:
                clinical_file = request.FILES['clinical_file']
                clinical_file_path = os.path.join(training_set_dir, clinical_file.name)
                with open(clinical_file_path, 'wb') as f:
                    for chunk in clinical_file.chunks():
                        f.write(chunk)

            # Handle raw validation data upload (if applicable)
            if test_set_dir and 'ext_raw_files[]' in request.FILES:
                ext_raw_files = request.FILES.getlist('ext_raw_files[]')
                handle_file_upload(ext_raw_files, test_set_dir)

            # Handle clinical validation data upload (if applicable)
            if test_set_dir and 'ext_clinical_file' in request.FILES:
                ext_clinical_file = request.FILES['ext_clinical_file']
                ext_clinical_file_path = os.path.join(test_set_dir, ext_clinical_file.name)
                with open(ext_clinical_file_path, 'wb') as f:
                    for chunk in ext_clinical_file.chunks():
                        f.write(chunk)

            # Handle submission
            if request.POST.get('submit') == 'true':
                # Validation logic
                validation_errors = validate_uploaded_files(training_set_dir, clinical_file_path)
                if validation_errors:
                    return JsonResponse({'status': 'error', 'message': "\n".join(validation_errors)}, status=400)

                # Prepare user inputs
                user_inputs = {
                    # General
                    "projectID": project.project_id,
                    "instrument": request.POST.get("radioGroup", "max-flex"),

                    # IO
                    "dir": training_set_dir,
                    "clinicaldata": clinical_file_path or "None",
                    "outdir": project_dir,
                    "control": request.POST.get("control", "Control"),
                    "condition": request.POST.get("condition", "Condition"),
                    "testtype": request.POST.get("testingType", "Split"),
                    "testrun": ", ".join(request.POST.getlist("run[]")),

                    # TestSet
                    "testdir": test_set_dir or "None",
                    "testclinicaldata": ext_clinical_file_path or "None",

                    # DE
                    "norm": request.POST.get("norm", "auto"),
                    "lfcthreshold": request.POST.get("lfcthreshold", "0.5"),
                    "padjusted": request.POST.get("padjusted", "0.05"),

                    # RUVSeq
                    "k_factor": request.POST.get("k_factor", "1"),
                    "refgenes": request.POST.get("refgenes", "hkNpos"),
                    "minref": request.POST.get("min_refgenes", "5"),

                    # Enrichment Analysis
                    "reforg": request.POST.get("reforg", "hsapiens"),

                    # Filters
                    "filter_lowlyExpr_genes": request.POST.get("filter_lowlyExpr_genes", "False"),
                    "filter_genes_on_negCtrl": request.POST.get("filter_genes_on_negCtrl", "True"),
                    "filter_samples_on_negCtrl": request.POST.get("filter_samples_on_negCtrl", "True"),
                    "remove_outlier_samples": request.POST.get("remove_outlier_samples", "False"),
                    "iqrcutoff": request.POST.get("iqrcutoff", "2"),

                    # ML
                    "correlated": request.POST.get("correlated", "True"),
                    "quasiconstant": request.POST.get("quasiconstant", "True"),
                    "featureselection": request.POST.get("featureselection", "RFE"),
                    "rfecrossval": request.POST.get("rfecrossval", "10CV"),
                    "minfeature": request.POST.get("minfeature", "5"),
                    "classifiers": ", ".join(request.POST.getlist("classifiers[]")),
                    "crossval": request.POST.get("crossval", "5CV"),
                }

                # Populate the configuration file
                populate_config_file(project_dir, user_inputs)

                # Save project path in the database
                project.input_data = project_dir
                project.save()

                # Clear the session to allow a new project submission
                del request.session['project_id']

                return JsonResponse({'status': 'success', 'project_id': project.project_id, 'project_dir': project_dir})

            return JsonResponse({'status': 'success', 'project_id': project.project_id})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'}, status=400)


def search_project_view(request, project_id):
    # Function to do the search of the projects

    if not re.match(r'^[a-zA-Z0-9]{14}$', project_id):
        return JsonResponse({'status': 'error', 'message': 'Project ID must be alphanumeric and exactly 14 characters.'}, status=400)

    try:
        project = get_object_or_404(Project, project_id=project_id)
        project_dir = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id)

        if not os.path.exists(project_dir):
            return JsonResponse({'status': 'error', 'message': 'Project directory does not exist.'}, status=404)

        six_months_ago = datetime.now() - timedelta(days=6 * 30)
        if project.date < six_months_ago.date():
            return JsonResponse({'status': 'error', 'message': 'Project is older than 6 months and no longer available.'}, status=404)

        return JsonResponse({'status': 'success', 'project_id': project.project_id, 'project_dir': project_dir})

    except Project.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Project not found.'}, status=404)


def delete_project_view(request):
    # Deleting  directories that didn't run
    
    if request.method == 'POST':
        project_id = request.POST.get('project_id')
        if project_id:
            project_dir = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id)

            # Check if the directory exists
            if os.path.exists(project_dir):
                config_file_path = os.path.join(project_dir, 'config.init')

                # Only delete if `config.init` does not exist
                if not os.path.exists(config_file_path):
                    shutil.rmtree(project_dir)
                    print(f"Deleted orphaned project directory: {project_dir}")
                    return JsonResponse({'status': 'success', 'message': 'Orphaned project directory deleted.'})
                else:
                    return JsonResponse({'status': 'error', 'message': 'Cannot delete a project with config.init.'}, status=400)

            return JsonResponse({'status': 'error', 'message': 'Project directory does not exist.'}, status=404)

        return JsonResponse({'status': 'error', 'message': 'Project ID not provided.'}, status=400)

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'}, status=400)


def example_view(request):

    # Example dataset paths
    example_raw_data = os.path.join(settings.EXAMPLE_DATA_DIR, 'training_set', 'example_raw_data.RCC')
    example_clinical_data = os.path.join(settings.EXAMPLE_DATA_DIR, 'training_set', 'example_clinical_data.csv')

    # Prefilled data
    example_data = {
        "control": "ControlGroup1",
        "condition": "ConditionGroup2",
        "norm": "auto",
        "lfcthreshold": 1.0,
        "padjusted": 0.05,
        "raw_files": example_raw_data,
        "clinical_file": example_clinical_data,
    }

    return render(request, 'example.html', {"prefilled_data": example_data})


# Render the analysis page for a specific project ID.
def analysis_view(request, project_id):
    def file_exists(base_path, filenames):
        """Helper function to check if files exist and return a dictionary of their statuses."""
        return {name: os.path.exists(os.path.join(base_path, filename)) for name, filename in filenames.items()}

    # Base paths
    base_training_path = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id, "html_results/trainingset_normalisation/")
    base_classification_path = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id, "html_results/classification/")
    
    # Post-filtering files
    post_filter_files = {
        'pqcplot2_exists': "pca.filtered.condition.html",
        'pqcplot3_exists': "mds.filtered.condition.html",
        'pqcplot4_exists': "sample.correlation.filtered.html",
    }
    post_filter_status = file_exists(base_training_path, post_filter_files)

    # Classification files
    classification_files = {
        'clplot1_exists': "overall_ROC_plot.html",
        # Random Forest
        'clplot11_exists': "RF.TrainingTest.ROCplot.html",
        'clplot12_exists': "RF.TestSet.ConfusionMatrix.html",
        'clplot13_exists': "RF.TestSet.ProbabilityPlot.html",
        # KNN
        'clplot21_exists': "KNN.TrainingTest.ROCplot.html",
        'clplot22_exists': "KNN.TestSet.ConfusionMatrix.html",
        'clplot23_exists': "KNN.TestSet.ProbabilityPlot.html",
        # Gradient Boosting
        'clplot31_exists': "GB.TrainingTest.ROCplot.html",
        'clplot32_exists': "GB.TestSet.ConfusionMatrix.html",
        'clplot33_exists': "GB.TestSet.ProbabilityPlot.html",
        # Extra Trees
        'clplot41_exists': "ET.TrainingTest.ROCplot.html",
        'clplot42_exists': "ET.TestSet.ConfusionMatrix.html",
        'clplot43_exists': "ET.TestSet.ProbabilityPlot.html",
        # Logistic Regression
        'clplot51_exists': "LG.TrainingTest.ROCplot.html",
        'clplot52_exists': "LG.TestSet.ConfusionMatrix.html",
        'clplot53_exists': "LG.TestSet.ProbabilityPlot.html",
        # Enrichment Analysis
        'enrplot1_exists': "enrichment_analysis_plot.html",
    }
    classification_status = file_exists(base_classification_path, classification_files)

    # Combine all statuses into the context
    context = {
        'project_id': project_id,  # Pass the dynamic project ID to the template
        **post_filter_status,      # Unpack post-filtering file statuses
        **classification_status,   # Unpack classification file statuses
    }
    
    return render(request, 'analysis.html', context)


def run_nanoinsights(request, project_id):
    
    if request.method == "POST":
        
        try:
            # Path to config
            config_path = os.path.join(settings.INPUT_PROJECTS_ROOT, project_id, "config.init")

            if not config_path:
                return JsonResponse({"success": False, "error": "Config path is missing from the request."})

            if not os.path.exists(config_path):
                return JsonResponse({"success": False, "error": f"Config file {config_path} does not exist."})

            # Build the command
            backend_script = os.path.join(settings.NANOINSIGHTS_BACKEND_DIR, "nanoinsights_init.py")
            # command = ["python3", backend_script, "--config", config_path]
            command = [sys.executable, backend_script, "--config", config_path]
            print(f"Running command: {' '.join(command)}")

            # Execute the script
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

            # Log outputs for debugging
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

            # Check the result and return appropriate response
            if result.returncode == 0:
                return JsonResponse({"success": True, "output": result.stdout.strip()})
            else:
                return JsonResponse({
                    "success": False,
                    "error": "NanoInsights script failed to execute.",
                    "stderr": result.stderr.strip(),
                    "stdout": result.stdout.strip(),
                })

        except Exception as e:
            # Catch unexpected errors
            return JsonResponse({"success": False, "error": f"An unexpected error occurred: {str(e)}"})

    # If not a POST request
    return JsonResponse({"success": False, "error": "Invalid request method. Use POST."})