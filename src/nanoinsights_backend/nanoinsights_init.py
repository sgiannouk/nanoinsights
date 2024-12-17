#!/usr/local/bin python3.11

### ANALYSING NANOSTRING DATA ###
## ESR11 - Stavros Giannoukakos 


#Version of the program
__version__ = "v1.0.0"

usage = "python3.11 nanoinsights_init.py [options] -c config.init"
epilog = " -- October 2023 | Stavros Giannoukakos -- "
description = "DESCRIPTION"

# Importing required libraries
import re
import glob
import json
import math
import shutil
import zipfile
import os, sys
import subprocess
import numpy as np
import pandas as pd
import configparser, argparse
from scipy.stats import linregress
from collections import defaultdict
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

startTime = datetime.now()  # Start timing the execution
CONFIG = {}  # Global configuration dictionary




# Argument parser for command-line options 
def parse_arguments():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, usage=usage, description=description, epilog=epilog)
	# Path of the input configuration file 
	parser.add_argument("-c", "--config",  required=True, help="Path to the INI configuration file")

	#### IO
	# Path of input directory hosting all RCC files
	parser.add_argument('-d', '--dir', dest='dir', help="Directory containing all raw *RCC files")
	# Path of the .txt file containing all necessary clinical data
	parser.add_argument('-cd', '--clinicaldata', dest='clinicaldata', help="Text file with clinical data for the input RCC files")
	# Path of the output directory hosting the results
	parser.add_argument('-o', '--outdir', dest='outdir', help="Path of the directory that will containing all the analyses")
	# Control group
	parser.add_argument('-ctrl', '--control', dest='control', 
						help="For the two-group comparison, which\nlabel should correspond to the CONTROL group (e.g. Healthy)")
	# Condition group
	parser.add_argument('-cond', '--condition', dest='condition', 
						help="For the two-group comparison, which\nlabel should correspond to the CONDITION (e.g. Cancer)")
	# Choose test set
	parser.add_argument('-tt', '--testtype', dest='testtype', default='split', choices=['split', 'run', 'extSet', 'onlyNorm'],
	                	help="Select the test type for the classifier's accuracy evaluation.\nChoices:\n1. Default: Data is divided into a training/test split of 80/20 (split)\
	                		  \n2. One or more RUNs to choose from (run)\n3. Use an external dataset for test (extSet)\n4. Performing only normalisation to the data (onlyNorm)")
	# Choose test run set
	parser.add_argument('-tr', '--testrun', dest='testrun', default=None, nargs="+", 
						help="If you choose \'run\' as the test set,\nplease specify which run or runs should be used")

	
	##### EXTERNAL TEST SET
	# Path of input directory hosting all RCC files of the external test set
	parser.add_argument('-td', '--testdir', dest='valdir',
	                	help="Directory containing all raw *RCC files of the external test set")
	# Path of the .txt file containing all necessary clinical data of the external test set
	parser.add_argument('-tcd', '--testclinicaldata', dest='valclinicaldata',
	                	help="Text file with clinical data of the external test set")


	##### DE
	# Type of normalisation
	parser.add_argument('-n', '--norm', dest='norm', default='auto', choices=['auto', 'nSolver', 'geNorm_housekeeping', 'housekeeping_scaled', 'all_endogenous_scaled', 'quantile', 'loess', 'vsn', 'ruv'],
	                	help="Normalization method to be used (default auto)")
	# The log2FC cutoff
	parser.add_argument('-lfc', '--lfcthreshold', dest='lfcthreshold', default=0.5,
	                	help="The  log2 fold change cutoff to identify DE genes (default 0.5)")
	# The adjusted p-value cutoff
	parser.add_argument('-apv', '--padjusted', dest='padjusted', default=0.05,
	                	help="The cutoff for the adjusted p-value to call DE genes (default 0.05)")


	##### RUVSEQ 
	# k factor for the RUVSEq/RUVg function
	parser.add_argument('-k', '--k_factor', dest='k_factor', default=1,
	                	help="Choose the k factor for the RUVg function\n(default 1)")
	# How should we calculate the reference genes? 
	parser.add_argument('-rg', '--refgenes', dest='refgenes', default='hkNpos', choices=['hkNpos', 'hk', 'posCtrl', 'endNhkNpos'],
	                	help="How shall we calculate the reference (stable/non-significant)\ngenes for the RUVg function? (default hkNpos)")
	# Min. number of reference genes to be considered
	parser.add_argument('-nr', '--minref', dest='minref', default=5,
	                	help="Minimum number of reference (stable/non-significant)\ngenes to consider for the RUVg function (default 5)")


	##### ENRICHMENT ANALYSIS
	# Reference organism for enrichment analysis
	parser.add_argument('-r', '--ref_organism', dest='ref_organism', default="hsapiens",
	                	help="Choose the reference organism for performing enrichment analysis\n(default hsapiens)")


	#### FILTERS 
	# Filtering out lowly expressed genes
	parser.add_argument('-fleg', '--filter_lowlyExpr_genes', dest='filter_lowlyExpr_genes', action="store_true",
	                	help="Apply the \'filterByExpr\' function from the edgeR package\nto filter out lowly expressed genes (default: False).")
	# Filtering out genes based on the Negative Control genes
	parser.add_argument('-fgnc', '--filter_genes_on_negCtrl', dest='filter_genes_on_negCtrl', action="store_false",
	                	help="Filter Endogenous genes based on the mean\nof the Negative Controls plus two times the\nSD (default True)")
	# Filter out samples 
	parser.add_argument('-fsnc', '--filter_samples_on_negCtrl', dest='filter_samples_on_negCtrl', action="store_false",
	                	help="Filter samples where the majority of the genes\nare 0 when subtracting the mean of the Negative\nControls plus two times the SD (default True)")
	# Remove outlier samples
	parser.add_argument('-ros', '--remove_outlier_samples', dest='remove_outlier_samples', action="store_true",
	                	help="Remove samples considered outliers based on IQR analysis (default: False)")
	# Choose Interquartile range cutoff 
	parser.add_argument('-iqr', '--iqrcutoff', dest='iqrcutoff', default=2,
	                	help="Choose the cutoff for the interquartile range\nanalysis (default 2)")

	
	##### ML options
	# Correlated features
	parser.add_argument('-co', '--correlated', dest='correlated', action="store_false",
	                	help="Filter out highly correlated features (>90%%)")
	# Quasi-constant features
	parser.add_argument('-qc', '--quasiconstant', dest='quasiconstant', action="store_false",
	                	help="Filter out quasi-constant features (>99%%)")
	# Feature selection type
	parser.add_argument('-fs', '--featureselection', dest="featureselection", default='RFE', choices=['RFE', 'PI', 'DE', 'noFS'],
	                	help="Type of feature selection (default RFE).\nChoices:\n1. Recursive Feature Elimination with cross validation (RFE)\n2. Permutation Importance (PI)\n3. Differential Expressed features (DE)\n4. No feature selection (noFS)")
	# RFE and PI CV choice
	parser.add_argument('-rfecv', '--rfecrossval', dest='rfecrossval', default='10CV', choices=['LOOCV', '5CV', '10CV'],
	                	help="Type of feature selection CV (default 10CV).\nChoices:\n1. LeaveOneOut CV (LOOCV)\n2. 5-fold CV (5CV)\n3. 10-fold CV (10CV)")
	# Minimum number of features to consider
	parser.add_argument('-mf', '--minfeature', dest='minfeature', default=5, type=int, 
	                	help="Minimum number of features to be considered\nduring feature selection (default 5)")
	# Type of CV when using RFE for feature selection
	parser.add_argument('-cl', '--classifiers', dest='classifiers', default='RF', choices=['RF', 'KNN', 'GB', 'ET', 'LG'], nargs="+",
	                	help="Type of classifier(s) to be used (default RF).\nYou can select one or more classifiers from the available ones.\nChoices:\
	                	     \n1. RF (RandomForestClassifier)\n2. KNN (KNeighborsClassifier)\n3. GB (GradientBoostingClassifier)\n4. ET (ExtraTreesClassifier)\n5. LG (LogisticRegression)")
	# Type of CV when training classifier
	parser.add_argument('-cv', '--crossval', dest='crossval', default='5CV', choices=['10CV', '5CV', '3CV'],
	                	help="Type of cross validation when evaluating\nthe classifier's performance (default 5CV).\nChoices:\n1. 10-fold CV (10CV)\n2. 5-fold CV (5CV)\n3. 3-fold CV (5CV)")

	
	##### GENERAL
	# Random 14-alphanumeric serving as project ID
	parser.add_argument('-id', '--projectID', dest='projectID', help="Random 14-alphanumeric serving as project ID")
	# nCounter instrument
	parser.add_argument('-i', '--instrument', dest='instrument', default='max-flex', choices=['max-flex', 'sprint'],
	                	help="Choose which nCounter Instrument you used (default max-flex)")
	# Number of threads to use
	parser.add_argument('-t', '--threads', dest='threads', default=10,
	                	help="Number of threads to use for the analysis (default 10)")
	# Display the version of the pipeline
	parser.add_argument('-v', '--version', action='version', version='%(prog)s {0}'.format(__version__))
	return parser.parse_args()

# Reading the input configuration file
def read_config(ini_path):
    
    config = configparser.ConfigParser()
    config.read(ini_path)
    return config

# Retrieving and define all variables from config file
def initialise_config(args):

	global CONFIG
	ini_config = read_config(args.config)

	CONFIG = {
		# General settings
        "projectID": ini_config['General'].get('projectID'),
        "instrument": ini_config['General'].get('instrument'),
        "threads": int(ini_config['General'].get('threads')),

        # IO settings
        "input_dir": ini_config['IO'].get('dir'),
        "clinical_data": ini_config['IO'].get('clinicaldata'),
        "output_dir": ini_config['IO'].get('outdir'),
        "control_group": ini_config['IO'].get('control'),
        "condition_group": ini_config['IO'].get('condition'),
        "test_type": ini_config['IO'].get('testtype'),
        "test_runs": [run.strip() for run in ini_config['IO'].get('testrun', '').split(",") if run.strip()],

        # EXTERNAL TEST SET settings
        "test_dir": ini_config['TestSet'].get('testdir'),
        "test_clinical_data": ini_config['TestSet'].get('testclinicaldata'),

        # DE settings
        "normalisation_method": ini_config['DE'].get('norm'),
        "lfc_threshold": float(ini_config['DE'].get('lfcthreshold')),
        "padjusted": float(ini_config['DE'].get('padjusted')),

        # RUVSeq settings
        "k_factor": int(ini_config['RUVSeq'].get('k_factor')),
        "reference_genes": ini_config['RUVSeq'].get('refgenes'),
        "min_reference_genes": int(ini_config['RUVSeq'].get('minref')),

        # Enrichment Analysis
        "ref_organism": ini_config['EnrichmentAnalysis'].get('reforg'),

        # Filter settings
        "filter_lowlyExpr_genes": ini_config.getboolean('Filters', 'filter_lowlyExpr_genes', fallback=True),
        "filter_genes_on_negCtrl": ini_config.getboolean('Filters', 'filter_genes_on_negCtrl', fallback=True),
        "filter_samples_on_negCtrl": ini_config.getboolean('Filters', 'filter_samples_on_negCtrl', fallback=True),
        "remove_outlier_samples": ini_config.getboolean('Filters', 'remove_outlier_samples', fallback=True),
        "iqr_cutoff": int(ini_config['Filters'].get('iqrcutoff')),

        # ML settings
        "remove_correlated": ini_config.getboolean('ML', 'correlated', fallback=True),
        "remove_quasiconstant": ini_config.getboolean('ML', 'quasiconstant', fallback=True),
        "feature_selection_method": ini_config['ML'].get('featureselection'),
        "rfe_cross_validation_type": ini_config['ML'].get('rfecrossval'),
        "min_num_features": int(ini_config['ML'].get('minfeature')),
        "classifiers": [cl.strip() for cl in ini_config['ML'].get('classifiers', '').split(",") if cl.strip()],
        "cross_validation_type": ini_config['ML'].get('crossval')
    	}

	return

# Function to write a log entry
def write_log(message, level="INFO", details=None):
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "message": message
    }
    if details:  # Add details if provided
        log_entry["details"] = details
    
    with open(log_file, 'a') as log:
        log.write(json.dumps(log_entry, indent=2) + '\n')  # Append each log entry as a JSON object

    return

# Initialize the JSON log file and write the initial configuration
def initialize_log():

	# Open the file in write mode to ensure it starts fresh
	with open(log_file, 'w') as log:
	    pass  # Create or clear the file without writing anything

	# Write the initialization log entry
	write_log(f"### INITIALISATION OF NANOINSIGHTS PROJECT {CONFIG['projectID']}", "INFO")

	# Log the entire CONFIG dictionary as JSON
	config_log_entry = {
	    "timestamp": datetime.now().isoformat(),
	    "level": "INFO",
	    "message": "Initial Configuration Settings",
	    "details": CONFIG
	}
	with open(log_file, 'a') as log:
	    log.write(json.dumps(config_log_entry, indent=2) + '\n')

	return

# Reading all RCC files
def read_RCC_files():
    
    write_log("### READING THE RCC FILES ###", "INFO")

    # Initialize variables
    files_in_batches = defaultdict(list)
    panel = None
    endogenous_genes, housekeeping_genes, positive_controls, negative_controls = set(), set(), set(), set()
    rcc_files = glob.glob(os.path.join(CONFIG["input_dir"], "*.RCC"))


    # Process each RCC file
    for file in rcc_files:
        try:
            with open(file, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    if line.startswith("GeneRLF"):
                        panel = panel or line.split(",")[1]
                    elif line.startswith("CartridgeID"):
                        run = line.split(",")[1].strip()
                        files_in_batches[run].append(os.path.basename(file))
                    elif line.startswith("Endogenous"):
                        endogenous_genes.add(line.split(",")[1])
                    elif line.startswith("Housekeeping"):
                        housekeeping_genes.add(line.split(",")[1])
                    elif line.startswith("Positive"):
                        positive_controls.add(line.split(",")[1])
                    elif line.startswith("Negative"):
                        negative_controls.add(line.split(",")[1])

        except Exception as e:
            write_log(f"Error processing file {file}: {str(e)}", "ERROR")


    # Log summary
    write_log(
    	"RCC file processing summary:",
    	"INFO", {
    		"Total RCC files detected": len(rcc_files),
	        "Panel detected": panel,
	        "Total batches detected": len(files_in_batches),
	        "Total endogenous genes": len(endogenous_genes),
	        "Total housekeeping genes": len(housekeeping_genes),
	        "Total positive controls": len(positive_controls),
	        "Total negative controls": len(negative_controls),
	})

    return dict(files_in_batches), rcc_files

# Reading the clinical data file
def read_clinical_data():
    
    write_log("### READING THE CLINICAL TABLE ###", "INFO")

    clinical_data_path = CONFIG["clinical_data"]
    if not clinical_data_path or not os.path.exists(clinical_data_path):
        write_log("Clinical data file is missing or not specified in the configuration.", "ERROR")

    try:
        # Load clinical data
        cl_data = pd.read_csv(clinical_data_path, delimiter=None, engine="python")
        write_log(f"Clinical data file loaded successfully: {clinical_data_path}", "INFO")

        # Validate number of columns
        if len(cl_data.columns) < 2:
            write_log(
                "Clinical Data Import Error: The clinical table import failed due to format issues, it seems that less than two columns exist. "
                "Please verify the format and re-upload.", 
                "ERROR"
            )

        # Validate unique conditions
        conditions = cl_data["Condition"].unique()
        if len(conditions) != 2:
            write_log(
                f"Clinical Data Error: Found {len(conditions)} unique conditions in the 'Condition' column. "
                "NanoInsights supports analysis of exactly 2 conditions. Please adjust and re-upload.",
                "ERROR"
            )

        # Check for control and condition labels
        if CONFIG["control_group"] in conditions and CONFIG["condition_group"] in conditions:
            # Count samples for each condition
            control_count = cl_data[cl_data["Condition"] == CONFIG["control_group"]].shape[0]
            condition_count = cl_data[cl_data["Condition"] == CONFIG["condition_group"]].shape[0]

            write_log(
            	"Balance check summary:",
            	"INFO", {
            		"Status": "Successful",
            		"Conditions examined": f"{CONFIG['control_group'].upper()} vs. {CONFIG['condition_group'].upper()}",
            		"Sample counts": f"Control group: {CONFIG['control_group']} ({control_count} samples) vs, Condition group: {CONFIG['condition_group']} ({condition_count} samples)"
        	})
        else:
            write_log(
                "BALANCE CHECK FAILED: Inconsistencies between provided conditions and entries in the 'Condition' column of the clinical data file.",
                "ERROR"
            )

        return cl_data

    except Exception as e:
        write_log(f"Error reading clinical data: {str(e)}", "ERROR")

    return

# Checking class balance between conditions
def check_class_balance(y_train):

	write_log("### IDENTIFYING THE CLASS BALANCE IN THE DATASET ###", "INFO")

	unique_classes, class_counts = np.unique(y_train, return_counts=True)

	# Calculate the percentage of samples in each condition
	cond1_prc = class_counts[0] / class_counts.sum() * 100
	cond2_prc = class_counts[1] / class_counts.sum() * 100
	write_log(f"Condition '{unique_classes[0]}': {class_counts[0]}/{len(y_train)} ({cond1_prc:.1f}%) Condition '{unique_classes[1]}': {class_counts[1]}/{len(y_train)} ({cond2_prc:.1f}%)", "INFO")

	write_log("Class balance check completed successfully.", "INFO")

	return

# Split data into training and test sets based on Runs or Split validation types
def split_training_test_sets(clinical_data, files_in_batches=None, overall_rcc_files=None):
    
    try:
        write_log(f"### SPLITTING DATASETS USING {CONFIG['test_type'].upper()} ###", "INFO")

        # Define output directories
        test_outdir = os.path.join(CONFIG["output_dir"], "test_set")
        training_outdir = CONFIG["input_dir"]  # Use input_dir for training set to avoid moving files
        os.makedirs(test_outdir, exist_ok=True)

        # Initialize variables
        trainingset_labels, testset_labels = [], []

        if CONFIG["test_type"] == "split":
            write_log("Performing 80/20 split for training and test sets.", "INFO")

            initial_clinical_labels = clinical_data[['Filename', 'Condition']]

            check_class_balance(initial_clinical_labels['Condition'].to_numpy())

            # Perform train-test split
            trainingset_labels, testset_labels = train_test_split(
                initial_clinical_labels,
                test_size=0.20,
                random_state=42,
                stratify=initial_clinical_labels["Condition"]
            )
            trainingset_labels = trainingset_labels["Filename"].tolist()
            testset_labels = testset_labels["Filename"].tolist()

            write_log(f"Number of samples after 80/20 split for test set: {len(testset_labels)}", "INFO")

        elif CONFIG["test_type"] == "run":
            write_log("Performing splitting based on specified runs.", "INFO")

            if not files_in_batches:
                error_msg = "Files in batches information is required for run-based splitting but not provided."
                write_log(error_msg, "ERROR")

            runs = [item for run in CONFIG["test_runs"] for item in files_in_batches.get(run, [])]
            if not runs:
                error_msg = "No matching files found for the specified runs. Please check your configuration."
                write_log(error_msg, "ERROR")

            write_log(f"Number of samples found in runs for test set: {len(runs)}", "INFO")
            trainingset_labels = list(set(os.path.basename(f) for f in overall_rcc_files) - set(runs))
            testset_labels = runs


        # Move files for test set and subset clinical data
        for test_file in testset_labels:
            shutil.move(os.path.join(CONFIG["input_dir"], test_file), test_outdir)

        # Overwrite the clinical data file with training set labels
        training_set_clinical_data = clinical_data[clinical_data['Filename'].isin(trainingset_labels)]
        training_set_clinical_data.to_csv(CONFIG["clinical_data"], sep=",", index=False)

        # Create clinical data for the test set
        test_set_clinical_data = clinical_data[clinical_data['Filename'].isin(testset_labels)]
        test_clinical_data_path = os.path.join(test_outdir, "clinical_data.csv")
        test_set_clinical_data.to_csv(test_clinical_data_path, sep=",", index=False)

        # Update CONFIG with test set paths
        CONFIG["test_dir"] = test_outdir
        CONFIG["test_clinical_data"] = test_clinical_data_path

        write_log("Training and test set files have been successfully created.", "INFO")

    except Exception as e:
        error_msg = f"An error occurred during dataset splitting: {str(e)}"
        write_log(error_msg, "ERROR")

    return

# Calculating the limit of detection QC
def calculate_limit_of_detection(raw_expression):
    
    try:
        write_log("### STARTING LIMIT OF DETECTION QC CALCULATION ###", "INFO")

        # Extract 'POS_E(0.5)' and transform to a DataFrame with BCAC_ID
        try:
            posE = raw_expression[raw_expression['Gene'] == 'POS_E(0.5)'].drop(columns=['Gene', 'Class']).transpose().apply(pd.to_numeric, errors='coerce').squeeze().reset_index()
            posE.columns = ['BCAC_ID', 'posE']
            write_log(f"'POS_E(0.5)' extracted with shape {posE.shape}.", "INFO")
        except Exception as e:
            write_log(f"Error extracting 'POS_E(0.5)': {e}", "ERROR")
            return pd.DataFrame(columns=['BCAC_ID', 'posE', 'logThreshold'])  # Return empty DataFrame in case of failure

        # Extract 'Negative' class genes and calculate threshold
        try:
            negatives = raw_expression[raw_expression['Class'] == 'Negative'].drop(columns='Class').sort_values(by='Gene')
            mean_values = negatives.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').mean()
            std_dev_values = negatives.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').std()
            threshold = (mean_values + 2 * std_dev_values).reset_index()
            threshold.columns = ['BCAC_ID', 'logThreshold']
            write_log(f"'Negative' class genes processed to calculate thresholds with shape {threshold.shape}.", "INFO")
        except Exception as e:
            write_log(f"Error processing 'Negative' class genes: {e}", "ERROR")
            return pd.DataFrame(columns=['BCAC_ID', 'posE', 'logThreshold'])  # Return empty DataFrame in case of failure

        # Merge posE and threshold on BCAC_ID
        try:
            combined_df = posE.merge(threshold, on='BCAC_ID', how='left')
            write_log(f"Combined QC DataFrame created with shape {combined_df.shape}.", "INFO")
            return combined_df
        except Exception as e:
            write_log(f"Error merging 'POS_E(0.5)' and threshold data: {e}", "ERROR")
            return pd.DataFrame(columns=['BCAC_ID', 'posE', 'logThreshold'])  # Return empty DataFrame in case of failure

    except Exception as e:
        write_log(f"Critical error in calculate_limit_of_detection: {e}", "ERROR")
        return pd.DataFrame(columns=['BCAC_ID', 'posE', 'logThreshold'])  # Return empty DataFrame in case of critical failure

    return 

# Calculating the r^2 for the positive linearity QC
def calculate_r_squared(col):
    
    x = np.arange(len(col))
    return linregress(x, col).rvalue ** 2

# Calculating the positive linearity QC
def calculate_positive_linearity(raw_expression, selected_columns):
    
    try:
        write_log("### STARTING POSITIVE LINEARITY QC CALCULATION ###", "INFO")

        # Define known log2 values
        log2known = [math.log2(x) for x in [128, 128 / 4, 128 / 16, 128 / 64, 128 / 256, 128 / (256 * 4)]]
        write_log(f"Known log2 values for positive linearity calculation: {log2known}", "INFO")

        # Filter for 'Positive' class and sort by 'Gene'
        log2pos = raw_expression[raw_expression['Class'] == 'Positive'].sort_values(by='Gene')
        if log2pos.empty:
            write_log("No 'Positive' class entries found in raw_expression. Returning empty DataFrame.", "WARNING")
            return pd.DataFrame(columns=['BCAC_ID', 'positiveLinearity'])

        write_log(f"Number of 'Positive' class genes found: {log2pos.shape[0]}", "INFO")

        # Convert selected columns to numeric and apply log2 transformation
        try:
            log2pos[selected_columns] = log2pos[selected_columns].apply(pd.to_numeric, errors='coerce')
            log2pos[selected_columns] = log2pos[selected_columns].apply(lambda x: x.apply(lambda y: math.log2(y) if y > 0 else 0))
            write_log(f"Log2 transformation applied to selected columns: {selected_columns}", "INFO")
        except Exception as e:
            write_log(f"Error during log2 transformation: {e}", "ERROR")

        # Calculate R-squared values and round
        try:
            r_squared_values = pd.DataFrame(log2pos.select_dtypes(include=[np.number]).apply(calculate_r_squared)).round(2)
            write_log("R-squared values calculated successfully.", "INFO")
        except Exception as e:
            write_log(f"Error during R-squared calculation: {e}", "ERROR")

        # Reset index and rename columns
        r_squared_values = r_squared_values.reset_index()
        r_squared_values.columns = ['BCAC_ID', 'positiveLinearity']
        write_log(f"Positive linearity QC DataFrame created with shape {r_squared_values.shape}.", "INFO")

        return r_squared_values

    except Exception as e:
        write_log(f"Critical error in calculate_positive_linearity: {e}", "ERROR")
        return pd.DataFrame(columns=['BCAC_ID', 'positiveLinearity'])  # Return empty DataFrame in case of error

    return

# Parsing a single RCC file and extract data
def parse_rcc_file(file_path, expression_dict, code_summary, is_first_file):

    filename = os.path.basename(file_path)[:-4]
    sample_attributes = {
        'BCAC_ID': filename, 'SampleID': filename, 'Owner': None, 'Comments': None, 'Date': None,
        'GeneRLF': None, 'SystemAPF': None, 'ID': None, 'FovCount': None, 'FovCounted': None,
        'ScannerID': None, 'StagePosition': None, 'BindingDensity': None, 'CartridgeID': None
    }
    
    count_data = []
    current_section = None

    with open(file_path, 'r') as file:
        for line in file:
            # Detect and handle section headers
            if line.startswith('<') and line.endswith('>\n'):
                current_section = line.strip('<>\n')
                continue

            # Extract data from relevant sections
            if current_section in ['Sample_Attributes', 'Lane_Attributes']:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    sample_attributes[key] = value if value else None

            elif current_section == 'Code_Summary':
                parts = line.strip().split(',')
                if not line.startswith('CodeClass') and len(parts) == 4:
                    # Add metadata only for the first file
                    if is_first_file:
                        code_summary['CodeClass'].append(parts[0])
                        code_summary['Name'].append(parts[1])
                        code_summary['Accession'].append(parts[2])
                    count_data.append(parts[3])

    # Append count data
    expression_dict[filename] = count_data
    return sample_attributes

# Extract all necessary info from the RCCs and output several files
def read_and_process_rcc_files(data_directory, datatype):
    
	try:
		write_log(f"### GENERATING THE RAW, pData AND RAW_EXPRESSION FOR THE {datatype.upper()} DATA SET ###", "INFO")

		# Gather RCC files
		rcc_files = glob.glob(os.path.join(data_directory, "*.RCC"))

		# Initialize data containers
		pData_list = []
		expression_dict = {}
		code_summary = defaultdict(list)

		# Process each RCC file
		write_log("Parsing RCC files", "INFO")
		for i, file_path in enumerate(rcc_files):
		    try:
		        sample_attributes = parse_rcc_file(file_path, expression_dict, code_summary, is_first_file=(i == 0))
		        pData_list.append(sample_attributes)
		    except Exception as e:
		        write_log(f"Error while parsing RCC file: {os.path.basename(file_path)} - {e}", "ERROR")
		write_log("Successfully parsed RCC files", "INFO")

		# Create summary DataFrame
		summary = pd.DataFrame(expression_dict)
		if code_summary:  # Populate metadata columns (CodeClass, Name, Accession)
		    for col_name in ['CodeClass', 'Name', 'Accession']:
		        summary.insert(0, col_name, code_summary[col_name])

		write_log(f"Summary DataFrame created with shape {summary.shape}.", "INFO")

		# Generate the raw DataFrame
		raw = summary.copy()
		try:
		    raw['Count'] = raw.iloc[:, 3:].apply(pd.to_numeric, errors='coerce').sum(axis=1)
		    raw.drop(raw.columns[3:-1], axis=1, inplace=True)
		    raw = raw[['CodeClass', 'Name', 'Accession', 'Count']]
		    raw_output_path = os.path.join(data_directory, "raw.tsv")
		    raw.to_csv(raw_output_path, sep="\t", index=False)
		    write_log(f"Raw data saved to {raw_output_path}", "INFO")
		except Exception as e:
		    write_log(f"Error while creating raw data: {e}", "ERROR")

		# Create the raw_expression DataFrame
		try:
		    raw_expression = summary.copy().drop(columns=['Accession']).rename(columns={'CodeClass': 'Class', 'Name': 'Gene'})
		    raw_expression = raw_expression[['Gene', 'Class'] + [col for col in raw_expression.columns if col not in ['Gene', 'Class']]]
		    raw_expression_output_path = os.path.join(data_directory, "raw_expression.tsv")
		    raw_expression.to_csv(raw_expression_output_path, sep="\t", index=False)
		    write_log(f"Raw Expression data saved to {raw_expression_output_path}", "INFO")
		except Exception as e:
		    write_log(f"Error while creating raw expression data: {e}", "ERROR")

		# Create and process pData
		try:
		    pData = pd.DataFrame(pData_list).replace('', 'NA')
		    pData = pData.rename(columns={'ID': 'LaneID', 'BindingDensity': 'bindingDensity'})
		    pData['SampleID'] = pData.apply(
		        lambda row: row['SampleID'].split(row['CartridgeID'])[1]
		        if row['CartridgeID'] in row['SampleID'] else row['SampleID'], axis=1
		    )
		    pData['SampleID'] = pData['SampleID'].str.replace(r'^[^A-Za-z0-9]+', '', regex=True)
		except Exception as e:
		    write_log(f"Error while preparing pData: {e}", "ERROR")
		    return

		# Add QC Calculations
		try:
		    selected_columns = [col for col in raw_expression.columns if col not in ['Gene', 'Class']]
		    r_squared_values = calculate_positive_linearity(raw_expression, selected_columns)
		    posE_and_threshold = calculate_limit_of_detection(raw_expression)

		    pData = pData.merge(r_squared_values, on='BCAC_ID', how='left')
		    pData['positiveLinearityQC'] = pData['positiveLinearity'] > 0.95

		    pData = pData.merge(posE_and_threshold, on='BCAC_ID', how='left')
		    pData['limitOfDetectionQC'] = pData['posE'] > pData['logThreshold']

		    # Add additional QC metrics
		    pData['bdThreshold'] = f"{bd_thresholds[0]}-{bd_thresholds[1]}"
		    pData['bindingDensityQC'] = pData['bindingDensity'].astype(float).between(bd_thresholds[0], bd_thresholds[1])
		    pData['CartridgeID'] = pData['CartridgeID'].str.replace(r'\s+', '', regex=True).str.replace(r'[Cc]atridge', '')
		    pData['imaging'] = (pData['FovCounted'].astype(float) / pData['FovCount'].astype(float)).round(2)
		    pData = pData.drop(columns=['FovCount', 'FovCounted'])
		    pData['imagingQC'] = pData['imaging'] >= 0.75

		    # Reorder and drop extra columns
		    desired_columns = [
		        'BCAC_ID', 'SampleID', 'Owner', 'Comments', 'Date', 'GeneRLF', 'SystemAPF', 'LaneID', 'ScannerID',
		        'StagePosition', 'CartridgeID', 'imagingQC', 'imaging', 'bindingDensityQC', 'bindingDensity',
		        'bdThreshold', 'positiveLinearityQC', 'positiveLinearity', 'limitOfDetectionQC', 'posE', 'logThreshold'
		    ]
		    pData = pData[desired_columns]

		    # Save the pData data to a TSV file
		    pData_output_path = os.path.join(data_directory, "pData.tsv")
		    pData.to_csv(pData_output_path, sep="\t", index=False)
		    write_log(f"pData dataframe was saved to {pData_output_path}", "INFO")
		except Exception as e:
		    write_log(f"Error while calculating QC or saving pData: {e}", "ERROR")

	except Exception as e:
	    write_log(f"Critical error in read_and_process_rcc_files: {e}", "ERROR")

	return

# Normalisation of the training or test set
def normalisation(data_dir, clinical_data, datatype, training_mat=None):
 
    write_log(f"### NORMALISATION ANALYSIS OF THE {datatype.upper()} SET ###", "INFO")

    # Determine R script and parameters based on datatype
    if datatype.lower() == "training":
        r_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trainingset_normalisation.R")
        additional_args = [
        	"--norm", CONFIG['normalisation_method'],  # Normalisation method to be applied
            "--filter_lowlyExpr_genes", str(CONFIG['filter_lowlyExpr_genes']).upper(),
            "--filter_genes_on_negCtrl", str(CONFIG['filter_genes_on_negCtrl']).upper(),
            "--remove_outlier_samples", str(CONFIG['remove_outlier_samples']).upper(),
            "--filter_samples_on_negCtrl", str(CONFIG['filter_samples_on_negCtrl']).upper(),
            "--iqrcutoff", str(CONFIG['iqr_cutoff']),  # IQR threshold
            "--lfcthreshold", str(CONFIG['lfc_threshold']),  # log2 FC threshold
            "--padjusted", str(CONFIG['padjusted']),  # FDR threshold
        ]
    elif datatype.lower() == "test":
        r_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testset_normalisation.R")
        additional_args = [
            "--training_mat", training_mat,  # Normalised matrix of the training set
        ]

    # Common arguments
    base_args = [
        "Rscript --vanilla", r_script,  # Calling the R normalisation script
        "--dir", data_dir,  # Path of dir hosting the RCC files
        "--clinicaldata", clinical_data,  # Clinical file
        "--currdir", os.path.abspath(os.path.dirname(r_script)),
        "--logfile", log_file,  # Input log file
        "--control", CONFIG['control_group'],  # Control group label
        "--condition", CONFIG['condition_group'],  # Condition group label
        "--upper_limit", str(bd_thresholds[1]), # Binding Density range upper limit
        "--k_factor", str(CONFIG['k_factor']),  # k factor for RUVSeq
        "--refgenes", CONFIG['reference_genes'],  # Ref. genes to choose for RUVSeq
        "--minref", str(CONFIG['min_reference_genes']), # Min. reference genes for RUVSeq
    ]

    # Combine all arguments
    command = " ".join(base_args + additional_args)

    # Execute the R script
    try:
        write_log(f"Running R script for {datatype} set: {command}", "INFO")

        subprocess.run(command, shell=True, check=True)
        write_log(f"{datatype.capitalize()} set normalisation completed successfully.", "INFO")
    
    except subprocess.CalledProcessError as e:
        write_log(f"NORMALISATION ({datatype.upper()} SET): ERROR - {e}", "ERROR")
        return None
    
    except Exception as e:
        write_log(f"NORMALISATION ({datatype.upper()} SET): ERROR - An unexpected error occurred: {e}", "ERROR")
        return None

    return

# Handle processing for 'onlyNorm' validation type
def handle_only_normalization():

    write_log("### VALIDATION TYPE: ONLY NORMALIZATION ###", "INFO")
    read_and_process_rcc_files(CONFIG["input_dir"], "Normalisation Only")
    normalisation(CONFIG["input_dir"], CONFIG["clinical_data"], "training")

    return 

# Handle processing for 'split', 'run' or 'extSet' test types
def handle_testset(clinical_data, files_in_batches, overall_rcc_files):
	
	if CONFIG["test_type"] == "split":
		write_log("### TEST TYPE: SPLIT ###", "INFO")
		split_training_test_sets(clinical_data)
	
	elif CONFIG["test_type"] == "run":
		write_log("### TEST TYPE: RUN ###", "INFO")
		split_training_test_sets(clinical_data, files_in_batches, overall_rcc_files)
	else:
		write_log("### TEST TYPE: EXTERNAL TEST SET ###", "INFO")

	read_and_process_rcc_files(CONFIG["input_dir"], "training")
	normalisation(CONFIG["input_dir"], CONFIG["clinical_data"], "training")

	read_and_process_rcc_files(CONFIG["test_dir"], "test")
	training_set_matrix = glob.glob(os.path.join(CONFIG["output_dir"], "output_results", "trainingset_normalisation", "*-normalised.matrix.ml.tsv"))
	normalisation(CONFIG["test_dir"], CONFIG["test_clinical_data"], "test", training_set_matrix[0])
	return

# Check if the specified file(s) are available. Log and raise an error if not
def check_file_availability(file_paths, message):

    if not file_paths:
        error_msg = f"{message} - Required file(s) not found."
        write_log(error_msg, "ERROR")
    else:
        write_log(f"{message} - File(s) found: {file_paths}", "INFO")

    return 

# Compress the entire output_results folder into a zip file
def compress_results():
    try:
        write_log("Starting compression of output results directory", "INFO")
        
        # Define the folder to compress and the output zip file path
        output_folder = os.path.join(CONFIG["output_dir"], "output_results")
        zip_file_name = f"{CONFIG['projectID']}_nanoinsights_results.zip"
        zip_file_path = os.path.join(CONFIG["output_dir"], zip_file_name)
        
        # Compress the folder
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_folder):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, output_folder)
                    zipf.write(full_path, arcname=relative_path)
        
        write_log(f"Output results directory successfully compressed to {zip_file_path}", "INFO")
    
    except Exception as e:
        write_log(f"Error while compressing results: {str(e)}", "ERROR")

    return

# Logs the elapsed time for the NanoInsights analysis
def log_elapsed_time(project_id, start_time):

    elapsed_time = datetime.now() - start_time

    # Convert elapsed time to days, hours, and minutes
    days = elapsed_time.days
    hours, remainder = divmod(elapsed_time.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    # Create the log message
    message = f"NanoInsights analysis of project {project_id} was finalised after {days} days, {hours} hours, and {minutes} minutes."

    # Write to the log
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": "INFO",
        "message": message
    }
    with open(log_file, 'a') as log:
        log.write(json.dumps(log_entry, indent=2) + '\n')  # Append log entry as JSON

    return



def main():

	global log_file, bd_thresholds

	args = parse_arguments()  # Initialise arguments
	initialise_config(args)  # Initialise configuration
	
	# JSON logging file
	log_file = os.path.join(CONFIG["output_dir"], f"{CONFIG['projectID']}_log.json")
	initialize_log()  # Call the function to initialize logging
	
	# Define binding density thresholds
	bd_thresholds = {"max-flex": [0.1, 2.25], "sprint": [0.1, 1.8]}.get(CONFIG['instrument'])

	# Reading the input RCC and clinical data files
	try:
		files_in_batches, overall_rcc_files = read_RCC_files()
		clinical_data = read_clinical_data()
	except Exception as e:
	    write_log(f"Error reading input files: {e}", "ERROR")


	# Determine validation type and process accordingly
	if CONFIG["test_type"] == "onlyNorm":
		handle_only_normalization()
	else:
		handle_testset(clinical_data, files_in_batches, overall_rcc_files)

		# Running classification analysis
		try:
		    write_log("### CLASSIFICATION ANALYSIS (PYTHON) ###", "INFO")
		    
		    # Dynamically determine the script location
		    ml_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nanoinsights_ml.py")

		    # Prepare the command using sys.executable for the current Python interpreter
		    ml_command = [sys.executable, ml_script, "--config", args.config]

		    # Log the assembled command
		    assembled_command = " ".join(ml_command)
		    write_log(f"Running Machine Learning command: {assembled_command}", "INFO")

		    # Execute the command
		    result = subprocess.run(ml_command, text=True, capture_output=True)

		    # Check subprocess results
		    if result.returncode != 0:
		        write_log(f"ERROR: Machine Learning script failed.\n{result.stderr}", "ERROR")
		    else:
		        write_log(f"Machine Learning script completed successfully:\n{result.stdout}", "INFO")

		except Exception as e:
		    write_log(f"CLASSIFICATION TASK ERROR: {str(e)}", "ERROR")

	compress_results()

	log_elapsed_time(CONFIG['projectID'], startTime)

if __name__ == "__main__": main()