### ML ANALYSING FOR NANOSTRING DATA ###
## ESR11 - Stavros Giannoukakos 


#Version of the program
__version__ = "v1.0.0"

usage = "python3.11 nanoinsights_ml.py [options] -c config.init"
epilog = " -- October 2023 | Stavros Giannoukakos -- "
description = "DESCRIPTION"

# Importing required libraries
import os
import json
import glob
import numpy as np
import pandas as pd
import urllib.parse
import configparser, argparse
from datetime import datetime
from gprofiler import GProfiler
# Importing plotting libraries
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.rcParams['font.family']='Arial'
from plotly.subplots import make_subplots
import seaborn as sns
# Importing ML libraries
import sklearn
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.metrics import auc, roc_curve, accuracy_score, roc_auc_score, balanced_accuracy_score, RocCurveDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, average_precision_score, class_likelihood_ratios
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, matthews_corrcoef, log_loss, brier_score_loss
from fast_ml.feature_selection import get_constant_features
# Importing ML classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
random_seed = 42


startTime = datetime.now()  # Start timing the execution
CONFIG = {}  # Global configuration dictionary



# Argument parser for command-line options 
def parse_arguments():

	parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, usage=usage, description=description, epilog=epilog)
	# Path of the input configuration file 
	parser.add_argument("-c", "--config",  required=True, help="Path to the INI configuration file")

	#### IO
	# Path of the output directory hosting the results
	parser.add_argument('-o', '--outdir', dest='outdir', help="Path of the directory that will containing all the analyses")
	# Control group
	parser.add_argument('-ctrl', '--control', dest='control', 
						help="For the two-group comparison, which\nlabel should correspond to the CONTROL group (e.g. Healthy)")
	# Condition group
	parser.add_argument('-cond', '--condition', dest='condition', 
						help="For the two-group comparison, which\nlabel should correspond to the CONDITION (e.g. Cancer)")

	##### Enrichment Analysis
	# Reference organism for enrichment analysis
	parser.add_argument('-r', '--ref_organism', dest='ref_organism', default="hsapiens",
	                	help="Choose the reference organism for performing enrichment analysis\n(default hsapiens)")

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
		## General settings
        "projectID": ini_config['General'].get('projectID'),
        "threads": int(ini_config['General'].get('threads')),

        ## IO settings
        "output_dir": ini_config['IO'].get('outdir'),
        "control_group": ini_config['IO'].get('control').capitalize(),
        "condition_group": ini_config['IO'].get('condition').capitalize(),
        "conditions": [ini_config['IO'].get('control').capitalize(), ini_config['IO'].get('condition').capitalize()],

        # Enrichment Analysis
        "ref_organism": ini_config['EnrichmentAnalysis'].get('reforg'),

        ## ML settings
        "remove_correlated": ini_config.getboolean('ML', 'correlated', fallback=True),
        "remove_quasiconstant": ini_config.getboolean('ML', 'quasiconstant', fallback=True),
        "feature_selection_method": ini_config['ML'].get('featureselection'),
        "rfe_cross_validation_type": ini_config['ML'].get('rfecrossval'),
        "min_num_features": int(ini_config['ML'].get('minfeature')),
        "classifiers": [cl.strip() for cl in ini_config['ML'].get('classifiers', '').split(",") if cl.strip()],
        "cross_validation_type": ini_config['ML'].get('crossval'),

        # Directories 
        # "html_dir": os.path.join(ini_config['IO'].get('outdir'), "html_results"),
        "html_training": os.path.join(ini_config['IO'].get('outdir'), "html_results", "trainingset_normalisation"),
        "html_test": os.path.join(ini_config['IO'].get('outdir'), "html_results", "testset_normalisation"),
        "html_classification": os.path.join(ini_config['IO'].get('outdir'), "html_results", "classification"),
        "html_feature_selection": os.path.join(ini_config['IO'].get('outdir'), "html_results", "classification", "feature_selection"),
        
        # "results_dir": os.path.join(ini_config['IO'].get('outdir'), "output_results"),
        "results_training": os.path.join(ini_config['IO'].get('outdir'), "output_results", "trainingset_normalisation"),
        "results_test": os.path.join(ini_config['IO'].get('outdir'), "output_results", "testset_normalisation"),
        "results_training_classification": os.path.join(ini_config['IO'].get('outdir'), "output_results", "trainingset_classification"),
        "results_classification": os.path.join(ini_config['IO'].get('outdir'), "output_results", "classification"),
        "results_feature_selection": os.path.join(ini_config['IO'].get('outdir'), "output_results", "classification", "feature_selection"),
    	}

	# Transform rfe_cross_validation_type into the appropriate object
	CONFIG["rfe_cross_validation_type"] = (
	    LeaveOneOut() if CONFIG["rfe_cross_validation_type"] == 'LOOCV' else
	    StratifiedKFold(5) if CONFIG["rfe_cross_validation_type"] == '5CV' else
	    StratifiedKFold(10)
	)

    # Transform cross_validation_type into the appropriate integer or object
	CONFIG["cross_validation_type"] = (
	    10 if CONFIG["cross_validation_type"] == '10CV' else
	    5 if CONFIG["cross_validation_type"] == '5CV' else
	    3 if CONFIG["cross_validation_type"] == '3CV' else
	    LeaveOneOut()
	)

	# Define classifiers with their full configurations
	available_classifiers = {
	    'RF': (RandomForestClassifier(random_state=random_seed, class_weight='balanced', n_jobs=int(CONFIG["threads"])), 'Random Forest Classifier'),
	    'ET': (ExtraTreesClassifier(random_state=random_seed, class_weight='balanced', n_jobs=int(CONFIG["threads"])), 'Extra Trees Classifier'),
	    'GB': (GradientBoostingClassifier(random_state=random_seed), 'Gradient Boosting Classifier'),
	    'KNN': (KNeighborsClassifier(n_jobs=int(CONFIG["threads"])), 'K-Nearest Neighbors Classifier'),
	    'LG': (LogisticRegression(random_state=random_seed, class_weight='balanced', solver='liblinear'), 'Logistic Regression Classifier')
	}


	# Filter only the chosen classifiers
	CONFIG["classifiers"] = {k: available_classifiers[k] for k in CONFIG["classifiers"] if k in available_classifiers}

    # Create necessary directories
	os.makedirs(CONFIG["html_classification"], exist_ok=True)
	os.makedirs(CONFIG["html_feature_selection"], exist_ok=True)
	os.makedirs(CONFIG["results_classification"], exist_ok=True)
	os.makedirs(CONFIG["results_feature_selection"], exist_ok=True)
	
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

# Reading the normalised data and labels for training and test datasets
def read_normalised_data():

    try:
        write_log("Reading normalised data and labels for training and test datasets...", "INFO")
        
        # File names for normalised data and labels
        files = {
			"All_Endogenous_Scaled": "all_endogenous_scaled-normalised.matrix.ml.tsv",
			"geNorm_Housekeeping": "geNorm_housekeeping-normalised.matrix.ml.tsv",
			"Housekeeping_Scaled": "housekeeping_scaled-normalised.matrix.ml.tsv",
			"LOESS": "loess-normalised.matrix.ml.tsv",
			"logTransformed": "logtransf-normalised.matrix.ml.tsv",
			"MinMax": "minmax-normalised.matrix.ml.tsv",
			"Quantile": "quantile-normalised.matrix.ml.tsv",
			"RUV": "ruv-normalised.matrix.ml.tsv",
			"TPM": "tpm-normalised.matrix.ml.tsv",
			"VSN": "vsn-normalised.matrix.ml.tsv",
		    "z-scores": "zscores-normalised.matrix.ml.tsv",
		    "labels": "labels.ml.tsv",
		}

        # Initialize storage for training and test datasets
        training_data = {}
        test_data = {}
        training_labels = None
        test_labels = None

        for norm_method, filename in files.items():
            
            try:
                training_file_path = os.path.join(CONFIG['results_training_classification'], filename)
                test_file_path = os.path.join(CONFIG['results_test'], filename)

                if norm_method == "labels":  # Read labels from the training and test directories
                    training_labels = pd.read_csv(training_file_path, sep = "\t", index_col = 0)
                    test_labels = pd.read_csv(test_file_path, sep = "\t", index_col = 0)
                    write_log(f"Labels loaded successfully for {norm_method}.", "INFO")
                else:  # Read normalised matrices for training and test
                    train_matrix = pd.read_csv(training_file_path, sep = "\t", index_col = 0).transpose().sort_index()
                    test_matrix = pd.read_csv(test_file_path, sep = "\t", index_col = 0).transpose().sort_index()

                    # Apply feature filtering if conditions are met
                    if (CONFIG['remove_correlated'] or CONFIG['remove_quasiconstant']) and CONFIG['feature_selection_method'] != "DE":
                        write_log(f"Applying feature filtering for {norm_method} normalisation...", "INFO")
                        train_matrix = filter_features(train_matrix, remove_correlated = CONFIG['remove_correlated'], remove_quasi_constant = CONFIG['remove_quasiconstant'])
                        
                        # Ensure test data matches filtered training features
                        test_matrix = test_matrix[train_matrix.columns]

                    # Store the processed data
                    training_data[norm_method] = train_matrix
                    test_data[norm_method] = test_matrix
                    write_log(f"Data loaded and filtered for {norm_method}.", "INFO")

            except Exception as e:
                write_log(f"Error loading or processing {norm_method} data: {str(e)}", "ERROR")

        write_log("All datasets loaded and processed successfully.", "INFO")
        return training_data, test_data, training_labels, test_labels

    except Exception as e:
        write_log(f"Error in read_normalised_data: {str(e)}", "ERROR")

    return

# Filtering out correlated and quasi-constant features
def filter_features(X, remove_correlated=True, remove_quasi_constant=True, correlation_threshold=0.95, quasi_constant_threshold=0.99):
    
    try:
        write_log("Starting feature filtering...", "INFO")
        X_filtered = X.copy()

        # Step 1: Remove highly correlated features
        if remove_correlated:
            write_log(f"Removing features with correlation > {correlation_threshold}...", "INFO")
            
            corr_matrix = X_filtered.corr(method = 'spearman').abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            correlated_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
            X_filtered.drop(columns = correlated_features, inplace=True)
            
            write_log(f"Removed {len(correlated_features)} correlated features.", "INFO")

        # Step 2: Remove quasi-constant features
        if remove_quasi_constant:
        	write_log(f"Removing quasi-constant features with threshold: {quasi_constant_threshold}...", "INFO")

        	constant_features = get_constant_features(X_filtered, threshold=quasi_constant_threshold)['Var'].to_list()
        	X_filtered.drop(columns=constant_features, inplace=True)  # Drop quasi-constant features
        	write_log(f"Removed {len(constant_features)} quasi-constant features.", "INFO")

        write_log(f"Feature filtering completed. {X.shape[1]} -> {X_filtered.shape[1]} remaining features.", "INFO")
        return X_filtered

    except Exception as e:
        write_log(f"Error during feature filtering: {str(e)}", "ERROR")

    return

# Evaluate multiple normalisation methods to determine which performs the best
def evaluate_normalisation_methods(training_data, training_labels, test_data, test_labels):

	try:
		write_log("Starting evaluation of normalisation methods...", "INFO")

		results = {}

		# Split the training data into sub-training and validation sets
		train_idx, val_idx = train_test_split(training_labels.index, test_size=0.2, stratify=training_labels["Condition"], random_state=random_seed)

		# Subset labels for training and validation
		y_train = training_labels.loc[train_idx, "Condition"]
		y_val = training_labels.loc[val_idx, "Condition"]

		# Iterate over normalisation methods
		for norm_method, X_train_full in training_data.items():
            
			write_log(f"Evaluating normalisation method: {norm_method}", "INFO")

			X_train = X_train_full.loc[train_idx, :]
			X_val = X_train_full.loc[val_idx, :]
			
			try:
				# Apply feature selection
				if CONFIG["feature_selection_method"] == "RFE":  # Recursive Feature Elimination (with CV)

					try:
						write_log(f"Performing RFECV for normalisation method: {norm_method}...", "INFO")

						# Set up RFECV
						selector = RFECV(estimator=RandomForestClassifier(random_state=random_seed, class_weight="balanced"),
						                 min_features_to_select=CONFIG["min_num_features"],
						                 scoring="balanced_accuracy",
						                 cv=CONFIG["rfe_cross_validation_type"],
						                 step=1, n_jobs=CONFIG["threads"])

						selector.fit(X_train, y_train)
						selected_features = X_train.columns[selector.support_]

						# Summarize results
						feature_importances = np.zeros(X_train.shape[1])
						if hasattr(selector.estimator_, "feature_importances_"):
						    feature_importances[selector.support_] = selector.estimator_.feature_importances_

						features_df = pd.DataFrame({"Feature": X_train.columns,
												    "Ranking": selector.ranking_,
												    "Selected": selector.support_,
												    "Feature_Importance": feature_importances})

						# Save to file
						features_file = os.path.join(CONFIG["results_feature_selection"], f"rfecv.selected_features.{norm_method}.tsv")
						features_df.to_csv(features_file, sep="\t", index=False)
						write_log(f"RFECV results saved to {features_file}", "INFO")

						generate_rfecv_plot(selector, X_train, norm_method)

					except Exception as e:
					    write_log(f"Error during RFECV for {norm_method}: {str(e)}", "ERROR")
					    selected_features = []

				elif CONFIG["feature_selection_method"] == "PI":  # Permutation Importance

					try:
						write_log(f"Performing Permutation Importance for normalisation method: {norm_method}", "INFO")

						# Train RandomForestClassifier
						model = RandomForestClassifier(random_state=random_seed, class_weight="balanced", n_jobs=CONFIG["threads"])
						model.fit(X_train, y_train)
						write_log("RandomForestClassifier trained successfully for Permutation Importance.", "INFO")

						# Compute Permutation Importance
						n_repeats = 150
						result = permutation_importance(model, X_val, y_val, n_repeats=n_repeats, 
						                                scoring="balanced_accuracy", random_state=random_seed, 
						                                n_jobs=CONFIG["threads"])

						# Extract statistics
						importance_mean = result.importances_mean
						importance_std = result.importances_std
						features_ranking = np.argsort(-importance_mean)  # Descending sort based on mean importance

						# Identify selected features
						selected_features = X_train.columns[importance_mean > 0]

						# Summarize feature importance results
						features_df = pd.DataFrame({"Feature": X_train.columns,
												    "Importance_Mean": importance_mean,
												    "Importance_Std": importance_std,
												    "Selected": importance_mean > 0}).sort_values(by="Importance_Mean", ascending=False)

						# Save the results to a file
						feature_file = os.path.join(CONFIG["results_feature_selection"], f"permFeatImp.selected_features.{norm_method}.tsv")
						features_df.to_csv(feature_file, sep="\t", index=False)
						write_log(f"Permutation Importance results saved to {feature_file}", "INFO",
						          details={"num_selected_features": len(selected_features)})

						# Generate an importance plot
						generate_permutation_importance_plots(features_df, norm_method)

					except Exception as e:
					    write_log(f"Error during Permutation Importance for {norm_method}: {str(e)}", "ERROR")
					    selected_features = []

				elif CONFIG["feature_selection_method"] == "DE":  # DE Genes
				    
				    try:
				        write_log(f"Performing feature selection using DE results for normalisation method: {norm_method}", "INFO")
				        
				        # Locate the DE file
				        de_files = glob.glob(os.path.join(CONFIG["results_training"], "*-normalised.matrix.de.ml.tsv"))
				        if not de_files:
				        	write_log("No DE file found matching the pattern '*-normalised.matrix.de.ml.tsv' in results_training.", "ERROR")
				        
				        de_file = de_files[0]
				        write_log(f"Using DE file: {de_file}", "INFO")
				        
				        # Read the DE file to obtain selected features
				        de_matrix = pd.read_csv(de_file, sep="\t", usecols=[0])
				        selected_features = de_matrix.iloc[:, 0].tolist()
				        
				        write_log(f"DE-based feature selection completed", "INFO",
				        	details={"num_selected_features": len(selected_features), "selected_features": selected_features})
				    
				    except Exception as e:
				        write_log(f"Error during DE-based feature selection: {str(e)}", "ERROR")

				elif CONFIG["feature_selection_method"] == "noFS":  # No Feature Selection

					write_log(f"NO feature selection was applied for normalisation method: {norm_method}", "INFO")

					# No feature selection, use all features
					selected_features = X_train.columns

				# Log the number of selected features and the features themselves
				log_details = {"normalisation_method": norm_method, "num_selected_features": len(selected_features)}

				if CONFIG["feature_selection_method"] != "noFS":
				    log_details["selected_features"] = list(selected_features)

				write_log(f"Feature selection using {CONFIG['feature_selection_method']} completed.", "INFO", details=log_details)

				# Reduce training and validation sets to selected features
				X_train_fs = X_train[selected_features]
				X_val_fs = X_val[selected_features]

				# Train and evaluate classifiers
				for clf_name, (clf, _) in CONFIG["classifiers"].items():
				    write_log(f"Training {clf_name} with {norm_method} normalisation...", "INFO")
				    clf.fit(X_train_fs, y_train)

				    # Validation predictions
				    y_val_pred = clf.predict(X_val_fs)
				    y_val_proba = clf.predict_proba(X_val_fs)[:, 1]

				    # Calculate metrics
				    accuracy = accuracy_score(y_val, y_val_pred)
				    balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
				    auc = roc_auc_score(y_val, y_val_proba)

				    # Compute composite score
				    composite_score = 0.6 * balanced_acc + 0.4 * auc

				    # Log metrics
				    write_log(f"Metrics for {clf_name} on {norm_method}: Accuracy={accuracy}, Balanced Accuracy={balanced_acc}, AUC={auc}", "INFO")

				    # Store results
				    results[(norm_method, clf_name)] = {"accuracy": accuracy, 
				    									"balanced_accuracy": balanced_acc, 
				    									"roc_auc": auc, 
				    									"composite_score": composite_score,
				    									"num_of_selected_features": len(selected_features),
				    									"selected_features": ", ".join(selected_features)}

			except Exception as e:
			    write_log(f"Error during evaluation of {norm_method}: {str(e)}", "ERROR")

		# Determine the best normalisation method based on composite score
		best_norm, best_clf = max(results, key=lambda x: results[x]["composite_score"])
		write_log(f"Best normalisation: {best_norm}, Best classifier: {best_clf}", "INFO")

		# Retrieve selected features for the best normalization and classifier
		selected_features = results[(best_norm, best_clf)].get("selected_features", "N/A").split(", ")

		# Retrain the best classifier on full training data and evaluate on test set
		X_train_best = training_data[best_norm]
		X_test_best = test_data[best_norm]

		X_train_best_fs = X_train_best[selected_features]
		X_test_best_fs = X_test_best[selected_features]

		best_model = CONFIG["classifiers"][best_clf][0]
		best_model.fit(X_train_best_fs, training_labels["Condition"])

		y_test_pred = best_model.predict(X_test_best_fs)
		y_test_proba = best_model.predict_proba(X_test_best_fs)[:, 1]

		# Calculate test metrics
		test_accuracy = accuracy_score(test_labels["Condition"], y_test_pred)
		test_balanced_accuracy = balanced_accuracy_score(test_labels["Condition"], y_test_pred)
		test_auc = roc_auc_score(test_labels["Condition"], y_test_proba)
		test_composite_score = 0.6 * test_balanced_accuracy + 0.4 * test_auc
		
		write_log(f"Test set metrics for best normalisation ({best_norm}) and classifier ({best_clf}):", "INFO",
		          details={"accuracy": test_accuracy, "balanced_accuracy": test_balanced_accuracy, "auc": test_auc, "composite_score": test_composite_score})

		# Add final test results to results
		results["final_test"] = {"best_normalisation": best_norm,
							    "best_classifier": best_clf,
							    "test_accuracy": test_accuracy,
							    "test_balanced_accuracy": test_balanced_accuracy,
							    "test_auc": test_auc,
							    "test_composite_score": test_composite_score,
							    "selected_features": selected_features}
		
		# Prepare results data for saving
		results_data = []
		for key, res in results.items():
		    if key == "final_test":
		        continue  # Skip "final_test" key
		    norm, clf = key  # Unpack normalisation method and classifier
		    results_data.append({"Normalization": norm,
						    	"Classifier": clf,
						        "Accuracy": res["accuracy"],
						        "Balanced Accuracy": res["balanced_accuracy"],
						        "AUC": res["roc_auc"],
						        "Composite Score": res["composite_score"],
						        "Num. of Selected Features": res["num_of_selected_features"],
						        "Selected Features": res.get("selected_features", "N/A")})

		# Add the final test results explicitly
		final_test = results["final_test"]
		results_data.append({"Normalization": final_test["best_normalisation"],
						    "Classifier": final_test["best_classifier"],
						    "Accuracy": final_test["test_accuracy"],
						    "Balanced Accuracy": final_test["test_balanced_accuracy"],
						    "AUC": final_test["test_auc"],
						    "Composite Score": final_test["test_composite_score"],
						    "Num. of Selected Features": "N/A (Test Set)",
						    "Selected Features": "N/A (Test Set)"})

		# Convert to DataFrame and save
		try:
		    results_file = os.path.join(CONFIG['results_classification'], "evaluate_normalisation_methods_results.tsv")
		    results_df = pd.DataFrame(results_data)

		    # Remove extra/empty rows or columns
		    results_df = results_df.dropna(how='all')  # Drop rows where all values are NaN
		    results_df = results_df.drop_duplicates()  # Remove duplicate rows

		    # Save to TSV file
		    results_df.to_csv(results_file, sep="\t", index=False)
		    write_log(f"Results saved to {results_file}", "INFO")

		except Exception as e:
		    write_log(f"Error saving results to file: {str(e)}", "ERROR")


		return results

	except Exception as e:
	    write_log(f"Error during normalisation evaluation: {str(e)}", "ERROR")

	return

# Train, validate, and test a classifier using cross-validation and generate performance plots
def train_validate_test(X_train, y_train, X_test, y_test, classifier, classifier_name, classifier_abbr, optimal_features):

    try:
        write_log(f"Starting training, validating, and testing the selected classifier: {classifier_name}", "INFO")
        
        # Safeguard: Check optimal features
        if not optimal_features:
            write_log(f"No optimal features were selected for training", "ERROR")
            return

        # Subset features
        X_train_selected = X_train[optimal_features]
        X_test_selected = X_test[optimal_features]

        # Initialize cross-validation structures
        skf = StratifiedKFold(n_splits=CONFIG["cross_validation_type"], shuffle=True, random_state=random_seed)
        tprs, mean_fpr = [], np.linspace(0, 1, 100)
        training_stats, training_roc = {}, {}

        # Prepare ROC curve plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.rcParams.update({'font.size': 9})

        # Cross-validation loop
        for i, (train_idx, val_idx) in enumerate(skf.split(X_train_selected, y_train), 1):
            X1_train, X1_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
            y1_train, y1_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            classifier.fit(X1_train, y1_train)

            # Calculate metrics
            if hasattr(classifier, "predict_proba"):
                y_proba = classifier.predict_proba(X1_val)[:, 1]
            else:
            	write_log(f"Classifier {classifier_name} does not support predict_proba.", "ERROR")

            scores = calculate_scores(y1_val, classifier.predict(X1_val), y_proba)
            training_stats[f"Fold{i}"] = scores
            
            # Static ROC plot for each fold
            viz = RocCurveDisplay.from_estimator(classifier, X1_val, y1_val, ax=ax, alpha=0.2, lw=1, label=f"Fold {i} | AUC={scores['auc_score']:.2f}", pos_label=CONFIG["condition_group"])
            
            # Interactive ROC preparation (calculate fpr and tpr)
            fpr, tpr, _ = roc_curve(y1_val, y_proba, pos_label=CONFIG["condition_group"])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            training_roc[f"Fold{i}"] = {"fpr": fpr, "tpr": tpr, "auc": scores['auc_score']}


        # Compute mean ROC and standard deviation
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std([fold['auc_score'] for fold in training_stats.values()])

        # Plot mean ROC curve
        ax.plot(mean_fpr, mean_tpr, color='#eb5e28', label=f"Mean Training Set | AUC={mean_auc:.2f} ± {std_auc:.2f}", lw=1.5, alpha=0.6)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#C9CCD5', alpha=0.2, label="± 1 std. dev.")

        # Plot chance line
        ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#7D1935', label='Chance', alpha=0.5)

        # Final training and testing
        classifier.fit(X_train_selected, y_train)
        y_test_pred = classifier.predict(X_test_selected)
        y_test_proba = classifier.predict_proba(X_test_selected)[:, 1]
        test_scores = calculate_scores(y_test, classifier.predict(X_test_selected), classifier.predict_proba(X_test_selected)[:, 1])

        # Compute ROC curve for the test set
        fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba, pos_label=CONFIG["condition_group"])

        # Prepare training_roc dictionary for interactive plot
        training_roc["Mean TrainingSet"] = {"fpr": mean_fpr, "tpr": mean_tpr, "auc": mean_auc, "std_auc": std_auc}
        training_roc["TestSet"] = {"fpr": fpr_test, "tpr": tpr_test, "auc": test_scores['auc_score']}

        # Save ROC stats for overall ROC plot
        rocStats[(classifier_name, "TrainingSet")] = (mean_fpr, mean_tpr, mean_auc, std_auc)
        rocStats[(classifier_name, "TestSet")] = (fpr_test, tpr_test, test_scores['auc_score'], 0.0) 


        # Generate testing stats
        class_labels = classifier.classes_.tolist()
        testing_stats = pd.DataFrame({"Sample": X_test_selected.index.values,
						            "True_label": y_test.tolist(),
						            "Predicted_label": y_test_pred.tolist(),
						            f"Probability_{class_labels[0]}": np.round(classifier.predict_proba(X_test_selected)[:, 0], 2),
						            f"Probability_{class_labels[1]}": np.round(classifier.predict_proba(X_test_selected)[:, 1], 2)})
        testing_stats["Results"] = testing_stats["True_label"] == testing_stats["Predicted_label"]

        # Save test stats to file
        testing_stats_file = os.path.join(CONFIG['results_classification'], f"{classifier_abbr}.TestSet.PredResults.tsv")
        testing_stats.to_csv(testing_stats_file, sep="\t", index=False)
        write_log(f"Training and Testing stats saved to {testing_stats_file}", "INFO")
        
        # Generate advanced classification metrics
        advanced_classification_metrics(test_scores, classifier_abbr=classifier_abbr, y_true=y_test, y_pred=y_test_pred, y_prob=y_test_proba)

        # Generate Model Evaluation plots
        write_log("Generating model evaluation plots (confusion matrix and probability plots)...", "INFO")
        model_evaluation_plots(test_scores, classifier_name, classifier_abbr, testing_stats)
        write_log("Model evaluation plots generated successfully.", "INFO")

        # Finalize ROC plot
        ax.plot(fpr_test, tpr_test, color="#226f54", lw=1.5, alpha=0.6, label=f"Test Set | AUC={test_scores['auc_score']:.2f}")
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve comparison\nfor {classifier_name} in Training and Test sets")
        ax.legend(loc='lower right', prop={'size': 8})
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        save_plot(fig, f"{classifier_abbr}.TrainingTest.ROC.png", "Training and Test ROC Curve")

        # Save results
        save_results(training_stats, test_scores, classifier_name, classifier_abbr, X_test_selected)

        # Generate interactive ROC plot
        interactive_roc(training_roc, classifier_name, classifier_abbr)


        write_log(f"Training, validation and testing completed for classifier: {classifier_name}", "INFO")
        return {"training_stats": training_stats, "test_scores": test_scores}

    except Exception as e:
        write_log(f"Error during training and validation for {classifier_name}: {str(e)}", "ERROR")
        return None

    return 

# Generates and saves RFECV performance plots (static and interactive)
def generate_rfecv_plot(rfecv, X_train, norm_method):

    try:
        write_log(f"Generating RFECV plots for method: {norm_method}", "INFO")

        # Selected CV
        method = None
        if isinstance(CONFIG["rfe_cross_validation_type"], LeaveOneOut):
        	method = "Leave-One-Out"
        elif isinstance(CONFIG["rfe_cross_validation_type"], StratifiedKFold):
        	if CONFIG["rfe_cross_validation_type"].n_splits == 5:
        		method = "5-Fold"
        	else:
		        method = "10-Fold"

        # Calculate the actual number of features at each step
        num_features = np.linspace(CONFIG["min_num_features"], X_train.shape[1], num=len(rfecv.cv_results_["mean_test_score"]), dtype=int)

        if len(num_features) != len(rfecv.cv_results_["mean_test_score"]):
        	write_log(f"Mismatch between calculated feature counts and RFECV results.", "ERROR")

        # Static RFECV Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(num_features, 
        				rfecv.cv_results_["mean_test_score"] - rfecv.cv_results_["std_test_score"], 
						rfecv.cv_results_["mean_test_score"] + rfecv.cv_results_["std_test_score"], 
						alpha=0.1)
        ax.plot(num_features, rfecv.cv_results_["mean_test_score"], "o-")
        ax.axvline(rfecv.n_features_, 
				   c="k", 
				   ls="--",
				   label=f"n_features={rfecv.n_features_}\nscore={rfecv.cv_results_['mean_test_score'].max():.3f}")

        plt.title(f"RFECV (with {method} CV) and Random Forest Classifier model feature selection")
        plt.xlabel("Number of features")
        plt.ylabel("Score (Balanced Accuracy)")
        plt.legend(frameon=True, loc="best")

        # removing right and top margin boarders
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(CONFIG["results_feature_selection"], f"rfecv_feature_selection.{norm_method}.png"), dpi=300)

        # Interactive RFECV Plot
        fig = go.Figure()

        # Plot mean_test_score and standard deviation as a shaded area
        fig.add_trace(go.Scatter(x=num_features, 
						         y=rfecv.cv_results_["mean_test_score"], 
						         showlegend=False, mode='lines+markers', 
						         hovertemplate=f"<b>Num. of Features: %{{x}}<br>Score: %{{y:.4f}}", 
						         name=''))

        # Add shaded area for standard deviation
        fig.add_trace(go.Scatter(x=num_features,
								 y=rfecv.cv_results_["mean_test_score"] + rfecv.cv_results_["std_test_score"],
								 mode='lines',
								 showlegend=False,
								 line=dict(width=0),
								 hoverinfo='none',
								 fill='tonexty',
								 fillcolor='rgba(2, 114, 162, 0.05)'))
		
		# Highlight the optimal number of features
        fig.add_trace(go.Scatter(x=num_features, 
					        	 y=rfecv.cv_results_["mean_test_score"] - rfecv.cv_results_["std_test_score"], 
					        	 mode='lines', showlegend=False, line=dict(width=0), 
					        	 hoverinfo='none', 
					        	 fill='tonexty', 
					        	 fillcolor='rgba(2, 114, 162, 0.2)'))
        
        optimal_index = np.where(num_features == rfecv.n_features_)[0][0]
        fig.add_trace(go.Scatter(x=[rfecv.n_features_], 
        						 y=[rfecv.cv_results_["mean_test_score"][optimal_index]],
					        	 mode='markers', 
					        	 marker=dict(color='#FA7070', size=12), 
					        	 hovertemplate=f"<b>Selected Features: %{{x}}<br>Score: %{{y:.4f}}", 
					        	 name='', 
					        	 showlegend=False))

        # Customize layout
        fig.update_layout(title=f'RFECV (with {method} CV) and Random Forest Classifier model feature selection',
		                  xaxis=dict(title='Number of Features'),
		                  yaxis=dict(title='Cross-Validation Score (Balanced Accuracy)'),
		                  legend=dict(x=0.90, y=0.95),
		                  template="plotly_white")

        fig.write_html(os.path.join(CONFIG["html_feature_selection"], f"rfecv_feature_selection.{norm_method}.html"))
        
        write_log(f"RFECV plots generated successfully for {norm_method}", "INFO")

    except Exception as e:
        write_log(f"Error during RFECV plot generation for {norm_method}: {str(e)}", "ERROR")

    return

# Generates and saves PI performance plots (static and interactive)
def generate_permutation_importance_plots(features_df, norm_method):
    
    try:
        # Determine whether to show the top 20 or all features
        if len(features_df) > 20:
            top_features = features_df.head(20)
            title_suffix = " (Top 20 Features)"
        else:
            top_features = features_df
            title_suffix = " (All Features)"

        write_log(f"Generating Permutation Importance plots for {norm_method} using {len(top_features)} features.", "INFO")

        # Static Bar Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x="Importance_Mean", 
		            y="Feature", 
		            data=top_features, 
		            hue="Feature", 
		            palette="viridis", 
		            dodge=False, 
		            legend=False)

        plt.xlabel("Mean Importance Score")
        plt.ylabel("Features")
        plt.title(f"Permutation Importance Mean Scores of {norm_method.replace('_', ' ')}{title_suffix}")

        static_plot_file = os.path.join(CONFIG["results_feature_selection"], f"permFeatImp.{norm_method}.png")
        plt.tight_layout()
        plt.savefig(static_plot_file, dpi=600)
        plt.close()

        write_log(f"Static Permutation Importance plot saved to {static_plot_file}", "INFO")

        # Interactive Plot
        fig = px.bar(top_features, 
		            x="Importance_Mean", 
		            y="Feature", 
		            orientation="h", 
		            color="Importance_Mean", 
		            color_continuous_scale="viridis_r", 
		            labels={'Importance': 'Importance Score'}, 
		            hover_data=[])

        # Customize the layout
        fig.update_layout(title=f"Permutation Importance Mean Scores of {norm_method.replace('_', ' ')}{title_suffix}",
			            xaxis_title='Importance Score',
			            yaxis_title='Features',
			            yaxis=dict(autorange="reversed"),
			            template="plotly_white")

        fig.update_traces(hovertemplate='Gene Name: %{y}<br>Importance Score: %{x:.4f}')
        fig.update_coloraxes(showscale=False)

        interactive_plot_file = os.path.join(CONFIG["html_feature_selection"], f"permFeatImp.{norm_method}.html")
        fig.write_html(interactive_plot_file)

        write_log(f"Interactive Permutation Importance plot saved to {interactive_plot_file}", "INFO")

    except Exception as e:
        write_log(f"Error generating plots for Permutation Importance: {str(e)}", "ERROR")

    return

# Calculating basic classification evaluation metrics
def calculate_scores(y_true, y_pred, y_scores):

    basic_metrics = {}
    
    # Precision, Recall, Average Precision
    basic_metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0, pos_label=CONFIG["condition_group"]), 2)
    basic_metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0, pos_label=CONFIG["condition_group"]), 2)
    basic_metrics["average_pr"] = round(average_precision_score(y_true, y_scores, pos_label=CONFIG["condition_group"]), 2)

    # Accuracy and Balanced Accuracy
    basic_metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 2)
    basic_metrics["balanced_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), 2)

    # AUC Score
    basic_metrics["auc_score"] = round(roc_auc_score(y_true, y_scores), 2)

    # F1 Score
    basic_metrics["f1_score"] = round(f1_score(y_true, y_pred, zero_division=0, pos_label=CONFIG["condition_group"]), 2)

    # Confusion Matrix Breakdown: TN, FP, FN, TP
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=CONFIG["conditions"]).ravel()
    basic_metrics["cm"] = [tn, fp, fn, tp]

    # Sensitivity and Specificity with ZeroDivision Protection
    basic_metrics["sensitivity"] = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 2)
    basic_metrics["specificity"] = round(tn / (tn + fp) if (tn + fp) > 0 else 0, 2)

    # Total Misclassifications
    basic_metrics["missclassifications"] = fp + fn

    return basic_metrics

# Calculating and extracting advanced classification metrics
def advanced_classification_metrics(basic_metrics, classifier_abbr, y_true, y_pred, y_prob):

    try:
        # Unpack confusion matrix
        tn, fp, fn, tp = basic_metrics['cm']

        # Ensure y_prob values are valid
        if not (0 <= np.min(y_prob) <= 1 and 0 <= np.max(y_prob) <= 1):
        	write_log("Predicted probabilities must be in range [0, 1]", "ERROR")

        # Advanced metrics calculation
        advanced_metrics = {"True Positive": tp,
				            "False Positive": fp,
				            "True Negative": tn,
				            "False Negative": fn,
				            "Population": tn + fp + fn + tp,
				            "Accuracy": basic_metrics['accuracy'],
				            "Balanced Accuracy": basic_metrics['balanced_accuracy'],
				            "False Positive Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
				            "False Negative Rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
				            "True Negative Rate": tn / (tn + fp) if (tn + fp) > 0 else 0,
				            "Negative Predictive Value": tn / (tn + fn) if (tn + fn) > 0 else 0,
				            "False Discovery Rate": fp / (fp + tp) if (fp + tp) > 0 else 0,
				            "True Positive Rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
				            "Positive Predictive Value": tp / (tp + fp) if (tp + fp) > 0 else 0,
				            "F1 Score": basic_metrics['f1_score'],
				            "F2 Score": fbeta_score(y_true, y_pred, beta=2, zero_division=0, pos_label=CONFIG["condition_group"]),
				            "Cohen Kappa Metric": cohen_kappa_score(y_true, y_pred),
				            "Matthews Correlation Coefficient": matthews_corrcoef(y_true, y_pred),
				            "ROC AUC Score": basic_metrics['auc_score'],
				            "Average Precision": basic_metrics['average_pr'],
				            "Log Loss": log_loss(y_true, y_prob),
				            "Brier Score": brier_score_loss(y_true, y_prob, pos_label=CONFIG["condition_group"]),
				            "Negative Likelihood Ratios": (fn / (fn + tp)) / (tn / (tn + fp)) if (tn + fp) > 0 else 0,
				            "Positive Likelihood Ratios": (tp / (tp + fn)) / (fp / (fp + tn)) if (fp + tn) > 0 else 0,
				            "Diagnostic Odds Ratio": ((tp / (tp + fn)) / (fp / (fp + tn)) / ((fn / (fn + tp)) / (tn / (tn + fp))) if (tn + fp) > 0 else 0)}

        # Format results into a DataFrame
        df = pd.DataFrame(list(advanced_metrics.items()), columns=['Metric', 'Value']).round(2)
        df['Abbrev.'] = ["TP", "FP", "TN", "FN", "Pop", "Acc", "BA", "FPR", "FNR", "TNR", 
        				"NPV", "FDR", "TPR", "PPV", "F1", "F2", "Kappa", "MCC", "ROC AUC", 
            			"AP", "LogLoss", "BS", "LR-", "LR+", "DOR"]
        
        df['Description'] = ["Number of true positive instances. Count of correctly predicted positive instances.",
				            "Number of false positive instances. Count of negative instances incorrectly predicted as positive.",
				            "Number of true negative instances. Count of correctly predicted negative instances.",
				            "Number of false negative instances. Count of positive instances incorrectly predicted as negative.",
				            "Total number of instances in the dataset (Population size).",
				            "Proportion of correctly classified instances. Calculated as (TP + TN) / Total population.",
				            "Balanced accuracy: Average of sensitivity (True Positive Rate) and specificity (True Negative Rate).",

				            "False Positive Rate (FPR): Proportion of negative instances incorrectly predicted as positive. Also known as Fall-Out or Type I error rate.",
				            "False Negative Rate (FNR): Proportion of positive instances incorrectly predicted as negative. Also known as Miss Rate or Type II error rate.",
				            "True Negative Rate (TNR): Proportion of correctly identified negatives out of all actual negative instances. Also known as Specificity.",
				            "Negative Predictive Value (NPV): Proportion of correctly predicted negatives out of all predicted negatives.",
				            "False Discovery Rate (FDR): Proportion of incorrectly predicted positives out of all predicted positives.",
				            "True Positive Rate (TPR): Proportion of correctly predicted positives out of all actual positive instances. Also known as Sensitivity, Recall, or Hit Rate.",
				            "Positive Predictive Value (PPV): Proportion of correctly predicted positives out of all predicted positives. Also known as Precision.",

				            "F1 Score: Harmonic mean of precision (PPV) and recall (TPR), balancing both metrics.",
				            "F2 Score: Weighted harmonic mean of precision and recall, with greater emphasis on recall.",
				            "Cohen's Kappa: Measures agreement between predictions and actual values, adjusted for chance agreement.",
				            "Matthews Correlation Coefficient (MCC): Balanced measure of correlation between predictions and actual values, suitable for imbalanced datasets.",
				            "ROC AUC Score: Area under the Receiver Operating Characteristic curve, summarizing the trade-off between TPR and FPR.",
				            "Average Precision (AP): Summarizes the precision-recall curve into a single value. Useful for imbalanced datasets.",

				            "Log Loss: Measures the accuracy of predicted probabilities by penalizing incorrect predictions more heavily.",
				            "Brier Score: Measures the mean squared difference between predicted probabilities and true outcomes.",
				            "Negative Likelihood Ratio (LR-): Ratio of the probability of a false negative to the probability of a true negative.",
				            "Positive Likelihood Ratio (LR+): Ratio of the probability of a true positive to the probability of a false positive.",
				            "Diagnostic Odds Ratio (DOR): Ratio of the odds of a positive test in diseased individuals to the odds of a positive test in non-diseased individuals."]

        # Save to file
        output_path = os.path.join(CONFIG["results_classification"], f"{classifier_abbr}.TestSet.AdvancedClassificationMetrics.tsv")
        df.to_csv(output_path, sep="\t", index=False)
        write_log(f"Advanced classification metrics saved to {output_path}", "INFO")

    except Exception as e:
        write_log(f"Error in advanced_classification_metrics: {str(e)}", "ERROR")

    return

# Converting the ROC plot to interactive using Plotly
def interactive_roc(training_roc, classifier_name, classifier_abbr):

    # Define color scheme and attributes for the ROC curves
    colors = {'Fold1' 			 : ('#4682B4' , 1, 0.6),
    		  'Fold2' 			 : ('#D2691E' , 1, 0.6),
    		  'Fold3' 			 : ('#D2B48C' , 1, 0.6),
    		  'Fold4' 			 : ('#F0E68C' , 1, 0.6),
    		  'Fold5' 			 : ('#9ACD32' , 1, 0.6),
    		  'Fold6' 			 : ('#008080' , 1, 0.6),
    		  'Fold7' 			 : ('#483D8B' , 1, 0.6),
    		  'Fold8' 			 : ('#B8860B' , 1, 0.6),
    		  'Fold9' 			 : ('#FA8072' , 1, 0.6),
    		  'Fold10' 			 : ('#FFC0CB' , 1, 0.6),
    		  'Mean TrainingSet' : ('#eb5e28' , 2, 0.8),
    		  'TestSet' 		 : ('#226f54' , 2, 0.8)}

    # Initialize the Plotly figure
    fig = make_subplots(rows=1, cols=1)

    # Add the chance line to the plot
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#7D1935'), name='Chance', opacity=0.5))

    # Add ROC curves for each fold, mean training, and test set
    for name, stats in training_roc.items():
        line_color, line_width, opacity = colors.get(name, ('#333333', 1, 0.6))
        
        # Label generation
        if name == "Mean TrainingSet":
            label = f"Mean Training Set | AUC: {stats['auc']:.2f} ± {stats['std_auc']:.2f}"
        elif name == "TestSet":
            label = f"Test Set | AUC: {stats['auc']:.2f}"
        else:
            label = f"{name} | AUC: {stats['auc']:.2f}"

        # Add the curve to the interactive plot
        fig.add_trace(go.Scatter(x=stats["fpr"], y=stats["tpr"], mode='lines', name=label, opacity=opacity,
            line=dict(color=line_color, width=line_width), 
            hovertemplate=f"<b>{label}</b><br>False Positive Rate: %{{x:.2f}}<br>True Positive Rate: %{{y:.2f}}"))

    # Update layout settings
    fig.update_layout(title=f"ROC Curve Comparison for {classifier_name} (Training and Test Sets)",
				        xaxis=dict(title="False Positive Rate", range=[-0.05, 1.05]),
				        yaxis=dict(title="True Positive Rate", range=[-0.05, 1.05]),
				        template="plotly_white")

    # Save as HTML file
    output_file = os.path.join(CONFIG["html_classification"], f"{classifier_abbr}.TrainingTest.ROCplot.html")
    fig.write_html(output_file)
    write_log(f"Interactive ROC plot saved to {output_file}", "INFO")

    return

# Generating confusion matrix and probability plots
def model_evaluation_plots(all_metrics, classifier_name, classifier_abbr, stats):
    
    try:
        write_log("Generating confusion matrix heatmap...", "INFO")
        
        ### CONFUSION MATRIX
        # Extract values and prepare labels
        tn, fp, fn, tp = all_metrics['cm']
        total = tn + fp + fn + tp
        group_names = ['True Positive', 'False Negative', 'False Positive', 'True Negative']
        group_counts = [tp, fn, fp, tn]
        group_percentages = [f"{value/total:.2%}" for value in group_counts]
        group_labels = [f"{name}\n{count}/{total}\n{percent}" 
                        for name, count, percent in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(group_labels).reshape(2, 2)

        # Plot static confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap([[tp, fn], [fp, tn]], annot=labels, fmt='', cmap='BuPu', cbar=False, ax=ax)
        ax.set_title(f"{classifier_name} Confusion Matrix\nAUC: {all_metrics['auc_score']:.2f}  "
                     f"BA: {all_metrics['balanced_accuracy']:.2f}  "
                     f"Miscls: {all_metrics['missclassifications']}/{total} "
                     f"({(all_metrics['missclassifications']/total*100):.1f}%)")
        ax.set_xlabel('\nPredicted Class')
        ax.set_ylabel('Actual Class')
        ax.xaxis.set_ticklabels(reversed(CONFIG["conditions"]))
        ax.yaxis.set_ticklabels(reversed(CONFIG["conditions"]))
        fig.savefig(os.path.join(CONFIG["results_classification"], f'{classifier_abbr}.TestSet.ConfusionMatrix.png'), format='png', dpi=600)
        plt.close(fig)
        write_log("Static confusion matrix heatmap saved successfully.", "INFO")

        # Interactive confusion matrix
        write_log("Generating interactive confusion matrix...", "INFO")
        fig_interactive = go.Figure(data=go.Heatmap(z=[[tp, fn], [fp, tn]],
										            x=list(reversed(CONFIG["conditions"])),
										            y=list(reversed(CONFIG["conditions"])),
										            text=labels,
										            hoverinfo="text",
										            colorscale="BuPu"))

        fig_interactive.update_layout(title=f"{classifier_name} Confusion Matrix",
						            xaxis_title='Predicted Class',
						            yaxis_title='Actual Class',
						            height=600,
						            width=600,
						            template="plotly_white")

        fig_interactive.write_html(os.path.join(CONFIG["html_classification"], f'{classifier_abbr}.TestSet.ConfusionMatrix.html'))
        write_log("Interactive confusion matrix saved successfully.", "INFO")

        ### PREDICTED PROBABILITIES
        write_log("Generating predicted probability plots...", "INFO")
        
        # Prepare probability data
        data = stats[["Sample", "Predicted_label", f'Probability_{CONFIG["conditions"][0]}', f'Probability_{CONFIG["conditions"][1]}', "Results"]].copy()
        data["Probability"] = np.where(data['Results'], data[f'Probability_{CONFIG["conditions"][1]}'], data[f'Probability_{CONFIG["conditions"][0]}'])
        data = data[["Sample", "Predicted_label", "Probability"]]
        data.columns = ["Sample", "Class", "Probability"]

        # Static probability boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="Class", y="Probability", data=data, width=0.3, whis=1.5, palette=['#8FBDD3', '#BB6464'], hue="Class", ax=ax)
        sns.swarmplot(x="Class", y="Probability", data=data, color="#383838", edgecolor="#7F7F7F", alpha=0.7, size=6, ax=ax)
        ax.axhline(y=0.5, color='#1b263b', linestyle='--', linewidth=1, alpha=0.8)
        ax.set(ylim=(0, 1), xticks=range(len(CONFIG["conditions"])), xticklabels=CONFIG["conditions"])
        ax.set_title(f'{classifier_name} Predicted Probability Plot')
        ax.set_xlabel('Class')
        ax.set_ylabel('Classification Probabilities')

        # Styling the box edges
        for box in ax.artists:
            box.set_edgecolor('black')
            box.set_facecolor('white')
        fig.savefig(os.path.join(CONFIG["results_classification"], f'{classifier_abbr}.TestSet.ProbabilityPlot.png'), format='png', dpi=600)
        plt.close(fig)
        write_log("Static predicted probability plot saved successfully.", "INFO")

        # Interactive probability plot
        fig_interactive_prob = px.box(
            data, x="Class", y="Probability", color="Class", points="all", width=800, height=500,
            category_orders={"Class": CONFIG["conditions"]},
            title=f'{classifier_name} Predicted Probability Plot',
            hover_data={"Sample": True},
            color_discrete_map={CONFIG["conditions"][0]: '#8FBDD3', CONFIG["conditions"][1]: '#BB6464'}
        )
        fig_interactive_prob.add_hline(y=0.5, line_dash="dash", line_color="#1b263b")
        fig_interactive_prob.update_traces(hovertemplate='Class: %{x}<br>Probability: %{y:.2f}<br>Sample: %{customdata[0]}')
        fig_interactive_prob.update_layout(
            yaxis=dict(range=[0, 1]),
            xaxis_title="Class",
            yaxis_title="Classification Probabilities",
            template="plotly_white",
            showlegend=False
        )
        fig_interactive_prob.write_html(os.path.join(CONFIG["html_classification"], f'{classifier_abbr}.TestSet.ProbabilityPlot.html'))
        write_log("Interactive predicted probability plot saved successfully.", "INFO")

    except Exception as e:
        write_log(f"Error generating auxiliary plots: {str(e)}", "ERROR")

# Output figures
def save_plot(fig, filename, title):
    try:
        path = os.path.join(CONFIG["results_classification"], filename)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        plt.close(fig)
        write_log(f"{title} saved to {path}", "INFO")
    except Exception as e:
        write_log(f"Error saving {title}: {str(e)}", "ERROR")

    return 

# Output stats
def save_results(training_stats, test_scores, classifier_name, classifier_abbr, X_test_selected):
    # Rename columns for clarity
    column_mapping = {"precision": "Precision",
			        "recall": "Recall",
			        "average_pr": "Avg Precision",
			        "accuracy": "Accuracy",
			        "balanced_accuracy": "Balanced Accuracy",
			        "auc_score": "AUC Score",
			        "f1_score": "F1 Score",
			        "cm": "Confusion Matrix (TN, FP, FN, TP)",
			        "sensitivity": "Sensitivity",
			        "specificity": "Specificity",
			        "missclassifications": "Misclassifications"}

    # Process training results
    train_results = pd.DataFrame(training_stats).T.rename(columns=column_mapping)
    train_results.index.name = "Fold"
    train_results["Type"] = "Training"

    # Process test results
    test_results = pd.DataFrame([test_scores]).rename(columns=column_mapping)
    test_results.index = ["Test Set"]
    test_results["Type"] = "Testing"

    # Combine results
    combined_results = pd.concat([train_results, test_results], axis=0)
    combined_results.reset_index(inplace=True)
    combined_results.rename(columns={"index": "Dataset"}, inplace=True)

    # Save results to a single TSV file
    results_file = os.path.join(CONFIG["results_classification"], f"{classifier_abbr}.TrainingTest.Stats.tsv")
    combined_results.to_csv(results_file, sep="\t", index=False)

    write_log(f"Training and test results saved for {classifier_name} to {results_file}", "INFO")

    return

# Combining all classifiers into one ROC plot
def overall_ROCplot(rocStats, output_dir):
    
    try:
        write_log("Starting the generation of overall ROC plots", "INFO")

        # Define color scheme and attributes
        colors = {('RandomForestClassifier', 'TrainingSet'): ('Random Forest', '#a9d6e5', 0.8),
                  ('RandomForestClassifier', 'TestSet'): ('Random Forest', '#013a63', 1.2),

                  ('ExtraTreesClassifier', 'TrainingSet'): ('Extra Trees', '#b7e4c7', 0.8),
                  ('ExtraTreesClassifier', 'TestSet'): ('Extra Trees', "#2d6a4f", 1.2),

                  ('GradientBoostingClassifier', 'TrainingSet'): ('Gradient Boosting', '#c1121f', 0.8),
                  ('GradientBoostingClassifier', 'TestSet'): ('Gradient Boosting', '#780000', 1.2),

                  ('KNeighborsClassifier', 'TrainingSet'): ('KNNeighbors', '#936639', 0.8),
                  ('KNeighborsClassifier', 'TestSet'): ('KNNeighbors', '#99582a', 1.2),

                  ('LogisticRegression', 'TrainingSet'): ('Logistic Regression', '#ffa5ab', 0.8),
                  ('LogisticRegression', 'TestSet'): ('Logistic Regression', '#da627d', 1.2)}

        # Define a mapping to normalize names to match colors
        name_mapping = {
            "Random Forest Classifier": "RandomForestClassifier",
            "Extra Trees Classifier": "ExtraTreesClassifier",
            "Gradient Boosting Classifier": "GradientBoostingClassifier",
            "K-Nearest Neighbors Classifier": "KNeighborsClassifier",
            "Logistic Regression Classifier": "LogisticRegression"
        }

        ### Static ROC Plot
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#7D1935', label='Chance', alpha=0.5)

        for name, stats in rocStats.items():
            classifier_key = name_mapping.get(name[0], name[0])  # Normalize name
            key = (classifier_key, name[1])  # Create a key matching the colors dict
            
            if key not in colors:
                write_log(f"Warning: {key} not found in colors dictionary. Skipping...", "WARNING")
                continue
            
            if name[1] == "TrainingSet":
                ax1.plot(stats[0], stats[1], 
                         label=f"{colors[key][0]} ({name[1]}) | AUC: {stats[2]:.2f} ± {stats[3]:.2f}",
                         color=colors[key][1], lw=colors[key][2], alpha=0.7)
            else:
                ax1.plot(stats[0], stats[1], 
                         label=f"{colors[key][0]} ({name[1]}) | AUC: {stats[2]:.2f}",
                         color=colors[key][1], lw=colors[key][2], alpha=0.9)

        ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Overall ROC Plot")
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right", prop={'size': 8})
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)

        # Save static plot
        static_file = os.path.join(output_dir, "overall_ROC_plot.png")
        fig1.savefig(static_file, format='png', dpi=600)
        write_log(f"Static overall ROC plot saved to {static_file}", "INFO")

        ### Interactive ROC Plot
        fig2 = make_subplots(rows=1, cols=1)

        # Add the chance line
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  line=dict(dash='dash', color='#7D1935'),
                                  name='Chance', opacity=0.5))

        # Add ROC curves for all classifiers
        for name, stats in rocStats.items():
            classifier_key = name_mapping.get(name[0], name[0])  # Normalize name
            key = (classifier_key, name[1])  # Create a key matching the colors dict

            if key not in colors:
                write_log(f"Warning: {key} not found in colors dictionary. Skipping...", "WARNING")
                continue

            if name[1] == "TrainingSet":
                label = f"{colors[key][0]} ({name[1]}) | AUC: {stats[2]:.2f} ± {stats[3]:.2f}"
            else:
                label = f"{colors[key][0]} ({name[1]}) | AUC: {stats[2]:.2f}"

            fig2.add_trace(go.Scatter(x=stats[0], y=stats[1], mode='lines',
                                      name=label, line=dict(color=colors[key][1], width=colors[key][2]),
                                      hovertemplate=f"<b>{label}</b><br>False Positive Rate: %{{x:.2f}}<br>True Positive Rate: %{{y:.2f}}"))

        # Update layout for the interactive plot
        fig2.update_layout(title="Overall ROC Plot",
                           xaxis=dict(title="False Positive Rate", range=[-0.05, 1.05]),
                           yaxis=dict(title="True Positive Rate", range=[-0.05, 1.05]),
                           template="plotly_white")

        # Save interactive plot
        interactive_file = os.path.join(CONFIG["html_classification"], "overall_ROC_plot.html")
        fig2.write_html(interactive_file)
        write_log(f"Interactive overall ROC plot saved to {interactive_file}", "INFO")

    except Exception as e:
        write_log(f"Error while generating overall ROC plots: {str(e)}", "ERROR")
        return None

    write_log("Overall ROC plots generated successfully", "INFO")
    return

# Perform enrichment analysis using gProfiler for the selected features
def perform_enrichment_analysis(selected_features):

    try:
        write_log("Starting enrichment analysis using gProfiler", "INFO")

        # Safeguard for empty feature list
        if not selected_features:
            write_log("No features provided for enrichment analysis.", "ERROR")
            return

        # Initialize gProfiler
        gp = GProfiler(return_dataframe=True)

        # Perform enrichment analysis
        enrichment_results = gp.profile(organism=CONFIG["ref_organism"], query=selected_features)

        # Check if results are empty
        if enrichment_results.empty:
            write_log("No enrichment results were found.", "WARNING")
            return

        # Save enrichment results to a file
        enrichment_file = os.path.join(CONFIG["results_classification"], "enrichment_analysis_results.tsv")
        enrichment_results.to_csv(enrichment_file, sep="\t", index=False)
        write_log(f"Enrichment analysis results saved to {enrichment_file}", "INFO")

        # Generate the gProfiler link for online results
        base_url = "https://biit.cs.ut.ee/gprofiler"
        query = {"organism": CONFIG["ref_organism"], "query": "\n".join(selected_features)}
        encoded_query = urllib.parse.urlencode(query)
        analysis_link = f"{base_url}?{encoded_query}"

        # Save the link to a file
        link_file = os.path.join(CONFIG["results_classification"], "gprofiler_analysis_link.txt")
        with open(link_file, "w") as f:
            f.write(f"Access the gProfiler enrichment analysis here:\n{analysis_link}\n")
        write_log(f"gProfiler enrichment analysis link saved to {link_file}", "INFO")
        write_log(f"Access the enrichment results here: {analysis_link}", "INFO")


        # Select top 15 terms based on p-value
        top_terms = enrichment_results.sort_values(by="p_value").head(15)

        # Matplotlib Static Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x="p_value", y="name", data=top_terms, hue="name", palette="coolwarm", dodge=False, legend=False)
        plt.title("Top Enriched Terms")
        plt.xlabel("Adjusted p-value (log scale)")
        plt.ylabel("Enriched Terms")
        plt.xscale("log")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save static plot
        static_plot_file = os.path.join(CONFIG["results_classification"], "enrichment_analysis_plot.png")
        plt.savefig(static_plot_file, dpi=600)
        plt.close()
        write_log(f"Static enrichment analysis plot saved to {static_plot_file}", "INFO")

        # Plotly Interactive Plot
        fig = px.bar(
            top_terms,
            x="p_value",
            y="name",
            color="p_value",
            color_continuous_scale="Viridis",
            title="Top Enriched Terms",
            labels={"p_value": "Adjusted p-value", "name": "Enriched Terms"},
            template="plotly_white",
            orientation="h",
        )
        fig.update_xaxes(type="log", title="Adjusted p-value (log scale)")
        fig.update_yaxes(title="Enriched Terms")
        fig.update_layout(coloraxis_colorbar=dict(title="P-value"))

        # Save as interactive HTML
        interactive_file = os.path.join(CONFIG["html_classification"], "enrichment_analysis_plot.html")
        fig.write_html(interactive_file)
        write_log(f"Interactive enrichment analysis plot saved to {interactive_file}", "INFO")

    except Exception as e:
        write_log(f"Error during enrichment analysis: {str(e)}", "ERROR")
        return None

    write_log("Enrichment analysis completed successfully", "INFO")
    
    return





def main():
    
    global log_file, rocStats

    try:
        # Initialise configuration and logging
        args = parse_arguments()  # Initialise arguments
        initialise_config(args)  # Initialise configuration

        # JSON logging file
        log_file = os.path.join(CONFIG["output_dir"], f"{CONFIG['projectID']}_log.json")
        write_log("### INITIALISING CLASSIFICATION ANALYSIS", "INFO")

        # Read the normalised data and labels
        training_data, test_data, training_labels, test_labels = read_normalised_data()

        # Evaluate normalisation methods
        results = evaluate_normalisation_methods(training_data, training_labels, test_data, test_labels)

        # Retrieve the best normalisation method and classifier
        best_norm = results["final_test"]["best_normalisation"]
        selected_features = results["final_test"]["selected_features"]

        if selected_features:
            write_log("Performing enrichment analysis for selected features", "INFO")
            perform_enrichment_analysis(selected_features)
        else:
            write_log("No features available for enrichment analysis.", "WARNING")

        write_log(f"Best Normalisation: {best_norm}", "INFO")
        write_log(f"Number of Selected Features: {len(selected_features)}", "INFO")

        # Retrieve best normalization data
        X_train_best = training_data[best_norm]
        X_test_best = test_data[best_norm]

        # Initialize global ROC stats dictionary
        rocStats = {}

        # Iterate through all classifiers chosen by the user
        for clf_abbrev, (clf, clf_name) in CONFIG["classifiers"].items():
            write_log(f"Training, validating, and testing classifier: {clf_name}", "INFO")

            train_validate_test(X_train=X_train_best, y_train=training_labels["Condition"], 
                                X_test=X_test_best, y_test=test_labels["Condition"],
                                classifier=clf, classifier_name=clf_name, classifier_abbr=clf_abbrev, optimal_features=selected_features)

        # Generate an overall ROC plot if multiple classifiers are tested
        if len(CONFIG["classifiers"]) > 1:
            write_log("Generating overall ROC plot for all classifiers", "INFO")
            overall_ROCplot(rocStats, CONFIG["results_classification"])


        # Log successful completion
        write_log("Classification analysis completed successfully.", "INFO")

    except Exception as e:
        # Log any exception that occurs
        write_log(f"Error in main(): {str(e)}", "ERROR")


if __name__ == "__main__": main()