### NORMALISATION ANALYSING FOR NANOSTRING TEST SET DATA [NanoInsights] ###
## version v5.0

### need to install pandoc

suppressPackageStartupMessages(
  suppressMessages(suppressWarnings({
    library("reticulate")
    library("optparse")
    library("ggplot2")
    library("plotly")
    library("reshape2")
    library("tidyverse")
    library("RColorBrewer")
    library("NanoStringClustR")
    library("NanoTube")
    library("RUVSeq")
    library("ctrlGene")
    library("htmltools")
    library("formattable")
    library("DT")
    library("ggrepel")
    library("bslib")
    set.seed(42)
})))



# Function to create directories
create_directory <- function(path, description) {
  tryCatch({
    dir.create(path, showWarnings = FALSE, recursive = TRUE)
    write_log(paste("Successfully created", description, "at:", path), level = "INFO")
  }, error = function(e) {
    write_log(paste("Failed to create", description, ":", e$message), level = "ERROR")
  })
}

# Enhanced Function to Read and Check Data with Specific File Extension Logic and NA Column Check
read_and_check <- function(file_path, var_name, description) {
  tryCatch({
    # Determine delimiter based on file extension
    delimiter <- if (grepl("\\.(txt|csv)$", file_path, ignore.case = TRUE)) {
      ","
    } else if (grepl("\\.tsv$", file_path, ignore.case = TRUE)) {
      "\t"
    } else {
      stop("Unsupported file format. Supported formats are .txt, .csv, and .tsv.")
    }
    
    # Read the file
    data <- read.table(file_path, sep = delimiter, header = TRUE, check.names = FALSE)
    
    # Check for columns entirely filled with NA
    na_columns <- which(colSums(is.na(data)) == nrow(data))
    if (length(na_columns) > 4) {
      stop(paste("The data contains", length(na_columns), "columns entirely filled with NA values. Please check the file:", file_path))
    }
    
    # Log success
    write_log(paste("Successfully imported", description, "from:", file_path), level = "INFO")
    return(data)
  }, error = function(e) {
    # Log and re-raise error
    write_log(paste("Error importing", description, "from:", file_path, "-", e$message), level = "ERROR")
  })
}



### Input Arguments
option_list = list(
  # Input directory with RCC files
  make_option(c("-d", "--dir"), type = "character", default = NULL,
              help="Path of the directory hosting the raw RCC files", metavar = "<directory path>"),
  # Input file containing the clinical data
  make_option(c("-c", "--clinicaldata"), type = "character", default = NULL,
              help="Path of the clinical data file", metavar = "<directory path>"),
  # Current directory path
  make_option(c("-y", "--currdir"), type = "character", default = NULL,
              help="Path of the source directory", metavar = "<directory path>"),
  # Input log file
  make_option(c("-e", "--logfile"), type = "character", default = NULL,
              help="Path of the log file", metavar = "<directory path>"),
  # Control group
  make_option(c("-a", "--control"), type = "character", default = NULL,
              help="In the two-group comparison, which label should be considered as CONTROL (e.g. Healthy)", metavar = "string"),
  # Condition group
  make_option(c("-b", "--condition"), type = "character", default = NULL,
              help="In the two-group comparison, which label should be considered as CONDITION (e.g. Cancer)", metavar = "string"),
  # Upper limit
  make_option(c("-u", "--upper_limit"), type = "numeric", default = NULL,
              help="Upper threshold in Binding Density (MAX/FLEX instruments: 2.25, SPRINT instruments: 1.8)", metavar = "number"),
  # k factor for the RUVSEq/RUVg function
  make_option(c("-k", "--k_factor"), type = "numeric", default = 1,
              help="Choose the k factor for the RUVg function [default %default]", metavar = "integer"),
  # How should we calculate the reference genes?
  make_option(c("-g", "--refgenes"), type = "character", default = "hkNpos",
              help="How shall we calculate the reference (stable/non significant) genes for the RUVg function?  [default %default]", metavar = "character"),
  # Min. number of reference genes to be considered
  make_option(c("-m", "--minref"), type = "numeric", default = 5,
              help="Minimum number of reference (stable/non-significant) genes to consider for the RUVg function (default 5 [default %default]", metavar = "integer"),
  # Training Set
  make_option(c("-t", "--training_mat"), type = "character", default = NULL,
              help="Path of the normalised training matrix", metavar = "<directory path>"))

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Defining arguments
raw_data <- opt$dir
clinical_data <- opt$clinicaldata
log_file <- opt$logfile
groups_capital <- c(str_to_title(opt$control), str_to_title(opt$condition))
bdlimits <- c(0.1, opt$upper_limit)
results <- file.path(dirname(raw_data), "output_results", "testset_normalisation")
html_results <- file.path(dirname(raw_data), "html_results", "testset_normalisation")




# ######### TO DELETE
# raw_data <- "/Users/stavris/Desktop/NanoInsights_Uploads/I7UH1YP2BL1LAD/test_set"
# clinical_data <- "/Users/stavris/Desktop/NanoInsights_Uploads/I7UH1YP2BL1LAD/test_set/clinical_data.csv"
# log_file <- "/Users/stavris/Desktop/NanoInsights_Uploads/I7UH1YP2BL1LAD/I7UH1YP2BL1LAD_log.json"
# groups_capital <- c(str_to_title("negative"), str_to_title("positive"))
# opt <- data.frame(1,6)
# opt$currdir <- "~/Desktop/Projects/nStringAnalysis/nanoinsights/src/nanoinsights_backend"
# opt$training_mat <- "/Users/stavris/Desktop/NanoInsights_Uploads/I7UH1YP2BL1LAD/output_results/trainingset_normalisation/quantile-normalised.matrix.ml.tsv"
# opt$refgenes <- "hkNpos"
# opt$minref <- 5
# opt$k_factor <- 1
# opt$upper_limit <- 2.25
# bdlimits <- c(0.1, opt$upper_limit)
# results <- file.path(dirname(raw_data), "output_results", "testset_normalisation")
# html_results <- file.path(dirname(raw_data), "html_results", "testset_normalisation")
# ######### TO DELETE


# Import all necessary for the analysis functions
source(file.path(opt$currdir, "helper_functions.R"))

# Initialize Logging
write_log("### INITIALISING TEST SET ANALYSIS (R) ###", level = "INFO")

# Create Directories
create_directory(results, "results directory")
create_directory(html_results, "HTML results directory")



### Import Data
write_log("Importing necessary data", level = "INFO")
pData <- read_and_check(file.path(raw_data, "pData.tsv"), "pData", "metadata (pData)")
raw <- read_and_check(file.path(raw_data, "raw.tsv"), "raw", "raw data")
raw_expression <- read_and_check(file.path(raw_data, "raw_expression.tsv"), "raw_expression", "raw expression data")
clinical_info <- read_and_check(clinical_data, "clinical_info", "clinical sample sheet")

# Process Clinical Data
write_log("Processing clinical data", level = "INFO")
tryCatch({
  
  # Remove file extensions from filenames
  clinical_info$Filename <- gsub(".{4}$", "", clinical_info$Filename)
  
  # Merge clinical_info with pData
  clinical_info <- merge(clinical_info, pData[, c("BCAC_ID", "CartridgeID")], by.x = "Filename", by.y = "BCAC_ID")
  clinical_info <- clinical_info %>% select(Sample = Filename, Condition, CartridgeID)
  row.names(clinical_info)  <- clinical_info$Sample
  
  # Convert Condition to title case
  clinical_info$Condition <- str_to_title(clinical_info$Condition)
  
  # Set Condition as a factor with levels from groups_capital
  clinical_info$Condition <- factor(clinical_info$Condition, levels = groups_capital)
  
  clinical_info$CartridgeID <- factor(clinical_info$CartridgeID, levels = unique(clinical_info$CartridgeID))
  write_log("Successfully processed clinical data.", level = "INFO")
}, error = function(e) {
  write_log(paste("Error processing clinical data:", e$message), level = "ERROR")
})

### Generate Initial QC Report
write_log("Generating initial QC report", level = "INFO")

tryCatch({
  # Define selected columns for QC stats
  selected_columns <- c("BCAC_ID", "Date", "CartridgeID", "imagingQC", "bindingDensityQC", "positiveLinearityQC", "limitOfDetectionQC")
  
  # Calculate library size metrics
  library_size <- tryCatch({
    data.frame(
      Endogenous = colSums(raw_expression[raw_expression$Class == "Endogenous", 3:ncol(raw_expression)]),
      Housekeepings = colSums(raw_expression[raw_expression$Class == "Housekeeping", 3:ncol(raw_expression)]),
      Positives = colSums(raw_expression[raw_expression$Class == "Positive", 3:ncol(raw_expression)]),
      Negatives = colSums(raw_expression[raw_expression$Class == "Negative", 3:ncol(raw_expression)]),
      Total = colSums(raw_expression[, 3:ncol(raw_expression)])
    )
  }, error = function(e) {
    write_log("Error calculating library size metrics", level = "ERROR", details = e$message)
  })
  write_log("Library size metrics calculated successfully.", level = "INFO")
  
  # Merge with metadata and clinical info
  library_size <- merge(library_size, pData[, selected_columns], by.x = 0, by.y = "BCAC_ID")
  library_size <- merge(library_size, clinical_info[, c("Sample", "Condition")], by.x = "Row.names", by.y = "Sample")
  row.names(library_size) <- library_size$Row.names
  library_size$Row.names <- NULL
  
  write_log("Library size merged with metadata and clinical info.", level = "INFO")
  
  # Reorder data based on clinical info
  clinical_info <- clinical_info[order(clinical_info$Condition), ]
  library_size <- library_size[match(row.names(clinical_info), row.names(library_size)), ]
  pData <- pData[match(row.names(clinical_info), pData$BCAC_ID), ]
  raw_expression <- raw_expression[, match(c("Gene", "Class", row.names(clinical_info)), colnames(raw_expression))]
  write_log("Data reordered based on clinical info.", level = "INFO")
  
  # Prepare interactive stats table for primary QC
  output_stats <- pData[, c("BCAC_ID", "Date", "CartridgeID", "imaging", "bindingDensity", "positiveLinearity", "limitOfDetectionQC")]
  
  visual_color <- function(value, threshold) {
    ifelse(value < threshold, scales::col_numeric(c("#a4161a", "#e5383b", "#ff758f"), domain = c(0, threshold))(value), "white")
  }
  
  custom_css <- "table.dataTable {font-family: 'Helvetica', sans-serif; font-size: 14px;}
                 p, ul {font-family: 'Helvetica', sans-serif; font-size: 12px; margin-top: 10px;}
                 ul {padding-left: 40px;}
                 li {font-size: 12px; line-height: 1.5;}"
  
  # Apply the visual colour formatting to a DT table
  datatable_with_colors <- datatable(output_stats, options = list(
    pageLength = 10,  # Number of samples to be shown
    autoWidth = TRUE,  # Automatically adjust column widths
    ordering = TRUE),
    rownames = FALSE) %>% formatStyle("imaging",
                                      backgroundColor = styleInterval(c(0.75), c("#ff758f", "white"))) %>%
    formatStyle("bindingDensity",
                color = styleInterval(bdlimits, c("#e5383b", "black", "#e5383b"))) %>%
    formatStyle("positiveLinearity",
                backgroundColor = styleInterval(c(0.95), c("#ff758f", "white"))) %>%
    formatStyle("limitOfDetectionQC",
                color = styleEqual(c("True", "False"), c("black", "#e5383b")))
  
  # Add explanatory text
  explanatory_text <- HTML(sprintf(
    "<p style='font-family: Helvetica, sans-serif; font-size: 12px; margin-top: 20px;'>
  This table provides the initial quality control (QC) metrics for the input dataset. 
  Samples highlighted in red indicate values requiring attention, but note that some highlighted values might be close to the thresholds and therefore acceptable. 
  The thresholds used for each metric are as follows:
  <ul>
    <li><strong>Imaging:</strong> Values must be higher than 0.75.</li>
    <li><strong>Binding Density:</strong> Values must be between 0.1 and %s.</li>
    <li><strong>Positive Linearity:</strong> Values must be higher than 0.95.</li>
    <li><strong>Limit of Detection QC:</strong> TRUE/FALSE (indicating pass/fail).</li>
  </ul>
  </p>",
    opt$upper_limit
  ))
  
  # Combine CSS, datatable, and explanatory text into a single HTML layout
  html_output <- tagList(tags$style(HTML(custom_css)), datatable_with_colors, explanatory_text)
  
  # Save the combined HTML content using htmltools::save_html
  save_html(html_output, file = file.path(html_results, "initial_stats.html"))
  
  write_log("Interactive QC report generated and saved as HTML.", level = "INFO")
  
  # Save initial stats as TSV
  write.table(pData, file = file.path(results, "initial_stats.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  write_log("Initial stats saved to TSV.", level = "INFO")
  
  # Update QC flags
  selected_columns <- c('imagingQC', 'bindingDensityQC', 'positiveLinearityQC', 'limitOfDetectionQC')
  library_size[selected_columns] <- lapply(library_size[selected_columns], function(x) ifelse(x == "True", "No flag", ifelse(x == "False", "Potential outlier sample", x)))
  
  # Save library size with updated QC flags
  write.table(data.frame("SampleNames" = rownames(library_size), library_size), file = file.path(results, "library_size.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  write_log("Library size with QC flags saved to TSV.", level = "INFO")
  rm(datatable_with_colors, html_output, output_stats, visual_color, explanatory_text, custom_css)
  
}, error = function(e) {
  write_log("Error generating initial QC report", level = "ERROR", details = e$message)
})


######################### INITIAL NANOSTRING QC
write_log("Generating NanoString standard QC metric plots", level = "INFO")

tryCatch({
  # Attempt to run the initial QC process
  initialqc()
  
  # Log a success message upon completion
  write_log("Successfully completed NanoString QC metrics generation.", level = "INFO")
}, error = function(e) {
  # Log a detailed error message with the error message from the exception
  write_log("Error during NanoString QC metrics generation", level = "ERROR", details = e$message)
})



######################### NORMALISATION
write_log("Data normalisation", level = "INFO")

# Log the start of data filtering and preparation
write_log("Starting data filtering and preparation", level = "INFO")

# Filter out genes excluded in the training matrix
tryCatch({
  write_log("Reading training matrix and filtering genes", level = "INFO")
  training_genes <- read.table(opt$training_mat, header = TRUE, sep = "\t")[, 1]
  raw_expression.filt <- raw_expression %>%
                         filter(Class %in% c("Housekeeping", "Negative", "Positive") | 
                               (Class == "Endogenous" & Gene %in% training_genes))
  write_log("Successfully filtered raw_expression data", level = "INFO")
}, error = function(e) {
  write_log("Error reading or filtering training matrix", level = "ERROR", details = e$message)
})

# Process the raw expression data frame to isolate Endogenous genes
tryCatch({
  write_log("Isolating Endogenous genes from raw expression data", level = "INFO")
  rawCounts <- raw_expression %>%
               filter(Class == "Endogenous") %>%
               select(-Class)  # Remove 'Class' column to keep gene counts only
  row.names(rawCounts) <- rawCounts$Gene  # Set 'Gene' as row names
  rawCounts$Gene <- NULL  # Remove the 'Gene' column
  
  # Filter rows in rawCounts based on row names matching training_genes
  rawCounts.filt <- rawCounts[rownames(rawCounts) %in% training_genes, ]
  
  write_log("Successfully isolated Endogenous genes and prepared rawCounts", level = "INFO")
}, error = function(e) {
  write_log("Error isolating Endogenous genes from raw expression data", level = "ERROR", details = e$message)
})

write_log("Data filtering and preparation completed successfully", level = "INFO")

# Obtain housekeeping genes
write_log("Obtaining housekeeping genes", level = "INFO")
hk_genes <- raw[raw$CodeClass == "Housekeeping", 2]


# Perform other normalisation methods
tryCatch({
  write_log("Preparing data for other normalisation methods", level = "INFO")
  Rnf5 <- merge(raw_expression.filt, raw[, 2:3], by.x = "Gene", by.y = "Name")
  Rnf5 <- Rnf5[, match(c("Gene", "Class", "Accession", row.names(clinical_info)), colnames(Rnf5))]
  colnames(Rnf5)[1:2] <- c("Gene_Name", "Code_Class")
  
  nano_data <- count_set(
    count_data = Rnf5,
    group = clinical_info$Condition,
    batch = clinical_info$CartridgeID,
    samp_id = row.names(clinical_info)
  )
  
  rnf5_norm <- multi_norm(
    count_set = nano_data,
    norm_method = "all",
    background_correct = "none",
    positive_control_scaling = TRUE,
    count_threshold = -1,
    geNorm_n = 5
  )
  
  methods <- c("geNorm_housekeeping", "housekeeping_scaled", 
               "all_endogenous_scaled", "quantile", "loess", "vsn", "ruv")
  
  for (method in methods) {
    if (method == "ruv") {
      ruv_norm <- ruvseq_norm(raw_expression.filt, rawCounts, clinical_info, opt$refgenes, opt$minref, opt$k_factor)
      ruv_norm <- ruv_norm[row.names(ruv_norm), colnames(ruv_norm)]
      write.table(ruv_norm, file = paste0(results, "/ruv-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
    } else {
      norm_data <- data.frame(assays(rnf5_norm)[[method]])
      norm_data <- norm_data[!rownames(norm_data) %in% hk_genes, ]
      write.table(norm_data, file = paste0(results, "/", method, "-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
    }
  }
  
  # Additional normalissations: TPM, Min-Max Scaling, Z-Score, and Log Transformation
  tpm <- sweep(rawCounts.filt, 2, colSums(rawCounts.filt), "/") * 1e6
  write.table(tpm, file = paste0(results, "/tpm-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
  minmax <- as.data.frame(scale(rawCounts.filt, center = FALSE, scale = apply(rawCounts.filt, 2, max) - apply(rawCounts.filt, 2, min)))
  write.table(minmax, file = paste0(results, "/minmax-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
  zscore <- as.data.frame(scale(rawCounts.filt))
  write.table(zscore, file = paste0(results, "/zscores-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
  logtrans <- log1p(rawCounts.filt)
  write.table(logtrans, file = paste0(results, "/logtransf-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
}, error = function(e) {
  write_log("Error in normalisation methods", level = "ERROR", details = e$message)
})

# Output samples and labels for ML analysis
write.table(clinical_info[ ,c("Sample", "Condition")], file=paste(results,"/labels.ml.tsv", sep=""), sep="\t", row.names = F, quote=FALSE)

rm(Rnf5, nano_data, hk_genes, rnf5_norm)
write_log("Normalisation process completed", level = "INFO")

write_log("### TEST SET ANALYSIS COMPLETED ###", level = "INFO")