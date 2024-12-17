### DE ANALYSING FOR NANOSTRING DATA [NanoInsights] ###
## version v5.0

### need to install pandoc / remotes::install_github('rstudio/DT')

suppressPackageStartupMessages(
  suppressMessages(suppressWarnings({
    library("reticulate")
    library("optparse")
    library("ggplot2")
    library("plotly")
    library("reshape2")
    library("tidyverse")
    library("DESeq2")
    library("edgeR")
    library("pheatmap")
    library("heatmaply")
    library("RColorBrewer")
    library("NanoStringClustR")
    library("NanoTube")
    library("RUVSeq")
    library("ggdendro")
    library("ctrlGene")
    library("htmltools")
    library("formattable")
    library("DT")
    library("ggrepel")
    library("bslib")
    library("grid")
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
  # Filter lowly expressed genes
  make_option(c("-l", "--filter_lowlyExpr_genes"), action="store_false", default = FALSE,
              help="Filter lowly expressed genes. This action is held by the \'filterByExpr\' function of the edgeR package [default %default]"),
  # Filtering out genes based on the Negative Control genes
  make_option(c("-f", "--filter_genes_on_negCtrl"), action="store_true", default = TRUE,
              help="Filter Endogenous genes based on the mean of the Negative Controls plus two times the SD [default %default]", metavar = "character"),
   # Filter out samples
  make_option(c("-s", "--filter_samples_on_negCtrl"), action="store_true", default = TRUE,
              help="Filter samples that the majority of the genes are 0 when subtracting the mean of the Negative Controls plus two times the SD [default %default]", metavar = "character"),
  # Remove outlier samples
  make_option(c("-r", "--remove_outlier_samples"), action="store_false", default = FALSE,
              help="Remove the samples which are considered as outliers based on the IQR analysis [default %default]", metavar = "character"),
  # Choose Interquartile range cutoff
  make_option(c("-q", "--iqrcutoff"), type = "numeric", default = 2,
              help="Choose the cutoff of the interquartile range analysis [default %default]", metavar = "number"),
  # k factor for the RUVSEq/RUVg function
  make_option(c("-k", "--k_factor"), type = "numeric", default = 1,
              help="Choose the k factor for the RUVg function [default %default]", metavar = "integer"),
  # How should we calculate the reference genes?
  make_option(c("-g", "--refgenes"), type = "character", default = "hkNpos",
              help="How shall we calculate the reference (stable/non significant) genes for the RUVg function?  [default %default]", metavar = "character"),
  # Min. number of reference genes to be considered
  make_option(c("-m", "--minref"), type = "numeric", default = 5,
              help="Minimum number of reference (stable/non-significant) genes to consider for the RUVg function (default 5 [default %default]", metavar = "integer"),
  # Type of normalisation
  make_option(c("-n", "--norm"), type = "character", default = "auto",
              help="Type of normalisation to be used [default %default]", metavar = "character"),
  # The log2FC cutoff
  make_option(c("-o", "--lfcthreshold"), type = "numeric", default = 0.5,
              help="The log2 fold change cutoff to call the DE genes [default %default]", metavar = "character"),
  # The adjusted p-value cutoff
  make_option(c("-p", "--padjusted"), type = "numeric", default = 0.05,
              help="The cutoff of the adjusted p-value in order to call the DE genes [default %default]", metavar = "character"))

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Set up paths and variables
raw_data <- opt$dir
clinical_data <- opt$clinicaldata
log_file <- opt$logfile
groups_capital <- c(str_to_title(opt$control), str_to_title(opt$condition))
bdlimits <- c(0.1, opt$upper_limit)
results <- file.path(dirname(raw_data), "output_results", "trainingset_normalisation")
ml_results <- file.path(dirname(raw_data), "output_results", "trainingset_classification")
html_results <- file.path(dirname(raw_data), "html_results", "trainingset_normalisation")


# ######### TO DELETE
# raw_data <- "~/Desktop/NanoInsights_Uploads/1G5CAZJLEEN0JG/training_set"
# clinical_data <- "/Users/stavris/Desktop/NanoInsights_Uploads/1G5CAZJLEEN0JG/training_set/clinical_data.csv"
# log_file <- "~/Desktop/NanoInsights_Uploads/1G5CAZJLEEN0JG/1G5CAZJLEEN0JG_log.json"
# groups_capital <- c(str_to_title("negative"), str_to_title("positive"))
# opt <- data.frame(1,14)
# opt$currdir <- "~/Desktop/Projects/nStringAnalysis/nanoinsights/src/nanoinsights_backend"
# opt$iqrcutoff <- 2
# opt$filter_lowlyExpr_genes <- FALSE
# opt$filter_genes_on_negCtrl <- TRUE
# opt$filter_samples_on_negCtrl <- TRUE
# opt$remove_outlier_samples <- TRUE
# opt$refgenes <- "hkNpos"
# opt$minref <- 5
# # opt$norm <- "nSolver"
# opt$norm <- "auto"
# opt$k_factor <- 1
# opt$lfcthreshold <- 0.5
# opt$padjusted <- 0.05
# opt$upper_limit <- 2.25
# bdlimits <- c(0.1, opt$upper_limit)
# results <- file.path(dirname(raw_data), "output_results", "trainingset_normalisation")
# ml_results <- file.path(dirname(raw_data), "output_results", "trainingset_classification")
# html_results <- file.path(dirname(raw_data), "html_results", "trainingset")
# ######### TO DELETE


# Import all necessary for the analysis functions
source(file.path(opt$currdir, "helper_functions.R"))


# Initialize Logging
write_log("### INITIALISING TRAINING SET ANALYSIS (R) ###", level = "INFO")

# Create Directories
create_directory(results, "results directory")
create_directory(ml_results, "ml results directory")
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


######################### EXPLORATORY ANALYSIS
# Performing exploratory analysis on the dataset
write_log("Starting exploratory analysis", level = "INFO")

# Define colour palette for plots
colors <- c('#8FBDD3', '#BB6464')

tryCatch({
  # Editing the raw expression data frame (ignoring Class and obtaining only the Endogenous genes)
  rawCounts <- raw_expression[raw_expression$Class == "Endogenous", c(1, 3:length(raw_expression))]
  row.names(rawCounts) <- rawCounts$Gene
  rawCounts$Gene <- NULL  # Setting Gene as row names
  write_log("Created rawCounts dataframe", level = "INFO")
  
  # PCA plot with library size and heatmap
  dds <- DESeqDataSetFromMatrix(countData = rawCounts, colData = clinical_info, design = ~ Condition)
  write_log("Imported rawCounts to DESeq2", level = "INFO")
  
  suppressMessages(suppressWarnings(rld <- rlog(dds, blind = TRUE, fitType = 'local')))
  write_log("Converted DDS to rlog", level = "INFO")
  
  data_pca <- plotPCA(rld, intgroup = "Condition", returnData = TRUE)
  percentVar <- round(100 * attr(data_pca, "percentVar"))
  write_log("Calculated principal components and saved in data_pca", level = "INFO")
  
  data_pca <- merge(data_pca, library_size[, c("CartridgeID", "Endogenous", "Total", "Date")], by.x = 0, by.y = 0)
  row.names(data_pca) <- data_pca$Row.names
  data_pca$Row.names <- NULL
  data_pca <- data_pca[match(row.names(clinical_info), row.names(data_pca)), ]
  write_log("Merged data_pca with library_size and matched with clinical_info", level = "INFO")
  
  # Call the exploratory analysis function
  exploratory_analysis()
  write_log("Successfully completed exploratory analysis", level = "INFO")
  
  # Re-create rawCounts for edgeR object preparation
  rawCounts <- raw_expression[raw_expression$Class == "Endogenous", c(1, 3:length(raw_expression))]
  row.names(rawCounts) <- rawCounts$Gene
  rawCounts$Gene <- NULL
  write_log("Re-created rawCounts dataframe for edgeR object", level = "INFO")
  
  # Preparing edgeR object
  edgeR_table <- DGEList(counts = rawCounts, group = factor(clinical_info$Condition))
  write_log("Imported rawCounts into edgeR_table", level = "INFO")
  
}, error = function(e) {
  write_log("Error during exploratory analysis", level = "ERROR", details = e$message)
})


tryCatch({
  # Perform Interquartile Range Analysis
  potential_outliers <- iqr_analysis()
  write_log("Successfully completed IQR analysis", level = "INFO")
  
  # Clean up objects to free memory
  rm(rld, percentVar, data_pca, dds, selected_columns)
  write_log("Removed unnecessary objects from memory", level = "INFO")
  
}, error = function(e) {
  # Log the error if IQR analysis fails
  write_log("Error during IQR analysis or cleanup", level = "ERROR", details = e$message)
})


######################### PERFORMING FILTERING ON GENE/SAMPLE LEVEL
write_log("Starting filtering steps on gene/sample levels", level = "INFO")


filt_genes <- NA
filt_samples <- NA

## 1. Filtering out lowly expressed genes
if (opt$filter_lowlyExpr_genes) {
  write_log("Filtering out lowly expressed genes by filterByExpr", level = "INFO")
  
  tryCatch({
    keep <- filterByExpr(edgeR_table, min.count = 5)
    edgeR_table2 <- edgeR_table[keep, , keep.lib.sizes = FALSE]
    nofilt_genes <- rownames(data.frame(edgeR_table2$counts))
    filt_genes <- setdiff(rownames(rawCounts), nofilt_genes)
    
    if (length(filt_genes) > 0) {
      log_elements_string <- paste(filt_genes, collapse = ",")
      write_log(sprintf("In total %d/%d genes did not pass the lowly expressed gene filtering step", length(filt_genes), nrow(rawCounts)), level = "INFO")
      write_log("The following genes were filtered out (filter_lowlyExpr_genes):", level = "INFO", details = log_elements_string)
      
      write(filt_genes, file = paste(results, "/filter_lowlyExpr_genes.tsv", sep = ""), append = FALSE, sep = ",")
    } else {
      write_log("No genes were filtered out (filter_lowlyExpr_genes)", level = "INFO")
    }
  }, error = function(e) {
    write_log("Error filtering lowly expressed genes", level = "ERROR", details = e$message)
  })
  
  rm(keep, edgeR_table2, nofilt_genes)
}

## 2. Filtering out genes based on Negative Control genes
if (opt$filter_genes_on_negCtrl) {
  write_log("Filtering out genes based on Negative Controls", level = "INFO")
  
  tryCatch({
    orig_matrix <- raw_expression
    rownames(orig_matrix) <- orig_matrix$Gene
    orig_matrix$Gene <- NULL
    
    meanNC <- apply(orig_matrix[orig_matrix$Class == "Negative", 2:ncol(orig_matrix)], 2, mean)
    standardDev <- apply(orig_matrix[orig_matrix$Class == "Negative", 2:ncol(orig_matrix)], 2, sd)
    final_threshold <- meanNC + (2 * standardDev)
    
    total_num <- nrow(orig_matrix[orig_matrix$Class == "Endogenous", ])
    orig_matrix <- orig_matrix[orig_matrix$Class == "Endogenous", 2:ncol(orig_matrix)] - final_threshold[col(orig_matrix[2:ncol(orig_matrix)])]
    has.neg <- apply(orig_matrix, 1, function(row) sum(row <= 0) > ceiling((ncol(orig_matrix) * 85) / 100))
    has.neg <- names(which(has.neg))
    
    if (length(has.neg) > 0) {
      log_elements_string <- paste(has.neg, collapse = ",")
      write_log(sprintf("In total %d/%d genes did not pass the NC filtering step", length(has.neg), total_num), level = "INFO")
      write_log("The following genes were filtered out (filter_genes_on_negCtrl):", level = "INFO", details = log_elements_string)
      
      write(has.neg, file = paste(results, "/filter_genes_on_negCtrl.tsv", sep = ""), append = FALSE, sep = ",")
    } else {
      write_log("No genes were filtered out (filter_genes_on_negCtrl)", level = "INFO")
    }
    
    filt_genes <- unique(c(filt_genes, has.neg))
  }, error = function(e) {
    write_log("Error filtering genes based on Negative Controls", level = "ERROR", details = e$message)
  })
  
  rm(orig_matrix, meanNC, standardDev, final_threshold, has.neg)
}

## 3. Remove samples based on Negative Controls
if (opt$filter_samples_on_negCtrl) {
  write_log("Removing samples based on Negative Controls", level = "INFO")
  
  tryCatch({
    orig_matrix <- raw_expression
    rownames(orig_matrix) <- orig_matrix$Gene
    orig_matrix$Gene <- NULL
    orig_matrix <- orig_matrix[!(orig_matrix$Class == "Endogenous" & rownames(orig_matrix) %in% filt_genes), ]
    
    meanNC <- apply(orig_matrix[orig_matrix$Class == "Negative", 2:ncol(orig_matrix)], 2, mean)
    standardDev <- apply(orig_matrix[orig_matrix$Class == "Negative", 2:ncol(orig_matrix)], 2, sd)
    final_threshold <- meanNC + (2 * standardDev)
    
    total_num <- ncol(orig_matrix) - 1
    orig_matrix <- orig_matrix[orig_matrix$Class == "Endogenous", 2:ncol(orig_matrix)] - final_threshold[col(orig_matrix[2:ncol(orig_matrix)])]
    removed_samples <- apply(orig_matrix, 2, function(col) sum(col < 1) > ceiling((nrow(orig_matrix) * 85) / 100))
    removed_samples <- names(which(removed_samples))
    
    raw_expression <- raw_expression[, !(colnames(raw_expression) %in% removed_samples)]
    clinical_orig <- clinical_info
    clinical_info <- clinical_info[!(rownames(clinical_info) %in% removed_samples), ]
    pData <- pData[!(pData$BCAC_ID %in% removed_samples), ]
    library_size <- library_size[!(rownames(library_size) %in% removed_samples), ]
    filt_samples <- unique(c(filt_samples, removed_samples))
    
    if (length(removed_samples) > 0) {
      log_elements_string <- paste(removed_samples, collapse = ",")
      write_log(sprintf("In total %d/%d samples did not pass the NC filtering step", length(removed_samples), total_num), level = "INFO")
      write_log("The following samples were filtered out (filter_samples_on_negCtrl):", level = "INFO", details = log_elements_string)
      
      write(removed_samples, file = paste(results, "/filter_samples_on_negCtrl.tsv", sep = ""), append = FALSE, sep = ",")
    } else {
      write_log("No samples were filtered out (filter_samples_on_negCtrl)", level = "INFO")
    }
  }, error = function(e) {
    write_log("Error removing samples based on Negative Controls", level = "ERROR", details = e$message)
  })
  
  rm(orig_matrix, meanNC, standardDev, final_threshold, removed_samples)
}

## 4. Remove outlier samples
if (opt$remove_outlier_samples) {
  write_log("Removing outlier samples based on IQR", level = "INFO")
  
  tryCatch({
    raw_expression <- raw_expression[, !(colnames(raw_expression) %in% potential_outliers)]
    clinical_info <- clinical_info[!(rownames(clinical_info) %in% potential_outliers), ]
    pData <- pData[!(pData$BCAC_ID %in% potential_outliers), ]
    library_size <- library_size[!(rownames(library_size) %in% potential_outliers), ]
    filt_samples <- unique(c(filt_samples, potential_outliers))
  }, error = function(e) {
    write_log("Error removing outlier samples", level = "ERROR", details = e$message)
  })
}


##### Applying Gene-Based Filtering #####
write_log("Applying gene-based filters if opted", level = "INFO")

tryCatch({
  # Removing filtered genes and samples
  raw <- raw[!(raw$Name %in% filt_genes), ]
  write_log("Removed filtered genes from the raw dataframe", level = "INFO")
  
  raw_expression <- raw_expression[!(raw_expression$Gene %in% filt_genes), ]
  write_log("Excluded filtered genes from the raw_expression dataframe", level = "INFO")
  
  rawCounts.filt <- data.frame(edgeR_table$counts)
  rawCounts.filt <- rawCounts.filt[!(rownames(rawCounts.filt) %in% filt_genes), !(colnames(rawCounts.filt) %in% filt_samples)]
  write_log("Excluded filtered genes and samples from the rawCounts.filt dataframe", level = "INFO")
  
}, error = function(e) {
  write_log("Error during gene-based filtering", level = "ERROR", details = e$message)
})

# Clean up
rm(edgeR_table, rawCounts, filt_samples, filt_genes)

##### Reordering All Filtered Dataframes #####
write_log("Reordering all filtered dataframes", level = "INFO")

tryCatch({
  clinical_info <- clinical_info[order(clinical_info$Condition), ]
  write_log("Reordered clinical_info by the Condition column", level = "INFO")
  
  pData <- pData[match(row.names(clinical_info), pData$BCAC_ID), ]
  write_log("Reordered pData based on clinical_info", level = "INFO")
  
  raw_expression <- raw_expression[, match(c("Gene", "Class", row.names(clinical_info)), colnames(raw_expression))]
  write_log("Reordered raw_expression based on clinical_info", level = "INFO")
  
  rawCounts.filt <- rawCounts.filt[, match(row.names(clinical_info), colnames(rawCounts.filt))]
  write_log("Reordered rawCounts.filt based on clinical_info", level = "INFO")
  
}, error = function(e) {
  write_log("Error during reordering of filtered dataframes", level = "ERROR", details = e$message)
})

# Clean up
rm(initialqc, exploratory_analysis, iqr_analysis, pData)


# Perform post-filtering exploratory analysis if any filtering options are enabled
if (opt$filter_lowlyExpr_genes || opt$filter_genes_on_negCtrl || opt$filter_samples_on_negCtrl || opt$remove_outlier_samples) {
  write_log("Starting post-filtering exploratory analysis", level = "INFO")
  
  # Call the post-filtering exploratory analysis function
  postfilt_expl_analysis()
}

# Cleanup unnecessary objects from memory
rm(library_size, postfilt_expl_analysis, paletteColors)
write_log("Cleaned up memory by removing post-filtering exploratory analysis objects", level = "INFO")


#################################### NORMALISATION
write_log("Starting normalisation process", level = "INFO")

# Create an empty list to store normalisation methods
norm_list <- list()

# Obtain housekeeping genes
write_log("Obtaining housekeeping genes", level = "INFO")
hk_genes <- raw[raw$CodeClass == "Housekeeping", 2]

# Perform nSolver normalisation
tryCatch({
  write_log("Performing nSolver normalisation", level = "INFO")
  
  suppressMessages(suppressWarnings(
  nsolver_data <- processNanostringData(nsFiles = raw_data,
                                        sampleTab = clinical_data,
                                        idCol = "Filename",
                                        groupCol = "Condition",
                                        normalization = "nSolver",
                                        bgType = "t.test",
                                        bgPVal = 0.01,
                                        skip.housekeeping = FALSE,
                                        output.format = "ExpressionSet")))
  
  # Extract normalized matrix
  nsolver <- data.frame(nsolver_data@featureData@data$Name,
                        nsolver_data@featureData@data$CodeClass,
                        nsolver_data@assayData[["exprs"]])
  
  colnames(nsolver)[1:2] <- c("Name", "CodeClass")
  nsolver <- nsolver[!nsolver$CodeClass %in% c("Positive", "Negative", "Housekeeping"), ]
  colnames(nsolver) <- sub("\\.RCC$", "", colnames(nsolver))
  row.names(nsolver) <- nsolver$Name
  nsolver$Name <- NULL
  nsolver$CodeClass <- NULL
  
}, error = function(e) {
  write_log("Error in nSolver normalisation", level = "ERROR", details = e$message)
})


# Perform other normalisation methods
tryCatch({
  write_log("Preparing data for other normalisation methods", level = "INFO")
  Rnf5 <- merge(raw_expression, raw[, 2:3], by.x = "Gene", by.y = "Name")
  Rnf5 <- Rnf5[, match(c("Gene", "Class", "Accession", row.names(clinical_info)), colnames(Rnf5))]
  colnames(Rnf5)[1:2] <- c("Gene_Name", "Code_Class")
  
  suppressMessages(suppressWarnings(
  nano_data <- count_set(
    count_data = Rnf5,
    group = clinical_info$Condition,
    batch = clinical_info$CartridgeID,
    samp_id = row.names(clinical_info)
  )))
  
  suppressMessages(suppressWarnings(
  rnf5_norm <- multi_norm(
    count_set = nano_data,
    norm_method = "all",
    background_correct = "none",
    positive_control_scaling = TRUE,
    count_threshold = -1,
    geNorm_n = 5
  )))
  
  methods <- c("geNorm_housekeeping", "housekeeping_scaled", 
               "all_endogenous_scaled", "quantile", "loess", "vsn", "ruv", "nSolver")
  
  for (method in methods) {
    if (method == "nSolver") {
      nsolver <- nsolver[order(row.names(nsolver)), order(colnames(nsolver))]
      norm_list[["nSolver"]] <- log2(nsolver + 1)
      rm(nsolver, nsolver_data)
    } else if (method == "ruv") {
      ruv_norm <- ruvseq_norm(raw_expression, rawCounts.filt, clinical_info, opt$refgenes, opt$minref, opt$k_factor)
      ruv_norm <- ruv_norm[row.names(norm_list$vsn), colnames(norm_list$vsn)]
      norm_list[["ruv"]] <- ruv_norm
      rm(ruv_norm)
    } else {
      norm_data <- data.frame(assays(rnf5_norm)[[method]])
      norm_data <- norm_data[!rownames(norm_data) %in% hk_genes, ]
      norm_list[[method]] <- norm_data
    }
    
    # Save the normalisation method to a file
    write.table(norm_list[[method]], file = paste0(ml_results, "/", method, "-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  }
  
  # Additional normalisations: TPM, Min-Max Scaling, Z-Score, and Log Transformation
  tpm <- sweep(rawCounts.filt, 2, colSums(rawCounts.filt), "/") * 1e6
  write.table(tpm, file = paste0(ml_results, "/tpm-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
  minmax <- as.data.frame(scale(rawCounts.filt, center = FALSE, scale = apply(rawCounts.filt, 2, max) - apply(rawCounts.filt, 2, min)))
  write.table(minmax, file = paste0(ml_results, "/minmax-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
  zscore <- as.data.frame(scale(rawCounts.filt))
  write.table(zscore, file = paste0(ml_results, "/zscores-normalised.matrix.ml.tsv"), sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
  logtrans <- log1p(rawCounts.filt)
  write.table(logtrans,file = paste0(ml_results, "/logtransf-normalised.matrix.ml.tsv"),sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE)
  
}, error = function(e) {
  write_log("Error in normalisation methods", level = "ERROR", details = e$message)
})

rm(Rnf5, nano_data, hk_genes, rnf5_norm)

# Determine the best normalisation method
if (opt$norm == "auto") {
  write_log("Automatically selecting the best normalisation method", level = "INFO")
  results_df <- data.frame(row.names = methods, MRLE = rep(0, length(methods)))
  
  for (method in methods) {
    current_df <- norm_list[[method]]
    results_df[method, 1] <- check_norm(current_df, method)
    rm(current_df)
  }
  
  best_norm <- rownames(results_df)[which.min(results_df$MRLE)]
  score <- results_df[best_norm, 1]
  write_log(sprintf("Best normalisation method: %s with MRLE score: %.2f", best_norm, score), level = "INFO")
  norm_data <- norm_list[[best_norm]]
  norm <- best_norm
} else if (opt$norm %in% names(norm_list)) {
  norm <- opt$norm
  norm_data <- norm_list[[norm]]
  score <- check_norm(norm_data, norm)
}

# Downstream analysis
if (norm == "nSolver") {
  clinical_info <- clinical_orig
  rm(clinical_orig)
}

write_log("Generating normalisation plots", level = "INFO")
norm_plots(norm_data, clinical_info, norm, score)

# Output labels for ML
write.table(clinical_info[, c("Sample", "Condition")], file = paste(results, "/labels.ml.tsv", sep = ""), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(clinical_info[, c("Sample", "Condition")], file = paste(ml_results, "/labels.ml.tsv", sep = ""), sep = "\t", row.names = FALSE, quote = FALSE)

# Differential Expression Analysis
write_log("Performing differential expression analysis", level = "INFO")
de_results <- limma_de_analysis(norm_data, clinical_info, norm)

write.table(
  round(norm_data, 3),
  file = paste(results, "/", norm, "-normalised.matrix.ml.tsv", sep = ""),
  sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE
)

volcano_plot(de_results, clinical_info, norm, opt$lfcthreshold, opt$padjusted)

sig_genes <- subset(de_results, abs(log2FC) >= opt$lfcthreshold & padj <= opt$padjusted)
if (nrow(sig_genes) > 0) {
  write.table(
    sig_genes,
    file = paste(results, "/", norm, "-normalised.matrix.de.ml.tsv", sep = ""),
    sep = "\t", row.names = TRUE, col.names = NA, quote = FALSE
  )
}

write_log("Normalisation process completed", level = "INFO")

rm(best_norm, score, norm_plots, results_df, norm_list, check_norm, limma_de_analysis, colors)
write_log("### TRAINING SET ANALYSIS COMPLETED ###", level = "INFO")
