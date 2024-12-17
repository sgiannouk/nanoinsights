suppressPackageStartupMessages(
  suppressMessages(suppressWarnings({
      library(jsonlite)
  })))



# Custom colours for visualisation
paletteColors <- c("#52D3D8", "#799351", "#FFB830", "#726A95", "#C69774",
                   "#FF9DA7", "#65ADC2", "#6E8E84", "#AAD9BB", "#826276",
                   "#233B43", "#800000", "#656A59", "#46B2B5", "#8CAA7E",
                   "#6F5438", "#C29365", "#C17529", "#AD84C6", "#8A8B79",
                   "#DD8047", "#6F8183", "#8784C7", "#84ACB6", "#785D37",
                   "#E63946", "#6A0572", "#2A9D8F", "#F4A261", "#264653", 
                   "#E76F51", "#FFC300", "#A020F0", "#008080", "#FF6347",
                   "#4682B4", "#228B22", "#FF4500", "#B22222", "#9400D3",
                   "#00FA9A", "#DAA520", "#DC143C", "#708090", "#7CFC00",
                   "#FFA07A", "#87CEFA", "#7B68EE", "#FFD700", "#468847",
                   "#556B2F", "#6B8E23", "#C71585", "#00BFFF", "#FFDAB9",
                   "#F5FFFA", "#8B008B", "#E0FFFF", "#FF1493", "#6495ED",
                   "#B0C4DE", "#A9A9A9", "#F0FFF0", "#9932CC", "#FAFAD2",
                   "#8FBC8F", "#20B2AA", "#66CDAA", "#7FFFD4", "#BDB76B")


# Define the log function
write_log <- function(message, level = "INFO", details = NULL) {
  log_entry <- list(timestamp = Sys.time(), level = level, message = message)
  if (!is.null(details)) {log_entry$details <- details}
  # Append the JSON log entry to the log file
  write(toJSON(log_entry, auto_unbox = TRUE, pretty = TRUE), file = log_file, append = TRUE)
}


# Performing standard initial NanoString QC of the input data based on nCounter metrics such as Imaging QC
initialqc <- function() {
  
  nanoqc <- pData[, c('BCAC_ID', 'LaneID', 'CartridgeID', 'imaging', 'imagingQC', 'bindingDensity',
                      'bindingDensityQC', 'positiveLinearity', 'positiveLinearityQC', 'logThreshold',
                      'limitOfDetectionQC')]
  nanoqc[selected_columns] <- lapply(nanoqc[selected_columns], function(x) ifelse(x == "True", "No flag", x))
  nanoqc[selected_columns] <- lapply(nanoqc[selected_columns], function(x) ifelse(x == "False", "Potential outlier sample", x))
  row.names(nanoqc) <- nanoqc$BCAC_ID; nanoqc$BCAC_ID <- NULL
  
  tryCatch({
    ## Imaging QC
    write_log("Creating Imaging QC plot", level = "INFO")
    
    suppressWarnings(
      imgqc <- ggplot(nanoqc, aes(x=CartridgeID, y=imaging, color=CartridgeID)) +
                      geom_boxplot(aes(fill=CartridgeID, alpha = 0.3), outlier.shape = NA, width=.5) +
                      geom_point(aes(text = paste("Sample:", rownames(nanoqc), "<br>CartridgeID:", CartridgeID, 
                                                  "<br>FOV:", imaging, "<br>FovFlag:", imagingQC)),
                                 size = 1.8, position = position_jitterdodge(dodge.width=0.2), alpha = 0.7) +
                      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
                      scale_fill_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
                      coord_cartesian(ylim = c(0.74, 1)) +
                      scale_y_continuous(breaks = seq(0.75, 1, by = 0.05)) +
                      geom_hline(yintercept = 0.75, color="#C70039", linewidth=0.4, alpha=0.4) +
                      annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = 0.75, fill = "#C70039", alpha = 0.1) +
                      theme_classic() +
                      theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                            text = element_text(size = 12, colour="#494949"),
                            axis.text.x = element_text(angle=25, vjust = 0.3, hjust=0.5),
                            axis.title.x = element_text(vjust=-1),
                            panel.grid.major.x = element_line(linewidth=.05, colour="#E6E6E6"),
                            panel.grid.major.y = element_line(linewidth=.4, colour="#E6E6E6"),
                            axis.line = element_line(linewidth = 0.35, colour = "#494949"),
                            legend.position = "None") +
                      labs(title = "Imaging QC",
                           y = "Field Of View (FOV)",
                           x = "CartridgeID"))
    
    ggsave(imgqc, file = paste(results, "/Fig1.A.imagingQC.png", sep = ""), device = "png", width = 14, height = 6, units = "in", dpi = 600)
    write_log("Successfully created Imaging QC plot", level = "INFO")
    
    # Converting to interactive plot
    write_log("Creating interactive Imaging QC plot", level = "INFO")
    
    # Add `text` directly in the Plotly object
    imgpq_plt <- ggplotly(imgqc, tooltip = "text") %>% 
                          highlight(on = "plotly_hover", selectize = TRUE)  %>%
                          layout(showlegend = FALSE, boxpoints = F, font = list(family = "Arial", size = 10))
    
    # Add the tooltip text programmatically
    imgpq_plt$x$data <- lapply(imgpq_plt$x$data, function(trace) {
      if (trace$type == "box") {
        # Disable the marker (outlier) properties
        trace$marker <- list(color = 'rgba(0,0,0,0)') # Make outlier points transparent
        trace$hoverinfo <- "none"
      }
      trace
    })
    
    suppressWarnings(htmlwidgets::saveWidget(widget = imgpq_plt, file = paste(html_results, "/imagingQC.html", sep = ""), selfcontained = TRUE))
    write_log("Interactive Imaging QC plot saved", level = "INFO")
    
    rm(imgqc, imgpq_plt)
    
  }, error = function(e) {
    write_log("Error in Imaging QC plot creation", level = "ERROR", details = e$message)
  })
  
  tryCatch({
    ## Binding Density QC
    write_log("Creating Binding Density QC plot", level = "INFO")
    
    bd_thresholds <- as.numeric(unlist(strsplit(pData$bdThreshold[1], "-")))
    
    suppressWarnings(
    bdenqc <- ggplot(nanoqc, aes(x = CartridgeID, y = bindingDensity, color = CartridgeID)) +
      geom_boxplot(aes(fill = CartridgeID, alpha = 0.3), outlier.shape = NA, width = .5) +
      geom_point(aes(text = paste("Sample:", rownames(nanoqc), "<br>CartridgeID:", CartridgeID,
                                  "<br>BindingDensity:", bindingDensity, "<br>BindingDensityFlag:", bindingDensityQC)),
                 size = 1.5, position = position_jitterdodge(dodge.width = 0.2), alpha = 0.7) +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      scale_fill_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      geom_hline(yintercept = bd_thresholds[1], color = "#C70039", linewidth = 0.4, alpha = 0.4) +
      annotate("rect", xmin = -Inf, xmax = Inf, ymin = 0, ymax = bd_thresholds[1], fill = "#C70039", alpha = 0.1) +
      geom_hline(yintercept = bd_thresholds[2], color = "#C70039", linewidth = 0.4, alpha = 0.4) +
      annotate("rect", xmin = -Inf, xmax = Inf, ymin = bd_thresholds[2], ymax = Inf, fill = "#C70039", alpha = 0.1) +
      theme_classic() +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12, colour = "#494949"),
            axis.text.x = element_text(angle = 25, vjust = 0.3, hjust = 0.5),
            axis.title.x = element_text(vjust = -1),
            panel.grid.major.x = element_line(linewidth = .05, colour = "#E6E6E6"),
            panel.grid.major.y = element_line(linewidth = .4, colour = "#E6E6E6"),
            axis.line = element_line(linewidth = 0.35, colour = "#494949"),
            legend.position = "None") +
      labs(title = "Binding Density QC",
           y = "Binding Density",
           x = "CartridgeID"))
    
    ggsave(bdenqc, file = paste(results, "/Fig1.B.bindingDensityQC.png", sep = ""), device = "png", width = 14, height = 6, units = "in", dpi = 600)
    write_log("Successfully created Binding Density QC plot", level = "INFO")
    
    # Converting to interactive plot
    write_log("Creating interactive Binding Density QC plot", level = "INFO")
    bdenqc_plt <- ggplotly(bdenqc, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = FALSE, boxpoints = FALSE, font = list(family = "Arial", size = 10))
    
    bdenqc_plt$x$data <- lapply(bdenqc_plt$x$data, function(trace) {
      if (trace$type == "box") {
        trace$marker <- list(color = 'rgba(0,0,0,0)') # Make outlier points transparent
        trace$hoverinfo <- "none"
      }
      trace
    })
    
    suppressWarnings(htmlwidgets::saveWidget(widget = bdenqc_plt, file = paste(html_results, "/bindingDensityQC.html", sep = ""), selfcontained = TRUE))
    write_log("Interactive Binding Density QC plot saved", level = "INFO")
    
    rm(bd_thresholds, bdenqc, bdenqc_plt)
  }, error = function(e) {
    write_log("Error in Binding Density QC plot creation", level = "ERROR", details = e$message)
  })
  
  tryCatch({
    ## Positive Control Linearity QC
    write_log("Creating Positive Control Linearity QC plot", level = "INFO")
    
    min_value <- ifelse(min(nanoqc$positiveLinearity) >= 0.95, 0.94, min(nanoqc$positiveLinearity))
    
    suppressWarnings(
    linearityqc <- ggplot(nanoqc, aes(x = CartridgeID, y = positiveLinearity, color = CartridgeID)) +
      geom_boxplot(aes(fill = CartridgeID, alpha = 0.3), outlier.shape = NA, width = .5) +
      geom_point(aes(text = paste("Sample:", rownames(nanoqc), "<br>CartridgeID:", CartridgeID,
                                  "<br>PositiveLinearity:", positiveLinearity, "<br>PositiveLinearityFlag:", positiveLinearityQC)),
                 size = 1.5, position = position_jitterdodge(dodge.width = 0.2), alpha = 0.7) +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      scale_fill_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      geom_hline(yintercept = 0.95, color = "#C70039", linewidth = 0.4, alpha = 0.4) +
      annotate("rect", xmin = -Inf, xmax = Inf, ymin = min_value, ymax = 0.95, fill = "#C70039", alpha = 0.1) +
      theme_classic() +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12, colour = "#494949"),
            axis.text.x = element_text(angle = 25, vjust = 0.3, hjust = 0.5),
            axis.title.x = element_text(vjust = -1),
            panel.grid.major.x = element_line(linewidth = .05, colour = "#E6E6E6"),
            panel.grid.major.y = element_line(linewidth = .4, colour = "#E6E6E6"),
            axis.line = element_line(linewidth = 0.35, colour = "#494949"),
            legend.position = "None") +
      labs(title = "Positive Control Linearity QC",
           y = "Positive Linearity (r2)",
           x = "CartridgeID"))
    
    ggsave(linearityqc, file = paste(results, "/Fig1.C.positiveLinearityQC.png", sep = ""), device = "png", width = 14, height = 6, units = "in", dpi = 600)
    write_log("Successfully created Positive Control Linearity QC plot", level = "INFO")
    
    # Converting to interactive plot
    write_log("Creating interactive Positive Control Linearity QC plot", level = "INFO")
    
    linearityqc_plt <- ggplotly(linearityqc, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = FALSE, boxpoints = FALSE, font = list(family = "Arial", size = 10))
    
    linearityqc_plt$x$data <- lapply(linearityqc_plt$x$data, function(trace) {
      if (trace$type == "box") {
        trace$marker <- list(color = 'rgba(0,0,0,0)') # Make outlier points transparent
        trace$hoverinfo <- "none"
      }
      trace
    })
    
    suppressWarnings(htmlwidgets::saveWidget(widget = linearityqc_plt, file = paste(html_results, "/positiveLinearityQC.html", sep = ""), selfcontained = TRUE))
    write_log("Interactive Positive Control Linearity QC plot saved", level = "INFO")
    
    rm(min_value, linearityqc, linearityqc_plt)
  }, error = function(e) {
    write_log("Error in Positive Control Linearity QC plot creation", level = "ERROR", details = e$message)
  })
  
  tryCatch({
    ## Limit of Detection QC
    write_log("Creating Limit of Detection QC plot", level = "INFO")
    mean_min_lod <- round(mean(pData$posE), 0)
    
    suppressWarnings(
    limitofdetqc <- ggplot(nanoqc, aes(x = CartridgeID, y = logThreshold, color = CartridgeID)) +
      geom_boxplot(aes(fill = CartridgeID, alpha = 0.3), outlier.shape = NA, width = .5) +
      geom_point(aes(text = paste("Sample:", rownames(nanoqc), "<br>CartridgeID:", CartridgeID,
                                  "<br>limitOfDetection:", logThreshold, "<br>limitOfDetectionFlag:", limitOfDetectionQC)),
                 size = 1.5, position = position_jitterdodge(dodge.width = 0.2), alpha = 0.7) +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      scale_fill_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      geom_hline(yintercept = mean_min_lod, color = "#C70039", linewidth = 0.4, alpha = 0.4) +
      annotate("rect", xmin = -Inf, xmax = Inf, ymin = mean_min_lod, ymax = Inf, fill = "#C70039", alpha = 0.1) +
      theme_classic() +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12, colour = "#494949"),
            axis.text.x = element_text(angle = 25, vjust = 0.3, hjust = 0.5),
            axis.title.x = element_text(vjust = -1),
            panel.grid.major.x = element_line(linewidth = .05, colour = "#E6E6E6"),
            panel.grid.major.y = element_line(linewidth = .4, colour = "#E6E6E6"),
            axis.line = element_line(linewidth = 0.35, colour = "#494949"),
            legend.position = "None") +
      labs(title = "Limit Of Detection QC",
           y = "Limit Of Detection",
           x = "CartridgeID"))
    
    ggsave(limitofdetqc, file = paste(results, "/Fig1.D.limitOfDetectionQC.png", sep = ""), device = "png", width = 14, height = 6, units = "in", dpi = 600)
    write_log("Successfully created Limit of Detection QC plot", level = "INFO")
    
    # Converting to interactive plot
    write_log("Creating interactive Limit of Detection QC plot", level = "INFO")
    limitofdetqc_plt <- ggplotly(limitofdetqc, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = FALSE, boxpoints = FALSE, font = list(family = "Arial", size = 10))
    
    limitofdetqc_plt$x$data <- lapply(limitofdetqc_plt$x$data, function(trace) {
      if (trace$type == "box") {
        trace$marker <- list(color = 'rgba(0,0,0,0)') # Make outlier points transparent
        trace$hoverinfo <- "none"
      }
      trace
    })
    
    suppressWarnings(htmlwidgets::saveWidget(widget = limitofdetqc_plt, file = paste(html_results, "/limitOfDetectionQC.html", sep = ""), selfcontained = TRUE))
    write_log("Interactive Limit of Detection QC plot saved", level = "INFO")
    
    rm(mean_min_lod, limitofdetqc, limitofdetqc_plt)
  }, error = function(e) {
    write_log("Error in Limit of Detection QC plot creation", level = "ERROR", details = e$message)
  })
}


# Performing exploratory analysis on the input dataset.
exploratory_analysis <- function() {
  
  write_log("Starting exploratory analysis", level = "INFO")
  
  # Boxplot of the raw data - Endogenous genes only
  rawCounts_log <- tryCatch({
    data.frame(cbind(Genes = raw_expression[grepl('Endogenous', raw_expression$Class), 1],
                     log2(raw_expression[grepl('Endogenous', raw_expression$Class), 3:length(raw_expression)] + 1)))
  }, error = function(e) {
    write_log("Error creating the rawCounts_log dataframe", level = "ERROR", details = e$message)
    stop(e)
  })
    
  colnames(rawCounts_log)[2:length(rawCounts_log)] <- colnames(raw_expression[3:length(raw_expression)])
    
  rawCounts_log_melt <- tryCatch({
    rawCounts_log %>%
      melt(id.vars = "Genes") %>%
      merge(clinical_info[, c(1, 3)], by.x = "variable", by.y = "Sample") %>%
      mutate(CartridgeID = as.character(CartridgeID),
             CartridgeID = factor(CartridgeID, levels = as.vector(unique(clinical_info$CartridgeID))))
  }, error = function(e) {
    write_log("Error processing rawCounts_log_melt dataframe", level = "ERROR", details = e$message)
    stop(e)
  })
    
  rawCounts_log_melt <- rawCounts_log_melt %>% arrange(CartridgeID, variable)
  rawCounts_log_melt$variable <- factor(rawCounts_log_melt$variable, levels = unique(rawCounts_log_melt$variable))
    
  write_log("Starting box plot analysis", level = "INFO")
  tryCatch({
    # Box 1: Boxplot of all samples (Unnormalised Dataset)
    suppressWarnings(
    box1 <- rawCounts_log_melt %>%
      mutate(variable = fct_reorder2(variable, value, CartridgeID, .desc = FALSE)) %>%
      ggplot(aes(x = variable, y = value, color = CartridgeID)) +
      geom_point(aes(text = paste("Sample:", variable, "<br>log2Counts:", round(value, 2))),
                 alpha = 0.3, position = "jitter") +
      geom_boxplot(alpha = 0, colour = "black", width = 0.6) +
      theme_bw() +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            panel.border = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(linewidth = .1, color = "gray78"),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            axis.line = element_line(colour = "black"),
            legend.position = "right") +
      guides(colour = guide_legend(override.aes = list(alpha = 0.6))) +
      labs(title = "Box Plot of the Unnormalised Dataset (Endogenous only)",
           y = "log2(unnormalised counts)",
           x = "Datasets") +
      geom_hline(yintercept = median(rawCounts_log_melt$value), color = "#B67171", linewidth = .7))
    
    ggsave(box1, file = paste(results, "/Fig2.A.boxplot.allSamples.png", sep = ""), device = "png", width = 14, height = 6, units = "in", dpi = 600)
    write_log("Saved Box Plot (Fig2.A.boxplot.allSamples.png)", level = "INFO")
    
    # Outlier Genes (Box 11)
    outliers <- rawCounts_log_melt %>% group_by(variable) %>%
      summarize(lower = quantile(value, probs = 0.25) - 1.5 * IQR(value),
                upper = quantile(value, probs = 0.75) + 1.5 * IQR(value)) %>%
      left_join(rawCounts_log_melt, by = "variable") %>%
      filter(value < lower | value > upper) %>%
      select(-lower, -upper)
    
    suppressWarnings(
    box11 <- rawCounts_log_melt %>%
      ggplot(aes(x = variable, y = value, color = CartridgeID, fill = CartridgeID)) +
      geom_point(data = outliers, aes(text = paste("Sample:", variable, "<br>log2Counts:", round(value, 2)))) +
      geom_boxplot(alpha = 0.6, width = 0.6) +
      theme_bw() +
      scale_fill_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            panel.border = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(linewidth = .1, color = "gray78"),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            axis.line = element_line(colour = "black"),
            legend.position = "right") +
      guides(colour = guide_legend(override.aes = list(alpha = 0.6))) +
      labs(title = "Box Plot of the Unnormalised Dataset (Endogenous only)",
           y = "log2(unnormalised counts)",
           x = "Datasets") +
      geom_hline(yintercept = median(rawCounts_log_melt$value), color = "#B67171", linewidth = .7))
    
    write_log("Generated Interactive Box Plot 1", level = "INFO")
    
    box1_plt <- ggplotly(box11, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = TRUE, boxpoints = FALSE, font = list(family = "Arial", size = 12))
    
    htmlwidgets::saveWidget(widget = box1_plt, file = paste(html_results, "/boxplot1.html", sep = ""), selfcontained = TRUE)
    write_log("Saved interactive Box Plot 1 (boxplot1.html)", level = "INFO")
    
    rm(box1, box11, outliers, box1_plt)
    
    # Box 2: Boxplot per Batch
    suppressWarnings(
    box2 <- ggplot(rawCounts_log_melt, aes(x = CartridgeID, y = value, color = CartridgeID)) +
      geom_point(aes(text = paste("Sample:", variable, "<br>log2Counts:", round(value, 2))),
                 alpha = 0.3, position = "jitter") +
      geom_boxplot(alpha = 0, colour = "black", width = 0.6) +
      theme_bw() +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            panel.border = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(linewidth = .1, color = "gray78"),
            # axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            axis.text.x = element_text(angle = 25, vjust = 0.3, hjust = 0.5),
            axis.line = element_line(colour = "black"),
            legend.position = "none") +
      guides(colour = guide_legend(override.aes = list(alpha = 0.6))) +
      labs(title = "Box Plot of the Unnormalised Dataset per Batch (Endogenous only)",
           y = "log2(unnormalised counts)",
           x = "CartridgeID") +
      geom_hline(yintercept = median(rawCounts_log_melt$value), color = "#B67171", linewidth = .7))
    
    ggsave(box2, file = paste(results, "/Fig2.B.boxplot.cartridgeID.png", sep = ""), device = "png", width = 14, height = 6, units = "in", dpi = 600)
    write_log("Saved Box Plot (Fig2.B.boxplot.cartridgeID.png)", level = "INFO")
    
    # Outlier Samples (Box 22)
    outliers <- rawCounts_log_melt %>% group_by(CartridgeID) %>%
      summarize(lower = quantile(value, probs = 0.25) - 1.5 * IQR(value),
                upper = quantile(value, probs = 0.75) + 1.5 * IQR(value)) %>%
      left_join(rawCounts_log_melt, by = "CartridgeID") %>%
      filter(value < lower | value > upper) %>%
      select(-lower, -upper)
    
    suppressWarnings(
    box22 <- ggplot(rawCounts_log_melt, aes(x = CartridgeID, y = value, color = CartridgeID, fill = CartridgeID)) +
      geom_point(data = outliers, aes(x = CartridgeID, y = value, text = paste("Sample:", variable, "<br>log2Counts:", round(value, 2)))) +
      geom_boxplot(alpha = 0.6, width = 0.6) +
      theme_bw() +
      scale_fill_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            panel.border = element_blank(),
            panel.grid.minor = element_blank(),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(linewidth = .1, color = "gray78"),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            axis.line = element_line(colour = "black"),
            legend.position = "none") +
      guides(colour = guide_legend(override.aes = list(alpha = 0.6))) +
      labs(title = "Box Plot of the Unnormalised Dataset per Batch (Endogenous only)",
           y = "log2(unnormalised counts)",
           x = "CartridgeID") +
      geom_hline(yintercept = median(rawCounts_log_melt$value), color = "#B67171", linewidth = .7))
    
    write_log("Generated Interactive Box Plot 2 with Outlier Samples", level = "INFO")
    
    box2_plt <- ggplotly(box22, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = FALSE, font = list(family = "Arial", size = 12))
    
    htmlwidgets::saveWidget(widget = box2_plt, file = paste(html_results, "/boxplot2.html", sep = ""), selfcontained = TRUE)
    write_log("Saved interactive Box Plot 2 (boxplot2.html)", level = "INFO")
    
    rm(box2, box22, box2_plt, outliers, rawCounts_log_melt, rawCounts_log)
    write_log("Successfully completed box plot analysis", level = "INFO")
    
  }, error = function(e) {
    write_log("Error during box plot analysis", level = "ERROR", details = e$message)
    stop(e)
  })
  
  
  write_log("Starting PCA analysis", level = "INFO")
  tryCatch({
    # PCA Plot 1: Condition
    suppressWarnings(
    p1 <- ggplot(data_pca, aes(PC1, PC2, color = Condition, label = name)) +
      geom_point(aes(text = paste("Sample:", name, "<br>CartridgeID:", CartridgeID,
                                  "<br>Date:", Date, "<br>TotalCounts:", Total)), size = 2) +
      geom_text(size = 2, nudge_y = 0.15) +
      theme_bw() +
      scale_color_manual(values = c(colors[1], colors[2])) +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            panel.border = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black"),
            legend.position = "right",
            legend.justification = "center") +
      labs(title = "Principal Component Analysis Plot with Sample Name Annotations",
           x = paste0("PC1: ", percentVar[1], "% variance"),
           y = paste0("PC2: ", percentVar[2], "% variance")))
    
    ggsave(p1, file = paste(results, "/Fig2.C.pca.unfiltered.condition.png", sep = ""), device = "png", width = 10, height = 8, units = "in", dpi = 600)
    write_log("Saved PCA Plot 1 (Fig2.C.pca.unfiltered.condition.png)", level = "INFO")
    
    # Interactive PCA Plot 1
    p1_plt <- ggplotly(p1, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = TRUE, font = list(family = "Arial", size = 12), 
             legend = list(traceorder = "normal", title = list(text = "Condition")))
    
    htmlwidgets::saveWidget(widget = p1_plt, file = paste(html_results, "/pca.unfiltered.condition.html", sep = ""), selfcontained = TRUE)
    write_log("Saved interactive PCA Plot 1 (pca.unfiltered.condition.html)", level = "INFO")
    
    # PCA Plot 2: CartridgeID
    suppressWarnings(
    p2 <- ggplot(data_pca, aes(PC1, PC2, color = CartridgeID, label = name)) +
      geom_point(aes(text = paste("Sample:", name, "<br>CartridgeID:", CartridgeID,
                                  "<br>Date:", Date, "<br>TotalCounts:", Total)), size = 3, alpha = 0.7) +
      scale_color_manual(values = paletteColors[1:length(unique(clinical_info$CartridgeID))]) +
      theme_bw() +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            panel.border = element_blank(),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black"),
            legend.position = "right",
            legend.justification = "center") +
      labs(title = "Principal Component Analysis Plot with Batch Annotations",
           x = paste0("PC1: ", percentVar[1], "% variance"),
           y = paste0("PC2: ", percentVar[2], "% variance")))
    
    ggsave(p2, file = paste(results, "/Fig2.D.pca.unfiltered.cartridgeID.png", sep = ""), device = "png", width = 10, height = 8, units = "in", dpi = 600)
    write_log("Saved PCA Plot 2 (Fig2.D.pca.unfiltered.cartridgeID.png)", level = "INFO")
    
    # Interactive PCA Plot 2
    p2_plt <- ggplotly(p2, tooltip = "text") %>%
      highlight(on = "plotly_hover", selectize = TRUE) %>%
      layout(showlegend = TRUE, boxpoints = FALSE, font = list(family = "Arial", size = 12))
    
    htmlwidgets::saveWidget(widget = p2_plt, file = paste(html_results, "/pca.unfiltered.cartridge.html", sep = ""), selfcontained = TRUE)
    write_log("Saved interactive PCA Plot 2 (pca.unfiltered.cartridge.html)", level = "INFO")
  
    write_log("Successfully completed PCA analysis", level = "INFO")
    rm(p1, p1_plt, p2, p2_plt)
    
  }, error = function(e) {
    write_log("Error during PCA analysis", level = "ERROR", details = e$message)
    stop(e)
  })
  
  
  write_log("Starting MDS analysis", level = "INFO")
  tryCatch({
    
    # Calculate Euclidean distance matrix
    dist_matrix <- tryCatch({
      dist(t(assay(rld)))
    }, error = function(e) {
      write_log("Error calculating the distances for the MDS plot", level = "ERROR", details = e$message)
      stop(e)
    })
    
    # Perform MDS and manipulate the dataframe
    mds_result <- tryCatch({
      # Calculate MDS dimensions
      mds <- data.frame(cmdscale(dist_matrix, k = 2))
      colnames(mds) <- c("Dim1", "Dim2")
      
      # Merge with clinical_info
      merged_mds <- merge(mds, clinical_info, by = 0, all.x = TRUE)
      
      # Clean up merged dataframe
      rownames(merged_mds) <- merged_mds$Row.names
      merged_mds$Row.names <- NULL
      
      write_log("Successfully performed scaling and merged MDS results with clinical_info", level = "INFO")
      merged_mds
    }, error = function(e) {
      write_log("Error performing scaling or merging for the MDS plot", level = "ERROR", details = e$message)
      stop(e)
    })
    
    mds_result$Row.names <- NULL
    
    # Generate MDS Plot
    mdsplot <- tryCatch({
      suppressWarnings(
      ggplot(mds_result, aes(x = Dim1, y = Dim2, label = Sample, color = Condition)) +
        geom_point(aes(text = paste("Sample:", Sample, "<br>CartridgeID:", CartridgeID, "<br>Condition:", Condition)),
                   size = 2.5) +
        geom_text(size = 2, nudge_y = 0.15, color = "#404258") +
        theme_bw() +
        scale_color_manual(values = c(colors[1], colors[2])) +
        theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
              text = element_text(size = 12),
              panel.border = element_blank(),
              panel.grid.minor = element_blank(),
              axis.line = element_line(colour = "black"),
              legend.position = "right",
              legend.justification = "center") +
        labs(title = "Pre-filtering Multi-Dimensional Scaling Plot with Sample Name Annotations",
             x = "Dim1",
             y = "Dim2"))
    }, error = function(e) {
      write_log("Error generating the MDS plot", level = "ERROR", details = e$message)
      stop(e)
    })
    
    ggsave(mdsplot, file = paste(results, "/Fig2.E.mds.unfiltered.condition.png", sep = ""), device = "png", width = 10, height = 8, units = "in", dpi = 600)
    write_log("Saved MDS plot (Fig2.E.mds.unfiltered.condition.png)", level = "INFO")
    
    # Generate MDS Plot without text
    mdsplot2 <- tryCatch({
      suppressWarnings(
      ggplot(mds_result, aes(x = Dim1, y = Dim2, label = Sample, color = Condition)) +
        geom_point(aes(text = paste("Sample:", Sample, "<br>CartridgeID:", CartridgeID, "<br>Condition:", Condition)),
                   size = 2.5) +
        theme_bw() +
        scale_color_manual(values = c(colors[1], colors[2])) +
        theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
              text = element_text(size = 12),
              panel.border = element_blank(),
              panel.grid.minor = element_blank(),
              axis.line = element_line(colour = "black"),
              legend.position = "right",
              legend.justification = "center") +
        labs(title = "Pre-filtering Multi-Dimensional Scaling Plot",
             x = "Dim1",
             y = "Dim2"))
    }, error = function(e) {
      write_log("Error generating the MDS plot without text", level = "ERROR", details = e$message)
      stop(e)
    })
    
    # Converting to interactive plot
    mdsplot_interactive <- tryCatch({
      ggplotly(mdsplot2, tooltip = "text") %>%
        highlight(on = "plotly_hover", selectize = TRUE) %>%
        layout(showlegend = TRUE, font = list(family = "Arial", size = 12))
    }, error = function(e) {
      write_log("Error making MDS plot interactive", level = "ERROR", details = e$message)
      stop(e)
    })
    
    htmlwidgets::saveWidget(widget = mdsplot_interactive, file = paste(html_results, "/mds.unfiltered.condition.html", sep = ""), selfcontained = TRUE)
    write_log("Saved interactive MDS plot to HTML (mds.unfiltered.condition.html)", level = "INFO")
    
    # Cleanup
    rm(dist_matrix, mds_result, mdsplot, mdsplot2)
    
    write_log("Successfully completed MDS analysis", level = "INFO")
    
  }, error = function(e) {
    write_log("Error during MDS analysis", level = "ERROR", details = e$message)
    stop(e)
  })
  
}


# Performing Interquartile Range analysis to detect potential outlier samples in each of the two groups
iqr_analysis <- function() {
  write_log("Starting interquartile range (IQR) analysis", level = "INFO")
  
  # Define outlier detection function
  is_outlier <- function(x) {
    lower_bound <- quantile(x, 0.25) - opt$iqrcutoff * IQR(x)
    upper_bound <- quantile(x, 0.75) + opt$iqrcutoff * IQR(x)
    return(x < lower_bound | x > upper_bound)
  }
  
  # Create dataset for IQR analysis
  dat <- tryCatch({
    data_pca %>%
      tibble::rownames_to_column(var = "outlier") %>%
      group_by(Condition) %>%
      mutate(is_outlier = ifelse(is_outlier(Endogenous), outlier, NA))
  }, error = function(e) {
    write_log("Error creating dataset for IQR analysis", level = "ERROR", details = e$message)
    stop(e)
  })
  
  # Identify potential outliers
  potential_outliers <- na.omit(dat$is_outlier)
  log_elements_string <- if (length(potential_outliers) > 0) paste(potential_outliers, collapse = ",") else NA
  
  # Log potential outliers
  if (length(potential_outliers) > 0) {
    write_log(sprintf("Potential outlier samples detected using %s IQR rule: %s", opt$iqrcutoff, log_elements_string), level = "INFO")
    if (opt$remove_outlier_samples) {
      write(potential_outliers, file = paste(results, "/potential_outliers_based_on_iqr.tsv", sep = ""), append = FALSE, sep = ",")
    }
  } else {
    write_log(sprintf("No potential outliers detected using %s IQR rule", opt$iqrcutoff), level = "INFO")
  }
  
  tryCatch({
    
    # Create boxplot
    suppressWarnings(
    op <- ggplot(dat, aes(y = Endogenous, x = Condition, fill = Condition)) +
                 geom_boxplot(outlier.shape = NA, width = 0.15) +
                 geom_point(aes(color = factor(is_outlier)), position = position_jitterdodge(dodge.width = 0.02), size = 1.8) +
                 geom_text_repel(data = subset(dat, !is.na(is_outlier)), 
                                 aes(label = outlier),
                                 size = 2.5,
                                 max.overlaps = Inf,
                                 box.padding = 0.3,
                                 point.padding = 0.5,
                                 segment.color = '#2A3335') +
                 theme_minimal() +
                 scale_fill_manual(values = colors) +
                 scale_color_manual(values = c("FALSE" = "#7f7f7f", "TRUE" = "#8E7AB5"), guide = "none") + # Map colors for outlier status
                 theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
                       text = element_text(size = 12),
                       axis.title = element_text(size = 12),
                       axis.text = element_text(size = 12),
                       axis.line = element_line(colour = "black"),
                       legend.position = "none") +
                labs(title = "Boxplot based on the Endogenous gene counts",
                     y = "Total Endogenous Read Counts",
                     x = "",
                     caption = if (length(potential_outliers) > 0) {
                                  paste0("The indicated samples might be potential outliers using the interquartile range criterion (", opt$iqrcutoff, " IQR rule)")
                              } else {
                                  paste0("No sample was found to be potential outliers using the interquartile range criterion (", opt$iqrcutoff, " IQR rule)")}))
    
      ggsave(op, file = paste(results, "/Fig2.F.boxplot.outliers.png", sep = ""), device = "png", width = 10, height = 8, units = "in", bg = "white", dpi = 600)
      write_log("Saved IQR boxplot", level = "INFO")
      
    
      # Interactive plot without geom_text_repel
      suppressWarnings(
      op_interactive <- ggplot(dat, aes(y = Endogenous, x = Condition, fill = Condition)) +
                               geom_boxplot(outlier.shape = NA, width = 0.015) +
                               geom_point(aes(text = paste("Sample:", outlier, "<br>Condition:", Condition, "<br>Endogenous:", round(Endogenous, 2)),
                                              color = factor(is_outlier)), position = position_jitterdodge(dodge.width = 0.2), size = 1.8) +
          theme_minimal() +
          scale_fill_manual(values = colors) +
          scale_color_manual(values = c("FALSE" = "#7f7f7f", "TRUE" = "#8E7AB5"), guide = "none") +
          theme(
            plot.title = element_text(colour = "#a6a6a4", size = 13),
            text = element_text(size = 12),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 12),
            axis.line = element_line(colour = "black"),
            legend.position = "none") +
          labs(title = "Boxplot based on the Endogenous gene counts",
               y = "Total Endogenous Read Counts",
               x = ""))  
      
      op_plt <- ggplotly(op_interactive, tooltip = "text", preserve = "text") %>%
                         highlight(on = "plotly_hover", selectize = TRUE) %>%
                         layout(showlegend = FALSE, font = list(family = "Arial", size = 12))
      
      htmlwidgets::saveWidget(widget = op_plt, file = paste(html_results, "/boxplot.outliers.html", sep = ""), selfcontained = TRUE)
      
  }, error = function(e) {
    write_log("Error in IQR boxplot", level = "ERROR", details = e$message)
    stop(e)
  })

  
  # Cleanup
  rm(op, op_plt)
  write_log("Successfully completed IQR analysis", level = "INFO")
  
  return(potential_outliers)
}


# Performing PCA, MDS analysis as well as plotting a sample correlation heatmap
postfilt_expl_analysis <- function() {
  write_log("Starting post-filtering exploratory analysis", level = "INFO")
  
  
  tryCatch({
    ### PCA Plot
    write_log("Importing rawCounts.filt into DESeq2", level = "INFO")
    dds <- DESeqDataSetFromMatrix(countData = rawCounts.filt, colData = clinical_info, design = ~ Condition)
    
    write_log("Converting DDS to rlog", level = "INFO")
    suppressMessages(suppressWarnings(rld <- rlog(dds, blind = TRUE, fitType = "local")))
    rld_assay <- assay(rld)  # Cache for reuse
    
    write_log("Calculating PCA and saving it in data_pca", level = "INFO")
    data_pca <- plotPCA(rld, intgroup = "Condition", returnData = TRUE)
    percentVar <- round(100 * attr(data_pca, "percentVar"))
    
    write_log("Merging data_pca with library_size", level = "INFO")
    library_data <- library_size[, c("CartridgeID", "Endogenous", "Total", "Date")]
    data_pca <- merge(data_pca, library_data, by.x = 0, by.y = 0)
    row.names(data_pca) <- data_pca$Row.names
    data_pca$Row.names <- NULL
    data_pca <- data_pca[match(row.names(clinical_info), row.names(data_pca)), ]
    
    write_log("Creating PCA plot", level = "INFO")
    suppressWarnings(
    pca <- ggplot(data_pca, aes(PC1, PC2, color = Condition, label = name)) +
                  geom_point(aes(text = paste("Sample:", name, "<br>CartridgeID:", CartridgeID)), size = 2, alpha = 0.7) +
                  geom_text(size = 2, nudge_y = 0.15) +
                  theme_bw() +
                  scale_color_manual(values = colors[1:2]) +
                  theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
                        text = element_text(size = 12),
                        axis.text = element_text(size = 12),
                        panel.border = element_blank(),
                        panel.grid.minor = element_blank(),
                        axis.line = element_line(colour = "black"),
                        legend.position = "right",
                        legend.justification = "center") +
                  labs(title = "Post-filtering PCA with Sample Name Annotations",
                       x = paste0("PC1: ", percentVar[1], "% variance"),
                       y = paste0("PC2: ", percentVar[2], "% variance")))
    
    ggsave(pca, file = paste(results, "/Fig3.A.pca.filtered.condition.png", sep = ""), device = "png", width = 10, height = 8, units = "in", dpi = 600)
    
    
    pca2 <- ggplot(data_pca, aes(PC1, PC2, color=Condition, label=name)) +
                   geom_point(aes(text = paste("Sample:", name, "<br>CartridgeID:", CartridgeID)), size=2, alpha=0.7) +
                   theme_bw() +
                   scale_color_manual(values=c(colors[1], colors[2])) +
                   theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                         text = element_text(size = 12),
                         axis.text=element_text(size=12),
                         panel.border = element_blank(),
                         panel.grid.minor = element_blank(),
                         axis.line = element_line(colour = "black"),
                         legend.position = "right",
                         legend.justification = "center") +
                   labs(title = "Post-filtering Principal Component Analysis Plot with Sample Name Annotations",
                        x = paste0("PC1: ",percentVar[1],"% variance"),
                        y = paste0("PC2: ",percentVar[2],"% variance"))
    
    pca_plt <- ggplotly(pca2, tooltip = "text") %>% 
                        highlight(on = "plotly_hover", selectize = T)  %>%
                        layout(showlegend = T, font = list(family = "Arial", size = 12))
    
    suppressWarnings(htmlwidgets::saveWidget(widget = pca_plt, file = paste(html_results, "/pca.filtered.condition.html", sep=""), selfcontained = TRUE))
    rm(pca, pca2, pca_plt)
    
  }, error = function(e) {
    write_log("Error during PCA analysis", level = "ERROR", details = e$message)
  })
  
  tryCatch({
    ### MDS Plot
    write_log("Calculating distances for MDS plot", level = "INFO")
    dist_matrix <- dist(t(rld_assay))  # Use cached `rld_assay`
    
    write_log("Performing MDS scaling", level = "INFO")
    mds_result <- data.frame(cmdscale(dist_matrix, k = 2))
    colnames(mds_result) <- c("Dim1", "Dim2")
    
    write_log("Merging MDS results with clinical_info", level = "INFO")
    mds_result <- merge(mds_result, clinical_info, by = 0)
    mds_result$Row.names <- NULL
    
    write_log("Generating MDS plot", level = "INFO")
    suppressWarnings(
    mdsplot <- ggplot(mds_result, aes(x = Dim1, y = Dim2, label = Sample, color = Condition)) +
                      geom_point(size = 2.5) +
                      geom_text(size = 2, nudge_y = 0.15, color = "#404258") +
                      theme_bw() +
                      scale_color_manual(values = colors[1:2]) +
                      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
                            text = element_text(size = 12),
                            axis.text = element_text(size = 12),
                            panel.border = element_blank(),
                            panel.grid.minor = element_blank(),
                            axis.line = element_line(colour = "black"),
                            legend.position = "right",
                            legend.justification = "center") +
                      labs(title = "Post-filtering MDS Plot with Sample Name Annotations",
                           x = "Dim1",
                           y = "Dim2"))
    
    ggsave(mdsplot, file = paste(results, "/Fig3.B.mds.filtered.condition.png", sep = ""), device = "png", width = 10, height = 8, units = "in", dpi = 600)
    
    suppressWarnings(
    mdsplot2 <- ggplot(mds_result, aes(x = Dim1, y = Dim2, label = Sample, color = Condition)) +
                       geom_point(aes(text = paste("Sample:", Sample, "<br>CartridgeID:", CartridgeID, "<br>Condition:", Condition)), 
                                  size=2.5) +
                       theme_bw() +
                       scale_color_manual(values = colors[1:2]) +
                       theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                             text = element_text(size = 12),
                             axis.text=element_text(size=12),
                             panel.border = element_blank(),
                             panel.grid.minor = element_blank(),
                             axis.line = element_line(colour = "black"),
                             legend.position = "right",
                             legend.justification = "center") +
                       labs(title = "Post-filtering Multi-Dimensional Scaling Plot with Sample Name Annotations",
                            x = "Dim1",
                            y = "Dim2"))
    
    mdsplot_int <- ggplotly(mdsplot2, tooltip = "text") %>% 
                            highlight(on = "plotly_hover", selectize = T)  %>%
                            layout(showlegend = T, font = list(family = "Arial", size = 12))
    
    suppressWarnings(htmlwidgets::saveWidget(widget = mdsplot_int, file = paste(html_results, "/mds.filtered.condition.html", sep=""), selfcontained = TRUE))
    
    rm(dist_matrix, mds_result, mdsplot, mdsplot2, mdsplot_int)
    
  }, error = function(e) {
    write_log("Error during MDS analysis", level = "ERROR", details = e$message)
  })
  
  tryCatch({
    ### Correlation Heatmap
    write_log("Creating Correlation Heatmap", level = "INFO")
    
    # data_pca$Condition <- str_to_title(as.character(groups_capital))
    group_df <- data.frame(row.names = row.names(data_pca), 
                           Condition = data_pca$Condition, 
                           Cartridge = data_pca$CartridgeID)
    
    unique_runs <- as.vector(unique(clinical_info$CartridgeID))
    
    # Create batch list for color mapping
    batch_list <- setNames(paletteColors[1:length(unique_runs)], unique_runs)
    group_df_colors <- list("Condition" = setNames(c(colors[1], colors[2]), unique(data_pca$Condition)),
                            "Cartridge" = batch_list)
    
    # Generating Spearman correlation matrix
    write_log("Calculating Spearman correlation matrix", level = "INFO")
    spearman_correlation <- cor(assay(rld), method = "spearman")
    
    # Generate heatmap using pheatmap
    write_log("Generating Spearman correlation heatmap using pheatmap", level = "INFO")
    
    # Capture the pheatmap output without saving as a PDF
    suppressWarnings(
      corheat <- pheatmap(spearman_correlation,
                          main = "Sample correlation heatmap on filtered data\n(based on Spearman correlation)",
                          fontsize = 8,
                          border_color = NA,
                          angle_col = 90,
                          treeheight_row = 0,
                          annotation_col = group_df,
                          annotation_colors = group_df_colors,
                          color = colorRampPalette(brewer.pal(n = 7, name = "YlGnBu"))(100)))
    
    png(filename = paste0(results, "/Fig3.C.sample.correlation.filtered.png"), width = 18, height = 12, units = "in", res = 600)
    print(corheat)
    dev.off()
    
    # suppressWarnings(
    # corheat <- grid.grabExpr({
    #   pheatmap(spearman_correlation,
    #            main = "Sample correlation heatmap on filtered data\n(based on Spearman correlation)",
    #            fontsize = 8,
    #            border_color = NA,
    #            angle_col = 90,
    #            treeheight_row = 0,
    #            annotation_col = group_df,
    #            annotation_colors = group_df_colors,
    #            color = colorRampPalette(brewer.pal(n = 7, name = "YlGnBu"))(100))}))
    # 
    # # Save the captured plot as a PNG
    # ggsave(corheat, file = paste0(results, "/Fig3.C.sample.correlation.filtered.png"), width = 18, height = 12, units = "in", dpi = 600)
    
    
    # Generate interactive heatmap using heatmaply
    write_log("Generating interactive Spearman correlation heatmap", level = "INFO")
    
    suppressWarnings(
      intheatmap <- heatmaply_cor(spearman_correlation,
                                  file = paste0(html_results, "/sample.correlation.filtered.html"),
                                  limits = NULL,
                                  colors = colorRampPalette(brewer.pal(n = 7, name = "YlGnBu"))(100),
                                  main = "Sample correlation (Spearman) heatmap on filtered data",
                                  key.title = NULL,
                                  hide_colorbar = FALSE,
                                  # cellnote = group_df,
                                  # cellnote_color = group_df_colors,
                                  column_text_angle = 90,
                                  dendrogram = "column",
                                  fontsize_col = 7,
                                  fontsize_row = 6,
                                  showticklabels = c(TRUE, TRUE)))
    
    # Clean up variables
    rm(intheatmap, corheat, spearman_correlation, group_df, group_df_colors, unique_runs, batch_list)
    write_log("Successfully generated correlation heatmaps", level = "INFO")
    
  }, error = function(e) {
    write_log("Error during heatmap generation", level = "ERROR", details = e$message)
  })
  
  
  write_log("Completed post-filtering exploratory analysis", level = "INFO")
}


# Performing RUVSeq normalisation
ruvseq_norm <- function(raw_expression, rawCounts.filt, clinical_data, refgenes, minref, k_factor) {
  write_log("Starting RUVSeq normalisation", level = "INFO")
  
  tryCatch({
    write_log("Identifying reference genes based on selected reference type", level = "INFO")
    if (refgenes == "hkNpos") {
      ref_genes <- raw_expression[which(raw_expression$Class %in% c("Housekeeping", "Positive")), c(1, 3:ncol(raw_expression))]
      raw_mat <- raw_expression[which(raw_expression$Class %in% c("Endogenous", "Housekeeping", "Positive")), c(1, 3:ncol(raw_expression))]
    } else if (refgenes == "hk") {
      ref_genes <- raw_expression[which(raw_expression$Class == "Housekeeping"), c(1, 3:ncol(raw_expression))]
      raw_mat <- raw_expression[which(raw_expression$Class %in% c("Endogenous", "Housekeeping")), c(1, 3:ncol(raw_expression))]
    } else if (refgenes == "posCtrl") {
      ref_genes <- raw_expression[which(raw_expression$Class == "Positive"), c(1, 3:ncol(raw_expression))]
      raw_mat <- raw_expression[which(raw_expression$Class %in% c("Endogenous", "Positive")), c(1, 3:ncol(raw_expression))]
    } else if (refgenes == "endNhkNpos") {
      ref_genes <- raw_expression[which(raw_expression$Class %in% c("Endogenous", "Housekeeping", "Positive")), c(1, 3:ncol(raw_expression))]
      raw_mat <- raw_expression[which(raw_expression$Class %in% c("Endogenous", "Housekeeping", "Positive")), c(1, 3:ncol(raw_expression))]
    }
    
    write_log("Preparing reference genes and raw matrix", level = "INFO")
    row.names(ref_genes) <- ref_genes$Gene
    ref_genes$Gene <- NULL
    ref_genes <- t(ref_genes)
    
    write_log("Performing geNorm analysis", level = "INFO")
    genorm_result <- geNorm(ref_genes, ctVal = FALSE)
    genorm_result <- genorm_result[order(genorm_result$Avg.M), ]
    
    min_ref_genes <- unlist(strsplit(as.character(genorm_result$Genes[1:(minref - 1)]), "[-]"))
    genorm_result <- genorm_result[!is.na(genorm_result$Avg.M), ]
    detected_ref_genes <- unlist(strsplit(as.character(genorm_result[genorm_result[, 2] < 1.0, ]$Genes), "[-]"))
    
    write_log("Determining final reference genes", level = "INFO")
    reference_genes <- if (length(detected_ref_genes) >= length(min_ref_genes)) {
      detected_ref_genes
    } else {
      min_ref_genes
    }
    
    rm(ref_genes, detected_ref_genes, genorm_result, min_ref_genes)
  }, error = function(e) {
    write_log("Error in geNorm analysis and reference gene selection", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Performing RUVSeq normalisation", level = "INFO")
    row.names(raw_mat) <- raw_mat$Gene
    raw_mat$Gene <- NULL
    set <- newSeqExpressionSet(as.matrix(raw_mat), phenoData = clinical_data)
    set <- RUVg(set, reference_genes, k = k_factor, round = FALSE)
    norm_data <- data.frame(round(set@assayData$normalizedCounts, 2))
    
    write_log("Filtering normalised data for endogenous genes", level = "INFO")
    norm_data <- norm_data[which(row.names(norm_data) %in% row.names(rawCounts.filt)), ]
    norm_data <- log2(norm_data + 1)
    
    rm(set)
    write_log("RUVSeq normalisation completed successfully", level = "INFO")
    return(norm_data)
  }, error = function(e) {
    write_log("Error in RUVSeq normalisation", level = "ERROR", details = e$message)
    return(NULL)
  })
}


# Getting a mean of all median score to identify which normalisation worked best on the normalised input data
check_norm <- function(normalised_data, norm_method) {
  write_log(paste0("Starting MRLE calculation for the ", norm_method, " normalisation"), level = "INFO")
  
  tryCatch({
    write_log("Converting normalised data to matrix", level = "INFO")
    normalised_data <- data.matrix(normalised_data)
  }, error = function(e) {
    write_log("Error converting normalised data to matrix", level = "ERROR", details = e$message)
    return(NA)
  })
  
  tryCatch({
    write_log("Calculating the median of each gene", level = "INFO")
    features_medians <- rowMedians(normalised_data)
  }, error = function(e) {
    write_log("Error calculating the median of each gene", level = "ERROR", details = e$message)
    return(NA)
  })
  
  tryCatch({
    write_log("Calculating RLE for each gene", level = "INFO")
    med_devs <- normalised_data - features_medians
  }, error = function(e) {
    write_log("Error calculating RLE for each gene", level = "ERROR", details = e$message)
    return(NA)
  })
  
  tryCatch({
    write_log("Obtaining the median of each sample", level = "INFO")
    sample_medians <- colMedians(med_devs)
  }, error = function(e) {
    write_log("Error obtaining the median of each sample", level = "ERROR", details = e$message)
    return(NA)
  })
  
  tryCatch({
    write_log("Calculating the MRLE (mean of absolute sample medians)", level = "INFO")
    rle_metric <- round(mean(abs(sample_medians)), digits = 3)
    write_log(paste0("MRLE for ", norm_method, " normalisation: ", rle_metric), level = "INFO")
    return(rle_metric)
  }, error = function(e) {
    write_log("Error calculating the MRLE", level = "ERROR", details = e$message)
    return(NA)
  })
  
  # Cleanup
  rm(features_medians, med_devs, sample_medians)
}


# Performing RLE, PCA, and hierarchical clustering analyses, as well as creating Density plots
norm_plots <- function(normalised_data, clinical_data, method, score) {
  write_log("Starting normalisation plots", level = "INFO")

  #### RLE Plot
  tryCatch({
    write_log("Generating RLE plot", level = "INFO")
    
    exprs_mat <- as.matrix(normalised_data)
    features_meds <- rowMedians(exprs_mat)
    med_devs <- exprs_mat - features_meds
    
    # Prepare data for ggplot
    df_to_plot <- data.frame(Group = rep(seq_len(ncol(med_devs)), each = nrow(med_devs)),
                             Score = as.numeric(med_devs),
                             Sample = rep(colnames(normalised_data), each = nrow(med_devs)),
                             Gene = rep(rownames(normalised_data), times = ncol(med_devs)),
                             Condition = rep(clinical_data$Condition, each = nrow(med_devs)))
    
    df_to_plot <- df_to_plot %>% arrange(Condition, Sample)
    df_to_plot$Sample <- factor(df_to_plot$Sample, levels = unique(df_to_plot$Sample))
    
    # Generate RLE plot
    plot_out <- ggplot(df_to_plot, aes(x = Sample, group = Group, y = Score, color = Condition, fill = Condition)) +
                       geom_boxplot() +
                       stat_summary(geom = "crossbar", width = 0.65, fatten = 0, color = "white", 
                                    fun.data = function(x){ c(y = median(x), ymin = median(x), ymax = median(x)) }) +
                       theme_classic() +
                       scale_color_manual(values=colors) +
                       scale_fill_manual(values=colors) +
                       theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                             text = element_text(size = 12),
                             axis.text = element_text(size=12),
                             axis.title = element_text(size=12),
                             panel.border = element_blank(),
                             panel.grid.minor = element_blank(),
                             panel.grid.major.x = element_blank(),
                             panel.grid.major.y = element_line(linewidth=.1, color="gray78"),
                             axis.text.x = element_text(angle = 90, hjust = 1),
                             legend.position = "right") +
                       labs(title = paste0("Relative Log Expression plot with ", str_to_title(norm), " Normalisation (with MRLE score: ",score,")"),
                            y = "Relative log expression",
                            x = "Samples")
    
    ggsave(plot_out, file = paste(results, "/Fig4.A.", method, ".normalisation.rle.png", sep = ""), device = "png", width = 14, height = 8, dpi = 600)
    
    # Outliers and interactive RLE plot
    outliers <- df_to_plot %>% 
                group_by(Sample, Condition) %>%
                summarize(lower = quantile(Score, probs = 0.25) - 1.5 * IQR(Score),
                          upper = quantile(Score, probs = 0.75) + 1.5 * IQR(Score)) %>%
                left_join(df_to_plot, by = c("Sample", "Condition")) %>%
                filter(Score < lower | Score > upper) %>%
                select(-lower, -upper)
    
    
    plot_out2 <- ggplot(df_to_plot, aes(x = Sample, group = Group, y = Score, fill = Condition)) +
                        geom_boxplot(aes(color = Condition)) +
                        geom_point(data = outliers, aes(x = Sample, y = Score, color = Condition)) +
                        stat_summary(geom = "crossbar", width = 0.65, fatten = 0, color = "white",
                                     fun.data = function(x){ c(y = median(x), ymin = median(x), ymax = median(x)) }) +
                        theme_classic() +
                        scale_color_manual(values=colors) +
                        scale_fill_manual(values=colors) +
                        theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                              text = element_text(size = 12),
                              axis.text = element_text(size=10),
                              axis.title = element_text(size=12),
                              panel.border = element_blank(),
                              panel.grid.minor = element_blank(),
                              panel.grid.major.x = element_blank(),
                              panel.grid.major.y = element_line(linewidth=.1, color="gray78"),
                              axis.text.x = element_text(angle = 90, hjust = 1),
                              legend.position = "right") +
                        labs(title = paste0("Relative Log Expression plot with ", str_to_title(norm), " Normalisation (with MRLE score: ",score,")"),
                             y = "Relative log expression",
                             x = "Samples")
    
    plot_out_plt <- ggplotly(plot_out2, tooltip = c("Sample", "Score", "Condition"), preserve = "text") %>%
                             highlight(on = "plotly_hover", selectize = T)  %>%
                             layout(showlegend = F, boxpoints = T, highlight = FALSE, font = list(family = "Arial", size = 12))
    
    htmlwidgets::saveWidget(widget = plot_out_plt, file = paste(html_results, "/", method, ".normalisation.rle.html", sep = ""), selfcontained = TRUE)
    
    rm(exprs_mat, features_meds, med_devs, df_to_plot, plot_out, plot_out2, outliers, plot_out_plt)
  }, error = function(e) {
    write_log("Error in RLE plot generation", level = "ERROR", details = e$message)
  })
  
  #### PCA Plot
  tryCatch({
    write_log("Generating PCA plot", level = "INFO")
    
    pca_result <- prcomp(t(normalised_data), scale. = TRUE)
    data_pca <- as.data.frame(pca_result$x[, 1:2])
    variance_explained <- round((pca_result$sdev^2 / sum(pca_result$sdev^2)) * 100, 1)
    data_pca <- merge(data_pca, clinical_data, by.x = 0, by.y = 0)
    row.names(data_pca) <- data_pca$Row.names
    data_pca$Row.names <- NULL
    
    p1 <- ggplot(data_pca, aes(PC1, PC2, color = Condition, label = Sample)) +
                 geom_point(aes(text = paste("Sample:", Sample, "<br>CartridgeID:", CartridgeID,
                                             "<br>Condition:", Condition)), size = 2, alpha = 0.9) +
                 theme_bw() +
                 scale_color_manual(values = colors) +
                 theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                       text = element_text(size = 12),
                       panel.border = element_blank(),
                       panel.grid.minor = element_blank(),
                       axis.line = element_line(colour = "black"),
                       legend.position = "right",
                       legend.justification = "center") +
                 labs(title = paste0("PCA plot after ", str_to_title(method), " Normalisation"),
                      x = paste0("PC1: ", variance_explained[1], "% variance"),
                      y = paste0("PC2: ", variance_explained[2], "% variance"))
    
    ggsave(p1, file = paste(results, "/Fig4.B.", method, ".normalisation.pca.png", sep = ""), device = "png", bg = "white", width = 10, height = 8, dpi = 600)
    
    p1_plt <- ggplotly(p1, tooltip = "text") %>%
              highlight(on = "plotly_hover", selectize = TRUE) %>%
              layout(font = list(family = "Arial", size = 12))
    
    htmlwidgets::saveWidget(widget = p1_plt, file = paste(html_results, "/", method, ".normalisation.pca.html", sep = ""), selfcontained = TRUE)
    
    rm(pca_result, data_pca, variance_explained, p1, p1_plt)
  }, error = function(e) {
    write_log("Error in PCA plot generation", level = "ERROR", details = e$message)
  })
  
  
  #### Hierarchical Clustering Plot
  tryCatch({
    write_log("Generating hierarchical clustering plot", level = "INFO")
    
    # Compute distances and hierarchical clustering
    sampleDists <- dist(scale(t(normalised_data)), method = "euclidean")
    clusters <- hclust(sampleDists, method = "ward.D")
    
    # Convert hclust object into dendrogram for ggplot
    dendro_df <- dendro_data(as.dendrogram(clusters))
    dendro_df$labels <- merge(dendro_df$labels, clinical_data, by.x = "label", by.y = "Sample", sort = FALSE)
    
    avg_dist <- ceiling(mean(dist(matrix(dendro_df$segments$y)))) + nchar(dendro_df$labels$label[1])

    # Create the dendrogram plot
    dendro_plot <- ggplot(dendro_df$segments) +
                          geom_segment(aes(x=x, y=y, xend=xend, yend=yend)) +
                          geom_text(data = dendro_df$labels, aes(x, y, label = label),
                                    hjust = 1.1, angle = 90, size = 3) +
                          geom_point(data = dendro_df$labels, aes(x = x, y = y, color = Condition, text = paste("Sample:", label, "<br>CartridgeID:", CartridgeID, "<br>Condition:", Condition, "<br>Branch:", x)), size = 3) +
                          scale_color_manual(values=c(colors[1], colors[2])) +
                          theme_dendro() +
                          theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                                text = element_text(size = 12),
                                axis.text=element_text(size=12),
                                axis.title=element_text(size=12),
                                legend.position = "bottom") +
                          labs(title = paste0("Hierarchical Clustering Plot after ", 
                                              str_to_title(norm), " Normalisation")) +
                          ylim(floor(min(dendro_df$segments$y))-avg_dist, ceiling(max(dendro_df$segments$y)))
    
    # Save the static plot as PNG
    ggsave(dendro_plot, file = paste(results, "/Fig4.C.", method, ".normalisation.hclust.png", sep = ""), device = "png", bg = "white", width = 14, height = 8, dpi = 600)
    
    dendro_plot$layers[[2]] <- NULL
    
    # Make the dendrogram interactive
    dendro_plt <- ggplotly(dendro_plot, tooltip = "text") %>% 
                            highlight(on = "plotly_hover", selectize = T)  %>%
                            layout(font = list(family = "Arial", size = 12), showlegend = TRUE)
    
    # Save the interactive plot as an HTML widget
    htmlwidgets::saveWidget(widget = dendro_plt, file = paste(html_results, "/", method, ".normalisation.hclust.html", sep = ""), selfcontained = TRUE)
    
    write_log("Completed hierarchical clustering plot", level = "INFO")
    rm(sampleDists, clusters, dendro_df, dendro_plot, dendro_plt)
    
  }, error = function(e) {
    write_log("Error in hierarchical clustering plot generation", level = "ERROR", details = e$message)
  })
  
  
  ### Density Plot
  tryCatch({
    write_log("Generating density plot", level = "INFO")
    
    density_df <- reshape2::melt(as.data.frame(normalised_data))
    density_df <- merge(density_df, clinical_data, by.x = "variable", by.y = "Sample", sort = FALSE)

    dplot <- ggplot(density_df, aes(x = value, color = variable)) +
                    geom_density(aes(text = paste("Sample:", variable, "<br>CartridgeID:", CartridgeID,"<br>Condition:", Condition)), alpha = 0.5) +
                    labs(title = paste0("Density Plot after ", str_to_title(norm), " Normalisation"), 
                         x = "log2(Normalised Values)", 
                         y = "Density") +
                    theme_minimal() +
                    theme(plot.title = element_text(colour = "#a6a6a4", size=13),
                          text = element_text(size = 12),
                          axis.text=element_text(size=12),
                          axis.title=element_text(size=12),
                          axis.line = element_line(colour = "black"),
                          legend.position = "none")

    ggsave(dplot, file = paste(results, "/Fig4.D.", method, ".normalisation.density.png", sep = ""), device = "png", bg = "white", width = 10, height = 8, dpi = 600)

    dplot_plt <- ggplotly(dplot, tooltip = "text") %>% 
                          highlight(on = "plotly_hover", selectize = T)  %>%
                          layout(font = list(family = "Arial", size = 12))
    
    htmlwidgets::saveWidget(widget = dplot_plt, file = paste(html_results, "/", method, ".normalisation.density.html", sep = ""), selfcontained = TRUE)

    rm(density_df, dplot, dplot_plt)
  }, error = function(e) {
    write_log("Error in density plot generation", level = "ERROR", details = e$message)
  })
  
  write_log("Completed normalisation plots", level = "INFO")
}


# Performing de analysis using the limma package
limma_de_analysis <- function(normalised_data, clinical_data, norm) {
  write_log("Starting differential expression analysis using limma", level = "INFO")
  
  tryCatch({
    write_log("Extracting Sample and Condition columns from clinical data", level = "INFO")
    clinical_design <- clinical_data[, 1:2]
  }, error = function(e) {
    write_log("Error extracting Sample and Condition columns from clinical data", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Binarizing the Condition column", level = "INFO")
    clinical_design <- clinical_design %>%
      mutate(Condition = ifelse(Condition == groups_capital[1], 0,
                                ifelse(Condition == groups_capital[2], 1, Condition)))
  }, error = function(e) {
    write_log("Error binarizing the Condition column", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Generating the design matrix for the experiment", level = "INFO")
    design <- model.matrix(~ clinical_design$Condition)
  }, error = function(e) {
    write_log("Error generating the design matrix", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Fitting a linear model using limma", level = "INFO")
    fit <- lmFit(normalised_data, design = design)
  }, error = function(e) {
    write_log("Error fitting the linear model", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Applying empirical Bayes statistics to fit", level = "INFO")
    fit <- eBayes(fit, robust = TRUE)
  }, error = function(e) {
    write_log("Error applying empirical Bayes statistics", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Extracting DE results and sorting by adjusted p-value", level = "INFO")
    de_results <- data.frame(topTable(fit, n = Inf, adjust.method = "BH"))[, c(1, 4, 5)]
    colnames(de_results) <- c("log2FC", "pvalue", "padj")
  }, error = function(e) {
    write_log("Error extracting DE results", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Merging normalised data with DE statistics", level = "INFO")
    de_results <- data.frame(merge(round(normalised_data, 3), de_results, by = "row.names"))
    row.names(de_results) <- de_results$Row.names
    de_results$Row.names <- NULL
    de_results <- de_results[order(de_results$padj), ]
  }, error = function(e) {
    write_log("Error merging normalised data with DE statistics", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Writing DE results to file", level = "INFO")
    write.table(data.frame("Genes" = rownames(de_results), de_results), 
                file = paste0(results, "/", norm, "-normalised.matrix.de.tsv"), 
                sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
  }, error = function(e) {
    write_log("Error writing DE results to file", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  write_log("Successfully completed differential expression analysis", level = "INFO")
  return(de_results)
}


# Creating a Volcano plot to visualizations Differential Expression results
volcano_plot <- function(de_data, clinical_data, method, lfcthreshold, padjusted) {
  write_log("Starting volcano plot generation", level = "INFO")
  
  tryCatch({
    write_log("Copying the DE data to a new dataframe", level = "INFO")
    res_overall <- de_data
  }, error = function(e) {
    write_log("Error copying the DE data to a new dataframe", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Adding the significant column to the data", level = "INFO")
    res_overall$significant <- -log10(res_overall$padj)
    res_overall[is.infinite(res_overall$significant), "significant"] <- 350
  }, error = function(e) {
    write_log("Error adding the significant column", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Categorizing expression levels in the data", level = "INFO")
    res_overall <- res_overall %>% 
      mutate(Expression = case_when(
        log2FC >= lfcthreshold & padj <= padjusted ~ "upRegulated",
        log2FC <= -lfcthreshold & padj <= padjusted ~ "downRegulated",
        TRUE ~ "noDifference"
      ))
  }, error = function(e) {
    write_log("Error categorizing expression levels", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Identifying significant genes based on thresholds", level = "INFO")
    sig_genes <- subset(res_overall[which(abs(res_overall$log2FC) >= lfcthreshold & res_overall$padj <= padjusted), ])
  }, error = function(e) {
    write_log("Error identifying significant genes", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  # Helper functions for axis limits
  my.max <- function(x) ifelse(!all(is.na(x)), max(x, na.rm = TRUE), NA)
  my.min <- function(x) ifelse(!all(is.na(x)), min(x, na.rm = TRUE), NA)
  min.pos <- floor(my.min(res_overall$log2FC))
  max.pos <- ceiling(my.max(res_overall$log2FC))
  
  tryCatch({
    write_log("Generating the volcano plot", level = "INFO")
    suppressWarnings(
    volc_plot <- ggplot(res_overall, aes(x = log2FC, y = significant)) + 
      geom_point(aes(color = Expression, text = paste(
        "GeneName:", row.names(res_overall),
        "<br>log2FC:", round(log2FC, 2),
        "<br>FDR:", round(padj, 5),
        "<br>Significance:", Expression
      )), size = 2, alpha = .8) +
      ggrepel::geom_text_repel(data = sig_genes, aes(label = rownames(sig_genes)), color = "black", size = 2.5, segment.color = "grey") +
      geom_hline(yintercept = -log10(padjusted), linetype = "longdash", colour = "#835858", linewidth = 0.2) +
      geom_vline(xintercept = c(-lfcthreshold, lfcthreshold), linetype = "longdash", colour = "#835858", linewidth = 0.2) +
      geom_vline(xintercept = 0, colour = "black", linetype = 0.09) +
      theme_bw() +
      scale_color_manual(values = c("#A00000", "#B7C4CF", "#2a850e")) +
      scale_x_continuous(breaks = seq.int(min.pos, max.pos, by = .1)) +
      expand_limits(y = c(0, my.max(-log10(res_overall$padj)) + 0.15)) +
      theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
            axis.line = element_line(),
            panel.grid.minor = element_blank(),
            panel.border = element_blank(),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 12),
            text = element_text(size = 12),
            panel.grid.major.x = element_line(linewidth = .27, color = "gray65", linetype = "dotted"),
            panel.grid.major.y = element_line(linewidth = .27, color = "gray65", linetype = "dotted")) +
      labs(title = paste0("Volcano plot after ", str_to_title(method), " Normalisation"),
           x = "log2(fold change)",
           y = "-log10(adjusted p-value)"))
    
    ggsave(volc_plot, file = paste(results, "/Fig5.A.", norm, ".deanalysis.volcano.png", sep = ""), device = "png", width = 14, height = 8, units = "in", dpi = 600)
  }, error = function(e) {
    write_log("Error generating the volcano plot", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  tryCatch({
    write_log("Converting volcano plot to interactive Plotly object", level = "INFO")
    
    # Interactive volcano plot without geom_text_repel
    suppressWarnings(
      volc_plot_interactive <- ggplot(res_overall, aes(x = log2FC, y = significant)) +
        geom_point(aes(color = Expression, text = paste(
          "GeneName:", row.names(res_overall),
          "<br>log2FC:", round(log2FC, 2),
          "<br>FDR:", round(padj, 5),
          "<br>Significance:", Expression
        )), size = 2, alpha = .8) +
        geom_hline(yintercept = -log10(padjusted), linetype = "longdash", colour = "#835858", linewidth = 0.2) +
        geom_vline(xintercept = c(-lfcthreshold, lfcthreshold), linetype = "longdash", colour = "#835858", linewidth = 0.2) +
        geom_vline(xintercept = 0, colour = "black", linetype = 0.09) +
        theme_bw() +
        scale_color_manual(values = c("#A00000", "#B7C4CF", "#2a850e")) +
        scale_x_continuous(breaks = seq.int(min.pos, max.pos, by = .1)) +
        expand_limits(y = c(0, my.max(-log10(res_overall$padj)) + 0.15)) +
        theme(plot.title = element_text(colour = "#a6a6a4", size = 13),
              axis.line = element_line(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              axis.title = element_text(size = 12),
              axis.text = element_text(size = 12),
              text = element_text(size = 12),
              panel.grid.major.x = element_line(linewidth = .27, color = "gray65", linetype = "dotted"),
              panel.grid.major.y = element_line(linewidth = .27, color = "gray65", linetype = "dotted")) +
        labs(title = paste0("Volcano plot after ", str_to_title(method), " Normalisation"),
             x = "log2(fold change)",
             y = "-log10(adjusted p-value)"))
    
    volc_plot <- ggplotly(volc_plot_interactive, tooltip = "text", preserve = "text") %>%
      layout(
        xaxis = list(showgrid = FALSE),
        yaxis = list(showgrid = FALSE),
        font = list(family = "Arial", size = 12)
      ) %>%
      highlight(on = "plotly_hover", selectize = TRUE)
    
    htmlwidgets::saveWidget(widget = volc_plot, file = paste(html_results, "/", norm, ".deanalysis.volcano.html", sep = ""), selfcontained = TRUE)
  }, error = function(e) {
    write_log("Error converting volcano plot to interactive Plotly object", level = "ERROR", details = e$message)
    return(NULL)
  })
  
  write_log("Successfully completed volcano plot generation", level = "INFO")
  rm(res_overall, sig_genes, my.max, my.min, min.pos, max.pos)
}
