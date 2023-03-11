# Load libraries
library(tidyverse)  # Load core packages: 
# ggplot2,   for data visualization.
# dplyr,     for data manipulation.
# tidyr,     for data tidying.
# purrr,     for functional programming.
# tibble,    for tibbles, a modern re-imagining of data frames.
# stringr,   for strings.
# forcats,   for factors.
# lubridate, for date/times.
# readr,     for reading .csv, .tsv, and .fwf files.
# readxl,    for reading .xls, and .xlxs files.
# feather,   for sharing with Python and other languages.
# haven,     for SPSS, SAS and Stata files.
# httr,      for web apis.
# jsonlite   for JSON.
# rvest,     for web scraping.
# xml2,      for XML.
# modelr,    for modelling within a pipeline
# broom,     for turning models into tidy data
# hms,       for times.

library(magrittr)   # Pipeline operator
library(lobstr)     # Visualizing abstract syntax trees, stack trees, and object sizes
library(pander)     # Exporting/converting complex pandoc documents, EX: df to Pandoc table
library(ggforce)    # More plot functions on top of ggplot2
library(ggpubr)     # Automatically add p-values and significance levels  plots. 
# Arrange and annotate multiple plots on the same page. 
# Change graphical parameters such as colors and labels.
library(sf)         # Geo-spatial vector manipulation: points, lines, polygons
library(kableExtra) # Generate 90 % of complex/advanced/self-customized/beautiful tables
library(cowplot)    # Multiple plots arrangement
library(gridExtra)  # Multiple plots arrangement
library(animation)  # Animated figure container
library(latex2exp)  # Latex axis titles in ggplot2
library(ellipse)    # Simultaneous confidence interval region to check C.I. of 2 slope parameters
library(plotly)     # User interactive plots
library(ellipse)    # Simultaneous confidence interval region to check C.I. of 2 regressors
library(olsrr)      # Model selections 
library(leaps)      # Regression subsetting 
library(pls)        # Partial Least squares
library(MASS)       # LDA, QDA, OLS, Ridge Regression, Box-Cox, stepAIC, etc,.
library(e1071)      # Naive Bayesian Classfier,SVM, GKNN, ICA, LCA
library(class)      # KNN, SOM, LVQ
library(ROCR)       # Precision/Recall/Sensitivity/Specificity performance plot 
library(boot)       # LOOCV, Bootstrap,
library(caret)      # Classification/Regression Training, run ?caret::trainControl
library(corrgram)   # for correlation matrix
library(corrplot)   # for graphical display of correlation matrix

set.seed(1234)        # make random results reproducible

current_dir <- getwd()

if (!is.null(current_dir)) {
  setwd(current_dir)
  remove(current_dir)
}


#########################
## K-fold CV (`caret`) ##
#########################
#---------------------------#
#----Model Construction-----#
#---------------------------#
set.seed(1234)
train_control <- trainControl(method = "cv", number = 10)

set.seed(1234)
qda_model <- train(good ~ ., 
                   data = train, 
                   method = "qda", 
                   trControl = train_control)

save(qda_model, file = "dataset\\qda.model_kfoldCV.Rdata")


# Data Import
load("dataset\\wine.data_cleaned.Rdata")
load("dataset\\train.Rdata")
load("dataset\\test.Rdata")

# Function Import
load("dataset\\function\\accu.kappa.plot.Rdata")

# Model import
load("dataset\\model\\qda.model_kfoldCV.Rdata")

qda.predictions <- predict(qda_model, newdata = test)

confusionMatrix(qda.predictions, test$good)

k <- 10
qda.predictions <- as.numeric(qda.predictions)
pred_obj <- prediction(qda.predictions, test$good)
# Compute the RMSE and MAE
RMSE <- caret::RMSE(as.numeric(unlist(pred_obj@predictions)), as.numeric(test$good))
MAE <- caret::MAE(as.numeric(unlist(pred_obj@predictions)), as.numeric(test$good))

# Compute AUC value
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

qda.perf <- performance(pred_obj, "tpr", "fpr")
plot(qda.perf, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "caret::qda ROC Curves")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(qda.perf@x.values))
y_values <- as.numeric(unlist(qda.perf@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
qda.kfoldCV_caret.ROC.plot <- recordPlot()

qda_df <- data.frame(k = 1:k,
                     Accuracy = qda_model$results$Accuracy,
                     Kappa = qda_model$results$Kappa)

pander::pander(data.frame("Accuracy" = qda_model$results$Accuracy, 
                          "RMSE" = RMSE, 
                          "MAE" = MAE,
                          "Kappa" = qda_model$results$Kappa), 
               caption = "caret::qda Performance (10-fold CV)")


########################
## K-fold CV (`MASS`) ##
########################
# Set the number of folds
k <- 10

# Randomly assign each row in the data to a fold
set.seed(1234) # for reproducibility
fold_indices <- sample(rep(1:k, length.out = nrow(wine.data_cleaned)))

# Initialize an empty list to store the folds
folds <- vector("list", k)

# Assign each row to a fold
for (i in 1:k) {
  folds[[i]] <- which(fold_indices == i)
}

#To store the error rate of each fold
error_rate <- numeric(k)
rmse <- numeric(k)
mae <- numeric(k)
confusion_matrices <- vector("list", k)
kappa <- numeric(k)


# Loop through each fold
for (i in 1:10) {
  # Extract the i-th fold as the testing set
  test_indices <- unlist(folds[[i]])
  
  test <- wine.data_cleaned[test_indices, ]
  train <- wine.data_cleaned[-test_indices, ]
  
  # Fit the model on the training set
  qda_model <- qda(good ~ ., data = train, family = binomial)
  
  # Make predictions on the testing set and calculate the error rate
  qda.pred <- predict(qda_model, newdata = test, type = "response")
  predicted_classes <- ifelse(qda.pred$posterior[,2] > 0.7, 1, 0)
  
  # Compute RMSE
  rmse[i] <- sqrt(mean((predicted_classes - as.numeric(test$good)) ^ 2))
  
  # Compute MAE
  mae[i] <- mean(abs(predicted_classes - as.numeric(test$good)))
  
  # Compute OER
  error_rate[i] <- mean((predicted_classes > 0.7) != as.numeric(test$good))
  
  # Compute confusion matrix
  test$good <- as.factor(test$good)
  predicted_classes <- factor(predicted_classes, levels = c(0, 1))
  confusion_matrices[[i]] <- caret::confusionMatrix(predicted_classes, test$good)
  
  # Compute Kappa value
  kappa[i] <- confusion_matrices[[i]]$overall[[2]]
  
  # Print the error rates for each fold
  cat(paste0("Fold ", i, ": ", "OER:", error_rate[i], " RMSE:", rmse[i], " MAE:", mae[i], "\n"))
}

best_confmat_index <- which.min(error_rate)
best_confmat_index
best_confmat_indexi <- which.min(rmse)
best_confmat_index
best_confmat_index <- which.min(mae)
best_confmat_index
confusion_matrices[best_confmat_index]

#AUC and Performance Plot
predicted_classes <- as.numeric(predicted_classes)
pred_obj <- prediction(predicted_classes, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
qda.perf <- performance(pred_obj,"tpr","fpr")
auc_val

plot(qda.perf, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "MASS::qda ROC Curves")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(qda.perf@x.values))
y_values <- as.numeric(unlist(qda.perf@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
qda.kfoldCV_MASS.ROC.plot <- recordPlot()

qda_df <- data.frame(k = 1:k,
                     Accuracy = 1-error_rate, 
                     Kappa = kappa)

qda.kfoldCV_MASS.plot <- accu.kappa.plot(qda_df) + 
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)), vjust = -1) +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)), vjust = -1) +
  ggtitle("MASS::qda Model Performance (10-Fold CV)")


#############
## Summary ##
#############
cowplot::plot_grid(qda.kfoldCV_caret.ROC.plot,
                   qda.kfoldCV_MASS.ROC.plot,
                   ncol = 2, align = "hv", scale = 0.8)


# | Resampling Method | Error Rate | Sensitivity | Specificity | AUC       |
# | ----------------- | ---------- | ----------- | ----------- | --------- |
# | QDA (`caret`)     | 0.2399     | 0.7743      | 0.7013      | 0.7377967 |
# | QDA (`MASS`)      | 0.1692     | 0.8934      | 0.5714      | 0.7324227 |
#   

save(qda.kfoldCV_MASS.plot, file = "dataset\\plot\\qda.kfoldCV_MASS.plot.Rdata")
save(qda.kfoldCV_caret.ROC.plot, file = "dataset\\plot\\qda.kfoldCV_caret.ROC.plot.Rdata")
save(qda.kfoldCV_MASS.ROC.plot, file = "dataset\\plot\\qda.kfoldCV_MASS.ROC.plot.Rdata")


