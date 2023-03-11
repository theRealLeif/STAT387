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

#--------------------#
#-----KNN Model------#
#--------------------#

## Model Construction
#--------------------#
#-----K-fold CV------#
#--------------------#
set.seed(1234)
# Define the training control object for 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train the KNN model using 10-fold cross-validation
# tuneLength argument to specify the range of values of K to be considered for tuning
set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = train_control,
                   tuneGrid = data.frame(k = 1:10))

# Save the model into .Rdata for future import 
save(knn_model, file = "dataset\\knn.model_kfoldCV.Rdata")


#--------------------------#
#-----K-fold CV (Mod)------#
#--------------------------#
set.seed(1234)
train_control <- trainControl(method = "cv", number = 10)

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = train_control, 
                   tuneGrid = data.frame(k = 1:30))

# Save the model into .Rdata for future import 
save(knn_model, file = "dataset\\knn.model_kfoldCV_mod.Rdata")


#--------------------#
#----Hold-out CV-----#
#--------------------#
set.seed(1234)
train_control <- trainControl(method = "none",)

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn",
                   tuneGrid = data.frame(k = 1:10))

save(knn_model, file = "dataset\\knn.model_holdoutCV.Rdata")


#--------------------------#
#----Hold-out CV (Mod)-----#
#--------------------------#
set.seed(1234)
train_control <- trainControl(method = "none",)

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn",
                   tuneGrid = expand.grid(k=1:30))

save(knn_model, file = "dataset\\knn.model_holdoutCV_mod.Rdata")


#--------------------#
#-------LOOCV--------#
#--------------------#
set.seed(1234)
train_control <- trainControl(method = "LOOCV")

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = train_control,
                   tuneGrid = data.frame(k = 1:10))

save(knn_model, file = "dataset\\knn.model_looCV.Rdata")


#--------------------------#
#-------LOOCV (Mod)--------#
#--------------------------#
set.seed(1234)
train_control <- trainControl(method = "LOOCV")

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = train_control,
                   tuneLength = 10,
                   tuneGrid = expand.grid(k = 1:20))

save(knn_model, file = "dataset\\knn.model_looCV_mod.Rdata")


#--------------------#
#----Repeated CV-----#
#--------------------#
set.seed(1234)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "knn", 
                   trControl = train_control)

save(knn_model, file = "dataset\\knn.model_repeatedCV.Rdata")


#--------------------------#
#----Repeated CV (Mod)-----#
#--------------------------#
set.seed(1234)
train_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

kknn.grid <- expand.grid(kmax = c(3, 5, 7 ,9, 11), distance = c(1, 2, 3),
                         kernel = c("rectangular", "gaussian", "cos"))

set.seed(1234)
knn_model <- train(good ~ ., 
                   data = train, 
                   method = "kknn",
                   trControl = train_control, 
                   tuneGrid = kknn.grid,
                   preProcess = c("center", "scale"))

save(knn_model, file = "dataset\\knn.model_repeatedCV_mod.Rdata")

###############
## K-fold CV ##
###############
# Data Import
load("dataset\\train.Rdata")
load("dataset\\test.Rdata")

# Model Import
load("dataset\\model\\knn.model_kfoldCV.Rdata")

# Make predictions on the test data using the trained model and calculate the test error rate
knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$good)


# Convert predictions to a numeric vector
knn.predictions <- as.numeric(knn.predictions)

# Calculate the AUC using the performance() and auc() functions:
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

# Performance plot for TP and FP
roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with 10-fold CV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.kfoldCV.ROC.plot<- recordPlot()

knn_df <- data.frame(k = knn_model$results$k, 
                     Accuracy = knn_model$results$Accuracy,
                     Kappa = knn_model$results$Kappa)

# Accuracy and Kappa value plot
accu.kappa.plot <- function(model_df) {
  p <- ggplot(data = model_df) +
    geom_point(aes(x = k, y = Accuracy, color = "Accuracy")) +
    geom_point(aes(x = k, y = Kappa, color = "Kappa")) +
    geom_line(aes(x = k, y = Accuracy, linetype = "Accuracy", color = "Accuracy")) +
    geom_line(aes(x = k, y = Kappa, linetype = "Kappa", color = "Kappa")) +
    scale_color_manual(values = c("#98c379", "#e06c75"),
                       guide = guide_legend(override.aes = list(linetype = c(1, 0)) )) +
    scale_linetype_manual(values=c("solid", "dotted"),
                          guide = guide_legend(override.aes = list(color = c("#98c379", "#e06c75")))) +
    labs(x = "K value", 
         y = "Accuracy / Kappa") +
    ylim(0, 1) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    guides(color = guide_legend(title = "Metric"),
           linetype = guide_legend(title = "Metric"))
  return(p)
}

knn.kfoldCV.plot <- accu.kappa.plot(knn_df) + 
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)), vjust = -1) +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)), vjust = -1) +
  ggtitle("KNN Model Performance (10-Fold CV)")

#############
### Tuned ###
#############
load("dataset\\model\\knn.model_kfoldCV_mod.Rdata")

knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with Tuned 10-fold CV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.kfoldCV_mod.ROC.plot <- recordPlot()

knn_df <- data.frame(k = knn_model$results$k, 
                     Accuracy = knn_model$results$Accuracy,
                     Kappa = knn_model$results$Kappa)

knn.kfoldCV_mod.plot <- accu.kappa.plot(knn_df) +
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)),  hjust = -0.3, angle=90)  +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)),  hjust = -0.3, angle=90) +
  ggtitle("KNN Model Performance (Tuned 10-Fold CV)")

###########################################
## Hold-out CV (Validation Set Approach) ##
###########################################
load("dataset\\model\\knn.model_holdoutCV.Rdata")


knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with Hold-out CV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.holdoutCV.ROC.plot <- recordPlot()

knn_df <- data.frame(k = knn_model$results$k, 
                     Accuracy = knn_model$results$Accuracy,
                     Kappa = knn_model$results$Kappa)

knn.holdoutCV.plot <- accu.kappa.plot(knn_df) +
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)), vjust = -1) +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)), vjust = -1) +
  ggtitle("KNN Model Performance (Hold-out CV)")

#############
### Tuned ###
#############
load("dataset\\model\\knn.model_holdoutCV_mod.Rdata")

knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with Tuned Hold-out CV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.holdoutCV_mod.ROC.plot <- recordPlot()

knn_df <- data.frame(k = knn_model$results$k, 
                     Accuracy = knn_model$results$Accuracy,
                     Kappa = knn_model$results$Kappa)

knn.holdoutCV_mod.plot <- accu.kappa.plot(knn_df) + 
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)), hjust = -0.3, angle=90) +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)), hjust=-0.3, angle=90) +
  ggtitle("KNN Model Performance (Tuned Hold-out CV)")


###########
## LOOCV ##
###########
load("dataset\\model\\knn.model_looCV.Rdata")

knn.predictions <- predict(knn_model, newdata = test)
confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with LOOCV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.looCV.ROC.plot <- recordPlot()

knn_df <- data.frame(k = knn_model$results$k, 
                     Accuracy = knn_model$results$Accuracy,
                     Kappa = knn_model$results$Kappa)

knn.looCV.plot <- accu.kappa.plot(knn_df) + 
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)), vjust = -1) +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)), vjust = -1) +
  ggtitle("KNN Model Performance (LOOCV)")

#############
### Tuned ###
#############
load("dataset\\model\\knn.model_looCV_mod.Rdata")

knn.predictions <- predict(knn_model, newdata = test)
confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "Knn ROC Curves Tuned LOOCV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.looCV_mod.ROC.plot <- recordPlot()

knn_df <- data.frame(k = knn_model$results$k, 
                     Accuracy = knn_model$results$Accuracy,
                     Kappa = knn_model$results$Kappa)

knn.looCV_mod.plot <- accu.kappa.plot(knn_df) + 
  geom_text(aes(x = k, y = Accuracy, label = round(Accuracy, 3)), hjust = -0.3, angle=90) +
  geom_text(aes(x = k, y = Kappa, label = round(Kappa, 3)), hjust = -0.3, angle=90) +
  ggtitle("KNN Model Performance (Tuned LOOCV)")


#################
## Repeated CV ##
#################
load("dataset\\model\\knn.model_repeatedCV.Rdata")

knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with Repeated CV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.repeatedCV.ROC.plot <- recordPlot()

knn_df <- knn_model$results
knn.repeatedCV.plot <- ggplot(data=knn_df, aes(x = kmax, y = Accuracy)) +
  geom_point(aes(color = "Accuracy")) +
  geom_point(aes(color = "Kappa")) +
  geom_line(aes(linetype = "Accuracy", color = "Accuracy")) +
  geom_line(aes(y = Kappa, linetype = "Kappa", color = "Kappa")) +
  geom_text(aes(label = round(Accuracy, 3)), vjust = -1) +
  geom_text(aes(y = Kappa, label = round(Kappa, 3)), vjust = -1) +
  scale_color_manual(values = c("#98c379", "#e06c75"),
                     guide = guide_legend(override.aes = list(linetype = c(1, 0)) )) +
  scale_linetype_manual(values=c("solid", "dotted"),
                        guide = guide_legend(override.aes = list(color = c("#98c379", "#e06c75")))) +
  labs(x = "K value", 
       y = "Accuracy / Kappa",
       title = "KNN Model Performance (Repeated CV)") +
  ylim(0, 1) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  guides(color = guide_legend(title = "Metric"),
         linetype = guide_legend(title = "Metric"))

#############
### Tuned ###
#############
load("dataset\\model\\knn.model_repeatedCV_mod.Rdata")

knn.predictions <- predict(knn_model, newdata = test)

confusionMatrix(knn.predictions, test$good)

knn.predictions <- as.numeric(knn.predictions)
pred_obj <- prediction(knn.predictions, test$good)
auc_val <- performance(pred_obj, "auc")@y.values[[1]]
auc_val

roc_obj <- performance(pred_obj, "tpr", "fpr")
plot(roc_obj, colorize = TRUE, lwd = 2,
     xlab = "False Positive Rate", 
     ylab = "True Positive Rate",
     main = "KNN ROC Curves with Tuned Repeated CV")
abline(a = 0, b = 1)
x_values <- as.numeric(unlist(roc_obj@x.values))
y_values <- as.numeric(unlist(roc_obj@y.values))
polygon(x = x_values, y = y_values, 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
polygon(x = c(0, 1, 1), y = c(0, 0, 1), 
        col = rgb(0.3803922, 0.6862745, 0.9372549, alpha = 0.3),
        border = NA)
text(0.6, 0.4, paste("AUC =", round(auc_val, 4)))
knn.repeatedCV_mod.ROC.plot <- recordPlot()

knn.repeatedCV_mod.plot <- ggplot(knn_model) +
  labs(x = "K value", 
       y = "Accuracy", 
       title = "KNN Model Performance (Tuned Repeated CV)") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) 


#############
## Summary ##
#############
ggarrange(knn.kfoldCV.plot,
          knn.kfoldCV_mod.plot,
          knn.holdoutCV.plot,
          knn.holdoutCV_mod.plot,
          knn.looCV.plot,
          knn.looCV_mod.plot,
          knn.repeatedCV.plot,
          knn.repeatedCV_mod.plot,
          ncol = 2, nrow = 4)


cowplot::plot_grid(knn.kfoldCV.ROC.plot, knn.kfoldCV_mod.ROC.plot,
                   ncol = 2, align = "hv", scale = 0.8)
cowplot::plot_grid(knn.holdoutCV.ROC.plot, knn.holdoutCV_mod.ROC.plot,
                   ncol = 2, align = "hv", scale = 0.8)
cowplot::plot_grid(knn.looCV.ROC.plot, knn.looCV_mod.ROC.plot,
                   ncol = 2, align = "hv", scale = 0.8)
cowplot::plot_grid(knn.repeatedCV.ROC.plot, knn.repeatedCV_mod.ROC.plot,
                   ncol = 2, align = "hv", scale = 0.8)


# | Resampling Method    | Error Rate | Sensitivity | Specificity | AUC       |
# | -------------------- | ---------- | ----------- | ----------- | --------- |
# | K-Fold CV            | 0.1692     | 0.9615      | 0.2775      | 0.6195157 |
# | K-Fold CV (Tuned)    | 0.1953     | 0.9750      | 0.0837      | 0.5293632 |
# | Hold-out CV          | 0.1768     | 0.9563      | 0.2599      | 0.6081037 |
# | Hold-out CV  (Tuned) | 0.1944     | 0.9886      | 0.0308      | 0.5096953 |
# | LOOCV                | 0.1692     | 0.9605      | 0.2819      | 0.6211981 |
# | LOOCV (Tuned)        | 0.1961     | 0.9740      | 0.0837      | 0.5288429 |
# | Repeated CV          | 0.1069     | 0.9542      | 0.6344      | 0.7942878 |
# | Repeated CV (Tuned)  | 0.1204     | 0.9584      | 0.5463      | 0.7523161 |
  
save(knn.kfoldCV.ROC.plot, file = "dataset\\plot\\knn.kfoldCV.ROC.plot.Rdata")
save(knn.kfoldCV_mod.ROC.plot, file = "dataset\\plot\\knn.kfoldCV_mod.ROC.plot.Rdata")
save(knn.holdoutCV.ROC.plot, file = "dataset\\plot\\knn.holdoutCV.ROC.plot.Rdata.Rdata")
save(knn.holdoutCV_mod.ROC.plot, file = "dataset\\plot\\knn.holdoutCV_mod.ROC.plot.Rdata")
save(knn.looCV.ROC.plot, file = "dataset\\plot\\knn.looCV.ROC.plot.Rdata")
save(knn.looCV_mod.ROC.plot, file = "dataset\\plot\\knn.looCV_mod.ROC.plot.Rdata")
save(knn.repeatedCV.ROC.plot, file = "dataset\\plot\\knn.repeatedCV.ROC.plot.Rdata")
save(knn.repeatedCV_mod.ROC.plot, file = "dataset\\plot\\knn.repeatedCV_mod.ROC.plot.Rdata")

save(knn.kfoldCV.plot, file = "dataset\\plot\\knn.kfoldCV.plot.Rdata")
save(knn.kfoldCV_mod.plot, file = "dataset\\plot\\knn.kfoldCV_mod.plot.Rdata")
save(knn.holdoutCV.plot, file = "dataset\\plot\\knn.holdoutCV.plot.Rdata")
save(knn.holdoutCV_mod.plot, file = "dataset\\plot\\knn.holdoutCV_mod.plot.Rdata")
save(knn.looCV.plot, file = "dataset\\plot\\knn.looCV.plot.Rdata")
save(knn.looCV_mod.plot, file = "dataset\\plot\\knn.looCV_mod.plot.Rdata")
save(knn.repeatedCV.plot, file = "dataset\\plot\\knn.repeatedCV.plot.Rdata")
save(knn.repeatedCV_mod.plot, file = "dataset\\plot\\knn.repeatedCV_mod.plot.Rdata")

save(accu.kappa.plot, file = "dataset\\function\\accu.kappa.plot.Rdata")
