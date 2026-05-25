############################################################################################################### 
# STAT606 Practical 4: Ensemble Methods: Random forests, Gradient Boosted Machines (GBM), Stacked Ensembles   #         
###############################################################################################################

# This practical covers 3 ensemble methods all fitted using H2O:

# 1. Random forest
# 2. Gradient boosted machine (GBM)
# 3. Stacked ensemble

#The demonstration uses the same dataset as practical 3 (credit_card_approval_clean). NOTE: Tree-based ensemble methods such as random forests and GBM generally do not require
#preprocessing (normalization and dummy variable encoding). However, in many workflows, preprocessing is still applied before fitting these models so that the same dataset can
#be used consistently across multiple algorithms. Thus, preprocessing is included in this practical as per the prac for SVMs and NNs. 

## NB: Only a few of the hyperparameters are considered here for tuning. This list is not exhaustive. See the H2O documentation for each algorithm for their respective
# hyperparameters that can be tuned. 

# Currently, H2O does not have support for running the XGBoost platform in Windows, so we will not consider it at this stage. It is coming soon though. We will consider
# GBM instead.

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- # 

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC
library(h2o)

# additional package for this prac:
library(recipes) # a package from tidymodels to preprocess data

# to graph the combined results

library(ggplot2)

# this is to turn scientific notation off so the output is easier to interpret:
options(scipen = 999) # turn back on by changing to 0

# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters -------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# User-specified parameters
seed = 606     # Seed for reproducibility (students can change)
train_frac <- 0.7  # Proportion of data in training set
metric <- "auc" # "auc", "aucpr" (Area Under Precision–Recall Curve), "logloss", "Accuracy", "Specificity", "Precision", "Recall", "F1" 
folds <- 5 # for 5-fold CV, or change to 10


# ----------------------------------------------------------------------------- #
# 2. Load, Inspect and Format Data ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# For this demonstration, we will make use of the same credit card approval data as practical 1. However, we will use the data that has already been structured and cleaned (as per lines 43 to 95 in Prac 1) and saved as an RDATA file (called credit_card_approval_clean.RDATA). This is the file we will load for this prac:

# load("credit_card_approval_clean.RDATA")
load("C:/Users/mambaza/Desktop/UKZN/PGDM - Data Science/Semester 1/STAT606 - Applied Binary Classification and Matching/R Practicals/Practical 4 - Ensemble Methods/credit_card_approval_clean.RDATA")

# Recall what the target looks like: 

summary(credit_card_approval_clean$Approved)

# ----------------------------------------------------------------------------- #
# 3. Specify df and target --------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# Instead of writing code that only works for one dataset, we define named objects here that act as settings/place-holders. Then we reuse those throughout the code (rather than hard-coding):

df <- credit_card_approval_clean

target <- "Approved" 

summary(df[[target]])

# NOTE: We can use the following to convert the class labels to 0 and 1 only if they are Yes/No (or any version of yes/no):

# Check if target is already coded as 0/1
if (!all(sort(unique(na.omit(df[[target]]))) %in% c(0, 1))) {
  
  # Convert Yes/No (case insensitive - changes all to lower case) to 1/0
  if (all(tolower(na.omit(unique(df[[target]]))) %in% c("yes", "no"))) {
    
    df[[target]] <- factor(
      ifelse(tolower(df[[target]]) == "yes", 1, 0),
      levels = c(0, 1)
    )
    
  } else {
    stop("Target variable is neither 0/1 nor Yes/No.")
  }
}

# check to make sure the change has been implemented (there must be two classes, 0 and 1)
summary(df[[target]])


# ----------------------------------------------------------------------------- #
# 4. Train/Test Split --------------------------------------------------------- 
# ----------------------------------------------------------------------------- #

set.seed(seed)  

# stratified sampling is used to maintain the proportion of class labels in your training and test sets:
split=sample.split(df[[target]],SplitRatio = train_frac) # train_frac was specified under the setup above 

training_set=subset(df,split==TRUE) 
test_set=subset(df,split==FALSE)

summary(training_set[[target]]) # check the class imbalance in the training set

# ----------------------------------------------------------------------------- #
# 5. Data preprocessing  ------------------------------------------------------
# ----------------------------------------------------------------------------- #

## NOTE: For SVMs and NNs AND ensemble models, the data MUST be pre-processed (scaled, centered AND dummy variables)

# 1. define the model so that the function knows what is the target (this defines the recipe)
rec <- recipe(as.formula(paste(target, "~ .")), 
              data = training_set) %>%
  
  # Scale and center all numeric predictors
  step_normalize(all_numeric_predictors()) %>%
  
  # Dummy encode categorical predictors 
  step_dummy(all_nominal_predictors(), one_hot = FALSE)

# to avoid perfect linearity, one of the categories of the variable during the encoding is dropped. This also speeds up the training and improves the stability of the ML model. This is done by setting one_hot = FALSE in the above.

# 2. Prep the recipe
rec_prep <- prep(rec)

# the normalization parameters (extracted from the training set) get stored in the rec_prep list object, specifically in  rec_prep[["steps"]][[1]][["means"]] for the means and rec_prep[["steps"]][[1]][["sds"]] for the standard deviations (run tidy(rec_prep, number = 1) too see), and tidy(rec_prep, number = 2) for dummy variable encoding.

# 3. Apply (bake) the prepped recipe on the training set 
train_processed <- bake(rec_prep, new_data = training_set)

# fix any names of columns which may include a period:

colnames(train_processed) <- make.names(colnames(train_processed))

# 4. Apply (bake) the same transformations to the scaled test set
test_processed <- bake(rec_prep, new_data = test_set)

# fix names of columns which may include a period:

colnames(test_processed) <- make.names(colnames(test_processed))

summary(train_processed)
summary(test_processed)

# ----------------------------------------------------------------------------- #
# 6. Balancing of the training set ---------------------------------------------
# ----------------------------------------------------------------------------- #

# NB: If balancing had to be applied, we would do so here. However, for this demonstration balancing is not required (the target is fairly well balanced).

# ----------------------------------------------------------------------------- #
# 7. Specify final training data and predictors ---------------------------------
# ----------------------------------------------------------------------------- #

# set the name of the final training set here (we now use the preprocessed training set)

training_set_final <- train_processed

# To use generic code below, we will also specify the final name of the test set as the preprocessed test set is used)

test_set_final <- test_processed

# specificy predictors

predictors <- setdiff(names(training_set_final), target)

# ----------------------------------------------------------------------------- #
# 8. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

# just to maintain the same structure as the other pracs, H2O is initialized here even though the first model (SVM) will be fitted using a different package.

h2o.init() 

# Convert  data to H2O dataframe
train_h2o <- as.h2o(training_set_final)
test_h2o  <- as.h2o(test_set_final) # NOTE: final test set which has been processed is being used

# ----------------------------------------------------------------------------- #
# 9. Random Forest ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

# H2O fits distributed random forests (DRF) - Many independent decision trees trained on different “views” of the data, built simultaneously across a computing cluster, then combined into one prediction.

# The random forest has numerous hyperparameters. Not all will be run here otherwise the model takes too long. 

# In addition to those defined in the lecture notes for section 4.5, we have:
# sample_rate controls how much of the training data each tree in the forest is allowed to see, directly affecting model diversity and overfitting.
# col_sample_rate_per_tree is the fraction of predictor variables (features) randomly selected and used to build each individual tree. This differs from mtries (number of variables selected at EACH split)

# uncomment or comment (remember the comma after each line) as required

#Define hyperparameter grid
hyper_params_drf <- list(
  ntrees = c(50, 100, 200),
  #max_depth = c(10, 20, 30),
  #min_rows = c(5, 10),
  sample_rate = seq(0.6, 1, by = 0.1),
  col_sample_rate_per_tree  = seq(0.8, 1, by = 0.1)
)


h2o.rm("drf_grid") # remove anything stored in this grid ID 


# Perform grid search using H2OGrid
grid_search_drf <- h2o.grid(
  algorithm = "drf",
  grid_id = "drf_grid",
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params_drf,
  search_criteria = list(strategy = "Cartesian"),
  nfolds = folds,
  seed = seed
)

# View grid sorted by metric of choice to be maximized (if metric should be minimized, then change decreasing = FALSE)
grid_results_drf <- h2o.getGrid(grid_id = "drf_grid", 
                                sort_by = metric, 
                                decreasing = TRUE)

print(grid_results_drf)

# extract best model
best_model_drf <- h2o.getModel(grid_results_drf@model_ids[[1]])

best_params_drf <- best_model_drf@allparameters
print(best_params_drf)


# Build and train the drf model based on the tuned hyperparameters:
drf <- h2o.randomForest(
  x = predictors,
  y = target,
  ntrees = best_params_drf$ntrees,
  max_depth = best_params_drf$max_depth,
  min_rows = best_params_drf$min_rows,
  sample_rate = best_params_drf$sample_rate,
  col_sample_rate_per_tree = best_params_drf$col_sample_rate_per_tree,
  training_frame = train_h2o,
  nfold =folds,
  seed=seed,
  keep_cross_validation_predictions = TRUE ### this option is so that we can use this model in stacking
)


h2o.performance(drf) # view cross-validated model performances


########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_drf_train <- h2o.predict(drf, train_h2o)
preds_drf_test <- h2o.predict(drf, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_drf_train <- as.data.frame(preds_drf_train)
preds_drf_test <- as.data.frame(preds_drf_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets as usual:

train_drf_pred <- cbind(training_set_final,
                       setNames(preds_drf_train[, 3, drop = FALSE], "pred_prob")) 

test_drf_pred <- cbind(test_set_final, 
                      setNames(preds_drf_test[, 3, drop = FALSE], "pred_prob"))  

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_drf_pred$pred_class <- factor(ifelse(train_drf_pred$pred_prob > threshold,"1","0"))

# test
test_drf_pred$pred_class <- factor(ifelse(test_drf_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
drf_train_CM <- caret::confusionMatrix(
  train_drf_pred$pred_class,
  train_drf_pred[[target]],
  positive = "1",
  mode = "everything"
)
drf_train_CM

# actual classes first then predicted probabilities for the training set
roc_drf_train <- pROC::roc(train_drf_pred[[target]], train_drf_pred$pred_prob)
drf_train_auc <- pROC::auc(roc_drf_train)
drf_train_auc

plot(roc_drf_train)


# test set

# predicted classes first then actual classes for the test set
drf_test_CM <- caret::confusionMatrix(
  test_drf_pred$pred_class,
  test_drf_pred[[target]],
  positive = "1",
  mode = "everything"
)

drf_test_CM

# actual classes first then predicted probabilities for the test set
roc_drf_test <- pROC::roc(test_drf_pred[[target]], test_drf_pred$pred_prob)
drf_test_auc <- pROC::auc(roc_drf_test)
drf_test_auc

plot(roc_drf_test)

# ----------------------------------------------------------------------------- #
# 10. Gradient Boosted Machines (GBM) ------------------------------------------
# ----------------------------------------------------------------------------- #

# Both GBM and XGboost use gradient boosting, but XGBoost enhances the process with second-order optimization, post-pruning, and explicit regularization. These additions make XGBoost more flexible and often more accurate, but also more complex and computationally intensive - hence it requires more powerful systems.

# We will fit a GBM (some hyperparameters are commented out so the model runs quicker for demonstraion purposes)

hyper_params_gbm <- list(
  ntrees = c(100, 200),
  #max_depth = c(3, 8),
  learn_rate = c(0.01, 0.1, 0.3), # The range is 0.0 to 1.0, and the default value is 0.1.
  sample_rate = c(0.7, 1),
  col_sample_rate = c(0.7, 1) # add a comma here is the next line is uncommented
  #min_rows = c(5, 10),
  #min_split_improvement = c(0, 0.1)  
)


h2o.rm("gbm_grid") # remove anything stored in this grid ID 


grid_search_gbm <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params_gbm,
  nfolds = folds,
  seed = seed,
  search_criteria = list(strategy = "Cartesian")
)

# View grid sorted by metric of choice to be maximized (if metric should be minimized, then change decreasing = FALSE)
grid_results_gbm <- h2o.getGrid(grid_id = "gbm_grid", 
                                sort_by = metric, 
                                decreasing = TRUE)

print(grid_results_gbm)

# extract best model
best_model_gbm <- h2o.getModel(grid_results_gbm@model_ids[[1]])

best_params_gbm <- best_model_gbm@allparameters
print(best_params_gbm)


# Build and train the gbm model based on the tuned hyperparameters:
gbm <- h2o.gbm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  ntrees = best_params_gbm$ntrees,
  max_depth = best_params_gbm$max_depth,
  learn_rate = best_params_gbm$learn_rate,
  sample_rate = best_params_gbm$sample_rate,
  col_sample_rate = best_params_gbm$col_sample_rate,
  min_rows = best_params_gbm$min_rows,
  min_split_improvement = best_params_gbm$min_split_improvement,
  seed = seed,
  nfold = folds,
  keep_cross_validation_predictions = TRUE ### this option is so that we can use this model in stacking
)


h2o.performance(gbm) # view cross-validated model performances


########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_gbm_train <- h2o.predict(gbm, train_h2o)
preds_gbm_test <- h2o.predict(gbm, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_gbm_train <- as.data.frame(preds_gbm_train)
preds_gbm_test <- as.data.frame(preds_gbm_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets as usual:

train_gbm_pred <- cbind(training_set_final,
                        setNames(preds_gbm_train[, 3, drop = FALSE], "pred_prob")) 

test_gbm_pred <- cbind(test_set_final, 
                       setNames(preds_gbm_test[, 3, drop = FALSE], "pred_prob"))  

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_gbm_pred$pred_class <- factor(ifelse(train_gbm_pred$pred_prob > threshold,"1","0"))

# test
test_gbm_pred$pred_class <- factor(ifelse(test_gbm_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
gbm_train_CM <- caret::confusionMatrix(
  train_gbm_pred$pred_class,
  train_gbm_pred[[target]],
  positive = "1",
  mode = "everything"
)
gbm_train_CM

# actual classes first then predicted probabilities for the training set
roc_gbm_train <- pROC::roc(train_gbm_pred[[target]], train_gbm_pred$pred_prob)
gbm_train_auc <- pROC::auc(roc_gbm_train)
gbm_train_auc

plot(roc_gbm_train)


# test set

# predicted classes first then actual classes for the test set
gbm_test_CM <- caret::confusionMatrix(
  test_gbm_pred$pred_class,
  test_gbm_pred[[target]],
  positive = "1",
  mode = "everything"
)

gbm_test_CM

# actual classes first then predicted probabilities for the test set
roc_gbm_test <- pROC::roc(test_gbm_pred[[target]], test_gbm_pred$pred_prob)
gbm_test_auc <- pROC::auc(roc_gbm_test)
gbm_test_auc

plot(roc_gbm_test)

# ----------------------------------------------------------------------------- #
# 11. Stacked Ensemble --------------------------------------------------------
# ----------------------------------------------------------------------------- #

# Stacking is a class of algorithms that involves training a second-level “metalearner” to find the optimal combination of the base learners. Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together. See more info here https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html.

#Before training a stacked ensemble, you will need to train and cross-validate a set of “base models” which will make up the ensemble. In order to stack these models together, a few things are required:

# - The models must be cross-validated using the same number of folds (e.g. nfold = 5 or use the same fold_column across base learners).
# - The cross-validated predictions from all of the models must be preserved by setting keep_cross_validation_predictions = TRUE. This is the data which is used to train the metalearner, or “combiner”, algorithm in the ensemble.

# - The models must be trained on the same training_frame. The rows must be identical, but you can use different sets of predictor columns, x, across models if you choose.


# Train a stacked ensemble using the RF and GBM from above. The metalearner_algorithm option allows you to specify a different metalearner algorithm. See https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/metalearner_algorithm.html for the options. 

stacked <- h2o.stackedEnsemble(x = predictors,
                               y = target,
                               training_frame = train_h2o,
                               #metalearner_algorithm = "gbm", # default is GLM
                               base_models = list(drf, gbm))

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_stacked_train <- h2o.predict(stacked, train_h2o)
preds_stacked_test <- h2o.predict(stacked, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_stacked_train <- as.data.frame(preds_stacked_train)
preds_stacked_test <- as.data.frame(preds_stacked_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets as usual:

train_stacked_pred <- cbind(training_set_final,
                        setNames(preds_stacked_train[, 3, drop = FALSE], "pred_prob")) 

test_stacked_pred <- cbind(test_set_final, 
                       setNames(preds_stacked_test[, 3, drop = FALSE], "pred_prob"))  

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_stacked_pred$pred_class <- factor(ifelse(train_stacked_pred$pred_prob > threshold,"1","0"))

# test
test_stacked_pred$pred_class <- factor(ifelse(test_stacked_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
stacked_train_CM <- caret::confusionMatrix(
  train_stacked_pred$pred_class,
  train_stacked_pred[[target]],
  positive = "1",
  mode = "everything"
)
stacked_train_CM

# actual classes first then predicted probabilities for the training set
roc_stacked_train <- pROC::roc(train_stacked_pred[[target]], train_stacked_pred$pred_prob)
stacked_train_auc <- pROC::auc(roc_stacked_train)
stacked_train_auc

plot(roc_stacked_train)


# test set

# predicted classes first then actual classes for the test set
stacked_test_CM <- caret::confusionMatrix(
  test_stacked_pred$pred_class,
  test_stacked_pred[[target]],
  positive = "1",
  mode = "everything"
)

stacked_test_CM

# actual classes first then predicted probabilities for the test set
roc_stacked_test <- pROC::roc(test_stacked_pred[[target]], test_stacked_pred$pred_prob)
stacked_test_auc <- pROC::auc(roc_stacked_test)
stacked_test_auc

plot(roc_stacked_test)


# ----------------------------------------------------------------------------- #
# 12. Combine and compare all results  --------------------------------------
# ----------------------------------------------------------------------------- #

models <- c("drf", "gbm", "stacked")
metrics <- c("Sensitivity", "Specificity", "Precision", "F1", "AUC")

get_metrics <- function(model, dataset) {
  
  cm_obj <- get(paste0(model, "_", dataset, "_CM"))
  roc_obj <- get(paste0("roc_", model, "_", dataset))
  
  data.frame(
    Model = model,
    Dataset = dataset,
    
    Sensitivity = cm_obj$byClass["Sensitivity"],
    Specificity = cm_obj$byClass["Specificity"],
    Precision   = cm_obj$byClass["Pos Pred Value"],
    F1          = cm_obj$byClass["F1"],
    AUC         = as.numeric(pROC::auc(roc_obj)),
    row.names = NULL 
  )
}

train_results <- do.call(
  rbind,
  lapply(models, function(m) {
    get_metrics(m, "train")
  })
)

test_results <- do.call(
  rbind,
  lapply(models, function(m) {
    get_metrics(m, "test")
  })
)

results_df <- as.data.frame(rbind(train_results,test_results))

results_df$Model <- factor(results_df$Model, levels = models)
results_df$Dataset <- factor(results_df$Dataset, levels = c("train", "test"))

results_df <- results_df[order(results_df$Model, results_df$Dataset), ]
results_df

# export into an Excel file if desired:

write.csv(results_df, "model_performance_results.csv", row.names = FALSE)

# graph the results:

performance_measure <- "Sensitivity" #change this to any of the columns in results_df


ggplot(results_df, aes(x = Model, y = AUC, fill = Dataset)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(label = round(AUC, 3)),
    position = position_dodge(width = 0.9),
    vjust = -0.3,
    size = 3
  ) +
  labs(
    title = "Model Performance Comparison",
    x = "Model",
    y = performance_measure
  ) +
  theme_minimal()


# ----------------------------------------------------------------------------- #
# Shutdown H2O ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

# If a mistake was made along the way or H2O model needed to be run again, shut H2o down and start again.

h2o.shutdown(prompt = FALSE)

