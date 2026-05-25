#-----------------------------------------------------------------------------------------------------#


############################################################################# 
# STAT606 Practical 1: Naive Bayes, Decision trees and Logistic regression  #                  
#############################################################################

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- # 

library(dplyr)
library(caTools)
library(caret)
library(pROC)
library(h2o)
library(rpart)
library(rpart.plot)
library(readr)

# turn scientific notation off
options(scipen = 999)

# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters -------------------------------------------------- 
# ----------------------------------------------------------------------------- #

seed <- 606
train_frac <- 0.7
metric <- "F1"
folds <- 5

# ----------------------------------------------------------------------------- #
# 2. Load, Inspect and Format Data --------------------------------------------
# ----------------------------------------------------------------------------- #

credit_card_approval <- read_csv(
  "credit_card_approval.csv",
  show_col_types = FALSE
)

credit_card_approval <- data.frame(credit_card_approval)

summary(credit_card_approval)

# convert character variables to factors
credit_card_approval <- credit_card_approval %>%
  mutate(across(where(is.character), as.factor))

# manually convert additional categorical variables
credit_card_approval$Gender <- factor(credit_card_approval$Gender)
credit_card_approval$Married <- factor(credit_card_approval$Married)
credit_card_approval$BankCustomer <- factor(credit_card_approval$BankCustomer)
credit_card_approval$PriorDefault <- factor(credit_card_approval$PriorDefault)
credit_card_approval$Employed <- factor(credit_card_approval$Employed)
credit_card_approval$DriversLicense <- factor(credit_card_approval$DriversLicense)
credit_card_approval$ZipCode <- factor(credit_card_approval$ZipCode)
credit_card_approval$Approved <- factor(
  credit_card_approval$Approved,
  levels = c("0", "1")
)

summary(credit_card_approval)

# inspect high-cardinality variables
table(credit_card_approval$ZipCode)
table(credit_card_approval$Industry)
table(credit_card_approval$Citizen)

# drop problematic variables
credit_card_approval <- credit_card_approval %>%
  dplyr::select(
    -ZipCode,
    -Industry,
    -Citizen
  )

summary(credit_card_approval)
summary(credit_card_approval$Approved)

# ----------------------------------------------------------------------------- #
# 3. Specify df and target ---------------------------------------------------- 
# ----------------------------------------------------------------------------- #

df <- credit_card_approval
target <- "Approved"

# ----------------------------------------------------------------------------- #
# 4. Train/Test Split --------------------------------------------------------- 
# ----------------------------------------------------------------------------- #

set.seed(seed)

split <- sample.split(df[[target]], SplitRatio = train_frac)

training_set_final <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# ----------------------------------------------------------------------------- #
# 5. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

# Convert data to H2O
train_h2o <- as.h2o(training_set_final)
test_h2o <- as.h2o(test_set)

# ----------------------------------------------------------------------------- #
# 6. Specify predictors -------------------------------------------------------
# ----------------------------------------------------------------------------- #

predictors <- setdiff(names(training_set_final), target)

# ----------------------------------------------------------------------------- #
# 7. Fit Naive Bayes Classifier -----------------------------------------------
# ----------------------------------------------------------------------------- #

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 0,
  nfolds = folds,
  seed = seed
)

h2o.performance(nb)

# predictions
preds_nb_train <- as.data.frame(h2o.predict(nb, train_h2o))
preds_nb_test <- as.data.frame(h2o.predict(nb, test_h2o))

# append probabilities
train_nb_pred <- cbind(
  training_set_final,
  pred_prob = preds_nb_train[, 3]
)

test_nb_pred <- cbind(
  test_set,
  pred_prob = preds_nb_test[, 3]
)

# threshold
threshold <- 0.5

# predicted classes
train_nb_pred$pred_class <- factor(
  ifelse(train_nb_pred$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

test_nb_pred$pred_class <- factor(
  ifelse(test_nb_pred$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

# confusion matrices
confusionMatrix(
  train_nb_pred$pred_class,
  train_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

confusionMatrix(
  test_nb_pred$pred_class,
  test_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

# ROC
roc_nb_train <- roc(
  train_nb_pred[[target]],
  train_nb_pred$pred_prob
)

roc_nb_test <- roc(
  test_nb_pred[[target]],
  test_nb_pred$pred_prob
)

auc(roc_nb_train)
auc(roc_nb_test)

plot(roc_nb_test)

# ----------------------------------------------------------------------------- #
# 8. Decision Tree using H2O --------------------------------------------------
# ----------------------------------------------------------------------------- #

hyper_params <- list(
  max_depth = seq(3, 21, by = 2),
  min_rows = c(1, 5, 10, 20, 50)
)

search_criteria <- list(
  strategy = "Cartesian"
)

# safely remove old grid if it exists
existing_keys <- h2o.ls()$key

if ("dtree_grid" %in% existing_keys) {
  h2o.rm("dtree_grid")
}

grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "dtree_grid",
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  ntrees = 1,
  learn_rate = 1,
  sample_rate = 1,
  col_sample_rate = 1,
  stopping_rounds = 0,
  seed = seed,
  nfolds = folds
)

model_results_dt <- h2o.getGrid(
  "dtree_grid",
  sort_by = metric,
  decreasing = TRUE
)

print(model_results_dt)

best_model_id <- model_results_dt@model_ids[[1]]

best_model <- h2o.getModel(best_model_id)

tuned_param_names <- names(hyper_params)

best_tuned_values <- lapply(
  tuned_param_names,
  function(param_name) {
    best_model@allparameters[[param_name]]
  }
)

names(best_tuned_values) <- tuned_param_names

final_dt_model <- do.call(
  h2o.decision_tree,
  c(
    list(
      x = predictors,
      y = target,
      training_frame = train_h2o,
      seed = seed
    ),
    best_tuned_values
  )
)

# predictions
preds_dt_train <- as.data.frame(
  h2o.predict(final_dt_model, train_h2o)
)

preds_dt_test <- as.data.frame(
  h2o.predict(final_dt_model, test_h2o)
)

train_dt_pred <- cbind(
  training_set_final,
  pred_prob = preds_dt_train[, 3]
)

test_dt_pred <- cbind(
  test_set,
  pred_prob = preds_dt_test[, 3]
)

# predicted classes
train_dt_pred$pred_class <- factor(
  ifelse(train_dt_pred$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

test_dt_pred$pred_class <- factor(
  ifelse(test_dt_pred$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

# confusion matrices
confusionMatrix(
  train_dt_pred$pred_class,
  train_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

confusionMatrix(
  test_dt_pred$pred_class,
  test_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# ROC
roc_dt_train <- roc(
  train_dt_pred[[target]],
  train_dt_pred$pred_prob
)

roc_dt_test <- roc(
  test_dt_pred[[target]],
  test_dt_pred$pred_prob
)

auc(roc_dt_train)
auc(roc_dt_test)

plot(roc_dt_test)

# ----------------------------------------------------------------------------- #
# 9. Decision Tree using rpart ------------------------------------------------
# ----------------------------------------------------------------------------- #

set.seed(seed)

DT_rpart <- rpart(
  as.formula(paste(target, "~ .")),
  data = training_set_final,
  method = "class",
  xval = folds,
  control = rpart.control(
    maxdepth = 4
  )
)

printcp(DT_rpart)
plotcp(DT_rpart)

# predictions
pred_prob_DT_train <- predict(
  DT_rpart,
  newdata = training_set_final,
  type = "prob"
)

pred_prob_DT_test <- predict(
  DT_rpart,
  newdata = test_set,
  type = "prob"
)

train_DT_rpart <- cbind(
  training_set_final,
  pred_prob = pred_prob_DT_train[, 2]
)

test_DT_rpart <- cbind(
  test_set,
  pred_prob = pred_prob_DT_test[, 2]
)

# predicted classes
train_DT_rpart$pred_class <- factor(
  ifelse(train_DT_rpart$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

test_DT_rpart$pred_class <- factor(
  ifelse(test_DT_rpart$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

# confusion matrices
confusionMatrix(
  train_DT_rpart$pred_class,
  train_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

confusionMatrix(
  test_DT_rpart$pred_class,
  test_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# ROC
roc_DT_train_rpart <- roc(
  train_DT_rpart[[target]],
  train_DT_rpart$pred_prob
)

roc_DT_test_rpart <- roc(
  test_DT_rpart[[target]],
  test_DT_rpart$pred_prob
)

auc(roc_DT_train_rpart)
auc(roc_DT_test_rpart)

plot(roc_DT_test_rpart)

# visualize tree
rpart.plot(DT_rpart)

# ----------------------------------------------------------------------------- #
# 10. Logistic Regression -----------------------------------------------------
# ----------------------------------------------------------------------------- #

LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial",
  lambda = 0,
  compute_p_values = TRUE
)

# ----------------------------------------------------------------------------- #
# Extract coefficients safely
# ----------------------------------------------------------------------------- #

LR_results <- as.data.frame(
  LR@model[["coefficients_table"]]
)

# safely create odds ratios
LR_results$OR <- round(
  exp(as.numeric(LR_results$coefficients)),
  4
)

# safely round p-values
if ("p_value" %in% names(LR_results)) {
  
  LR_results$p_value <- round(
    as.numeric(LR_results$p_value),
    4
  )
  
}

print(LR_results)

# ----------------------------------------------------------------------------- #
# Predictions
# ----------------------------------------------------------------------------- #

preds_LR_train <- as.data.frame(
  h2o.predict(LR, train_h2o)
)

preds_LR_test <- as.data.frame(
  h2o.predict(LR, test_h2o)
)

train_LR_pred <- cbind(
  training_set_final,
  pred_prob = preds_LR_train[, 3]
)

test_LR_pred <- cbind(
  test_set,
  pred_prob = preds_LR_test[, 3]
)

# ----------------------------------------------------------------------------- #
# Predicted classes
# ----------------------------------------------------------------------------- #

train_LR_pred$pred_class <- factor(
  ifelse(train_LR_pred$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

test_LR_pred$pred_class <- factor(
  ifelse(test_LR_pred$pred_prob > threshold, "1", "0"),
  levels = c("0", "1")
)

# ----------------------------------------------------------------------------- #
# Confusion matrices
# ----------------------------------------------------------------------------- #

confusionMatrix(
  train_LR_pred$pred_class,
  train_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

confusionMatrix(
  test_LR_pred$pred_class,
  test_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

# ----------------------------------------------------------------------------- #
# ROC Curves
# ----------------------------------------------------------------------------- #

roc_LR_train <- roc(
  train_LR_pred[[target]],
  train_LR_pred$pred_prob
)

roc_LR_test <- roc(
  test_LR_pred[[target]],
  test_LR_pred$pred_prob
)

auc(roc_LR_train)
auc(roc_LR_test)

plot(roc_LR_test)
# ----------------------------------------------------------------------------- #
# 11. Compare ROC Curves ------------------------------------------------------
# ----------------------------------------------------------------------------- #

plot(
  roc_nb_test,
  col = "#458B74",
  lwd = 2,
  main = "ROC Curve Comparison"
)

lines(
  roc_DT_test_rpart,
  col = "#CD3333",
  lwd = 2
)

lines(
  roc_LR_test,
  col = "#009ACD",
  lwd = 2
)

legend(
  "bottomright",
  legend = c(
    "Naive Bayes",
    "Decision Tree",
    "Logistic Regression"
  ),
  col = c(
    "#458B74",
    "#CD3333",
    "#009ACD"
  ),
  lwd = 2
)

# print AUCs
cat("\nAUC Results\n")
cat("Naive Bayes:", auc(roc_nb_test), "\n")
cat("Decision Tree:", auc(roc_DT_test_rpart), "\n")
cat("Logistic Regression:", auc(roc_LR_test), "\n")

# ----------------------------------------------------------------------------- #
# 12. Shutdown H2O ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.shutdown(prompt = FALSE)