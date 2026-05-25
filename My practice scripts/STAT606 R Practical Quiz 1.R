
#############################################################################
# STAT606 Practical Example 1
# Heart Disease Classification
#############################################################################

# ----------------------------------------------------------------------------- #
# 0. Load Libraries ------------------------------------------------------------
# ----------------------------------------------------------------------------- #

library(dplyr)
library(caTools)
library(caret)
library(pROC)
library(h2o)
library(rpart)
library(rpart.plot)
library(recipes)
library(themis)
library(readr)

options(scipen = 999)

# ----------------------------------------------------------------------------- #
# 1. Setup Parameters ---------------------------------------------------------
# ----------------------------------------------------------------------------- #

seed       <- 765
train_frac <- 0.75
folds      <- 10
threshold  <- 0.53

set.seed(seed)

# ----------------------------------------------------------------------------- #
# 2. Load & Format Data -------------------------------------------------------
# ----------------------------------------------------------------------------- #

df <- read_csv("heart_data.csv", show_col_types = FALSE)
df <- data.frame(df)

df$HeartDisease <- factor(df$HeartDisease, levels = c("No", "Yes"))

df <- df %>%
  mutate(across(
    c(Smoking, AlcoholDrinking, Stroke, Sex, Diabetic,
      PhysicalActivity, Asthma, KidneyDisease, SkinCancer),
    as.factor
  ))

# ----------------------------------------------------------------------------- #
# QUESTION 1 -----------------------------------------------------------------
# ----------------------------------------------------------------------------- #

table(df$HeartDisease)

# ANSWER Q1:
# The number of individuals who reported coronary heart disease or myocardial
# infarction is the count under "Yes" in the table above.

# ----------------------------------------------------------------------------- #
# Train / Test Split (75:25) --------------------------------------------------
# ----------------------------------------------------------------------------- #

split <- sample.split(df$HeartDisease, SplitRatio = train_frac)
train <- subset(df, split == TRUE)
test  <- subset(df, split == FALSE)

# ----------------------------------------------------------------------------- #
# Balance Training Data (SMOTE) ----------------------------------------------
# ----------------------------------------------------------------------------- #

smote_recipe <- recipe(HeartDisease ~ ., data = train) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(HeartDisease, skip = TRUE)

train_bal <- prep(smote_recipe) %>% bake(new_data = NULL)

prop.table(table(train_bal$HeartDisease))

# COMMENT:
# The balanced training set has approximately equal proportions of
# HeartDisease = Yes and No, satisfying the balancing requirement.

# ----------------------------------------------------------------------------- #
# Initialise H2O --------------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

train_h2o <- as.h2o(train_bal)
test_h2o  <- as.h2o(test)

y <- "HeartDisease"
x <- setdiff(names(train_bal), y)

# ----------------------------------------------------------------------------- #
# QUESTION 2: H2O DECISION TREE ----------------------------------------------
# ----------------------------------------------------------------------------- #

if ("dt_grid" %in% h2o.ls()$key) {
  h2o.rm("dt_grid")
}

hyper_grid <- list(
  max_depth = 3:6,
  min_rows  = seq(1, 10, by = 3)
)

grid <- h2o.grid(
  algorithm = "drf",
  grid_id = "dt_grid",
  x = x,
  y = y,
  training_frame = train_h2o,
  ntrees = 100,
  nfolds = folds,
  seed = seed,
  stopping_metric = "AUC",
  hyper_params = hyper_grid
)

grid_perf <- h2o.getGrid("dt_grid", sort_by = "f1", decreasing = TRUE)
best_dt <- h2o.getModel(grid_perf@model_ids[[1]])

best_dt@allparameters$max_depth
best_dt@allparameters$min_rows

# ANSWER Q2(a):
# The optimal maximum depth is the value printed above.

# ANSWER Q2(b):
# The optimal minimum number of observations in a leaf node is the value
# printed above for min_rows.

train_perf_dt <- h2o.performance(best_dt, train_h2o)
test_perf_dt  <- h2o.performance(best_dt, test_h2o)

h2o.F1(train_perf_dt, thresholds = 0.53)
h2o.F1(test_perf_dt, thresholds = 0.53)

# ANSWER Q2(c):
# F1-score training set = value shown in first output (rounded to 4 d.p.)
# F1-score test set     = value shown in second output (rounded to 4 d.p.)

h2o.auc(test_perf_dt)

# ANSWER Q2(d):
# The AUC on the test set (value above) indicates the model has a FAIR ability
# to distinguish between individuals with and without heart disease.

# ----------------------------------------------------------------------------- #
# QUESTION 3: rpart DECISION TREE --------------------------------------------
# ----------------------------------------------------------------------------- #

DT_rpart <- rpart(
  HeartDisease ~ .,
  data = train,
  method = "class",
  xval = folds
)

printcp(DT_rpart)

opt_cp <- DT_rpart$cptable[which.min(DT_rpart$cptable[, "xerror"]), "CP"]

# ANSWER Q3(a):
# The optimal complexity parameter (CP) is the value printed above.

DT_pruned <- prune(DT_rpart, cp = opt_cp)

nrow(DT_pruned$frame)

# ANSWER Q3(b):
# The total number of nodes in the final decision tree is the value above.

rpart.plot(DT_pruned)

# ANSWER Q3(c):
# The individual's heart disease status can be determined by following
# the decision rules shown in the plotted tree.

DT_pruned$variable.importance

# ANSWER Q3(d):
# The most important attribute is the variable with the highest importance score.

pred_prob_train <- predict(DT_pruned, train, type = "prob")[,2]
pred_prob_test  <- predict(DT_pruned, test, type = "prob")[,2]

confusionMatrix(
  factor(ifelse(pred_prob_train > threshold, "Yes", "No")),
  train$HeartDisease,
  positive = "Yes"
)$byClass["F1"]

confusionMatrix(
  factor(ifelse(pred_prob_test > threshold, "Yes", "No")),
  test$HeartDisease,
  positive = "Yes"
)$byClass["F1"]

# ANSWER Q3(e):
# F1-score training set = first value above (rounded to 4 d.p.)
# F1-score test set     = second value above (rounded to 4 d.p.)

# ANSWER Q3(f):
# The difference between training and test F1-scores indicates the model’s
# generalisation performance (comment on overfitting or stability).

# ----------------------------------------------------------------------------- #
# QUESTION 4: LOGISTIC REGRESSION --------------------------------------------
# ----------------------------------------------------------------------------- #

LR <- h2o.glm(
  x = x,
  y = y,
  training_frame = train_h2o,
  family = "binomial",
  lambda = 0,
  compute_p_values = TRUE
)

LR_results <- as.data.frame(LR@model$coefficients_table)
LR_results$OR <- round(exp(as.numeric(LR_results$coefficients)), 4)
LR_results$p_value <- round(as.numeric(LR_results$p_value), 4)
LR_results

# ANSWER Q4(a):
# A variable with p-value > 0.05 (shown above) does not have a significant effect.
# Report the variable name and corresponding p-value.

# ANSWER Q4(b):
# The odds ratio for BMI is reported in the OR column for BMI.

perf_LR_train <- h2o.performance(LR, train_h2o)
# What is the sensitivity achieved using the better threshold? Round off to 4 decimal places
h2o.sensitivity(perf_LR_train, thresholds = 0.3)
h2o.sensitivity(perf_LR_train, thresholds = 0.6)

# ANSWER Q4(c):
# The threshold (0.3 or 0.6) that gives the higher sensitivity above should be
# selected. Report that threshold and its sensitivity value.

pred_LR_test <- as.vector(h2o.predict(LR, test_h2o)$p1)
y_test_true  <- as.data.frame(test_h2o$HeartDisease)[,1]

roc_LR_test <- roc(y_test_true, pred_LR_test, levels = c("No", "Yes"))
auc(roc_LR_test)


# ANSWER Q4(d):
# The AUC on the test set is the value shown above.

# ----------------------------------------------------------------------------- #
# QUESTION 8: PREDICTED PROBABILITY & CLASS ----------------------------------
# ----------------------------------------------------------------------------- #

new_individual <- data.frame(
  Smoking = "Yes",
  AlcoholDrinking = "No",
  Stroke = "No",
  Sex = "Male",
  AgeCategory = "55-59",
  Diabetic = "No",
  PhysicalActivity = "Yes",
  Asthma = "No",
  KidneyDisease = "No",
  SkinCancer = "No",
  PhysicalHealth = 5,
  MentalHealth = 2,
  SleepTime = 7,
  BMI = 28
)

new_individual_proc <- bake(prep(smote_recipe), new_data = new_individual)
new_individual_h2o  <- as.h2o(new_individual_proc)

pred_prob_individual <- h2o.predict(LR, new_individual_h2o)$p1
prob_value <- as.data.frame(pred_prob_individual)[1,1]

round(prob_value, 4)
ifelse(prob_value > 0.53, "Yes", "No")

# ANSWER Q8:
# Predicted probability of heart disease = value above
# Predicted class label = "No" if probability ≤ 0.53, otherwise "Yes"

# ----------------------------------------------------------------------------- #
# Shutdown H2O ---------------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.shutdown(prompt = FALSE)
