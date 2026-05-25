
#############################################################################
# STAT606 Practical: Naive Bayes, Decision Trees, Logistic Regression & Balancing
#############################################################################

# ----------------------------------------------------------------------------- #
# Q0. Load Libraries ---------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

library(dplyr)      # data manipulation and preprocessing
library(caTools)    # train/test split (sample.split)
library(caret)      # confusionMatrix + sampling utilities
library(pROC)       # ROC + AUC
library(h2o)        # H2O models

library(rpart)      # decision tree (post-pruning via cp)
library(rpart.plot) # tree plots

options(scipen = 999) # turn off scientific notation

# INTERPRETATION:
# Turning off scientific notation makes outputs easier to read and interpret,
# especially predicted probabilities (e.g., 0.000012 instead of 1.2e-05),
# logistic regression coefficients, and odds ratios.


# ----------------------------------------------------------------------------- #
# Q1. Setup & User Parameters ------------------------------------------------- #
# ----------------------------------------------------------------------------- #

seed       <- 606     # Seed for reproducibility
train_frac <- 0.7     # Proportion of data in training set
metric     <- "F1"    # "auc", "Accuracy", "Precision", "Recall", "Specificity", "F1"
folds      <- 5       # 5-fold CV (change to 10 if needed)

# INTERPRETATION:
# seed: ensures results are reproducible (same split, same CV randomness).
# train_frac: controls how much data is used to learn patterns (train) vs evaluate (test).
# metric: defines what “best model” means (e.g., F1 balances precision/recall).
# folds: higher folds often gives more stable CV estimates, but costs more compute.


# ----------------------------------------------------------------------------- #
# Q2. Load, Inspect and Format Data ------------------------------------------ #
# ----------------------------------------------------------------------------- #

library(readr)

df_raw <- read_csv("credit_card_approval.csv", show_col_types = FALSE)
df_raw <- data.frame(df_raw)

summary(df_raw)
str(df_raw)

# INTERPRETATION:
# (a) The target variable should be the binary outcome you want to predict (e.g., Approved).
# (b) Use summary/str to identify categorical variables (character/factor) vs numeric.
# (c) Look for missing values, strange encodings (e.g., "?" in numeric fields),
#     and high-cardinality categorical fields (e.g., ZipCode-like variables).


# ----------------------------------------------------------------------------- #
# Q3. Convert Characters to Factors + fix 0/1 categorical fields -------------- #
# ----------------------------------------------------------------------------- #

df_raw <- df_raw %>%
  mutate(across(where(is.character), as.factor))

# Example: Manually force known 0/1 coded categoricals to factor (edit to match your data)
# (If columns do not exist in your dataset, this will safely skip them.)
to_factor <- c("Gender","Married","BankCustomer","PriorDefault","Employed",
               "DriversLicense","ZipCode","Industry","Citizen","Approved")

for (v in to_factor) {
  if (v %in% names(df_raw)) df_raw[[v]] <- as.factor(df_raw[[v]])
}

summary(df_raw)

# INTERPRETATION:
# Factors tell R/H2O which variables are categorical (levels) rather than numeric magnitude.
# For NB and DT, categorical handling is natural; for LR, factors create indicator/dummy effects.
# If you incorrectly keep categoricals as numeric, the model may assume linear numeric meaning
# (e.g., category 3 is “bigger” than category 1), which is wrong.


# ----------------------------------------------------------------------------- #
# Q4. Factor Cardinality + drop/merge high-cardinality variables -------------- #
# ----------------------------------------------------------------------------- #

factor_levels <- sapply(Filter(is.factor, df_raw), nlevels)
factor_levels

# Identify high-cardinality factors (rule of thumb: > 20 levels; adapt as needed)
high_card <- names(factor_levels[factor_levels > 20])
high_card

# Quick frequency check for suspected high-cardinality variables:
if ("ZipCode" %in% names(df_raw)) table(df_raw$ZipCode)
if ("Industry" %in% names(df_raw)) table(df_raw$Industry)
if ("Citizen" %in% names(df_raw))  table(df_raw$Citizen)

# Drop typical high-cardinality / sparse variables (edit based on your dataset)
drop_vars <- intersect(c("ZipCode","Industry","Citizen"), names(df_raw))
df <- df_raw %>% dplyr::select(-all_of(drop_vars))

summary(df)

# INTERPRETATION:
# High-cardinality variables create many sparse categories (few observations per level),
# which can cause unstable probability estimates (NB), noisy splits (DT),
# and inflated/unstable coefficients (LR).
# Dropping/merging reduces sparsity and improves generalisation.


# ----------------------------------------------------------------------------- #
# Q5. Specify df, target, predictors ----------------------------------------- #
# ----------------------------------------------------------------------------- #

target <- "Approved"  # EDIT if your target has a different name

# Ensure target exists:
stopifnot(target %in% names(df))

# Ensure target is factor:
df[[target]] <- as.factor(df[[target]])

predictors <- setdiff(names(df), target)

# INTERPRETATION:
# Defining target and predictors once makes the script reusable across datasets.
# It avoids hardcoding column names in every model call.


# ----------------------------------------------------------------------------- #
# Q6. Train/Test Split (stratified) ------------------------------------------ #
# ----------------------------------------------------------------------------- #

set.seed(seed)
split <- sample.split(df[[target]], SplitRatio = train_frac)

training_set_final <- subset(df, split == TRUE)
test_set           <- subset(df, split == FALSE)

# Check stratification (class proportions in train vs test)
prop.table(table(training_set_final[[target]]))
prop.table(table(test_set[[target]]))

# INTERPRETATION:
# Stratified splitting preserves the class distribution across train and test,
# which gives a fair evaluation when classes are imbalanced. [1](https://corpdir-my.sharepoint.com/personal/mambaza_emea_corpdir_net/_layouts/15/Doc.aspx?sourcedoc=%7B0D4E6366-0656-41CA-892F-D6C63F2C8D50%7D&file=End_to_End_Classification_Workflow%201.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1)


# ----------------------------------------------------------------------------- #
# Q7. Baseline Model (majority class) ---------------------------------------- #
# ----------------------------------------------------------------------------- #

# Create baseline predictions: always predict the majority class from training set
maj_class <- names(which.max(table(training_set_final[[target]])))

baseline_pred <- factor(rep(maj_class, nrow(test_set)),
                        levels = levels(test_set[[target]]))

# caret confusionMatrix needs factor levels aligned
cm_baseline <- confusionMatrix(baseline_pred,
                               test_set[[target]],
                               positive = "1",
                               mode = "everything")
cm_baseline

# INTERPRETATION:
# Baseline tells you what performance you get with “no intelligence”.
# Your ML models must beat this to be considered useful.
# If baseline accuracy is high in imbalanced data, it may still be a poor model
# (e.g., it can have Recall = 0 for the minority/positive class).


# ----------------------------------------------------------------------------- #
# Helper functions for metrics summary --------------------------------------- #
# ----------------------------------------------------------------------------- #

get_cm_metrics <- function(cm_obj) {
  # Returns a named vector with common metrics
  out <- c(
    Accuracy    = unname(cm_obj$overall["Accuracy"]),
    Kappa       = unname(cm_obj$overall["Kappa"]),
    Precision   = unname(cm_obj$byClass["Precision"]),
    Recall      = unname(cm_obj$byClass["Recall"]),
    Specificity = unname(cm_obj$byClass["Specificity"]),
    F1          = unname(cm_obj$byClass["F1"])
  )
  return(out)
}

get_auc <- function(actual_factor, pred_prob) {
  # AUC from pROC; assumes positive class is "1"
  roc_obj <- pROC::roc(actual_factor, pred_prob, quiet = TRUE)
  auc_val <- as.numeric(pROC::auc(roc_obj))
  list(roc = roc_obj, auc = auc_val)
}

apply_threshold <- function(pred_prob, thr = 0.5) {
  factor(ifelse(pred_prob > thr, "1", "0"), levels = c("0","1"))
}


# ----------------------------------------------------------------------------- #
# Q8. Initialise H2O ---------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

h2o.init()

train_h2o <- as.h2o(training_set_final)
test_h2o  <- as.h2o(test_set)

# INTERPRETATION:
# H2O stores data in its own memory space (JVM) and runs modelling there,
# which is efficient and scalable for larger datasets compared with base R.


# ----------------------------------------------------------------------------- #
# Q9. Naive Bayes (laplace = 0 vs 1) ----------------------------------------- #
# ----------------------------------------------------------------------------- #

# Fit NB with laplace = 0
nb_l0 <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 0,
  nfolds = folds,
  seed = seed
)

# Fit NB with laplace = 1 (smoothing)
nb_l1 <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 1,
  nfolds = folds,
  seed = seed
)

h2o.performance(nb_l0)
h2o.performance(nb_l1)

# INTERPRETATION:
# Laplace smoothing fixes “zero probability” problems:
# If a category never appears with class 1 in training, NB can assign probability 0,
# which collapses predictions. Laplace adds a small count to avoid zeros.
# Compare laplace=0 vs laplace=1 performance; smoothing sometimes improves generalisation
# when data is sparse.


# ----------------------------------------------------------------------------- #
# Q10. NB predicted probabilities + confusion matrix (threshold=0.5) ---------- #
# ----------------------------------------------------------------------------- #

preds_nb_test <- as.data.frame(h2o.predict(nb_l1, test_h2o))
# H2O predict returns: predict, p0, p1 (commonly). We take p1:
nb_test_prob <- preds_nb_test[, 3]

nb_test_class <- apply_threshold(nb_test_prob, thr = 0.5)

cm_nb_test <- confusionMatrix(nb_test_class,
                              test_set[[target]],
                              positive = "1",
                              mode = "everything")
cm_nb_test

# INTERPRETATION:
# Accuracy: overall correct classification proportion.
# Precision: of predicted positives, how many are truly positive (controls false positives).
# Recall/Sensitivity: of actual positives, how many did we catch (controls false negatives).
# Specificity: of actual negatives, how many did we correctly label negative.
# F1: harmonic mean of Precision & Recall; good for imbalance.
# positive="1" means class "1" is treated as the “positive/event of interest”.


# ----------------------------------------------------------------------------- #
# Q11. NB ROC/AUC on test set ------------------------------------------------ #
# ----------------------------------------------------------------------------- #

auc_nb <- get_auc(test_set[[target]], nb_test_prob)
auc_nb$auc
plot(auc_nb$roc, main = "NB ROC Curve (Test Set)")

# INTERPRETATION:
# ROC shows the trade-off between true positive rate (Recall) and false positive rate.
# AUC summarises ranking ability:
# AUC ~ 0.5 = random guessing; AUC ~ 1.0 = near-perfect separation.


# ----------------------------------------------------------------------------- #
# Q12. NB threshold sensitivity (0.3, 0.5, 0.7) ------------------------------ #
# ----------------------------------------------------------------------------- #

thresholds <- c(0.3, 0.5, 0.7)

nb_thr_results <- lapply(thresholds, function(t) {
  pred_class <- apply_threshold(nb_test_prob, thr = t)
  cm <- confusionMatrix(pred_class, test_set[[target]], positive = "1", mode = "everything")
  c(threshold = t, get_cm_metrics(cm))
})

nb_thr_results <- as.data.frame(do.call(rbind, nb_thr_results))
nb_thr_results

# INTERPRETATION:
# Lower threshold (e.g., 0.3) -> more predicted positives -> higher Recall, lower Precision.
# Higher threshold (e.g., 0.7) -> fewer predicted positives -> higher Precision, lower Recall.
# If missing positives is costly (high FN cost), prefer lower threshold to increase Recall.


# ----------------------------------------------------------------------------- #
# Q13. H2O Decision Tree tuning (grid) --------------------------------------- #
# ----------------------------------------------------------------------------- #

hyper_params <- list(
  max_depth = seq(3, 21, by = 2),
  min_rows  = c(1, 5, 10, 20, 50)
)

search_criteria <- list(strategy = "Cartesian")

# Remove previous grid if it exists
h2o.rm("dtree_grid")

# Use GBM with ntrees=1 to behave like a single tree (common H2O approach)
grid_dt <- h2o.grid(
  algorithm = "gbm",
  grid_id = "dtree_grid",
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  ntrees = 1,
  learn_rate = 1.0,
  sample_rate = 1.0,
  col_sample_rate = 1.0,
  stopping_rounds = 0,
  seed = seed,
  nfolds = folds
)

model_results_dt <- h2o.getGrid("dtree_grid", sort_by = metric, decreasing = TRUE)
print(model_results_dt)

# INTERPRETATION:
# max_depth: controls how deep the tree can grow (complexity).
# min_rows: minimum observations in a terminal node (leaf) (controls overfitting).
# CV/grid search estimates performance on unseen folds to choose robust hyperparameters.


# ----------------------------------------------------------------------------- #
# Q14. Fit final H2O “tree” using best hyperparameters ----------------------- #
# ----------------------------------------------------------------------------- #

best_model_id <- model_results_dt@model_ids[[1]]
best_model    <- h2o.getModel(best_model_id)

# Extract tuned parameter values used in best model
tuned_param_names <- names(hyper_params)
best_tuned_values <- lapply(tuned_param_names, function(p) best_model@allparameters[[p]])
names(best_tuned_values) <- tuned_param_names

# Train final single-tree-like model (GBM with ntrees=1) using best params
final_dt_model <- do.call(h2o.gbm, c(
  list(
    x = predictors,
    y = target,
    training_frame = train_h2o,
    ntrees = 1,
    learn_rate = 1.0,
    sample_rate = 1.0,
    col_sample_rate = 1.0,
    seed = seed
  ),
  best_tuned_values
))

# Predict on train + test
preds_dt_train <- as.data.frame(h2o.predict(final_dt_model, train_h2o))
preds_dt_test  <- as.data.frame(h2o.predict(final_dt_model, test_h2o))

dt_train_prob <- preds_dt_train[, 3]
dt_test_prob  <- preds_dt_test[, 3]

dt_test_class <- apply_threshold(dt_test_prob, thr = 0.5)

cm_dt_test <- confusionMatrix(dt_test_class,
                              test_set[[target]],
                              positive = "1",
                              mode = "everything")
cm_dt_test

auc_dt <- get_auc(test_set[[target]], dt_test_prob)
auc_dt$auc
plot(auc_dt$roc, main = "Decision Tree-like Model ROC (Test Set)")

# INTERPRETATION:
# Compare train vs test performance to detect overfitting:
# A big drop from train to test suggests high variance/overfitting.
# Deep trees often overfit by learning very specific training patterns. [2](https://corpdir-my.sharepoint.com/personal/mambaza_emea_corpdir_net/_layouts/15/Doc.aspx?sourcedoc=%7B96D2805F-5E32-472F-9A3E-76C26E6785AC%7D&file=STAT606_Binary_Classification_R%201.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1)[3](https://corpdir.sharepoint.com/sites/05649/A104/2019-08-14%20RD-TWR_Machine%20Learning.pdf?web=1)


# ----------------------------------------------------------------------------- #
# Q15. rpart tree + cp table ------------------------------------------------- #
# ----------------------------------------------------------------------------- #

set.seed(seed)

DT_rpart <- rpart(
  as.formula(paste(target, "~ .")),
  data = training_set_final,
  method = "class",
  xval = folds
)

DT_rpart
printcp(DT_rpart)
plotcp(DT_rpart)

# INTERPRETATION:
# rel error: training error relative to the root-only model.
# xerror: cross-validated error estimate (generalisation error).
# xstd: standard error of xerror.
# To select tree size using 1-SE rule:
# choose the smallest tree with xerror <= (min xerror + xstd at min xerror).


# ----------------------------------------------------------------------------- #
# Q16. Prune rpart using 1-SE cp and compare performance --------------------- #
# ----------------------------------------------------------------------------- #

cp_tab <- DT_rpart$cptable
min_idx <- which.min(cp_tab[, "xerror"])
min_xerror <- cp_tab[min_idx, "xerror"]
min_xstd   <- cp_tab[min_idx, "xstd"]

# 1-SE threshold
one_se_cut <- min_xerror + min_xstd

# Choose smallest tree (largest cp) with xerror <= one_se_cut
candidate_rows <- which(cp_tab[, "xerror"] <= one_se_cut)
cp_1se <- cp_tab[min(candidate_rows), "CP"]

DT_pruned <- prune(DT_rpart, cp = cp_1se)

# Probabilities on test set (prob of class "1" is column 2)
prob_rpart_test_unpruned <- predict(DT_rpart,  newdata = test_set, type = "prob")[, 2]
prob_rpart_test_pruned   <- predict(DT_pruned, newdata = test_set, type = "prob")[, 2]

class_unpruned <- apply_threshold(prob_rpart_test_unpruned, thr = 0.5)
class_pruned   <- apply_threshold(prob_rpart_test_pruned,   thr = 0.5)

cm_rpart_unpruned <- confusionMatrix(class_unpruned, test_set[[target]],
                                     positive = "1", mode = "everything")
cm_rpart_pruned   <- confusionMatrix(class_pruned,   test_set[[target]],
                                     positive = "1", mode = "everything")

cm_rpart_unpruned
cm_rpart_pruned

auc_rpart_unpruned <- get_auc(test_set[[target]], prob_rpart_test_unpruned)
auc_rpart_pruned   <- get_auc(test_set[[target]], prob_rpart_test_pruned)

auc_rpart_unpruned$auc
auc_rpart_pruned$auc

# INTERPRETATION:
# Pruning usually reduces variance (less overfitting) but can increase bias.
# If pruned test performance improves (or stays similar with simpler tree),
# pruning is beneficial for generalisation.


# ----------------------------------------------------------------------------- #
# Q17. Plot final pruned tree ------------------------------------------------ #
# ----------------------------------------------------------------------------- #

dev.new(width = 12, height = 8)
rpart.plot(DT_pruned, type = 2, fallen.leaves = TRUE, yesno = 1)

# INTERPRETATION:
# A split condition (e.g., Feature < value) tells how the tree partitions the data.
# Leaf nodes give final predicted class and often class probabilities.
# Follow a path from root to leaf to explain a decision in human terms.


# ----------------------------------------------------------------------------- #
# Q18. Logistic Regression (H2O GLM) + odds ratios --------------------------- #
# ----------------------------------------------------------------------------- #

LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial",
  lambda = 0,
  compute_p_values = TRUE
)

LR_results <- LR@model[["coefficients_table"]]
LR_results$OR <- exp(LR_results[, "coefficients"])

# Round for readability
LR_results$p_value <- round(LR_results$p_value, 4)
LR_results$OR      <- round(LR_results$OR, 4)

LR_results

# INTERPRETATION:
# coefficient sign:
#   + increases log-odds of class "1"; - decreases log-odds of class "1"
# odds ratio (OR):
#   OR > 1 -> odds of class "1" increase as predictor increases
#   OR < 1 -> odds of class "1" decrease as predictor increases
# p-value:
#   small p-value suggests predictor contributes evidence of association (inference context)
# intercept:
#   baseline log-odds when all predictors are at reference/0 levels (depends on coding).


# ----------------------------------------------------------------------------- #
# Q19. Top 5 influential predictors + plain-English interpretation ----------- #
# ----------------------------------------------------------------------------- #

# Exclude intercept for ranking
LR_rank <- LR_results %>%
  dplyr::filter(names != "Intercept") %>%
  dplyr::mutate(OR_dist = abs(log(OR))) %>%    # distance from 1 on log-scale
  dplyr::arrange(desc(OR_dist)) %>%
  dplyr::slice(1:5)

LR_rank

# INTERPRETATION (example template you should adapt to your dataset):
# For a predictor X with OR = 1.25:
# “Holding other variables constant, a one-unit increase in X multiplies the odds of
# approval (class 1) by 1.25 (i.e., increases odds by 25%).”
#
# For OR = 0.80:
# “Holding other variables constant, a one-unit increase in X multiplies the odds of
# class 1 by 0.80 (i.e., decreases odds by 20%).”


# ----------------------------------------------------------------------------- #
# Q20. LR evaluation (confusion matrix + ROC/AUC) ---------------------------- #
# ----------------------------------------------------------------------------- #

preds_lr_test <- as.data.frame(h2o.predict(LR, test_h2o))
lr_test_prob  <- preds_lr_test[, 3]
lr_test_class <- apply_threshold(lr_test_prob, thr = 0.5)

cm_lr_test <- confusionMatrix(lr_test_class, test_set[[target]],
                              positive = "1", mode = "everything")
cm_lr_test

auc_lr <- get_auc(test_set[[target]], lr_test_prob)
auc_lr$auc
plot(auc_lr$roc, main = "Logistic Regression ROC (Test Set)")

# INTERPRETATION:
# Compare NB vs DT vs LR using:
# - Recall (catch positives)
# - Precision (avoid false alarms)
# - AUC (ranking/separation quality)
# Logistic regression is often the most interpretable because:
# coefficients and OR provide clear direction + magnitude of effects.


# ----------------------------------------------------------------------------- #
# Q21. Check class distribution (imbalance) ---------------------------------- #
# ----------------------------------------------------------------------------- #

train_dist <- prop.table(table(training_set_final[[target]]))
test_dist  <- prop.table(table(test_set[[target]]))

train_dist
test_dist

# INTERPRETATION:
# If one class dominates (e.g., 90/10), imbalance is present.
# Whether balancing is needed depends on business costs:
# - If false negatives are costly, focus on Recall and consider balancing.
# - If false positives are costly, focus on Precision / Specificity.


# ----------------------------------------------------------------------------- #
# Q22. Apply balancing ONLY to training data + refit LR and DT ---------------- #
# ----------------------------------------------------------------------------- #

# Use caret upSample (works without extra packages)
# upSample requires predictors as data.frame and y as vector/factor
train_x <- training_set_final %>% dplyr::select(-all_of(target))
train_y <- training_set_final[[target]]

up_train <- upSample(x = train_x, y = train_y, yname = target)

# Convert balanced train to H2O
train_h2o_bal <- as.h2o(up_train)

# Refit LR on balanced training data
LR_bal <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o_bal,
  family = "binomial",
  lambda = 0,
  compute_p_values = TRUE
)

# Refit DT-like model on balanced training data (single tree)
DT_bal <- h2o.gbm(
  x = predictors,
  y = target,
  training_frame = train_h2o_bal,
  ntrees = 1,
  learn_rate = 1.0,
  sample_rate = 1.0,
  col_sample_rate = 1.0,
  seed = seed
)

# INTERPRETATION:
# Balancing must be done ONLY on training data to avoid data leakage:
# If you balance using the test set, you “inject” information into evaluation,
# leading to overly optimistic performance (not a true unseen test).


# ----------------------------------------------------------------------------- #
# Q23. Compare unbalanced vs balanced models on original test set ------------- #
# ----------------------------------------------------------------------------- #

# Balanced LR predictions on original test set
preds_lr_bal_test <- as.data.frame(h2o.predict(LR_bal, test_h2o))
lr_bal_test_prob  <- preds_lr_bal_test[, 3]
lr_bal_test_class <- apply_threshold(lr_bal_test_prob, thr = 0.5)

cm_lr_bal_test <- confusionMatrix(lr_bal_test_class, test_set[[target]],
                                  positive = "1", mode = "everything")
cm_lr_bal_test

auc_lr_bal <- get_auc(test_set[[target]], lr_bal_test_prob)
auc_lr_bal$auc

# Balanced DT predictions on original test set
preds_dt_bal_test <- as.data.frame(h2o.predict(DT_bal, test_h2o))
dt_bal_test_prob  <- preds_dt_bal_test[, 3]
dt_bal_test_class <- apply_threshold(dt_bal_test_prob, thr = 0.5)

cm_dt_bal_test <- confusionMatrix(dt_bal_test_class, test_set[[target]],
                                  positive = "1", mode = "everything")
cm_dt_bal_test

auc_dt_bal <- get_auc(test_set[[target]], dt_bal_test_prob)
auc_dt_bal$auc

# INTERPRETATION:
# Balancing often increases Recall (more positives caught),
# but may reduce Precision (more false positives).
# Use F1 if you need balance between Precision and Recall.


# ----------------------------------------------------------------------------- #
# Q24. Best model under unbalanced vs balanced training (F1 focus) ------------ #
# ----------------------------------------------------------------------------- #

# Collect metrics for key test models
results <- rbind(
  NB_unbalanced   = c(get_cm_metrics(cm_nb_test),   AUC = auc_nb$auc),
  DT_unbalanced   = c(get_cm_metrics(cm_dt_test),   AUC = auc_dt$auc),
  LR_unbalanced   = c(get_cm_metrics(cm_lr_test),   AUC = auc_lr$auc),
  LR_balanced     = c(get_cm_metrics(cm_lr_bal_test), AUC = auc_lr_bal$auc),
  DT_balanced     = c(get_cm_metrics(cm_dt_bal_test), AUC = auc_dt_bal$auc)
)

results <- as.data.frame(results)
results

# Identify best model by chosen metric (e.g., F1)
best_by_metric <- rownames(results)[which.max(results[[metric]])]
best_by_metric

# INTERPRETATION:
# If metric = "F1", the best model maximises balance between Precision and Recall.
# Final recommendation should consider:
# (a) metric value, (b) business cost (FN vs FP), (c) interpretability requirements.


# ----------------------------------------------------------------------------- #
# Q25. Plot ROC curves of final test models on one chart ---------------------- #
# ----------------------------------------------------------------------------- #

plot(auc_nb$roc, col = "#458B74", lwd = 2,
     main = "ROC Curve Comparison (Test Set): NB vs DT vs LR")
lines(auc_dt$roc, col = "#CD3333", lwd = 2)
lines(auc_lr$roc, col = "#009ACD", lwd = 2)

legend("bottomright",
       legend = c("Naive Bayes", "Decision Tree-like", "Logistic Regression"),
       col = c("#458B74", "#CD3333", "#009ACD"),
       lwd = 2)

# INTERPRETATION:
# If one ROC curve is consistently above another, it dominates (better TPR at same FPR).
# AUC is a summary measure: higher AUC generally indicates better class separation.


# ----------------------------------------------------------------------------- #
# Q26. Compact results summary + conclusion ----------------------------------- #
# ----------------------------------------------------------------------------- #

results

# INTERPRETATION (write your conclusion here as comments):
# 1) Best model (by chosen metric): best_by_metric
# 2) Explain WHY using test-set metrics (F1/Recall/Precision/AUC).
# 3) Note overfitting evidence (if DT train >> DT test, likely overfit). [2](https://corpdir-my.sharepoint.com/personal/mambaza_emea_corpdir_net/_layouts/15/Doc.aspx?sourcedoc=%7B96D2805F-5E32-472F-9A3E-76C26E6785AC%7D&file=STAT606_Binary_Classification_R%201.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1)[3](https://corpdir.sharepoint.com/sites/05649/A104/2019-08-14%20RD-TWR_Machine%20Learning.pdf?web=1)
# 4) If balanced models improved Recall but reduced Precision, justify which trade-off
#    matches the real-world objective (e.g., minimise false negatives).


# ----------------------------------------------------------------------------- #
# Q27. Shutdown H2O ----------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

h2o.shutdown(prompt = FALSE)

# INTERPRETATION:
# Shutting down H2O frees memory/CPU resources. If you leave H2O running,
# it can continue consuming resources and slow down your machine.
