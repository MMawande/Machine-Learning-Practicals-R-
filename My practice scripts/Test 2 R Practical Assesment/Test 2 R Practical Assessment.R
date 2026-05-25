
# Load libraries
library(caret)        # model training, cross-validation, and performance evaluation
library(rpart)        # decision tree modelling (classification/regression trees)
library(rpart.plot)   # visualisation of decision trees
library(dplyr)        # data manipulation and transformation
library(ROSE)         # handling imbalanced data (over/under-sampling)
library(MLmetrics)    # performance metrics such as F1-score

# Set seed
set.seed(842)

# set threshold
threshold <- 0.19

# Load data
load("C:/Users/mambaza/Desktop/UKZN/PGDM - Data Science/Semester 1/STAT606 - Applied Binary Classification and Matching/R Practicals/My practice scripts/Test 2 R Practical Assesment/flight_data.RDATA")
View(flight_data)

# Ensure target is factor
flight_data$no_show <- as.factor(flight_data$no_show)

# -----------------------------
# BASIC QUESTIONS
# -----------------------------

# Answer 1
sum(flight_data$no_show == "Yes")

# Answer 2
mean(flight_data$ticket_price)

# -----------------------------
# TRAIN-TEST SPLIT (80:20)
# -----------------------------
trainIndex <- createDataPartition(flight_data$no_show, p = 0.8, list = FALSE)
train <- flight_data[trainIndex, ]
test <- flight_data[-trainIndex, ]

# -----------------------------
# PART A: DECISION TREE
# -----------------------------
tree_model <- rpart(
  no_show ~ ., 
  data = train,
  method = "class",
  control = rpart.control(cp = 0, minsplit = 10, maxdepth = 5)
)

# Leaf nodes
leaf_nodes <- sum(tree_model$frame$var == "<leaf>")
leaf_yes <- sum(tree_model$frame$var == "<leaf>" & 
                  tree_model$frame$yval == which(levels(train$no_show)=="Yes"))
leaf_nodes
leaf_yes

# Variable usage
table(tree_model$frame$var)

# ---- PLOT: UNPRUNED DECISION TREE ----
rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree (Unpruned)"
)

# -----------------------------
# PART A: PRUNED TREE
# -----------------------------
tree_model_pruned <- rpart(
  no_show ~ ., 
  data = train,
  method = "class",
  control = rpart.control(cp = 0.001, minsplit = 10, maxdepth = 5)
)

# ---- PLOT: PRUNED DECISION TREE ----
rpart.plot(
  tree_model_pruned,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree (Pruned)"
)

# -----------------------------
# PART B: OVERSAMPLING
# -----------------------------
train_balanced <- ovun.sample(no_show ~ ., data = train, method = "over")$data

# Logistic regression
log_model <- glm(no_show ~ ., data = train_balanced, family = binomial)

# -----------------------------
# CROSS-VALIDATION 
# -----------------------------
ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

set.seed(842)
model <- train(
  no_show ~ ., 
  data = train_balanced,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)

# Training probabilities
train_probs <- predict(log_model, train_balanced, type="response")

# Threshold tuning (for reporting only)
thresholds <- seq(0.1, 0.9, 0.01)

# maximum cross-validated & handle NaN to obtain correct F1_score
metrics <- sapply(thresholds, function(t) {
  preds <- ifelse(train_probs > t, "Yes", "No")
  preds <- factor(preds, levels = levels(train_balanced$no_show))
  
  score <- F1_Score(y_pred = preds, y_true = train_balanced$no_show, positive = "Yes")
  
  if (is.nan(score)) 0 else score
})

# Answers 11 & 12
max_metric <- max(metrics)
best_threshold <- thresholds[which.max(metrics)]

max_metric
best_threshold

# -----------------------------
# PROBABILITY PREDICTION
# -----------------------------
p_true <- mean(train$no_show == "Yes")
p_sample <- mean(train_balanced$no_show == "Yes")

correct_prob <- function(p) {
  (p * p_true / p_sample) /
    ((p * p_true / p_sample) + ((1 - p) * (1 - p_true) / (1 - p_sample)))
}

train_probs_corr <- correct_prob(train_probs)
test_probs <- predict(log_model, test, type="response")
test_probs_corr <- correct_prob(test_probs)

# -----------------------------
# THRESHOLD = 0.19
# -----------------------------
train_pred <- ifelse(train_probs_corr > threshold, "Yes", "No")
test_pred <- ifelse(test_probs_corr > threshold, "Yes", "No")

train_pred <- factor(train_pred, levels = levels(train$no_show))
test_pred <- factor(test_pred, levels = levels(train$no_show))

# -----------------------------
# FINAL METRICS
# -----------------------------
F1_train <- F1_Score(train_pred, train_balanced$no_show, positive="Yes")
F1_test <- F1_Score(test_pred, test$no_show, positive="Yes")

F1_train
F1_test

# Confusion matrix
cm <- table(test$no_show, test_pred)
cm

# Matthews Correlation Coefficient (MCC)
TP <- as.numeric(cm["Yes","Yes"])
TN <- as.numeric(cm["No","No"])
FP <- as.numeric(cm["No","Yes"])
FN <- as.numeric(cm["Yes","No"])

denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
MCC_value <- ifelse(denominator == 0, 0, (TP * TN - FP * FN) / denominator)

MCC_value
