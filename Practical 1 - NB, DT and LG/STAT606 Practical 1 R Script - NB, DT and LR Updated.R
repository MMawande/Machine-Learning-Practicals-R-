############################################################################# 
# STAT606 Practical 1: Naive Bayes, Decision trees and Logistic regression  #                  
#############################################################################

# Please always check that you have the latest version of this script (as per the date updated). This practical covers the following:

# 1. Fitting a Naive Bayes model using H2O
# 2. Fitting a decision tree using H2O, which includes hyperparameter tuning beforehand
# 3. Fitting a decision tree using Rpart (not H2O), which includes visualizing the fitted tree (and no hyperparameter tuning beforehand)
# 4. Fitting a logistic regression model using H2O
# Each are presented in different recordings below.

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- # 

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC
library(h2o)

# for fitting  DT in rprart
library(rpart)
library(rpart.plot)

# this is to turn scientific notation off so the output is easier to interpret:
options(scipen = 999) # turn back on by changing to 0

# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters -------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# User-specified parameters
seed = 606     # Seed for reproducibility (students can change)
train_frac <- 0.7  # Proportion of data in training set
metric <- "F1" # "auc", "aucpr" (Area Under Precision–Recall Curve), "logloss", "Accuracy", "Specificity", "Precision", "Recall", "F1" 
folds <- 5 # for 5-fold CV, or change to 10


# ----------------------------------------------------------------------------- #
# 2. Load, Inspect and Format Data ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# recall: data from different sources/files can be imported into R. We will use data on credit card approvals in a CSV file (download from Moodle and ensure it is the in the same folder as this project)

library(readr) # package used to import CSV files

# credit_card_approval <- read_csv("credit_card_approval.csv", show_col_types = FALSE) # turn off the output about the column specifications (change to TRUE to see difference in output for this step) 
library(readr)
credit_card_approval <- read_csv("Practical 1 - Naive Bayes, Decision Trees and Logistic Regression/credit_card_approval.csv", show_col_types = FALSE)

View(credit_card_approval)

credit_card_approval <- data.frame(credit_card_approval)

# look at the properties of the data
summary(credit_card_approval)

# We need to structure all categorical variables into factors (instead of characters which is default when a flat file/CSV/Excel/text file is imported):

# This next step tells R to find all the character variables and change them to factors:
credit_card_approval <- credit_card_approval %>%
  mutate(across(where(is.character), as.factor))

# If you have an ID variable/identifier variable, it is not necessary to retain the variable for analysis as it is arbitrary (it does not contain any meaningful information for modelling, it is not a characteristic/attribute), it is only used to merge datasets or identify observations for individuals

# check again to see distribution of factor variables
summary(credit_card_approval)

# some other categorical variables have been treated as numeric, let's convert the to factor as well:

credit_card_approval$Gender <- factor(credit_card_approval$Gender)
credit_card_approval$Married <- factor(credit_card_approval$Married)
credit_card_approval$BankCustomer <- factor(credit_card_approval$BankCustomer)
credit_card_approval$PriorDefault <- factor(credit_card_approval$PriorDefault)
credit_card_approval$Employed <- factor(credit_card_approval$Employed)
credit_card_approval$DriversLicense <- factor(credit_card_approval$DriversLicense)
credit_card_approval$ZipCode <- factor(credit_card_approval$ZipCode)
credit_card_approval$Approved <- factor(credit_card_approval$Approved) # this is actually the target

# check new factors variables again for sparsity and high cardinality:
summary(credit_card_approval)

# use the following to determine the number of levels (cardinality) of the factor variables:

sapply(Filter(is.factor, credit_card_approval), nlevels)

# Take note of ZipCode. While it provide info on geographic location, it has a high cardinality with very many sparse categories. 
table(credit_card_approval$ZipCode)


# Also note industry: It has a similar issue
table(credit_card_approval$Industry)

# We can combine some of the industries to reduce the number of categories, but for the purpose of this demonstration, we will simply drop it, 

# citizen is also dominated by one category and has very low frequency in the 'Temporary' category.
table(credit_card_approval$Citizen)

# So we will drop ZipCode, Industry and Citizen (using the select function from the dplyr package):

credit_card_approval <- credit_card_approval %>%
                              dplyr::select(
                                -ZipCode, 
                                -Industry,
                                -Citizen)


# check again:
summary(credit_card_approval)

# The target ('Approved') does not suffer from class imbalance:

summary(credit_card_approval$Approved) # take note the class labels are 0 and 1 (which can sometimes be of Yes or No instead)

# We are now read to split the data into the training and test sets.

# ----------------------------------------------------------------------------- #
# 3. Specify df and target --------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# Instead of writing code that only works for one dataset, we define named objects here that act as settings/place-holders. Then we reuse those throughout the code (rather than hard-coding):

df <- credit_card_approval

target <- "Approved" 

summary(df[[target]])


# NOTE: We can use the following to convert the class labels to 0 and 1 only if they are Yes/No (or any version of yes/no): so that generic code can be used:

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

# ----------------------------------------------------------------------------- #
# 5. Data preprocessing  ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# Note that in this demonstration, we will be fitting a Naive Bayes classifier, decision tree and a logistic regression model. None of these algorithms require data preprocessing of the attributes (normalization and dummmy variable encoding). However, for other algorithms (SVM, KNN, NN and various ensemble methods), this is the stage is applied (to both the training and test sets).


# ----------------------------------------------------------------------------- #
# 6. Balancing of the training set ---------------------------------------------
# ----------------------------------------------------------------------------- #

# NB: If balancing had to be applied, we would do so here. However, for this demonstration balancing is not required (the target is fairly well balanced).


# ----------------------------------------------------------------------------- #
# 7. Specify final training data and predictors ---------------------------------
# ----------------------------------------------------------------------------- #

# set the name of the final training set here (as this can change for preprocessed or balanced training data). Change this based on whether preprocessed and/or balanced data should be used
training_set_final <- training_set

# If data preprocessing (normalization and dummy variable encoding has been applied), the name of the test set is changed. To use generic code below, we will specify the final name of the test set:

test_set_final <- test_set

# The target was already specified in step 3, so now we need to specify the names of the columns that would be considered as the attributes. 

# Rather than writing out the column names manually, we use setdiff function, such as setdiff(x,y) to return the elements in x that are not in y. This removes the target from the list to only leave the predictors:

predictors <- setdiff(names(training_set_final), target)

# ----------------------------------------------------------------------------- #
# 8. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init() 

# To use data in H2O functions/models, it needs to be an H2O data frame. The following converts the built-in R data frame into an H2O frame and stores it in the H2O memory space. Now, all future processing (modeling, predictions) happens in H2O's memory space (inside the Java engine, not R’s memory). H2O is great at handling big datasets relative to RAM size due to its optimized data structures.


# Convert  data to H2O dataframe
train_h2o <- as.h2o(training_set_final)
test_h2o  <- as.h2o(test_set_final)


# ----------------------------------------------------------------------------- #
# 9. Fit Naive Bayes Classifier -----------------------------------------------
# ----------------------------------------------------------------------------- #

# Build and train the Naive Bayes Classifier (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/naive-bayes.html):

# No hyperparameter tuning is required so we can go straight into fitting the model:

########## --> Fit the model ----

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 0, # a smoothing parameter for categories with 0 observations to avoid zero probabilities
  nfolds = folds, # this is based on 5 or 10 folds as specified in the setup (section 1)
  seed = seed # based on the seed set
)

# check performance of model on the training set (which is based on optimal threshold from F1 score):
h2o.performance(nb)

# we will extract the predicted probabilities of class label = 1, append it to the original training and test sets to determine the predicted class for each based on the threshold and then create our confusion matrix and obtain the performance measures:

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_nb_train <- h2o.predict(nb, train_h2o)
preds_nb_test <- h2o.predict(nb, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_nb_train <- as.data.frame(preds_nb_train)
preds_nb_test <- as.data.frame(preds_nb_test)

# view the extracted predictions to see what we actually obtained:

View(preds_nb_train)

# Column 3 contained the predicted probabilities for out positive class (class 1).

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_nb_pred <- cbind(training_set_final,
                       setNames(preds_nb_train[, 3, drop = FALSE], "pred_prob")) # extract the predicted probabilities in column 3 of preds_nb_train, combine it with the original training set and call the column "pred_prob", this all is saved in a new dataframe called train_nb_pred

test_nb_pred <- cbind(test_set_final, 
                      setNames(preds_nb_test[, 3, drop = FALSE], "pred_prob")) # the same is done with the test set

# view the above:

View(train_nb_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# This following converts predicted probabilities into class labels by applying a threshold: observations with predicted probability above the threshold are classified as “1” (the positive class), and those below as “0”, with the result stored as a factor.

# training
train_nb_pred$pred_class <- factor(ifelse(train_nb_pred$pred_prob > threshold,"1","0"))

# test
test_nb_pred$pred_class <- factor(ifelse(test_nb_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
caret::confusionMatrix(
  train_nb_pred$pred_class,
  train_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_nb_train <- pROC::roc(train_nb_pred[[target]], train_nb_pred$pred_prob)
pROC::auc(roc_nb_train)
plot(roc_nb_train)


# test set

# predicted classes first then actual classes for the test set
caret::confusionMatrix(
  test_nb_pred$pred_class,
  test_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_nb_test <- pROC::roc(test_nb_pred[[target]], test_nb_pred$pred_prob)
pROC::auc(roc_nb_test)
plot(roc_nb_test)


# ----------------------------------------------------------------------------- #
# 10. Fit a decision tree using h2o --------------------------------------------
# ----------------------------------------------------------------------------- #

# We will demonstrate how to fit a DT using H2O which uses pre-pruning, as well as RPART which uses post-pruning

########## --> Hyperparameter tuning ----

# we will start by tuning the hyperparameter. The possible hyperparameters are:

# max_depth: Maximum depth of the tree
# min_rows: Minimum number of observations in a leaf
# min_split_improvement: Minimum reduction in error required to make a split


# Set up the hyperparameter search space (we will do an example for max_depth and min_rows)
hyper_params <- list(
  max_depth = seq(3, 21, by = 2), # from 3 to 21 in increments of 2
  min_rows = c(1, 5, 10, 20, 50)
)

# Define search criteria:
search_criteria <- list(
  strategy = "Cartesian" # Try "RandomDiscrete" for random search
)

# we use the h2o.grid function for cross validation and hyperparameter tuning. This function requires a grid_id which is a label for the entire grid search run so that H2O can store, retrieve, and reference it. 

#NB: Each grid search is saved as a named experiment in grid_id. Once a grid_id is used, it cannot be overwritten — it is permanently stored in the H2O session until we explicitly remove it. 

# Therefore, if we update the grid search or hyperparameter list and re-run the h2o.grid function, we need to remove the anything store in the grid_id first:

h2o.rm("dtree_grid")

# Run the grid search using a single decision tree, GBM (gradient boosting method) with ntrees = 1)
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "dtree_grid", # this is just the ID we are giving the grid search
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  ntrees = 1,
  learn_rate = 1.0, # Full weight per tree (since it's only one)
  sample_rate = 1.0,  # use 100% of the training data
  col_sample_rate = 1.0, # use 100% of the attributes
  stopping_rounds = 0, # no stopping rounds required as only 1 tree is being grown
  seed = seed,
  nfolds =folds
)

# Next we order the grid search results according to the best CV performance based on our selected metric and save it in model_results_dt

model_results_dt <- h2o.getGrid("dtree_grid", sort_by = metric, decreasing = TRUE) # arrange metric in descending order so that the first model in this object has the best CV performance 


# Let's view the CV results 
print(model_results_dt)

# Extract the best model ID _which is in the first row of model_results_dt
best_model_id <- model_results_dt@model_ids[[1]]

# Retrieve hyperparameter values associated with the best model
best_model <- h2o.getModel(best_model_id)

# Step 1: Automatically identify which hyperparameters were tuned
# (this pulls the names directly from the hyper_params list — no hardcoding)
tuned_param_names <- names(hyper_params)

# Step 2: Extract the actual tuned values that were used in the best model
# (best_model@parameters stores everything that was actually applied)
best_tuned_values <- lapply(tuned_param_names, function(param_name) { 
  best_model@allparameters[[param_name]]
})

# append the hyperparameter names to the tuned values:
names(best_tuned_values) <- tuned_param_names

########## --> Fit the model ----

# Step 3: Build and train the final decision tree using h2o.decision_tree()
# We use do.call so the extracted hyperparameters are passed automatically
# (you can add any other fixed parameters you want here)

final_dt_model <- do.call(h2o.decision_tree, c(
  list(
    x = predictors,
    y = target,
    training_frame = train_h2o,
    seed = seed                       # the seed
  ),
  best_tuned_values     # ← automatically inserts max_depth, min_rows, etc.
))

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_dt_train <- h2o.predict(final_dt_model, train_h2o)
preds_dt_test <- h2o.predict(final_dt_model, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_dt_train <- as.data.frame(preds_dt_train)
preds_dt_test <- as.data.frame(preds_dt_test)

#the structure of the predicted output is the same as that from the Naive Bayes:
View(preds_dt_train)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets in the exact same manner as that for the Naive bayes:

train_dt_pred <- cbind(training_set_final,
                       setNames(preds_dt_train[, 3, drop = FALSE], "pred_prob")) 

test_dt_pred <- cbind(test_set_final, 
                      setNames(preds_dt_test[, 3, drop = FALSE], "pred_prob"))  

# view the above:

View(train_dt_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_dt_pred$pred_class <- factor(ifelse(train_dt_pred$pred_prob > threshold,"1","0"))

# test
test_dt_pred$pred_class <- factor(ifelse(test_dt_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
caret::confusionMatrix(
  train_dt_pred$pred_class,
  train_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_dt_train <- pROC::roc(train_dt_pred[[target]], train_dt_pred$pred_prob)
pROC::auc(roc_dt_train)
plot(roc_dt_train)


# test set

# predicted classes first then actual classes for the test set
caret::confusionMatrix(
  test_dt_pred$pred_class,
  test_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_dt_test <- pROC::roc(test_dt_pred[[target]], test_dt_pred$pred_prob)
pROC::auc(roc_dt_test)
plot(roc_dt_test)



# ----------------------------------------------------------------------------- #
# 11. Fit a decision tree using rpart --------------------------------------------
# ----------------------------------------------------------------------------- #

# we will use another package that fits a DT that allows for a visualization. This package grows the tree by implementing Cost-Complexity pruning, where pre and post pruning is implemented using a complexity parameter (cp). The complexity parameter (cp) is used to control the size of the decision tree and to select the optimal tree size.

# cp sets the minimum improvement a split must provide to be worth making.
# Any split that does not decrease the overall lack-of-fit (error) by at least a factor of cp is not attempted.
# Higher cp → smaller, simpler trees (more pruning).
# Lower cp → larger, more complex trees (less pruning, higher risk of overfitting).

#rpart does not have functionality for specifying grid searches, only single values. 

########## --> Fit the DT using Rpart ----

# NOTE: We use the training and test sets that were originally split, not the ones from H2O.

set.seed(seed)


DT_rpart <- rpart(
                as.formula(paste(target, "~ .")),
                data = training_set_final,
                method = "class", # for classification
                xval = folds, # CV 
                control = rpart.control(
                 #cp = 0.02,             # complexity parameter for pruning
                  #minsplit = 20,         # minimum observations to attempt a split
                  maxdepth = 4           # maximum depth
                )
              ) # default attribute selection measure is Gini Index

DT_rpart # run this to see information about the fitted tree

# rpart automatically searches over values for cp to be tuned.

# rpart grows a large tree first (with a very small cp), then considers a whole sequence of smaller sub-trees by increasing the effective cp.

# --------------------------------------------  #
# While the tree is being grown, a samll cp acts as a stopping rule:

# A split is only made if it improves the model fit by at least cp amount.
# More precisely: the reduction in impurity must exceed cp.

# So if cp is large → fewer splits → smaller tree
# If cp is very small → tree grows much deeper

# This is pre-pruning (early stopping).
# -------------------------------------------- #
# After growing a large tree, rpart also uses cp in a cost-complexity pruning framework:

# It computes a sequence of nested subtrees
# Each subtree corresponds to a different cp value
# These are stored in the complexity parameter table (cptable)
# -------------------------------------------- #

printcp(DT_rpart)
plotcp(DT_rpart)

# The cp table includes:

# The cp penalty value associated with each subtree

# nsplit: Number of splits in the tree (more splits = more complex tree)

# The rel error is the total error of the model divided by the error of the initial model (a model with just the root node, predicting the most frequent class). It's a measure of the error relative to the simplest possible model.

# The xerror is the cross-validation error of the model relative to te root node model(0 splits). It is computed during the tree-building process if cross-validation is enabled (e.g., using the xval argument in rpart()). This error is estimated by applying the decision tree to each of the cross-validation folds used during tree construction. It provides a measure of how well the tree is likely to perform on unseen data, hence an estimate of the model's generalization error. Typically, it helps identify if the model is overfitting. If xerror starts to increase as the complexity of the model increases (more splits in the tree), it may suggest that simpler models are preferable.

# The xstd is the standard error of the cross-validation error (xerror). This value provides an indication of the variability of the cross-validation error estimate. A high standard error suggests that the cross-validation error might not be a reliable estimate of the model's error on new data, possibly due to the model being unstable across different subsets of the training data or due to a small number of cross-validation folds.

# rpart() automatically computes the optimal tree size (considering complexity cost) using these metrics. Specifically, xerror and xstd are used to determine the smallest tree that is within one standard error of the minimum cross-validation error (xerror + xstd). This criterion helps to balance model accuracy with complexity, aiming to avoid overfitting while maintaining sufficient explanatory power.


########## --> Extract predicted probabilities ----

### Extract predicted probabilities:
# Note: this provides TWO columns - the predicted probabilities for "0" in column 1 and "1" in column 2.

pred_prob_DT_train <- predict(DT_rpart, newdata = training_set_final, type = "prob")

train_DT_rpart <- cbind(training_set_final, 
                        setNames(data.frame(pred_prob_DT_train[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")

# View the results:

View(train_DT_rpart)


pred_prob_DT_test <- predict(DT_rpart, newdata = test_set_final, type = "prob")

test_DT_rpart <- cbind(test_set_final, 
                        setNames(data.frame(pred_prob_DT_test[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training set
train_DT_rpart$pred_class <- factor(ifelse(train_DT_rpart$pred_prob > threshold, "1","0"))

# test
test_DT_rpart$pred_class <- factor(ifelse(test_DT_rpart$pred_prob > threshold,"1","0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
caret::confusionMatrix(
  train_DT_rpart$pred_class,
  train_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_DT_train_rpart <- pROC::roc(train_DT_rpart[[target]], train_DT_rpart$pred_prob)
pROC::auc(roc_DT_train_rpart)
plot(roc_DT_train_rpart)

# test set

# predicted classes first then actual classes for the test set
caret::confusionMatrix(
  test_DT_rpart$pred_class,
  test_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_DT_test_rpart <- pROC::roc(test_DT_rpart[[target]], test_DT_rpart$pred_prob)
pROC::auc(roc_DT_test_rpart)
plot(roc_DT_test_rpart)


########## --> visualize the DT ----

dev.new(width = 15, height = 20) # This just allows the plot to be shown in a separate window (useful for small screens)

rpart.plot(DT_rpart)
rpart.plot(DT_rpart, yesno = 1, type = 2, fallen.leaves = FALSE) # add additional options to change the appearance.
# see http://www.milbo.org/rpart-plot/prp.pdf for more options to customize the plot



# ----------------------------------------------------------------------------- #
# 12. Fit a logistic regression model --------------------------------------------
# ----------------------------------------------------------------------------- #

# A logistic regression is in the class of a generalized linear model (GLM), various GLMs can be fitted in h2o for different types of responses (continuous, binary, count, multiple categories - multi-class classification)

# An LR model has no hyperparameters to tune.

########## --> Fit the LR model in H2O ----

# Fit the logistic regression model
LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial", # logistic regression
  lambda = 0, # no regularization (like classical GLM)
  compute_p_values = TRUE, # optional: get p-values
  nfolds = folds,
  seed = seed
)

########## --> Perform inference using the LR model ----

# extract p-values for inference and save into a df called LR_results

LR_results <- LR@model[["coefficients_table"]]

# create odds ratios from the regression coefficient estimates
LR_results$OR <- exp(LR_results[,2])

# round the p-values off to 4 decimal places
LR_results$p_value <- round(LR_results$p_value,4)
LR_results$OR <- round(LR_results$OR,4)

View(LR_results)

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_LR_train <- h2o.predict(LR, train_h2o)
preds_LR_test <- h2o.predict(LR, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_LR_pred <- cbind(training_set_final,
                       setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))  

test_LR_pred <- cbind(test_set_final,
                       setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob")) 



########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_LR_pred$pred_class <- factor(ifelse(train_LR_pred$pred_prob > threshold,"1","0"))

# test
test_LR_pred$pred_class <- factor(ifelse(test_LR_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
caret::confusionMatrix(
  train_LR_pred$pred_class,
  train_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_LR_train <- pROC::roc(train_LR_pred[[target]], train_LR_pred$pred_prob)
pROC::auc(roc_LR_train)
plot(roc_LR_train)

# test

# predicted classes first then actual classes
caret::confusionMatrix(
  test_LR_pred$pred_class,
  test_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_LR_test <- pROC::roc(test_LR_pred[[target]], test_LR_pred$pred_prob)
pROC::auc(roc_LR_test)
plot(roc_LR_test)


#################### Combine ROC curves of test set for all models ############

# Plot (see https://r-charts.com/colors/ for more colours)
plot(
  roc_nb_test,
  col = "#458B74",
  lwd = 2,
  main = "ROC Curve Comparison of test set for NB, DT and LR"
)
lines(roc_DT_test_rpart, col = "#CD3333", lwd = 2)
lines(roc_LR_test, col = "#009ACD", lwd = 2)

# Add legend
legend(
  "bottomright",
  legend = c("Naive Bayes", "Decision tree", "Logistic regression"),
  col = c("#458B74", "#CD3333", "#009ACD"),
  lwd = 2
)


############## Shut down H2O cluster so it doesn't use up any more resources ############

h2o.shutdown(prompt = FALSE)



