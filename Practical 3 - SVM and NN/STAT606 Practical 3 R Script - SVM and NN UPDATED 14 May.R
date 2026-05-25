############################################################################# 
# STAT606 Practical 3: Support Vector Machine (SVM) and Neural Network (NN) #                  
#############################################################################

# This practical covers 4 models:

# 1. Linear SVM
# 2. SVM with a radial basis function kernel function
# 3. SVM with a non-linear polynomial kernel function
# 4. Neural network
# The SVMs are fitted using the train() function from the CARET package (as H2O is limited with the kernel functions). The neural network is fitted using H2O (deep learning). 

# For all of the above models:

# The training and test datasets are preprocessed (numeric variables are scaled and centered, the categorical variables undergo dummy variable encoding). 
# Hyperparameter tuning is carried out

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

#load("credit_card_approval_clean.RDATA")
load("C:/Users/mambaza/Desktop/UKZN/PGDM - Data Science/Semester 1/STAT606 - Applied Binary Classification and Matching/R Practicals/Practical 3 - SVM and NN/credit_card_approval_clean.RDATA")

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

## NOTE: For SVMs and NNs, the data MUST be pre-processed (scaled, centered AND dummy variables)

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
# 9. Fit the SVM --------------------------------------------------------------
# ----------------------------------------------------------------------------- #

# H2O does not yet have full capacity to train all types of SVMs (with different kernel types). Therefore, we will use the CARET package, specifically which allows a linear; radial basis function; or polynomial kernel to be implemented. CARET also allows for hyperparameter tuning. CARET is already load on line 16 as we use it for the performance metrics (https://topepo.github.io/caret/train-models-by-tag.html#support-vector-machines).


# Recall that there is at least 1 hyperparameter for the SVM, C (known as the cost or regularization parameter). This determines the penalty for misclassifying a data point, which directly affects the slack variables (measures how much each data point is allowed to violate the margin, meaning how far a data point can lie on the wrong side of the margin or even within the hyperplane itself).

##################### CV + training set setup for ALL SVM models ######################

# we first set up our controls for cross validation (used for all of the SVM models here)

control <- trainControl(method = "cv", 
                         number = folds,       # number of k-folds 
                         classProbs = TRUE,
                         summaryFunction = twoClassSummary, # enable this to get threshold-dependent metrics
                         savePredictions = TRUE)

# see https://topepo.github.io/caret/model-training-and-tuning.html#metrics for metrics used in the trainControl function when used for the k-fold cross-validation method


# Fixing class labels: train() generates class probabilities internally, and it uses the factor levels as column names (in our case, 0 and 1). These factor levels cannot legally be used as R column names. So we can change this by running the following:

training_set_SVM <- training_set_final

training_set_SVM[[target]] <- factor(
  training_set_SVM[[target]],
  levels = c(0, 1),
  labels = c("X0", "X1")
)



###################################### Linear SVM ###############################

# For a linear SVM, the only hyperparameter is C. Let's explore 3 ways of specifying values for C.

# We can specify a grid of values for the SVM to search through:

grid1 <- expand.grid(C = c(0.75, 0.9, 1)) # specify a vector of possible values

grid2 <- expand.grid(C = seq(0, 2, length = 20)) # 20 values from 0 to 20

grid3 <- expand.grid(C = seq(0, 2, by = 0.1)) # values from 0 to 2 in increments of 0.1

# Recall: C in the cost function of the SVM is a regularization parameter (referred to as the tuning parameter here) that plays a critical role in controlling the trade-off between achieving a low training error and maintaining a low model complexity for better generalization to new data. 

# a high C means giving higher penalty to the errors (slack variables). This forces the SVM to classify all training examples correctly, which can lead to overfitting if the data is noisy or not linearly separable. A high value of C such as 10000 (10^4), leads to a HARDERmore strict margin. NOTE: Setting C = 0 would mean you're applying zero penalty for misclassification, which results in a completely unconstrained margin - so no solution. 

# a low C allows the optimizer to focus more on maximizing the margin and less on classifying all training points correctly. This can lead to a more generalized model but might increase the number of misclassifications.

# By default CARET builds the SVM linear classifier using C = 1. It's possible to automatically compute SVM for different values of C and choose the optimal one that maximizes the model's cross-validation performance measure (specified by user). 


set.seed(seed)

SVM_linear <- train(as.formula(paste(target, "~ .")),  # the usual way of specifying the model (target and predictors) 
                    data = training_set_SVM, 
                    method = "svmLinear", # for linear SVM
                    metric= "ROC", # Accuracy, Kappa, Sens, Spec, ROC - metrics differ to that in H2O
                    trControl = control,
                    tuneGrid = grid2   # change between grid1, grid2 and grid3 depending on preference
)

SVM_linear  

# Plot model's performance vs different values of Cost

plot(SVM_linear)

# Print the best tuning parameter C that maximizes the model's performance

SVM_linear$bestTune


########## --> Extract predicted probabilities ----

pred_train_SVM_linear = predict(SVM_linear,newdata=training_set_SVM, type="prob")

pred_test_SVM_linear = predict(SVM_linear,newdata=test_set_final, type="prob")

# Append column 2 (predicted probabilities for class label = 1) to original training and test sets:

# NOTE: SVM is not a probabilistic model. The train() function applies a probability calibration step to convert the SVM scores into probabilities using Platt Scaling (logistic calibration).

train_SVM_linear_pred <- cbind(training_set_final,
                       setNames(pred_train_SVM_linear[, 2, drop = FALSE], "pred_prob"))  

test_SVM_linear_pred <- cbind(test_set_final,
                      setNames(pred_test_SVM_linear[, 2, drop = FALSE], "pred_prob")) 


########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_SVM_linear_pred$pred_class <- factor(ifelse(train_SVM_linear_pred$pred_prob > threshold,"1","0"))

# test
test_SVM_linear_pred$pred_class <- factor(ifelse(test_SVM_linear_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
SVM_linear_train_CM <- caret::confusionMatrix(
                    train_SVM_linear_pred$pred_class,
                    train_SVM_linear_pred[[target]],
                    positive = "1",
                    mode = "everything"
                  )

SVM_linear_train_CM

# actual classes first then predicted probabilities
roc_SVM_linear_train <- pROC::roc(train_SVM_linear_pred[[target]], train_SVM_linear_pred$pred_prob)
SVM_linear_train_auc <- pROC::auc(roc_SVM_linear_train)
SVM_linear_train_auc

plot(roc_SVM_linear_train)

# test

# predicted classes first then actual classes
SVM_linear_test_CM <- caret::confusionMatrix(
                          test_SVM_linear_pred$pred_class,
                          test_SVM_linear_pred[[target]],
                          positive = "1",
                          mode = "everything"
                        )

SVM_linear_test_CM

# actual classes first then predicted probabilities
roc_SVM_linear_test <- pROC::roc(test_SVM_linear_pred[[target]], test_SVM_linear_pred$pred_prob)
SVM_linear_test_auc <- pROC::auc(roc_SVM_linear_test)
SVM_linear_test_auc

plot(roc_SVM_linear_test)

####################### SVM Radial ##############################

# For the Radial basis function (RBF) SVM, there is an additional hyperparameter: Gamma (called sigma here in the train function). It tells us how much each individual data point will influence the decision boundary.

# sigma is a scale parameter for the RBF kernel. It controls the width of the kernel and hence influences how the similarity between data points decreases with distance.

# A small sigma value leads to a narrower peak in the kernel function, meaning that the effect of a single training example is limited to a small neighborhood around it. This can lead to a model that fits the training data very closely, but may generalize poorly on new, unseen data (overfitting).

# A large sigma value results in a wider peak, meaning that the influence of each training example reaches further. This can cause smoother decision boundaries, potentially improving generalization but at the risk of underfitting if too large.

# In summary, low values of sigma typically produce highly non-linear decision boundaries, and large values of sigma often results in a decision boundary that is more linear. 

# Use the expand.grid to specify the search space	
grid1 <- expand.grid(sigma = c(.01, .015, 0.2),
                     C = c(0.75, 0.9, 1, 1.1, 1.25))

grid2 <- expand.grid(sigma = seq(1, 3, length = 10),
                     C = 10^6) # very high value of C 


set.seed(seed) 

SVM_radial <-  train(as.formula(paste(target, "~ .")),  
                   data = training_set_SVM, 
                   method = "svmRadial", # for RBF SVM
                   metric= "ROC", # Accuracy, Kappa, Sens, Spec, ROC - metrics differ to that in H2O
                   trControl = control,
                   tuneGrid = grid1   # change between grid1, grid2 and grid3 depending on preference
)


SVM_radial


# Plot model's performance vs different values of Cost

plot(SVM_radial)

# Print the best tuning parameters 

SVM_radial$bestTune

########## --> Extract predicted probabilities ----

pred_train_svm_radial = predict(SVM_radial,newdata=training_set_SVM, type="prob")

pred_test_svm_radial = predict(SVM_radial,newdata=test_set_final, type="prob")

# Append column 2 (predicted probabilities for class label = 1) to original training and test sets:

# NOTE: SVM is not a probabilistic model. The train() function applies a probability calibration step to convert the SVM scores into probabilities using Platt Scaling (logistic calibration).

train_SVM_radial_pred <- cbind(training_set_final,
                            setNames(pred_train_svm_radial[, 2, drop = FALSE], "pred_prob"))  

test_SVM_radial_pred <- cbind(test_set_final,
                           setNames(pred_test_svm_radial[, 2, drop = FALSE], "pred_prob")) 


########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_SVM_radial_pred$pred_class <- factor(ifelse(train_SVM_radial_pred$pred_prob > threshold,"1","0"))

# test
test_SVM_radial_pred$pred_class <- factor(ifelse(test_SVM_radial_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
SVM_radial_train_CM <- caret::confusionMatrix(
                  train_SVM_radial_pred$pred_class,
                  train_SVM_radial_pred[[target]],
                  positive = "1",
                  mode = "everything"
                )

SVM_radial_train_CM

# actual classes first then predicted probabilities
roc_SVM_radial_train <- pROC::roc(train_SVM_radial_pred[[target]], train_SVM_radial_pred$pred_prob)
SVM_radial_train_auc <- pROC::auc(roc_SVM_radial_train)
SVM_radial_train_auc

plot(roc_SVM_radial_train)

# test

# predicted classes first then actual classes
SVM_radial_test_CM <- caret::confusionMatrix(
                      test_SVM_radial_pred$pred_class,
                      test_SVM_radial_pred[[target]],
                      positive = "1",
                      mode = "everything"
                    )
SVM_radial_test_CM

# actual classes first then predicted probabilities
roc_SVM_radial_test <- pROC::roc(test_SVM_radial_pred[[target]], test_SVM_radial_pred$pred_prob)
SVM_radial_test_auc <- pROC::auc(roc_SVM_radial_test)
SVM_radial_test_auc

plot(roc_SVM_radial_test)


############ Non-linear Polynomial SVM ####################

# The SVM which  a non-linear polynomial  kernel has two additional hyperparameters (additional to C):
# - degree: Polynomial order - Controls curvature complexity (a degree increases, boundary becomes more “curved” and high-order)
# - scale: Kernel scaling - Controls sensitivity to feature similarity (as scale increases, boundary becomes more sensitive to small feature differences)

grid1 <- expand.grid(
  degree = c(1, 2, 3),
  scale = c(0.001, 0.01, 0.1),
  C = c(0.1, 1, 10)
)

# Fit the model on the training set
set.seed(seed)
SVM_poly <- train(as.formula(paste(target, "~ .")),  
                  data = training_set_SVM, 
                  method = "svmPoly", # for poly SVM
                  metric= "ROC", # Accuracy, Kappa, Sens, Spec, ROC - metrics differ to that in H2O
                  trControl = control,
                  tuneGrid = grid1   # change between grid1, grid2 and grid3 depending on preference
)
  

SVM_poly

SVM_poly$bestTune


########## --> Extract predicted probabilities ----

pred_train_svm_poly = predict(SVM_poly,newdata=training_set_SVM, type="prob")

pred_test_svm_poly = predict(SVM_poly,newdata=test_set_final, type="prob")

# Append column 2 (predicted probabilities for class label = 1) to original training and test sets:

# NOTE: SVM is not a probabilistic model. The train() function applies a probability calibration step to convert the SVM scores into probabilities using Platt Scaling (logistic calibration).

train_SVM_poly_pred <- cbind(training_set_final,
                               setNames(pred_train_svm_poly[, 2, drop = FALSE], "pred_prob"))  

test_SVM_poly_pred <- cbind(test_set_final,
                              setNames(pred_test_svm_poly[, 2, drop = FALSE], "pred_prob")) 


########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_SVM_poly_pred$pred_class <- factor(ifelse(train_SVM_poly_pred$pred_prob > threshold,"1","0"))

# test
test_SVM_poly_pred$pred_class <- factor(ifelse(test_SVM_poly_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
SVM_poly_train_CM <- caret::confusionMatrix(
                        train_SVM_poly_pred$pred_class,
                        train_SVM_poly_pred[[target]],
                        positive = "1",
                        mode = "everything"
                      )
SVM_poly_train_CM

# actual classes first then predicted probabilities
roc_SVM_poly_train <- pROC::roc(train_SVM_poly_pred[[target]], train_SVM_poly_pred$pred_prob)
SVM_poly_train_auc <- pROC::auc(roc_SVM_poly_train)
SVM_poly_train_auc

plot(roc_SVM_poly_train)

# test

# predicted classes first then actual classes
SVM_poly_test_CM <- caret::confusionMatrix(
                          test_SVM_poly_pred$pred_class,
                          test_SVM_poly_pred[[target]],
                          positive = "1",
                          mode = "everything"
                        )
SVM_poly_test_CM

# actual classes first then predicted probabilities
roc_SVM_poly_test <- pROC::roc(test_SVM_poly_pred[[target]], test_SVM_poly_pred$pred_prob)
SVM_poly_test_auc <- pROC::auc(roc_SVM_poly_test)
SVM_poly_test_auc

plot(roc_SVM_poly_test)


# ----------------------------------------------------------------------------- #
# 10. Fit the Neural Network --------------------------------------------------
# ----------------------------------------------------------------------------- #

# The H2O package supports deep learning and hyperparameter tuning for the NN. It uses the stochastic gradient descent (SDG) optimizer (weights are updated in batches of 1, i.e. after each training example).

# See https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html for all info. 

# we will consider 3 architecture types:
# - 2 hidden layers (one with 3 nodes and then next with 5 nodes)
# - 2 hidden layers, each with 10 nodes
# - 3 hidden layers, the first with 5 nodes, the second with 10 nodes and the last with 3 nodes

# Set hyperparameters for grid search
hyper_params <- list(
  hidden = list(c(3, 5), c(10, 10), c(5, 10, 3)),  
  activation = c("Rectifier", "Tanh", "Maxout"),  # activation functions for the hidden layers
  #epochs = c(10, 20),  # number of epochs - uncomment if included in grid search
  rate = c(0.001, 0.01)  # learning rate - can also use seq(0.00001,0.1, length=10)
)

h2o.rm("nn_grid") # remove anything stored in this grid ID 

# Perform grid search for hyperparameter tuning
grid_search <- h2o.grid(
  algorithm = "deeplearning", 
  grid_id = "nn_grid",
  hyper_params = hyper_params,
  x = predictors,
  y = target,
  standardize = FALSE, # this has already been done
  training_frame = train_h2o,
  search_criteria = list(strategy = "Cartesian"), # can change to "RandomDiscrete"
  adaptive_rate = FALSE, # turn on and off
  nfolds = folds,
  stopping_rounds = 0, # turn off early stopping
  seed = seed,
  reproducible = TRUE # turns off multi-threading for reproducibility, does make it slower. 
)


# View grid sorted by metric of choice to be maximized (if metric should be minimized, then change decreasing = FALSE)
model_results_nn <- h2o.getGrid("nn_grid", sort_by = metric, decreasing = TRUE)

model_results_nn

# extract best model
best_model <- h2o.getModel(model_results_nn@model_ids[[1]])

best_params <- best_model@allparameters
print(best_params) # see the default settings along with tuned hyperparameters

# train NN using full training set and tuned hyperparameters

NN <- h2o.deeplearning(
  x = predictors,
  y = target,
  training_frame = train_h2o,  # Combine training + validation if needed
  hidden = best_params$hidden,
  activation = best_params$activation, # defaults to rectifier if not included in hyper_params
  rate = best_params$rate, # defauls to 0.005 if not included in hyper_params
  adaptive_rate = FALSE,
  epochs = best_params$epochs, # defaults to 10 if not included in hyper_params
  seed = seed,
  reproducible = TRUE # turns off multi-threading for reproducibility
) 

h2o.performance(NN) # view cross-validated model performances


########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_NN_train <- h2o.predict(NN, train_h2o)
preds_NN_test <- h2o.predict(NN, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_NN_train <- as.data.frame(preds_NN_train)
preds_NN_test <- as.data.frame(preds_NN_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets as usual:

train_NN_pred <- cbind(training_set_final,
                       setNames(preds_NN_train[, 3, drop = FALSE], "pred_prob")) 

test_NN_pred <- cbind(test_set_final, 
                      setNames(preds_NN_test[, 3, drop = FALSE], "pred_prob"))  

# view the above:

View(train_NN_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_NN_pred$pred_class <- factor(ifelse(train_NN_pred$pred_prob > threshold,"1","0"))

# test
test_NN_pred$pred_class <- factor(ifelse(test_NN_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
NN_train_CM <- caret::confusionMatrix(
                  train_NN_pred$pred_class,
                  train_NN_pred[[target]],
                  positive = "1",
                  mode = "everything"
                )
NN_train_CM

# actual classes first then predicted probabilities for the training set
roc_NN_train <- pROC::roc(train_NN_pred[[target]], train_NN_pred$pred_prob)
NN_train_auc <- pROC::auc(roc_NN_train)
NN_train_auc

plot(roc_NN_train)


# test set

# predicted classes first then actual classes for the test set
NN_test_CM <- caret::confusionMatrix(
                    test_NN_pred$pred_class,
                    test_NN_pred[[target]],
                    positive = "1",
                    mode = "everything"
                  )

NN_test_CM

# actual classes first then predicted probabilities for the test set
roc_NN_test <- pROC::roc(test_NN_pred[[target]], test_NN_pred$pred_prob)
NN_test_auc <- pROC::auc(roc_NN_test)
NN_test_auc

plot(roc_NN_test)

# ----------------------------------------------------------------------------- #
# 11. Combine and compare all results  --------------------------------------
# ----------------------------------------------------------------------------- #

models <- c("SVM_linear", "SVM_radial", "SVM_poly", "NN")
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







