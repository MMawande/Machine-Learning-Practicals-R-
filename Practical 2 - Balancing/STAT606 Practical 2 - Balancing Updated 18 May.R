############################################################################# 
# STAT606 Practical 2: Balancing                                            #                  
#############################################################################

# This practical covers the 4 sampling techniques for balancing:

# 1. Under sampling
# 2, Over sampling
# 3. Combination of over and under
# 4. SMOTE
# In addition, the process of selecting the 'best' technique to use on the competing models is outlined. This consists of these steps:

# Repeatedly fit a simple model to each of the 4 balanced training sets. A logistic regression model using H2O is chosen.
# For each, extract the predicted probabilities for the balanced training and unbalanced test sets.
# Determine the AUC (a threshold-independent metric)
# Continue to append the AUC for the balanced training set and unbalanced test set of each technique. Choose the one based on the best AUC.
# For ANY model fitted with a balanced training set from under sampling, over sampling or a combination of the two, the process of correcting 
# the probabilities is shown based on a fitted model (from which the data sets combining the training/test sets with the predicted probabilities are obtained). 
# After which the confusion matrices and model performances are obtained for the both the balanced training set and unbalanced test set. ---> 
# This same process can be followed for ANY probabilistic model that where balancing has been applied to the training set (under sampling, over sampling or a combination).

# For models fitted using SMOTE, the above probability correction cannot be applied. Rather, a confusion matrix and threshold-dependent performance metrics can be obtained
# based on applying the optimal threshold to determine the predicted class labels (covered in practical 5).

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- # 

library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting into train and test sets
library(caret) # used for performance metric functions
library(pROC) # used for obtaining AUC
library(h2o)

# install the packages required for balancing (done once)
# install.packages("ROSE")
# install.packages("mltools")

# the packages used for balancing
library(ROSE)
library(mltools)

# for SMOTE that can handle mixed types off attributes:

# highlight the following then use SHIFT + CTRL + C to un-comment and then comment again once installed

# install.packages(c(
#   "sp",
#   "randomForest",
#   "gstat",
#   "MBA",
#   "automap"
# ))
 
# install.packages(
#   "https://cran.r-project.org/src/contrib/Archive/UBL/UBL_0.0.9.tar.gz",
#   repos = NULL,
#   type = "source"
#)       # IGNORE THE WARNING MESSAGE

# Load library used for SMOTE
library(UBL)


# this is to turn scientific notation off so the output is easier to interpret:
options(scipen = 999) # turn back on by changing to 0

# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters -------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# User-specified parameters
seed = 456     # Seed for reproducibility (students can change)
train_frac <- 0.7  # Proportion of data in training set
metric <- "F1" # "auc", "aucpr" (Area Under Precision–Recall Curve), "logloss", "Accuracy", "Specificity", "Precision", "Recall", "F1" 
folds <- 5 # for 5-fold CV, or change to 10


# ----------------------------------------------------------------------------- #
# 2. Load, Inspect and Format Data --------------------------------------------
# ----------------------------------------------------------------------------- #

# This demonstration makes use of depression data which has already been structured and cleaned. It is in an .RData file on Moodle. 

# This data contains information pertaining to an individual's depression status (based on self-reporting), in addition to some characteristics of the individual. The objective is to use the attributes to predict a person's depression status.
# The file called 'Variable Values for Depression Data' contains the category names and levels for the categorical attributes.


# Let's load the .RData file containing the data (an .RData file is an R file format used to store one or more objects (datasets, models, or variables etc.) from your workspace/global environment so they can be saved and loaded later). When you load an .RData file, R brings the objects back into your workspace with their original structure intact, so variables keep their types (e.g. factors vs numeric), levels, and any formatting that was previously applied before it as saved.

#load("depression_data.RData")
load("C:/Users/mambaza/Desktop/UKZN/PGDM - Data Science/Semester 1/STAT606 - Applied Binary Classification and Matching/R Practicals/Practical 2 - Balancing/depression_data.RData")
summary(depression_data)

#Note the severe class imbalance for the target (Depression):

round(prop.table(table(depression_data$Depression))*100,2)


# We are now read to split the data into the training and test sets.

# ----------------------------------------------------------------------------- #
# 3. Specify df and target --------------------------------------------------- 
# ----------------------------------------------------------------------------- #

# Instead of writing code that only works for one dataset, we define named objects here that act as settings/place-holders. Then we reuse those throughout the code (rather than hard-coding):

df <- depression_data

target <- "Depression" 

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

summary(training_set[[target]]) # check the class imbalance in the training set

# ----------------------------------------------------------------------------- #
# 5. Data preprocessing  ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# Any data pre-processing (normalization and dummy variable encoding) must be applied BEFORE balancing.


# ----------------------------------------------------------------------------- #
# 6. Balancing of the training set ---------------------------------------------
# ----------------------------------------------------------------------------- #

# We will consider 4 different sampling techniques: under, over, combination of over and under, and SMOTE

##################### Under-sampling ###############################

# The process of undersampling counts the number of minority samples in the dataset (given by for formula for total_under below), then randomly selects the same number from the majority sample. In our case we would end up with 70 randomly chosen non-depression cases ("0") and the original 70 depression cases ("1") resulting in a 50:50 split.
# This has a major drawback as we are only using a very small % of the original dataset.

# save the number of MINORITY cases in an object call total_under:
total_under <- nrow(training_set[training_set[[target]] == "1", ])

train_under <- ovun.sample(
  as.formula(paste(target, "~ .")), # this formula specifies what the target is and the predictors
  data = training_set,
  method = "under",
  N = 2 * total_under, # new total = multiply by 2 for the two classes
  seed = seed
)

# Extract and save the resulting under-sampled data:
train_under_data <- train_under$data

summary(train_under_data[[target]])

############################## Over-sampling ##############################################

# This method repeatedly duplicates randomly selected minority classes until there are an equal number of majority and minority samples. It does have its drawback as the duplicates may lead to generalizing of the minority class.

# save the number of MAJORITY cases in an object call total_over:
total_over <- nrow(training_set[training_set[[target]] == "0", ])

train_over <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "over",
  N = 2 * total_over, # multiply by 2 for the twp classes
  seed = seed
)

# Extract and save the resulting under-sampled data:
train_over_data <- train_over$data
summary(train_over_data[[target]])

###################### Combination of over and under #################################

# We can apply a combination of both over- and under-sampling, where the number of minority
# cases increases and the number of majority cases decreases.

total_both <- nrow(training_set) # specify the total sample size after the procedure, this can be changed to any value
fraction_new <- 0.50 # specify the approx proportion of minority cases to be produced

train_both <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "both",
  N = total_both,
  p = fraction_new,
  seed = seed
)

# Extract and save the resulting data (list):
train_both_data <- train_both$data
summary(train_both_data[[target]])

####################################### SMOTE ##################################


# We use the SmoteClassif function which allows us to specify the method of determining the synthetic observations based on the nearest neighbours. We use the dist option to specify the method to use based on the type of data (see https://rdrr.io/cran/UBL/man/smoteClassif.html)

# The depression data has mixed attributes (numerical and categorical), so we use HEOM or HVDM

set.seed(seed)
train_smote_data <- SmoteClassif(
  as.formula(paste(target, "~ .")),
  training_set,
  C.perc = "balance", # minority and majority classes 
  k = 5, # number of nearest neighbours,
  dist = "HVDM"
)

summary(train_smote_data[[target]])

###############################################################################


########################### Which technique to use?? ##########################

# We can use multiple models on each technique to find the best performing model/technique combination but this is very time-consuming. So we can fit a base model instead (usually a simple model) to each balanced data set to find the technique that produces the best performing model (based on the selected metric). We then take the balanced set based on that technique and fit all of the competing models. A logistic regression model is generally the easiest to fit. We will once again use the H2O package for this:

# ----------------------------------------------------------------------------- #
# 7. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

# ----------------------------------------------------------------------------- #
# 8. Specify the attribute names ----------------------------------------------
# ----------------------------------------------------------------------------- #

predictors <- setdiff(names(training_set), target)

# ----------------------------------------------------------------------------- #
# 9.Repeatedly fit the LR model ----------------------------------------------
# ----------------------------------------------------------------------------- #

# Recall that we now have 4 balanced training sets and the original unbalanced set:
# "train_under_data" (based on under-sampling),
# "train_over_data" (based on over-sampling),
# "train_both_data" (based on a combination of over- and under-sampling)
# "train_smote_data" (based on SMOTE)
# "training_set" (the unbalanced original training set)

# We will iterate through each of them, fitting an LR and determine their performances.

# Prepare the results data frame to append the results of each data set:
Performance_comparison <- data.frame()

#################  THE CODE BELOW IS RE-RUN FOR EACH TRAINING SET  (see lecture recording) ############

# Let's create a variable to use for the name of the balanced training set which we can update rather than repeating the code for each training set:

### Loop through this code to get appended results, just change the name of the data set:  


train_set_name <- "train_over_data"

# train_set_name just stores the name of the data set, not the data set itself. To use the actual data frame referenced by its name, you can use get():

training_set_final <- get(train_set_name)

# convert to h2o data sets:

balanced_train_set_h2o <- as.h2o(training_set_final)
test_h2o <- as.h2o(test_set) # convert the test set created on line 43 to an H2O data set 


# Fit the logistic regression model
LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = balanced_train_set_h2o,
  family = "binomial", # logistic regression
  lambda = 0, # no regularization (like classical GLM)
  compute_p_values = TRUE # optional: get p-values
)

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_LR_train <- h2o.predict(LR, balanced_train_set_h2o)
preds_LR_test <- h2o.predict(LR, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_LR_pred <- cbind(training_set_final,
                       setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))  

test_LR_pred <- cbind(test_set,
                      setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob")) 


# ----------------------------------------------------------------------------- #
# 10. Save model performance results repeatedly for each data set --------------
# ----------------------------------------------------------------------------- #

# When training a classification model on a balanced dataset (achieved through oversampling techniques such as random oversampling or SMOTE), the model learns from an artificial class distribution that does not reflect reality. As a result, the predicted probabilities will be miscalibrated, typically inflated for the minority class. 

# If the goal is simply to predict a class label and the decision threshold has been appropriately tuned, correction may not be necessary. Similarly, if evaluation relies solely on AUC-ROC, calibration does not affect the ranking of predictions and correction can be omitted. 

# Therefore, we will use the AUC to compare the performance of the LR models fitted with the different datasets:

######### Extract and append results to a data frame ############

######### REPEAT THESE STEPS FOR ALL BALANCED DATA SETS ######################### 

# Step 1: Extract metrics from confusion matrices

# Training AUC
roc_LR_train <- roc(train_LR_pred[[target]], train_LR_pred$pred_prob)
auc_train <- auc(roc_LR_train)

# Test AUC
roc_LR_test <- roc(test_LR_pred[[target]], test_LR_pred$pred_prob)
auc_test <- auc(roc_LR_test)

# Step 2: Create individual rows for train and test
performance_train <- data.frame(
  technique = train_set_name,
  dataset = "train",
  auc = auc_train
)

performance_test <- data.frame(
  technique = train_set_name,
  dataset = "test",
  auc = auc_test
)

# Step 3: Append into one data frame (joins results from previous models each time)
Performance_comparison <- rbind(Performance_comparison,performance_train, performance_test, make.row.names = FALSE)


# Step 4: View results
View(Performance_comparison)
print(Performance_comparison)

#### Use the balanced data set that produces the highest AUC.

# ----------------------------------------------------------------------------- #
# 11. Exporting the balanced data set for later use ---------------------------
# ----------------------------------------------------------------------------- #

# You can save the balanced set (and test set) for later use (saves to your working directory):

save(
  train_over_data, # change this based on the optimal balanced data set
  test_set, # save test set too
  file = "Unbalanced and balanced data.RDATA"
)

# Then to use the saves data objects when you start a new R session, load all the saved objects into your RStudio environment:

load("Unbalanced and balanced data.RDATA")

# ----------------------------------------------------------------------------- #
# 12. Correcting probabilities for over- or under- sampling -------------------
# ----------------------------------------------------------------------------- #

# If predicted probabilities are meaningful in the context of the problem, for example in risk scoring or any decision that depends on the magnitude of the probability (such as predicted class labels), then correction is essential. 

# We have two options to obtain threshold-dependent metrics for a model (from a confusion matrix):
# 1. Tune the threshold used to make the class predictions, then probability correction is not required - this will be covered in the last practical (prac 5)
# 2. For random over- or under-sampling, a Bayes-based correction (see slide 48 in chapter 2) can be applied using the known true and artificial class proportions to rescale the predicted probabilities back to reflect the real-world distribution. 

#For SMOTE, this correction is less theoretically clean because synthetic interpolated samples distort not just the class prior but the feature space itself. In this case, probability calibration techniques such as Platt scaling or isotonic regression, applied on a held-out validation set, are more appropriate for recovering well-calibrated probability estimates.

# We will manually correct the probabilities using R code for any model that has been fitted using the balanced training set (over-, under-, combination)


# -------------------------------------------------->
# 1. Specify the name of the dataframe containing the combined training/test sets with predicted probabilities:
# -------------------------------------------------->

combined_train <- train_LR_pred
combined_test <- test_LR_pred

# -------------------------------------------------->
# 2. True class proportions from the ORIGINAL training set (before balancing)
# -------------------------------------------------->
p_1 <- prop.table(table(training_set[[target]]))["1"]   # true minority class proportion

# -------------------------------------------------->
# 3. Balanced class proportions AFTER over/under sampling
# -------------------------------------------------->
p_asterisk_1 <- prop.table(table(combined_train[[target]]))["1"]

# -------------------------------------------------->
# 4. Correction function 
# -------------------------------------------------->
probs_correction <- function(predicted_probs, p_1, p_asterisk_1) {
  
  numerator   <- predicted_probs * (p_1 / p_asterisk_1)
  denominator <- numerator + (1 - predicted_probs) * ((1 - p_1)/(1 - p_asterisk_1))
  
  corrected   <- numerator / denominator
  return(corrected)
  
}

# -------------------------------------------------->
# 5. Apply to training set
# -------------------------------------------------->

# combined_train$pred_prob are the raw predicted probabilities from your model

combined_train$corrected_probs <- probs_correction(combined_train$pred_prob,
                                                   p_1, 
                                                   p_asterisk_1)


# -------------------------------------------------->
# 6. Apply to test set
# -------------------------------------------------->

# combined_test$pred_prob are the raw predicted probabilities from your model

combined_test$corrected_probs <- probs_correction(combined_test$pred_prob,
                                                   p_1, 
                                                   p_asterisk_1)

# -------------------------------------------------->
# 7. Look at model performance on the training and test sets:
# -------------------------------------------------->

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.003 # for now, let's use a threshold close to the original prevalence

########## --> Determine the predicted class labels ----

# training
combined_train$pred_class <- factor(ifelse(combined_train$corrected_probs > threshold,"1","0"))

# test
combined_test$pred_class <- factor(ifelse(combined_test$corrected_probs > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
metrics_train <- caret::confusionMatrix(
                          combined_train$pred_class,
                          combined_train[[target]],
                          positive = "1",
                          mode = "everything"
                        )

# calculate Matthews Correlation Coefficient (MCC) for the training set
mcc_train <- mcc(combined_train$pred_class,  combined_train[[target]])

# Recall: MCC produces a value between −1 and +1, where 
# +1 indicates perfect prediction, 
# 0 indicates performance no better than random guessing, and 
# −1 indicates total disagreement between predictions and actual outcomes. 


# test

# predicted classes first then actual classes
metrics_test <- caret::confusionMatrix(
                  combined_test$pred_class,
                  combined_test[[target]],
                  positive = "1",
                  mode = "everything"
                )

# calculate Matthews Correlation Coefficient (MCC) for the test set
mcc_test <- mcc(combined_test$pred_class,  combined_test[[target]])

# -------------------------------------------------->
# 8. Save model performance
# -------------------------------------------------->

metrics_combined <- rbind(train = c(as.list(metrics_train$byClass),MCC = mcc_train),  
                          test = c(as.list(metrics_test$byClass),MCC = mcc_test))

# or save as a CSV file

write.csv(metrics_combined, "model_performance_metrics.csv", row.names = TRUE)

############## Shut down H2O cluster so it doesn't use up any more resources ############

h2o.shutdown(prompt = FALSE)
