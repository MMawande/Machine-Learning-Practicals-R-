###################################################################
# STAT606 Practical 5: Variable importance and optimal cutoff     #         
###################################################################

# NB: to run this script, you must run the corresponding models from pracs 1, 3 and 4. Load the usual libraries and have an H2O cluster initiated. 

# Ensure you install any of the new packages first (uncomment the following and run):

#install.packages(c("iml","fastshap","shapviz","cutpointr"))

library(h2o)
library(caret)

# ----------------------------------------------------------------------------- #
# 1. Variable importance for some H2O models ----------------------------------------
# ----------------------------------------------------------------------------- #

# we need to supply the fitted model object - only some H2O models have built in variable importance for the function below o work (LR, NN, drf, gbm) An example will be done using the random forest from prac 4 where the model fitted was saved in drf;

model_object <- NN

### Variable importance for the NN (H2O object)

var_imp <- h2o.varimp(model_object)
View(var_imp)
h2o.varimp_plot(model_object)


# repeat the above - only for those H2O models that have been run and are in the global environment. 

# ----------------------------------------------------------------------------- #
# 2. Variable importance for a DT fitted in the Rpart function from prac 1 -----
# ----------------------------------------------------------------------------- #

varImp(DT_rpart)


# ----------------------------------------------------------------------------- #
# 3. Variable importance for SVM models fitted in train() ---------------------
# ----------------------------------------------------------------------------- #

# https://uc-r.github.io/iml-pkg

library(iml)

## This allows for model-agnostic variable importance of any model fitted. However, we will use it for the SVMs:

# specify the name of the object containing the SVM fitted model results:

model_name <- SVM_linear    # change this to SVM_poly or SVM_radial

predictor <- Predictor$new(model_name, 
                           data = as.data.frame(training_set_SVM[predictors]), 
                           y = training_set_SVM[[target]],  
                           type = "prob"
                           )

imp <- FeatureImp$new(predictor, loss = "ce")  # cross-entropy for classification
plot(imp)

# ----------------------------------------------------------------------------- #
# 4. SHAP plots for H2O tree-based ensemble models -----------------------------
# ----------------------------------------------------------------------------- #

# SHAP values in H2O are implemented using TreeSHAP, an algorithm developed specifically for tree-based ensemble models (drf, gbm). 

# ALWAYS use the TEST set for SHAP values!

h2o.shap_summary_plot(drf, # change the model according to the one you want to use
                      test_h2o) 

# ----------------------------------------------------------------------------- #
# 5. SHAP plots for models fitted using train() or rpart() ---------------------
# ----------------------------------------------------------------------------- #

# specify the name of the model  object that was fitted

model_name_2 <- SVM_poly # change to the other SVM models (SVM_radial, SVM_poly) or DT_rpart


library(fastshap)

shap_values <- fastshap::explain(
  object = model_name_2,
  X = test_set_final[, predictors],
  pred_wrapper = function(object, newdata) {
    predict(object, newdata, type = "prob")[,2]
  },
  nsim = 100
)


library(shapviz)

sv <- shapviz(
  shap_values,
  X = test_set_final[, predictors]
)

sv_importance(sv, kind = "beeswarm")


# ----------------------------------------------------------------------------- #
# 6. Optimal cutoff  ----------------------------------------------------------
# ----------------------------------------------------------------------------- #


library(cutpointr)
?cutpointr

# see https://cran.r-project.org/web/packages/cutpointr/vignettes/cutpointr.html for more metrics

# The cutpointr function requires the predicted probabilities, followed by the actual classes/response.

# Specify the training and test sets with the predicted probabilities from the the fitted model.

# We will use the example of random forest (drf):

train_results_df <- train_NN_pred

test_results_df <- test_NN_pred

### Let's use the predicted values for the training set from the GBM in prac 4:

cp <- cutpointr(test_results_df$pred_prob, 
                test_results_df[[target]], 
                pos_class = "1",
                method = maximize_metric, 
                metric = F1_score) # can change to youden, accuracy, F1_score, sensitivity etc.

# Youden- or J-Index = sensitivity + specificity - 1

summary(cp)

########## --> Specify threshold ----

# We can look at the performance of the model based on the obtained threshold in the object cp:

threshold <- cp$optimal_cutpoint # set it to the optimal cutoff


########## --> Determine the predicted class labels ----

# training
train_results_df$pred_class <- factor(ifelse(train_results_df$pred_prob > threshold,"1","0"))

# test
test_results_df$pred_class <- factor(ifelse(test_results_df$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 
caret::confusionMatrix(
  train_results_df$pred_class,
  train_results_df[[target]],
  positive = "1",
  mode = "everything"
)

# test
caret::confusionMatrix(
  test_results_df$pred_class,
  test_results_df[[target]],
  positive = "1",
  mode = "everything"
)














