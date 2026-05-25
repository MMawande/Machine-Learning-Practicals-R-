##########################################################################################
#                                                                                        #
#                                  H2O Package                                           #
#                                                                                        #
##########################################################################################

# H2O is an open-source, high-performance machine learning and AI platform. The R package provides an interface to the H2O backend engine, which runs in Java and is designed for distributed in-memory computing — making it extremely fast, even on large datasets.

# Supports many ML models:	GLMs, random forests, gradient boosting, deep learning (feed forward NNs only), naive Bayes, etc.
# AutoML	Automatically trains and tunes multiple models, stacking the best ones
# Handles imbalanced data	Built-in class balancing, weighted models
# Model interpretation: Variable importance, SHAP values, partial dependence plots
# Scales well:	Efficient with big datasets, supports multicore and distributed setups
# Cross-validation:	Built-in K-fold CV, early stopping, and grid search
# Supports regression, classification, clustering	Versatile across problem types

################################## Installing and loading H2O #############################

install.packages("h2o") # this only needs to be run once on your laptop
library(h2o)

# Start the H2O cluster to check that the installation worked:

# In H2O, starting a cluster means starting the H2O backend engine, which runs in Java. Even if you’re just using it on your laptop, H2O still sets up a local "cluster" — a computing environment that can process data and train models in memory. "In memory" means that your data and computations are stored and processed directly in your computer’s RAM (Random Access Memory), not on disk (like your hard drive or SSD). Accessing data from RAM is much faster than reading/writing to disk.

# Ensure you have the 64 bit version of JAVA: See this document on how to download and install it: https://drive.google.com/file/d/1nICPoxk45M26m9hhZgjdMTqxQdSyqZfk/view?usp=sharing


# start the cluster (ignore the warning message,  above it it should show ;Starting H2O JVM and connecting:  Connection successful!)
h2o.init()



############## Shut down H2O cluster so it doesn't use up any more resources ############

h2o.shutdown(prompt = FALSE)




