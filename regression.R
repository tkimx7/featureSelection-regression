### --------------------------------------------------------------- #
### --- OBJECTIVE ------------------------------------------------- #
### --------------------------------------------------------------- #

# --- COMMENT: We compare 3 models, which are
#             (1) GLM without Cross Validation, that is without Feature Selection (cause of overfitting)
#             (2) LASSO
#             (3) Ridge Regression
# --- COMMENT: Ridge regression, though better than model (1), discourages sparse models with diminishing returns.
# --- COMMENT: Therefore, we demonstrate that LASSO is optimal for our dataset.

### --------------------------------------------------------------- #
### --- Part 0: Clear Workspace/Import Libraries ------------------ #
### --------------------------------------------------------------- #

rm(list = ls())

install.packages("dplyr")
install.packages("ggplot2")
install.packages("caret")
install.packages("glmnet")
install.packages("randomForest")

library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(randomForest)

### --------------------------------------------------------------- #
### --- Part I: Importing Data ------------------------------------ #
### --------------------------------------------------------------- #

train_data <- as.data.frame(read.csv("train.csv", header = TRUE))
test_data  <- as.data.frame(read.csv("test.csv",  header = TRUE))

### --------------------------------------------------------------- #
### --- Part 1.5: True Values ------------------------------------- #
### --------------------------------------------------------------- #

# --- COMMENT: The "true values" of life expectancy in the test data are missing.
# --- COMMENT: We assume that the submission example constitutes the true values.
# --- COMMENT: We see that the true data and test data "row names" are in the same order.

true_data <- as.data.frame(read.csv("submission_example.csv",  header = TRUE))
true_values <- unlist(true_data[,2])

### --------------------------------------------------------------- #
### --- Part II: Imputation (Missing Values) ---------------------- #
### --------------------------------------------------------------- #

max(train_data$X)
max(test_data$X)
sum(c(train_data$X, test_data$X) %in% 1:461) == max(train_data$X)
dim(test_data)[1]/(dim(train_data)[1] + dim(test_data)[1])

# --- COMMENT: The "row names" in column "X" from the training and test datasets combined yield 1 to 461.
# --- COMMENT: It seems that the training and test data sets are from a single data source partitioned 8:2 randomly.
# --- COMMENT: We could simply combine the train and test datasets to impute missing values with overall mean for each column.
# --- COMMENT: But in the case data sources are different, missing values in test data are replaced with means from training data.
# --- COMMENT: To account for this case, we impute missing values in test data with means of TRAINING data.

# --- COMMENT: We leave out panel data analysis/time series for dataset is cross-sectional.
# --- COMMENT: There are no missing values for factors, i.e. "categorical variables", in both train and test datasets.
# --- COMMENT: In time series, we impute missing values via linear interpolation, brown_bridge, etc.
# --- COMMENT: In this case, we impute missing values with mean.

# --- COMMENT: Lots of missing values with inflation, especially weekly data.

dim(train_data)
names(train_data)
head(train_data)

classes <- vector()
categorical_missing <- vector()
j <- 1
numerical_missing <- vector()
numerical_names <- vector()
k <- 1

for (i in 1:dim(train_data)[2]) {

  classes[i] <- class(train_data[,i])
 
  if (classes[i] == "factor") {
 
   categorical_missing[j] <- sum(is.na(train_data[,i]))
   j <- j + 1
  }
 
  if (classes[i] == "integer" | classes[i] == "numeric") {
   numerical_names[k] <- names(train_data)[i]
   numerical_missing[k] <- sum(is.na(train_data[,i]))
   k <- k + 1
  }
}

numerical_names <- numerical_names[!(numerical_names == "X")]
train_all <- train_data

for (i in 1:length(numerical_names)) {

  train_temp <- as.vector(unlist(train_data %>% select(numerical_names[i])))
  missing_index <- which(is.na(train_temp))
  train_temp[missing_index] <- mean(train_temp, na.rm = TRUE)
  train_all[,names(train_all) == numerical_names[i]] <- train_temp
}

sum(is.na(train_all))

dim(test_data)
names(test_data)
head(test_data)

classes_test <- vector()
categorical_missing_test <- vector()
j <- 1
numerical_missing_test <- vector()
numerical_names_test <- vector()
k <- 1

for (i in 1:dim(test_data)[2]) {

  classes_test[i] <- class(test_data[,i])
 
  if (classes_test[i] == "factor") {
 
   categorical_missing_test[j] <- sum(is.na(test_data[,i]))
   j <- j + 1
  }
 
  if (classes_test[i] == "integer" | classes_test[i] == "numeric") {
   numerical_names_test[k] <- names(test_data)[i]
   numerical_missing_test[k] <- sum(is.na(test_data[,i]))
   k <- k + 1
  }
}

numerical_names_test <- numerical_names_test[!(numerical_names_test == "X")]
test_all <- test_data

for (i in 1:length(numerical_names_test)) {

  train_temp <- as.vector(unlist(train_data %>% select(numerical_names_test[i])))    
  test_temp <- as.vector(unlist(test_data %>% select(numerical_names_test[i])))
  missing_index_test <- which(is.na(test_temp))
  test_temp[missing_index_test] <- mean(train_temp, na.rm = TRUE)
  test_all[,names(test_all) == numerical_names_test[i]] <- test_temp
}

sum(is.na(test_all))

### --------------------------------------------------------------- #
### --- Part 2.5: Internet Users ---------------------------------- #
### --------------------------------------------------------------- #

# --- COMMENT: X per Y people ought to be quantified X/Y.
# --- COMMENT: We use regular expressions to text mine the integers preceding 100 or 1000.

train_df <- train_all
test_df <- test_all

train_internet_users_temp <- vector()
test_internet_users_temp <- vector()

for (i in 1:dim(train_df)[1]) {

    if (train_df$internet_users[i] == "unknown") {
       
        train_internet_users_temp[i] <- 0
        next
    }
   
    train_internet_users_temp[i] <-
    as.numeric(unlist(strsplit(gsub("people", "", gsub("per", "", train_df$internet_users[i])), "[[:blank:]]+")))[1]/
    as.numeric(unlist(strsplit(gsub("people", "", gsub("per", "", train_df$internet_users[i])), "[[:blank:]]+")))[2]
}

for (i in 1:dim(test_df)[1]) {

    if (test_df$internet_users[i] == "unknown") {
       
        test_internet_users_temp[i] <- 0
        next
    }
   
    test_internet_users_temp[i] <-
    as.numeric(unlist(strsplit(gsub("people", "", gsub("per", "", test_df$internet_users[i])), "[[:blank:]]+")))[1]/
    as.numeric(unlist(strsplit(gsub("people", "", gsub("per", "", test_df$internet_users[i])), "[[:blank:]]+")))[2]
}

train_df$internet_users <- train_internet_users_temp
test_df$internet_users <- test_internet_users_temp

### --------------------------------------------------------------- #
### --- Part III: Feature Selection/Extraction -------------------- #
### --------------------------------------------------------------- #

# --- COMMENT: We decide whether to proceed with either feature selection or extraction.
# --- COMMENT: Feature selection is the riddance of redundant parameters, e.g. multicollinear variables.
# --- COMMENT: Feature extraction is the linear combination of all parameters without loss of information, e.g. PCA.
# --- COMMENT: PCA linearly transforms all quantitative features by weight of eigenvalues (variance).
# --- COMMENT: Multi-Factor Analysis is PCA but for both quantitative and categorical variables.
# --- COMMENT: As a regression problem in our case, we proceed with feature selection for cross-validation purposes.
# --- COMMENT: Step-wise penalized regressions, e.g. LASSO and Ridge, are available with the "glmnet" package in R.
# --- COMMENT: Cross validation and therefore feature selection is part of these methods.
# --- COMMENT: But here in this step for visualization purposes, we perform an explicit feature selection procedure.

### --------------------------------------------------------------- #
### --- Part 3.5: Mean Absolute Error ----------------------------- #
### --------------------------------------------------------------- #

mean_absolute_error <- function(data1, data2, n) {
   
    MAE <- sum(abs(data1 - data2))/n
    return(MAE)
}

### --------------------------------------------------------------- #
### --- Part IV: Overfitted General Linear Model ------------------ #
### --------------------------------------------------------------- #

# --- COMMENT: A GLM without cross-validation generally overfits as accounting for all predictors naturally increases variance.

overfit_model <- lm(life_expectancy ~. - X, data = train_df)
overfit_predict <- predict(overfit_model, newdata = test_df)

MAE_overfit <- mean_absolute_error(true_values, overfit_predict, 100)

### --------------------------------------------------------------- #
### --- Part 4.5: LASSO/Ridge Predictors -------------------------- #
### --------------------------------------------------------------- #

# --- COMMENT: Standard glm/lm function in R accounts for dummy variables, but penalized regressions via glmnet package do not.
# --- COMMENT: Therefore, we must manually create dummy (categorical) variables used for both LASSO and Ridge Regressions.
# --- COMMENT: In the case matrix dimensions are large, we use regular expressions to list out all column names.
# --- COMMENT: For our case, manual listing is sufficient.

categorical_names <- names(train_df)[!(names(train_df) %in% numerical_names)]

train_categorical <- as.data.frame(model.matrix(life_expectancy ~
             national_income +
             mobile_subscriptions +
             internet_users +
             improved_sanitation +
             women_parliament_seats_rate, data = train_df)[, -1])

train_predictors <- data.frame(train_categorical, train_df[,names(train_df) %in% numerical_names_test])
train_predictors <- train_predictors[,!(names(train_predictors) %in% c("mobile_subscriptionsmore.than.3.per.person"))]

test_categorical <- as.data.frame(model.matrix(true_values ~
             national_income +
             mobile_subscriptions +
             internet_users +
             improved_sanitation +
             women_parliament_seats_rate, data = test_df)[, -1])

test_predictors <- data.frame(test_categorical, test_df[,names(train_df) %in% numerical_names_test])

### --------------------------------------------------------------- #
### --- Part V: L1 Lasso Regression ------------------------------- #
### --------------------------------------------------------------- #

# --- COMMENT: alpha = 1 for Lasso
# --- COMMENT: Assume lambda is constant for both Lasso and Ridge, i.e. same uncertainty of priors.

lasso_model <- caret::train(y = train_df$life_expectancy,
                 x = train_predictors,
                 method = 'glmnet',
                 tuneGrid = expand.grid(alpha = 1, lambda = 5))

lasso_predict <- predict.train(lasso_model, test_predictors)

MAE_lasso <- mean_absolute_error(true_values, lasso_predict, 100)

### --------------------------------------------------------------- #
### --- Part VI: L2 Ridge Regression ------------------------------ #
### --------------------------------------------------------------- #

# --- COMMENT: alpha = 0 for Ridge
# --- COMMENT: Assume lambda is constant for both Lasso and Ridge, i.e. same uncertainty of priors.

ridge_model <- caret::train(y = train_df$life_expectancy,
                 x = train_predictors,
                 method = 'glmnet',
                 tuneGrid = expand.grid(alpha = 0, lambda = 5))

ridge_predict <- predict.train(ridge_model, test_predictors)

MAE_ridge <- mean_absolute_error(true_values, ridge_predict, 100)

### --------------------------------------------------------------- #
### --- Conclusion and writing predictions to CSV files ----------- #
### --------------------------------------------------------------- #

# LASSO is superior, which can be true for sparse matrices.

MAE_overfit #14.4296837763496
MAE_ridge #14.2839492870866
MAE_lasso #13.1651388098435

write.csv(cbind(test_data$X, overfit_predict), "glm_predict.csv")
write.csv(cbind(test_data$X, ridge_predict), "ridge_predict.csv")
write.csv(cbind(test_data$X, lasso_predict), "lasso_predict.csv")

