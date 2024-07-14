# college

MODEL BUILDING
Code for all the models:
# Load necessary libraries
library(caret)
library(glmnet)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Data Preparation
# Print the structure of the data frame
print(str(mobile))

# Convert RAM and Battery to numeric by extracting the numeric values
mobile$Ram <- as.numeric(gsub("[^0-9.]", "", mobile$Ram))
mobile$Battery <- as.numeric(gsub("[^0-9.]", "", mobile$Battery))

# Convert Display to numeric by extracting the numeric values
mobile$Display <- as.numeric(gsub("[^0-9.]", "", mobile$Display))

# Convert Price to numeric by removing commas
mobile$Price <- as.numeric(gsub(",", "", mobile$Price))

# Handle missing values
mobile[is.na(mobile)] <- 0

# Convert 'No_of_sim' to character
mobile$No_of_sim <- as.character(mobile$No_of_sim)

# Identify numeric and categorical columns
numeric_cols <- sapply(mobile, is.numeric)
categorical_cols <- !numeric_cols & sapply(mobile, function(x) is.character(x) || is.factor(x))

# Convert categorical variables to factors (excluding 'No_of_sim')
for (col in names(mobile)[categorical_cols]) {
  if (col != "No_of_sim") {
    mobile[[col]] <- as.factor(mobile[[col]])
  }
}

# Print column types for verification
print(sapply(mobile, class))

# Split data into training and testing sets
trainIndex <- createDataPartition(mobile$Price, p = 0.7, list = FALSE)
trainData <- mobile[trainIndex, ]
testData <- mobile[-trainIndex, ]

# Function to perform one-hot encoding
one_hot_encode <- function(data, column) {
  unique_values <- unique(data[[column]])
  for (value in unique_values) {
    new_col_name <- paste0(column, "_", make.names(value))
    data[[new_col_name]] <- as.integer(data[[column]] == value)
  }
  data[[column]] <- NULL
  return(data)
}

# Apply one-hot encoding to 'No_of_sim' for both train and test data
trainData <- one_hot_encode(trainData, "No_of_sim")
testData <- one_hot_encode(testData, "No_of_sim")

# Ensure all one-hot encoded columns exist in both datasets
all_columns <- union(names(trainData), names(testData))
for (col in all_columns) {
  if (!(col %in% names(trainData))) {
    trainData[[col]] <- 0
  }
  if (!(col %in% names(testData))) {
    testData[[col]] <- 0
  }
}

# Ensure column order is the same in both datasets
trainData <- trainData[, all_columns]
testData <- testData[, all_columns]

# Single Linear Regression
single_lm <- lm(Price ~ Ram, data = trainData)
summary(single_lm)


# Prepare matrix for Lasso and Ridge regression
x <- model.matrix(Price ~ ., trainData)[,-1]
y <- trainData$Price

# Lasso Regression
lasso <- cv.glmnet(x, y, alpha = 1)
best_lambda <- lasso$lambda.min
lasso_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(lasso_model)

# Ridge Regression
ridge <- cv.glmnet(x, y, alpha = 0)
best_lambda_ridge <- ridge$lambda.min
ridge_model <- glmnet(x, y, alpha = 0, lambda = best_lambda_ridge)
coef(ridge_model)

# Predictions and Model Evaluation
test_x <- model.matrix(Price ~ ., testData)[,-1]
test_y <- testData$Price

# Single Linear Regression Predictions
single_pred <- predict(single_lm, testData)
single_rmse <- sqrt(mean((single_pred - test_y)^2))

# Lasso Predictions
lasso_pred <- predict(lasso_model, newx = test_x)
lasso_rmse <- sqrt(mean((lasso_pred - test_y)^2))

# Ridge Predictions
ridge_pred <- predict(ridge_model, newx = test_x)
ridge_rmse <- sqrt(mean((ridge_pred - test_y)^2))

# Print RMSE values
print(paste("Single Linear Regression RMSE:", single_rmse))
print(paste("Lasso Regression RMSE:", lasso_rmse))
print(paste("Ridge Regression RMSE:", ridge_rmse))
