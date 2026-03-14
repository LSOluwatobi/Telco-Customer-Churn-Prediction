# ==============================================================================
# PROJECT: Telco Customer Churn Prediction (Non-Parametric)
# Models: Random Forest, XGBoost, KNN, SVM
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: Setup
# ------------------------------------------------------------------------------
if(!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, randomForest, xgboost, e1071, pROC, kernlab)

# ------------------------------------------------------------------------------
# PHASE 2: Data Loading & Initial Cleaning
# ------------------------------------------------------------------------------
url <- "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
raw_churn <- read.csv(url, na.strings = c("", "NA"))

# Drop ID and handle the 11 missing TotalCharges rows
df <- raw_churn %>% dplyr::select(-customerID) %>% drop_na()
table(df$Churn)

# ------------------------------------------------------------------------------
# PHASE 3: Feature Engineering (One-Hot Encoding)
# ------------------------------------------------------------------------------
# XGBoost, KNN, and SVM require numeric inputs (Factors to Dummies)
# We use fullRank = T to avoid perfect multicollinearity
dummies <- dummyVars(Churn ~ ., data = df, fullRank = TRUE)
features_transformed <- data.frame(predict(dummies, newdata = df))

# Re-attach Target
final_df <- cbind(Churn = factor(df$Churn, levels = c("No", "Yes")), features_transformed)

# ------------------------------------------------------------------------------
# PHASE 4: Partitioning & Scaling
# ------------------------------------------------------------------------------
trainIndex <- createDataPartition(final_df$Churn, p = 0.7, list = FALSE)
train_set <- final_df[trainIndex, ]
test_set  <- final_df[-trainIndex, ]

# Pre-processing (Scaling) for KNN and SVM
scaler <- preProcess(train_set, method = c("center", "scale"))
train_scaled <- predict(scaler, train_set)
test_scaled  <- predict(scaler, test_set)

# ------------------------------------------------------------------------------
# PHASE 5: Model Training
# ------------------------------------------------------------------------------

# 1. Random Forest (Bagging)
fit_rf <- randomForest(Churn ~ ., data = train_set, ntree = 500)

# 2. XGBoost (Boosting) - Requires Matrix
dtrain <- xgb.DMatrix(data = train_x, label = train_y)
dtest  <- xgb.DMatrix(data = as.matrix(test_set[,-1]))
params <- list(
  objective = "binary:logistic", # This works inside xgb.train
  eval_metric = "auc"
)

# Train Model
fit_xgb <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 50,
  verbose = 0
)

p_xgb <- predict(fit_xgb, dtest)

# 3. KNN (Distance-based)
fit_knn <- train(Churn ~ ., data = train_scaled, method = "knn", tuneLength = 5)

# 4. SVM with RBF Kernel (Non-linear boundary)
fit_svm <- svm(Churn ~ ., data = train_scaled, probability = TRUE, kernel = "radial")

# ------------------------------------------------------------------------------
# PHASE 6: Model Evaluation & Comparison
# ------------------------------------------------------------------------------
p_rf  <- predict(fit_rf, test_set, type = "prob")[, "Yes"]
p_xgb <- predict(fit_xgb, as.matrix(test_set[,-1]))
p_knn <- predict(fit_knn, test_scaled, type = "prob")[, "Yes"]
p_svm <- attr(predict(fit_svm, test_scaled, probability = TRUE), "probabilities")[, "Yes"]

get_metrics <- function(probs, actual, label) {
  classes <- factor(ifelse(probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(classes, actual, positive = "Yes")
  roc_obj <- roc(actual, probs, quiet = TRUE)
  
  return(c(Model = label, Accuracy = cm$overall["Accuracy"], ROC_AUC = auc(roc_obj),
           Precision = cm$byClass["Precision"], Recall = cm$byClass["Recall"], F1 = cm$byClass["F1"]))
}

comparison_df <- rbind(
  get_metrics(p_rf, test_set$Churn, "Random Forest"),
  get_metrics(p_xgb, test_set$Churn, "XGBoost"),
  get_metrics(p_knn, test_scaled$Churn, "KNN"),
  get_metrics(p_svm, test_scaled$Churn, "SVM")
) %>% as.data.frame()

print(comparison_df)

# ------------------------------------------------------------------------------
# PHASE 7: Variable Importance
# ------------------------------------------------------------------------------
importance <- as.data.frame(importance(fit_rf))
print(importance %>% arrange(desc(MeanDecreaseGini)) %>% head(10))

# Transform comparison_df for plotting
plot_data <- comparison_df %>%
  mutate(across(Accuracy.Accuracy:F1.F1, as.numeric)) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Bar Chart: Accuracy vs Recall
ggplot(plot_data %>% filter(Metric %in% c("Accuracy.Accuracy", "Recall.Recall")), 
       aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  theme_minimal() +
  labs(title = "Model Comparison: Accuracy vs. Recall", y = "Score") +
  scale_fill_manual(values = c("#2c3e50", "#e74c3c"), labels = c("Accuracy", "Recall"))

# ROC Curves Comparison
plot(roc(test_set$Churn, p_rf), col="#1abc9c", lwd=2, main="ROC Curve: Churn Comparison")
plot(roc(test_set$Churn, p_xgb), col="#3498db", lwd=2, add=TRUE)
plot(roc(test_scaled$Churn, p_knn), col="#f1c40f", lwd=2, add=TRUE)
plot(roc(test_scaled$Churn, p_svm), col="#9b59b6", lwd=2, add=TRUE)
legend("bottomright", legend=c("Random Forest", "XGBoost", "KNN", "SVM"), 
       col=c("#1abc9c", "#3498db", "#f1c40f", "#9b59b6"), lwd=2)

# Variable Importance (The 'Business Story')
varImpPlot(fit_rf, n.var = 10, main = "Top 10 Drivers of Customer Churn")
