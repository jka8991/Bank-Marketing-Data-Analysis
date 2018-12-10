
### Required Packages
print("Install or load required packages")
library("dplyr")
library("ggplot2")
library("data.table")
library("VIM")
library("mice")
library("mlr")
library("parallelMap")
library("knitr")
library("xgboost")
library("glmnet")
library("caTools")
library("randomForest")


file_loc <- '~/bank-marketing-data/'
file_name <- 'bank-additional-full.csv'

### load data
dataset <- 
  fread(paste0(file_loc, file_name), stringsAsFactors = TRUE) %>%  
  as.data.frame()
dataset %>% glimpse()
##############################
### 1.a Missing values
###   Replace 'unknow' to NA
##############################
is.na(dataset) %>% sum()      # Number of na's
colSums(dataset=='unknown')   # Number of 'unknown' values

# save the columns that have 'unknown'
col_unknown <-
  which(colSums(dataset=='unknown')>0) %>% 
  as.data.frame %>%
  rownames()
dataset[dataset=='unknown'] = NA   # replace 'unknown' with NA

# drop the unused factors
dataset[col_unknown] <- droplevels(dataset[col_unknown])

natable <- colSums(is.na(dataset)) # table for the number of NAs
natable <- natable[natable!=0]
barplot(natable,legend.text = rownames(natable), main="missing (dataset)") # plot for missig vlalues

# Missing values plots
aggr_dataset <- aggr(x=dataset, sortVar=TRUE,
                     cex.axis=.7, gap=2, col=c("green", "red"), 
                     ylab=c("% missing (All customer)","Combinations (All customer)" ))

# replace response variable -> 'yes' = 1, 'no' = 0 as factor
dataset$y <- ifelse(dataset$y=='yes',1 , 0) %>% as.factor
table(dataset$y)/sum(table(dataset$y))


# split data into new and old customers
old_customer_data<-subset(dataset, dataset$poutcome != "nonexistent")
new_customer_data<-subset(dataset, dataset$poutcome == "nonexistent")

# % of 0 and 1 
table(dataset$y)/sum(table(dataset$y))
table(old_customer_data$y)/sum(table(old_customer_data$y))
table(new_customer_data$y)/sum(table(new_customer_data$y))

######################################################################
### 2. Data impuation
######################################################################
dataset %>% glimpse
# table for NAs
na_table <- colSums(is.na(dataset))
na_table <- na_table[na_table!=0]
barplot(na_table, legend.text = rownames(na_table), main="missing values")
table(dataset$y)
# clear summary of the features using mlr package 
summarizeColumns(dataset) %>% kable(digits=2)

# data imputation using 'mice'
# use m=1 and maxit=1 for the simplicity

# a) apply the imputation to dataset 
data_imputer <- mice(data=dataset, m=1, method="cart", maxit=1)
imp_data <- complete(data_imputer)
colSums(is.na(imp_data) )
aggr_imp_data <- aggr(x=imp_data, sortVar=TRUE,
                      cex.axis=.7, gap=2, col=c("green","black"))


# # b) apply the imputation to seperately to old an new customer data
# old_imputer <- mice(data=old_customer_data, m=1, method="cart", maxit=1)
# old_imp_data <- complete(old_imputer)
# 
# new_imputer <- mice(data=new_customer_data, m=1, method="cart", maxit=1)
# new_imp_data <- complete(new_imputer)
# 
# imp_data1 <- bind_rows(old_imp_data, new_imp_data)
# # shuffle the rows
# imp_data1 <- imp_data1[sample(nrow(imp_data1)),]

########################################
### 2. Some Visualizations
hist_duration <- ggplot(imp_data, aes(x=duration)) + 
  geom_histogram(bins=50, col="black", fill="white") +
  labs(title="hist of duration",
       x="duration",
       y="frequency")
hist_duration
print("Duration has a postive skewness")

hist_age <- ggplot(imp_data, aes(x=age)) + 
  geom_histogram(bins=50, col="black", fill="white") +
  facet_wrap(~y) +
  labs(title="hist of age",
       x="age",
       y="frequency") +
  geom_vline(xintercept=mean(imp_data$age), linetype="dotted",
             size=1.5, col="red")
hist_age
print("Age has a postive skewness")

hist_job <- ggplot(imp_data, aes(x=job, fill=y)) + 
  geom_histogram(stat="count", col="black") +
  facet_wrap(~y) +
  labs(title="hist of job",
       x="job",
       y="frequency") +
  theme(axis.text.x=element_text(angle = 45, hjust = 1))
hist_job

hist_edu <- ggplot(imp_data, aes(x=education, fill=y)) +
  geom_histogram(stat='count', col="black") +
  facet_wrap(~y) +
  labs(title="hist of education",
       x="education",
       y="frequency") +
  theme(axis.text.x=element_text(angle = 45, hjust = 1))
hist_edu

##################################
### 3. Data preprocess
###     - Normalize and create dummies
prep_data <-
  imp_data %>% 
  normalizeFeatures(target="y", method="standardize") %>% 
  createDummyFeatures(target="y")

### Split the dataset
set.seed(123)
split <- sample.split(prep_data$y, SplitRatio = 0.8)
training_set <- subset(prep_data, split == TRUE)
test_set <- subset(prep_data, split == FALSE)

##########################################################
### create Tasks
trainTask <- makeClassifTask(data=training_set,  
                             target="y")
testTask <- makeClassifTask(data=test_set,
                            target="y")
print(trainTask)
### Using 4-folds cross validation 
cv_folds <- makeResampleDesc(method="CV", iters=4)
print(cv_folds)
### Tune the parameters using random search,
### Using 3 iteration for the simplicity for now
control_random <- makeTuneControlRandom(maxit=1)


############################################################
### 3a. Models:XGboost
############################################################
# create Learner
xgb_learner <- makeLearner(cl="classif.xgboost", 
                           id="xgb_clf",
                           predict.type = "response")
# Set the tuning parameters
xgb_param_set<- makeParamSet(
  # makeIntegerParam("nrounds", lower=100, upper=1000),
  makeDiscreteParam("nrounds", values = c(300,500,1000)),
  makeNumericParam("eta", lower=0.001, upper=0.6),
  makeNumericParam("gamma", lower=0, upper=10),
  makeNumericParam("lambda", lower=0, upper=10),
  makeIntegerParam("max_depth", lower=1, upper=10),
  makeNumericParam("min_child_weight", lower=1, upper=5),
  makeNumericParam("subsample", lower=0.1, upper=0.8),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8),
  makeDiscreteParam("verbose", values = 0, tunable = FALSE)
)


parallelStartSocket(8)
# Tune the parameters using random search,
xgb_tuned <- tuneParams(learner = xgb_learner,
                        task = trainTask,
                        resampling = cv_folds,
                        par.set = xgb_param_set,
                        control = control_random,
                        measure = acc,
                        show.info = TRUE)
parallelStop()
# result for tuning
print("XGboost tuning results:")
print(xgb_tuned)
# set the parameters
xgb_tuned_model <- setHyperPars(xgb_learner,
                                par.vals = xgb_tuned$x)
# train the model to get final model
xgb_best_model <- train(xgb_tuned_model,
                        task = trainTask)
# test the model with test set
xgb_mod <- getLearnerModel(xgb_best_model)
xgb_pred <- predict(xgb_best_model, testTask)$data$response

# results and importance plot
# confusion matrix
xgb_cm <- table(true=test_set$y, pred=xgb_pred)
# misclassification rate
xgb_error <- 1 - sum(diag(xgb_cm))/sum(xgb_cm)
xgb_res <- (list(confusion_matrix=xgb_cm, misclassification=xgb_error))
print(xgb_res)
# importance plot
cols <- prep_data %>% colnames()
xgb.importance(cols, model = xgb_mod) %>%
  xgb.plot.importance(top_n = 25)


############################################################
### 3b. LASSO
############################################################
### build the lasso model using cv.glmnet
### it gives the lambda (regularization parameter)
lasso_model <- cv.glmnet(y=training_set$y, 
                         x=training_set %>% select(-y) %>% as.matrix(), 
                         nfolds=4,
                         family="binomial")
# plot for lambdas
plot(lasso_model)
# coefficients of the model
coef(lasso_model)
lasso_pred <- predict(lasso_model, 
                      s=lasso_model$lambda.1se,
                      newx=test_set %>% select(-y) %>% as.matrix(),
                      type="class") %>% as.vector() %>% as.numeric()
# confustion matrix
lasso_cm <- table(true=test_set$y, pred=lasso_pred)
# misclassification rate
lasso_error <- 1 - sum(diag(lasso_cm))/sum(lasso_cm)
lasso_res <- (list(confusion_matrix=lasso_cm, misclassification=lasso_error))
### plot lambda's
print(lasso_res)


############################################################
### 3c. ksvm ( using radial kernel ) 
############################################################
getParamSet("classif.ksvm") 
ksvm_learner <- makeLearner(cl = "classif.ksvm", 
                            id = "ksvm_clf",
                            predict.type = "response")
ksvm_param_set <- makeParamSet(
  makeNumericParam("C", lower = 0, upper = 10), #cost parameters
  makeNumericParam("sigma", lower=0.0001, upper=15),#RBF Kernel Parameter
  makeLogicalLearnerParam("scaled", default = FALSE))

parallelStartSocket(8)
ksvm_tuned <- tuneParams(learner=ksvm_learner, 
                         task = trainTask, 
                         resampling = cv_folds, 
                         par.set = ksvm_param_set, 
                         control = control_random,
                         measures = acc)
parallelStop()
print("ksvm tuning results: ")
print(ksvm_tuned)
ksvm_tuned_model <- setHyperPars(ksvm_learner, par.vals = ksvm_tuned$x)
ksvm_best_model <- train(ksvm_tuned_model, trainTask)
ksvm_pred <- predict(ksvm_best_model, testTask)$data$response

ksvm_cm <- table(true=test_set$y, pred=ksvm_pred)
ksvm_error <- 1 - sum(diag(ksvm_cm))/sum(ksvm_cm)
ksvm_res <- (list(confusion_matrix=ksvm_cm, misclassification=ksvm_error))
print(ksvm_res)

############################################################
### 3d. random forest
############################################################
getParamSet("classif.randomForest") 
rf_learner <- makeLearner(cl = "classif.randomForest", 
                          id = "rf_clf",
                          predict.type = "response")
rf_param_set <- makeParamSet(
  # makeIntegerParam("ntree", lower=200, upper=1200),
  makeDiscreteParam("ntree", values = c(300,500,1000)),  # number of trees
  makeIntegerParam("mtry", lower = 1, upper = ncol(imp_data)-1 ), # number of variables selected at each split
  makeIntegerParam("nodesize", lower=5, upper=50)  # minimum size of terminal nodes
)
parallelStartSocket(8)
rf_tuned <- tuneParams(learner=rf_learner, 
                       task = trainTask, 
                       resampling = cv_folds, 
                       par.set = rf_param_set, 
                       control = control_random,
                       measures = acc)
parallelStop()
print("rf tuning results: ")
print(rf_tuned)
rf_tuned_model <- setHyperPars(rf_learner, par.vals = rf_tuned$x)
rf_best_model <- train(rf_tuned_model, trainTask)
rf_pred <- predict(rf_best_model, testTask)$data$response

### results and importance plot 
rf_cm <- table(true=test_set$y, pred=rf_pred)
rf_error <- 1 - sum(diag(rf_cm))/sum(rf_cm)
rf_res <- (list(confusion_matrix=rf_cm, misclassification=rf_error))
print(rf_res)

rf_mod <- getLearnerModel(rf_best_model)
varImpPlot(rf_mod)

############################################################
### 4. results
############################################################
data.frame(xgb = xgb_res$misclassification,
           lasso = lasso_res$misclassification,
           ksvm = ksvm_res$misclassification,
           rf = rf_res$misclassification)


############################################################
### 4. Model comparisons using random splits
############################################################
# calculate the misclssficiation rate
mis_error <- function(true_y, pred_y){
  cm <- table(true=true_y, pred=pred_y)
  return(1 - sum(diag(cm))/sum(cm))
}
# print model
print_msg <- function(model_name, r){
  print(paste0("<<< ",model_name," >>>  #",r," -done"))
}

# test the final models with random split 
n_times <- 5
methods <- c("xgb", "lasso", "ksvm", "rf")
res_df <- matrix(NA, nrow=n_times, ncol=length(methods))
colnames(res_df) <- methods
for( r in (1:n_times) ){
  split <- sample.split(prep_data$y, SplitRatio = 0.8)
  training_set <- subset(prep_data, split == TRUE)
  test_set <- subset(prep_data, split == FALSE)
  
  trainTask <- makeClassifTask(data = training_set, target="y")
  testTask <- makeClassifTask(data = test_set, target="y")
  
  xgb_best_model <- train(xgb_tuned_model,
                          task = trainTask)
  xgb_pred <- predict(xgb_best_model, testTask)$data$response
  print_msg("XGB",r)
  lasso_model <- cv.glmnet(y=training_set$y, 
                           x=training_set %>% select(-y) %>% as.matrix(), 
                           nfolds=4,
                           family="binomial")
  lasso_pred <- predict(lasso_model, 
                        s=lasso_model$lambda.1se,
                        newx=test_set %>% select(-y) %>% as.matrix(),
                        type="class") %>% as.vector() %>% as.numeric()
  print_msg("LASSO",r)
  ksvm_best_model <- train(ksvm_tuned_model, trainTask)
  ksvm_pred <- predict(ksvm_best_model, testTask)$data$response
  print_msg("KSVM",r)
  rf_best_model <- train(rf_tuned_model, trainTask)
  rf_pred <- predict(rf_best_model, testTask)$data$response
  print_msg("RF",r)
  res_df[r,1] <- mis_error(test_set$y, xgb_pred)
  res_df[r,2] <- mis_error(test_set$y, lasso_pred)
  res_df[r,3] <- mis_error(test_set$y, ksvm_pred)
  res_df[r,4] <- mis_error(test_set$y, rf_pred)
}


##########################################
# Box plot - misclassification rates
par(mfrow=c(1,2))
boxplot((res_df), las=2, main="Test Error \n (misclassification rate)")
boxplot(sqrt(res_df), las=2, main="Test Error \n (sqrt(misclassification rate))")

##########################################
# scaled Box plot - misclassificatipn rats
rescaled_res_df <- res_df/apply(X=res_df, MARGIN=1, FUN=min)
# Box Plot
par(mfrow=c(1,2))
boxplot((rescaled_res_df), las=2, 
        main="Misclassification rate \n (Re-scaled)")

boxplot(sqrt(rescaled_res_df), las=2, 
        main="Misclassification rate \n sqrt(Re-scaled)")


