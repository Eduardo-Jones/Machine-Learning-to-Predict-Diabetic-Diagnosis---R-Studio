
library(caret)
library(mice)
library(tidyverse)
library(randomForest)


diabetes0 = read.csv('diabetes.csv')

# check for missing values
sapply(diabetes0, function(x) sum(is.na(x)))

# Make a copy of original data
original = diabetes0
summary(diabetes0)
##### Label 0 -> No and 1 -> Yes #####
diabetes0$Outcome[diabetes0$Outcome==0] = "No"
diabetes0$Outcome[diabetes0$Outcome==1] = "Yes"
diabetes0$Outcome = as.factor(diabetes0$Outcome)
# change 0's to NA
diabetes0[, 2:7][diabetes0[, 2:7] == 0] <- NA
# check for missing values
sapply(diabetes0, function(x) sum(is.na(x)))
str(diabetes0)


# convert integer to numeric
#diabetes0 <- data.frame(lapply(diabetes0, as.numeric))
#str(diabetes0)

library(finalfit)
diabetes0 %>% 
  missing_plot()

# install.packages('naniar')
# install.packages('visdat')
library(visdat)
vis_miss(diabetes0)

# overview of missing missing data
#introduce(diabetes0)

diabetes0 = subset(diabetes0, select = -c(SkinThickness ,Insulin) )
summary(diabetes0)
# Impute missing values. cart = classification and regression trees
imputed_data <-  mice(diabetes0[,1:6], method="cart")
# A CART is a predictive algorithm that determines how a given variable’s values can be predicted based on other values.
# It is composed of decision trees where each fork is a split in a predictor variable and each node at the end has a prediction for the target variable.


# use the complete() function and assign to a new object. 
diabetes <- complete(imputed_data) 
sapply(diabetes, function(x) sum(is.na(x)))
summary(diabetes)
# add pregnancy and outcome back to data frame
#diabetes$Pregnancies <- diabetes0$Pregnancies
diabetes$Outcome <- diabetes0$Outcome

# Reorder df so Pregnancies is in the first col:
#diabetes <- diabetes[, c(6,1,2,3,4,5,7)]

summary(diabetes)

# split data
set.seed(100)
inTrain <- createDataPartition(as.matrix(diabetes[,7]), p = .8, list=FALSE)
diab.train = diabetes[inTrain,]
diab.test = diabetes[-inTrain,]


# preprocess
diab.TrainPP <- preProcess(diab.train, method = c("BoxCox", "scale", "center","nzv","spatialSign"))
diab.TestPP <- preProcess(diab.test, method = c("BoxCox", "scale", "center","nzv","spatialSign"))

# Predict
diabTrainTrans <- predict(diab.TrainPP, diab.train)
diabTestTrans <- predict(diab.TestPP, diab.test)

# change outcome variable to factor variable
#diabTrainTrans$Outcome <- as.factor(diabTrainTrans$Outcome)

# Change binary number to binary character x1 = 1 x2 = 2
#diabTrainTrans <- diabTrainTrans %>%
#  mutate(Outcome = factor(Outcome,
#                            labels = make.names(levels(Outcome))))
                            

pairs(diabTrainTrans)
cor(diabTrainTrans[,-7])

#control the computational nuances of the train function
set.seed(100)
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)
                     


#################### Logistic Regression #################### 
set.seed(5)
logisticTune <- train(x = as.matrix(diabTrainTrans[,1:6]), 
                      y = diabTrainTrans$Outcome,
                      method = "glm",
                      metric = "ROC",
                      trControl = ctrl)
logisticTune 


test_results <- data.frame(obs = diabTestTrans$Outcome,
                           logistic = predict(logisticTune, diabTestTrans))
                           

#################### PLSDA Regression #################### 
set.seed(476)
plsdaTune <- train(x = as.matrix(diabTrainTrans[,1:6]),
                   y = diabTrainTrans$Outcome,
                   method = "pls", 
                   metric = "ROC",
                   tuneGrid = expand.grid(.ncomp = 1:5),
                   trControl = ctrl)

plsdaTune
plot(plsdaTune)

#################### SVM Radial ########################### 
set.seed(476)
sigmaRangeReduced <- sigest(as.matrix(diabTrainTrans[,1:6]))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 4)))
svmRTune <- train(x = as.matrix(diabTrainTrans[,1:6]),
                 y = diabTrainTrans$Outcome,
                 method = "svmRadial", #svmLinear #svmPoly
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 tuneGrid = svmRGridReduced,
                 fit = FALSE,
                 trControl = ctrl)
svmRTune

test_results$svm <- predict(svmRTune, newdata=diabTestTrans[,-7])
# DOES NOT WORK
# SVMR_Results = postResample(pred=svmR_testresults, obs=diabTestTrans$Outcome)
# SVMR_Results


## Test Results

#SVM_Results

######################### Random Forest #####################################
mtryGrid <- data.frame(mtry = 1:6) #since we only have 6 predictors

### Tune the model using cross-validation
set.seed(476)
rfTune <- train(x = as.matrix(diabTrainTrans[,1:6]),
                y = diabTrainTrans$Outcome,
                method = "rf",
                metric = "ROC",
                tuneGrid = mtryGrid,
                ntree = 150,
                importance = TRUE,
                trControl = ctrl)
rfTune

plot(rfTune)


######### Predict the test set based on four models ###############

#logistic 
diabTestTrans$logistic <- predict(logisticTune,diabTestTrans, type = "prob")[,1]

#PLSDA
diabTestTrans$plsda <- predict(plsdaTune,diabTestTrans[,1:6], type = "prob")[,1]

#SVM
diabTestTrans$SVMR <- predict(svmRTune, diabTestTrans[,1:6], type = "prob")[,1]

#Random Forest
diabTestTrans$RF <- predict(rfTune,diabTestTrans, type = "prob")[,1]


######################### ROC curves ######################
library(pROC)

#ROC for logistic model
logisticROC <- roc(diabTestTrans$Outcome, diabTestTrans$logistic)
plot(logisticROC, col=1, lty=1, lwd=2)

#ROC for PLSDA
plsdaROC <- roc(diabTestTrans$Outcome, diabTestTrans$plsda)
lines(plsdaROC, col=3, lty=3, lwd=2)

#ROC for SVM
SVMROC <- roc(diabTestTrans$Outcome, diabTestTrans$SVMR)
lines(SVMROC, col=8, lty=8, lwd=2)

#ROC for Random Forest
RFROC <- roc(diabTestTrans$Outcome, diabTestTrans$RF)
lines(RFROC, col=4, lty=4, lwd=2)

legend('bottomright', c('logistic','plsda','SVM', 'Random Forest'), col=1:6, lty=1:5,lwd=2)


########## Create the confusion matrix from the test set ############
#Confusion matrix of logistic model
confusionMatrix(data = predict(logisticTune, diabTestTrans), reference = diabTestTrans$Outcome)

#Confusion matrix of partial least squares discriminant analysis
confusionMatrix(data = predict(plsdaTune, diabTestTrans), reference = diabTestTrans$Outcome)

#Confusion matrix of SVM
confusionMatrix(data = predict(svmRTune, diabTestTrans[,1:6]), reference = diabTestTrans$Outcome)

#Confusion matrix of Random Forest
confusionMatrix(data = predict(rfTune, diabTestTrans), reference = diabTestTrans$Outcome)


#We can add the models from chapter 12
res1 = resamples(list(Logistic = logisticTune, PLSDA = plsdaTune, 
                      SVM = SVMTune ))
dotplot(res1)




