library(e1071)
library(kernlab)
library(randomForest)
library(mgcv)
library("verification")
library(nnet)
library(mgcv)
library(rpart)
library(tree)
library(maptree)
library(MASS)
library(ROCR)
setwd("C:/Coursework/01_Data_Mining/Project/data/")


# Read data
snet = read.csv("train_with_diff.csv", header = T)

snet_ab = snet[13:34]

d=data.frame(matrix(NA,nrow=nrow(snet),ncol=1+(ncol(snet_ab)/2)))
d[,1] = snet[,1]

half = ncol(snet_ab)/2

for (i in 1:half) {
  d[,(i+1)] = log(snet_ab[,i]+1) - log(snet_ab[,i+half]+1)
}

snet = d
snet$X1 = as.factor(snet$X1)
# Sampling data
set.seed(620345)
snet_sample = sample(nrow(snet), nrow(snet) * 0.90)
snet_train=snet[snet_sample, ]
snet_test=snet[-snet_sample, ]
str(snet_train)

# 1.GLM 

snet_glm=glm(X1~ ., family=binomial, data=snet_train)
summary(snet_glm)
AIC(snet_glm)
BIC(snet_glm)
deviance_glm=snet_glm$deviance/snet_glm$df.residual
deviance_glm

# 1.1 Stepwise variable selection
snet_step <- step(snet_glm,k=log(nrow(snet_train)), direction = c("both"))
summary(snet_step)
AIC(snet_step)
BIC(snet_step)

snet_step$deviance/snet_step$df.residual


# 1.2 Search grid to find the optimal cut off probability 
searchgrid = seq(0.01, 0.99, 0.01)  
result = cbind(searchgrid, NA)  
cost1 <- function(r, pi) 
{ 
  weight1 = 1
  weight0 = 1 
  c1 = (r == 1) & (pi < pcut) #logical vector - true if actual 1 but predict 0 
  c0 = (r == 0) & (pi > pcut) #logical vecotr - true if actual 0 but predict 1 
  return(mean(weight1 * c1 + weight0 * c0)) 
} 
for (i in 1:length(searchgrid)) 
{ 
  pcut <- result[i, 1] # assign the cost to the 2nd col 
  result[i, 2] <- cost1(snet_train$X1, predict(snet_step, type = "response"))
} 
plot(result, ylab = "Total Cost", xlab="Cut-off Probability")

index.min = which.min(result[, 2])
result[index.min, 2]
result[index.min, 1]




# 1.3 in sample performance
pcut_glm = result[index.min, 1] # from code above
snet_insample <- predict(snet_glm, snet_train, type = "response")
snet_predicted <- snet_insample > pcut_glm
snet_predicted <- as.numeric(snet_predicted)
table(snet_train$X1, snet_predicted, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_train$X1 != snet_predicted, 1, 0))
#cost
cost1(snet_train$X1, snet_predicted)
#deviance
deviance_glm=snet_glm$deviance/snet_glm$df.residual
deviance_glm

# 1.4 Out of sample performance
snet_outsample <- predict(snet_glm, snet_test, type = "response")
snet_predicted <- snet_outsample > pcut_glm
snet_predicted <- as.numeric(snet_predicted)
table(snet_test$X1, snet_predicted, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_test$X1 != snet_predicted, 1, 0))
#cost
cost1(snet_test$X1, snet_predicted)

# 1.5 ROC curve
#--in sample
roc.plot(snet_train$X1 == "1", snet_insample)
roc.plot(snet_train$X1 == "1", snet_insample)$roc.vol
#--out of sample
roc.plot(snet_test$X1 == "1", snet_outsample)
roc.plot(snet_test$X1 == "1", snet_outsample)$roc.vol

#############################################GAM#########################################################

gam_formula <- as.formula("X1 ~ X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + s(X2)+s(X3)+s(X4)+s(X5)+s(X6)+s(X7)+s(X8)+s(X9)+s(X10)+s(X11)+s(X12)")

snet_gam <- gam(formula = gam_formula, family = binomial, data = snet_train)
summary(snet_gam)
plot(snet_gam, seWithMean = TRUE, scale = 0, pages=1)
AIC(snet_gam)
BIC(snet_gam)

# 2.1 Search grid to find the optimal cut off probability 
searchgrid = seq(0.01, 0.99, 0.01)  
result = cbind(searchgrid, NA)  

for (i in 1:length(searchgrid)) 
{ 
  pcut <- result[i, 1] # assign the cost to the 2nd col 
  result[i, 2] <- cost1(snet_train$X1, predict(snet_gam, type = "response"))
} 
plot(result, ylab = "Total Cost", xlab="Cut-off Probability")

index.min = which.min(result[, 2])
result[index.min, 2]
result[index.min, 1]

# 2.2 in sample performance
pcut_gam = result[index.min, 1] # from code above
snet_gam_insample <- predict(snet_gam, snet_train, type = "response")
snet_gam_predicted <- snet_gam_insample > pcut_gam
snet_gam_predicted <- as.numeric(snet_gam_predicted)
table(snet_train$X1, snet_gam_predicted, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_train$X1 != snet_gam_predicted, 1, 0))
#cost
cost(snet_train$X1, snet_gam_predicted)
#deviance
deviance_gam=snet_gam$deviance/snet_gam$df.residual
deviance_gam

# 2.3 Out of sample performance
snet_gam_outsample <- predict(snet_gam, snet_test, type = "response")
snet_gam_predicted <- snet_gam_outsample > pcut_gam
snet_gam_predicted <- as.numeric(snet_gam_predicted)
table(snet_test$X1, snet_gam_predicted, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_test$X1 != snet_gam_predicted, 1, 0))
#cost
cost1(snet_test$X1, snet_gam_predicted)

# 2.4 ROC curve
#--insample
roc.plot(snet_train$X1 == "1", snet_gam_insample)
roc.plot(snet_train$X1 == "1", snet_gam_insample)$roc.vol
#--out of sample
roc.plot(snet_test$X1 == "1", snet_gam_outsample)
roc.plot(snet_test$X1 == "1", snet_gam_outsample)$roc.vol


###############################################CART TREE#####################################################


# 3. Tree
library(rpart)
#install.packages("maptree")
library(maptree)
snet_tree=rpart(formula = X1~., data=snet_train,parms=list(loss=matrix(c(0,1,1,0),nrow=2)))
printcp(snet_tree)
draw.tree(snet_tree, cex=0.7, digits=2)
plotcp(snet_tree)

# 3.1 prune tree
snet_tree_best=prune(snet_tree, cp=0.013)
draw.tree(snet_tree_best, cex=0.7, digits=2)
printcp(snet_tree_best)

snet_tree_pred = predict(snet_tree_best, snet_test, type = "prob")


pred = prediction(snet_tree_pred[, 2], snet_test$X1)
perf = performance(pred, "tpr", "fpr")
#ROC curve
plot(perf, colorize = TRUE)
#AUC Curve
slot(performance(pred, "auc"), "y.values")[[1]]

# 3.1 in sample performance
snet_tree_insample=predict(snet_tree,snet_train,type="class")
table(snet_train$X1, snet_tree_insample, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_train$X1 != snet_tree_insample, 1, 0))
#cost
cost_tree <- function(r, pi) {
  weight1 = 1
  weight0 = 1
  c1 = (r == 1) & (pi == 0)  #logical vector - true if actual 1 but predict 0
  c0 = (r == 0) & (pi == 1)  #logical vecotr - true if actual 0 but predict 1
  return(mean(weight1 * c1 + weight0 * c0))
}
cost_tree(snet_train$X1,snet_tree_insample)

# 3.2 out of sample performance
snet_tree_outsample=predict(snet_tree,snet_test,type = "class")
table(snet_test$X1, snet_tree_outsample, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_test$X1 != snet_tree_outsample, 1, 0))
#cost
cost_tree(snet_test$X1,snet_tree_outsample)


############################################LDA########################################################

library(MASS)
str(snet_train)
snet_lda=lda(X1~., data=snet_train)
summary(snet_lda)

# 4.1 Search grid to find the optimal cut off probability 
searchgrid = seq(0.01, 0.99, 0.01)  
result = cbind(searchgrid, NA)  

for (i in 1:length(searchgrid)) 
{ 
  pcut <- result[i, 1] # assign the cost to the 2nd col
  snet_lda_insample <- predict(snet_lda, data = snet_train)
  result[i, 2] <- cost1(snet_train$X1, snet_lda_insample$posterior[,2] )
} 
plot(result, ylab = "Total Cost", xlab="Cut-off Probability")

index.min = which.min(result[, 2])
result[index.min, 2]
result[index.min, 1]

# 4.2 in sample performance
pcut_lda = result[index.min, 1] # from code above
snet_lda_insample <- predict(snet_lda, snet_train, type = "response")
snet_lda_insample <- (snet_lda_insample$posterior[, 2] >= pcut_lda) * 1
table(snet_train$X1, snet_lda_insample, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_train$X1 != snet_lda_insample, 1, 0))
#cost
cost(snet_train$X1, snet_lda_insample)

# 4.3 Out of sample performance
snet_lda_outsample <- predict(snet_lda, snet_test, type = "response")
snet_lda_outsample <- (snet_lda_outsample$posterior[, 2] >= pcut_lda) * 1
table(snet_test$X1, snet_lda_outsample, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_test$X1 != snet_lda_outsample, 1, 0))
#cost
cost_tree(snet_test$X1, snet_lda_outsample)

#ROC In-Sample
snet_lda_insample <- predict(snet_lda, snet_train, type = "response")
roc.plot(snet_train$X1 == "1", snet_lda_insample$posterior[, 2])$roc.vol

#ROC In-Sample
snet_lda_outsample <- predict(snet_lda, snet_test, type = "response")
roc.plot(snet_test$X1 == "1", snet_lda_outsample$posterior[, 2])$roc.vol


##########################################NEURAL NETS########################################################

snet_net<-nnet(X1~.,size = 10, data=snet_train, rang = 0.00001, 
               linout = FALSE, maxit = 1000, decay = 0, skip = TRUE);



searchgrid = seq(0.01, 0.99, 0.01)  
result = cbind(searchgrid, NA)  

for (i in 1:length(searchgrid)) 
{ 
  pcut <- result[i, 1] # assign the cost to the 2nd col
  snet_net_insample<-predict(snet_net, snet_train);
  result[i, 2] <- cost1(snet_train$X1, snet_net_insample )
} 
plot(result, ylab = "Total Cost", xlab="Cut-off Probability")

index.min = which.min(result[, 2])
result[index.min, 2]
result[index.min, 1]

# in sample
pcut_nnet = result[index.min, 1] # from code above
snet_net_insample<-predict(snet_net, snet_train)


snet_net_insample <- (snet_net_insample > pcut_nnet)*1
table(snet_train$X1, snet_net_insample, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_train$X1 != snet_net_insample, 1, 0))
#cost
cost_tree(snet_train$X1, snet_net_insample)

snet_net_insample<-predict(snet_net, snet_train)
roc.plot(snet_train$X1 == "1", snet_net_insample)
roc.plot(snet_train$X1 == "1", snet_net_insample)$roc.vol

# out of sample


snet_net_outsample<-predict(snet_net, snet_test)
snet_net_outsample <- (snet_net_outsample > pcut_nnet) * 1
table(snet_test$X1, snet_net_outsample, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_test$X1 != snet_net_outsample, 1, 0))
#cost
cost_tree(snet_test$X1, snet_net_outsample)

snet_net_outsample<-predict(snet_net, snet_test)
roc.plot(snet_test$X1 == "1", snet_net_outsample)
roc.plot(snet_test$X1 == "1", snet_net_outsample)$roc.vol


############################################RANDOM FORESTS#######################################################

snet_rf<-randomForest(X1~., data=snet_train, mtry = 2, importance = TRUE,  do.trace = 100)
print(snet_rf) # view results 
A<-predict(snet_rf,snet_test,type="prob")


table(snet_rf$y, snet_rf$predicted, dnn = c("Truth", "Predicted"))
#misX1ification rate
mean(ifelse(snet_rf$y != snet_rf$predicted, 1, 0))
#cost
cost_tree(snet_rf$y,snet_rf$predicted)

snet_rf_outsample<-predict(snet_rf, snet_test,type="prob")
roc.plot(snet_test$X1 == "1", snet_rf_outsample[,2])
roc.plot(snet_test$X1 == "1", snet_rf_outsample[,2])$roc.vol

snet_rf_insample<-predict(snet_rf, snet_train,type="prob")
roc.plot(snet_train$X1 == "1", snet_rf_insample[,2])
roc.plot(snet_train$X1 == "1", snet_rf_insample[,2])$roc.vol

##############################################SVM (WIP)#######################################################

tuned <- tune.svm(X1~., data = snet, gamma = 10^(-6:-1), cost = 10^(-1:1))
summary(tuned)

snet_svm <- svm(X1 ~ ., data = snet_train,cost = tuned$best.parameters[2], gamma = tuned$best.parameters[1], probability=TRUE)
plot(snet_svm, snet_train)
print(snet_svm)
summary(snet_svm)


searchgrid = seq(0.01, 0.99, 0.01)  
result = cbind(searchgrid, NA)  

for (i in 1:length(searchgrid)) 
{ 
  pcut <- result[i, 1] # assign the cost to the 2nd col 
  result[i, 2] <- cost1(snet_train$X1, attr(predict(svm.model, snet_train, probability = TRUE), "probabilities")[, 2])
} 
plot(result, ylab = "Total Cost", xlab="Cut-off Probability")


prob.svm.train = predict(svm.model, snet_train, probability = TRUE)
prob.svm.train = attr(prob.svm.train, "probabilities")[, 2]  #This is needed because prob.svm gives a 
cost1(snet_train$X1, pred.svm.train)
pred.svm.train = as.numeric((prob.svm.train >= 0.50))
table(snet_train$X1, pred.svm.train, dnn = c("Obs", "Pred"))
mean(ifelse(snet_train$X1 != pred.svm.train, 1, 0))
cost_tree(snet_train$X1, pred.svm.train)

snet_svm_insample<-predict(snet_svm, snet_train,probability = TRUE)
roc.plot(snet_train$X1 == "1", attr(snet_svm_insample, "probabilities")[,1])
roc.plot(snet_train$X1 == "1", attr(snet_svm_insample, "probabilities")[,1])$roc.vol

snet_svm_outsample<-predict(snet_svm, snet_test,probability = TRUE)
roc.plot(snet_test$X1 == "1", attr(snet_svm_outsample, "probabilities")[,1])
roc.plot(snet_test$X1 == "1", attr(snet_svm_outsample, "probabilities")[,1])$roc.vol
