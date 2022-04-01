
#Loading all required Libraries
library(dplyr)
library(readr)
library(Boruta)

#Loading Train Dataset
train = read.csv("train.csv")
View(train)

#Converting all categorical variables to factors
train[sapply(train,is.character)] = lapply(train[sapply(train,is.character)],as.factor)

paste("Number of Rows: ",nrow(train))
paste("Number of Columns: ",ncol(train))
paste("Summary of Train Data")
summary(train)

#Plotting Histogram for missing Values
library(VIM)
library(ggplot2)
x1 = c("PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked")
p1 = aggr(train, col=c('black','red'), numbers=TRUE, sortVars=TRUE, labels=x1, cex.axis=.5, gap=2, ylab=c("Histogram of missing data","Pattern"))

#Dropping Ticket & Cabin due to more NA values
train = select(train,-Ticket,-Cabin)

#Dropping All NA Tuples
train = na.omit(train)
paste("Summary of Train Dataset Without Null Values")
summary(train)

#Selecting the confirmed features
train = select(train,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,Survived)
paste("Train Dataset")
View(train)
write.csv(train,"updated_train.csv")

#Loading Test Dataset
test = read.csv("test.csv")
paste("Test Dataset")
View(test)
paste("Summary of Test Dataset")
summary(test)

#Selecting only those features suggested by Boruta
test = select(test,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked)

#Converting all categorical variables to factors
test[sapply(test,is.character)] = lapply(test[sapply(test,is.character)],as.factor)

#Checking For Missing Values in Test Dataset
x2 = c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")
p2 = aggr(test, col=c('black','red'), numbers=TRUE, sortVars=TRUE, labels=x2, cex.axis=1, gap=2, ylab=c("Histogram of missing data","Pattern"))
paste("Red Boxes shows the Missing Values & We need to Impute them")

#Imputing the missing values in Test DataSet using MICE Imputation
library(mice)
my_imputation = mice(test,5,method = "pmm",maxit = 5,seed = 245435)

paste("5 Imputed datasets W.R.T Age feature:")
my_imputation$imp$Age
paste("Initially The Stats of Age were")
summary(test$Age)
paste("***** WE CHOOSE 1st Imputed Dataset as the values are closer to mean of initial Test Dataset *****")
completed_test_data=complete(my_imputation,1)
paste("Stats of Age after imputation")
summary(completed_test_data$Age)
paste("We See that mean of AGE after imputation is almost same w.r.t mean of AGE of initial Test Dataset")

#Checking For Missing Values in Test Dataset
x3 = c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")
p3 = aggr(completed_test_data, col=c('black','red'), numbers=TRUE, sortVars=TRUE, labels=x3, cex.axis=1, gap=2, ylab=c("Histogram of missing data","Pattern"))
paste("We See there are no missing values left")

#Writing the updated Dataset into csv
write.csv(completed_test_data,"updated_test.csv")

#Loading Train Dataset
train = read.csv("updated_train.csv")
#Selecting Train Label
train_label = as.factor(train$Survived)
train = select(train,-X,-Survived)

#Loading Test Dataset
test = read.csv("updated_test.csv")
test = select(test,-X)

#Converting all features to numeric in Train Dataset for XGBOOST ALGORITHM
train[sapply(train,is.character)] = lapply(train[sapply(train,is.character)],as.factor)
train = sapply(train,as.numeric)
paste("Updated Train Dataset")
View(train)

#Converting all features to numeric in Test Dataset for XGBOOST ALGORITHM
test[sapply(test,is.character)] = lapply(test[sapply(test,is.character)],as.factor)
test = sapply(test,as.numeric)
paste("Updated Test Dataset")
View(test)

#Applying XGB Algorithm
library(xgboost)
xgb = xgboost(data = as.matrix(train), 
              label = as.matrix(train_label), 
              max.depth = 3, #Max depth of each Decision Tree
              nround = 9, #Number of Boosting rounds
              early_stopping_rounds = 3, 
              objective = "binary:logistic", 
              gamma = 1,
)

#Predicting for the Target variable from Test Dataset i.e. Survived or Not
Survived = predict(xgb,data.matrix(test))
Survived = as.data.frame(Survived)

#Converting all values greater than 0.5 to 1 & less than or equal to 0.5 values to 0
#1 means survived & 0 means not survived
Survived[Survived > 0.5] = 1
Survived[Survived <= 0.5] = 0
#Getting Index attribute from original Test Dataset
old_test = read.csv("test.csv")
PassengerId = select(old_test,PassengerId)

#Binding Index Attribute with Target Attribute & Storing it in a .csv
x = cbind(PassengerId,Survived)
write.csv(x,"PRJ_Ans.csv",row.names = FALSE)
#xgb = read_csv("PRJ_Ans.csv")
#xgb = select(xgb,-X1)
#View(xgb)