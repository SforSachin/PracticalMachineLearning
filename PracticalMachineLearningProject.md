---
title: "Practial Machine Learning"
author: "Sachin"
date: "January 13, 2019"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



# Introduction:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# Data Source:
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Prepare the Dataset:


```r
library(caret);
```

```
## Warning: package 'caret' was built under R version 3.4.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.4.4
```

```r
library(rpart);
```

```
## Warning: package 'rpart' was built under R version 3.4.4
```

```r
library(rpart.plot);
```

```
## Warning: package 'rpart.plot' was built under R version 3.4.4
```

```r
library(RColorBrewer);
library(randomForest);
```

```
## Warning: package 'randomForest' was built under R version 3.4.4
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
url1<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv";

url2<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv";

DataTraining<-read.csv(url1)
DataTesting<-read.csv(url2)

dim(DataTraining)
```

```
## [1] 19622   160
```

```r
dim(DataTesting)
```

```
## [1]  20 160
```

# Data Cleaning:
There are many data fields which is having NA so those are of no use in the analysis. Also initial 7 fields are related to time series that are not of use because we are not considering that in this analysis.


```r
DataTraining <- DataTraining[, colSums(is.na(DataTraining)) == 0]

DataTesting <- DataTesting[, colSums(is.na(DataTesting)) == 0]

trainToRemove<- grepl("^X|timestamp|window", names(DataTraining))

DataTraining<-DataTraining[,!trainToRemove]

classe<- DataTraining$classe

CleanedTrainingData<- DataTraining[, sapply(DataTraining, is.numeric)]

CleanedTrainingData$classe<-classe

testToRemove<- grepl("^X|timestamp|window", names(DataTesting))

DataTesting<-DataTesting[,!testToRemove]

CleanedTestingData<- DataTesting[, sapply(DataTesting, is.numeric)]
```
# Data Partition
We partition Training data set into training and cross validation data set. As per instruction in our learning we use 60% for data partitioning.

```r
set.seed(56789)

trainIndex<-createDataPartition(CleanedTrainingData$classe,p=0.6,list=FALSE)

Training<-CleanedTrainingData[trainIndex,]

Testing<-CleanedTrainingData[-trainIndex,]
```
# Create and Predict using Decision Tree model:

```r
fitDT<-train(classe~.,data = Training,method="rpart")

predDT<-predict(fitDT, Testing)

confusionMatrix(Testing$classe,predDT)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2022   41  160    0    9
##          B  642  509  367    0    0
##          C  604   48  716    0    0
##          D  572  230  484    0    0
##          E  213  211  361    0  657
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4976          
##                  95% CI : (0.4865, 0.5087)
##     No Information Rate : 0.5166          
##     P-Value [Acc > NIR] : 0.9996          
##                                           
##                   Kappa : 0.3436          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4989  0.48989  0.34291       NA  0.98649
## Specificity            0.9446  0.85177  0.88677   0.8361  0.89067
## Pos Pred Value         0.9059  0.33531  0.52339       NA  0.45562
## Neg Pred Value         0.6382  0.91625  0.78821       NA  0.99859
## Prevalence             0.5166  0.13242  0.26612   0.0000  0.08488
## Detection Rate         0.2577  0.06487  0.09126   0.0000  0.08374
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7218  0.67083  0.61484       NA  0.93858
```

```r
##plotting the decision tree

rpart.plot(fitDT$finalModel)
```

![](PracticalMachineLearningProject_files/figure-html/DTmodel-1.png)<!-- -->
# Create and predict using Random forest

```r
fitRF<-train(classe~.,data=Training,method="rf",ntree=100)

predRF<-predict(fitRF,Testing)

confusionMatrix(Testing$classe,predRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228    4    0    0    0
##          B   22 1485    9    2    0
##          C    0   11 1354    3    0
##          D    0    2   13 1270    1
##          E    0    0    2    1 1439
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9911         
##                  95% CI : (0.9887, 0.993)
##     No Information Rate : 0.2868         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9887         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9902   0.9887   0.9826   0.9953   0.9993
## Specificity            0.9993   0.9948   0.9978   0.9976   0.9995
## Pos Pred Value         0.9982   0.9783   0.9898   0.9876   0.9979
## Neg Pred Value         0.9961   0.9973   0.9963   0.9991   0.9998
## Prevalence             0.2868   0.1914   0.1756   0.1626   0.1835
## Detection Rate         0.2840   0.1893   0.1726   0.1619   0.1834
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9948   0.9917   0.9902   0.9964   0.9994
```
# Predicting on Testing Dataset


```r
## Decision Tree Prediction
predictionDT <- predict(fitDT, CleanedTestingData)
predictionDT
```

```
##  [1] C A C A A C C A A A C C C A C A A A A C
## Levels: A B C D E
```

```r
## Random Forest Perdiction
predictionRF<- predict(fitRF,CleanedTestingData)
predictionRF
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

# Conclusion
As we can we from the result, the random forest algorithem far outperforms the decision tree in terms of accuracy. We are getting 99.25% in sample accuracy, while the decision tree gives us only nearly 50% in sample accuracy.
