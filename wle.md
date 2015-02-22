
```
## Warning: package 'doMC' was built under R version 3.1.2
```

```
## Loading required package: foreach
```

```
## Warning: package 'foreach' was built under R version 3.1.2
```

```
## Loading required package: iterators
```

```
## Warning: package 'iterators' was built under R version 3.1.2
```

```
## Loading required package: parallel
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.1.2
```
Predicting Exercise Quality
========================================================
#### Can you use data to predict how an exercise is performed? 

# Executive Summary
Exercise is an important part of a healthy lifestyle. While many people exercise frequent, they tend to forget that exercise done incorrectly can have long term repruccions on your body. Many athletes can hurniate disks or tear muscles if an exercise is not done with the appropriate form. This paper will analysis data gathered electronicly from a group of subjects performing an exercise. We will use this sensor data to determine if we can programatically predict how well the exercise was performed from the data alone. 

# Process
There are many prediction algorithms that can be used to estimate the quality of an exercise being perfomed. However, the first step in creating a prediction algorithm is to clean your data. If you have invalid data, it will greatly affect the accuracy of prediction. Second, we will need to use a form of cross validation to create a training data set and a test data set. Creating two data sets will give us a way to test if our algorithm is accurate on a new data set. Finally, we can begin to explore various algorithms to predict the quality of excersize. 

## Creating a Tidy Data Set
The first step in our process is to create a tidy, clean data set to predict with. Upon loading the data we see that there are many observations missing. While a few missing data points will not cause our model harm, many columns are missing a majority of data points. So the first thing we do is remove any columns that contain NULL values. 

After removing the NULL values we explore the data set and notice variables related to time. If a subject performs the exercise on minimal sleep, this could affect the quality of the exercise. Given this is a controlled test, we will assume that time will not affect the quality of the exercise and thus we remove this data.


```r
testData = read.csv("/Users/udumami/Dropbox/coursera/Machine Learning/Project/pml-testing.csv", na.strings=c("NA","", "#DIV/0!"))
testData = testData[,!sapply(testData,function(x) any(is.na(x)))]
testData = na.omit(testData)
testData = testData[, !names(testData) %in% c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'num_window', 'new_window','row.names')]

data = read.csv("/Users/udumami/Dropbox/coursera/Machine Learning/Project/pml-training.csv", na.strings=c("NA","", "#DIV/0!"))
data = data[,!sapply(data,function(x) any(is.na(x)))]
data = na.omit(data)
data = data[, !names(data) %in% c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp', 'num_window', 'new_window', 'row.names')]
```

## Cross Validation
With a tidy data set in hand, we can focus our efforts on cross validation. Cross validation is an important step in the process as it provides a way to test your algorithm on a untouched data set. We will proceed with a simple variant of cross validation. Instead of splitting our data set into 3 parts (training 60%, tes 20%t and validation 20%) we will use two sets with a 80/20 split. This is because the final portion of our paper will be to test the model on 20 data points that have already been set aside. 


```r
train = createDataPartition(data$classe, p=.8, list=FALSE)
training = data[train, ]
testing = data[-train, ]
```

## Algorithm Selection
When wanting to predict an outcome, there are many algorithms to choose from. By utlizing the in sample error rate, we can narrow down which algorithm will best suited to predict the quality of exercise being performed by a subject. For this prediction, we started with Decision Tree model and settled on a Random Forest model. 

### Decision Tree
Our first attempt at prediction was with decision trees. Overall accuracy is very low, less than 50%. This is no better than flipping a coin. While we may be able to tune this algorithm to perform better, we will shift our focus to see if a Random Forest model will perform better. 


```r
modFitDF = train(classe ~ ., method="rpart", data=training)
```

```
## Loading required package: rpart
```

```r
p = predict(modFitDF, training)
confusionMatrix(p, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4065 1293 1278 1179  411
##          B   71 1016   87  449  400
##          C  316  729 1373  945  753
##          D    0    0    0    0    0
##          E   12    0    0    0 1322
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4953          
##                  95% CI : (0.4875, 0.5032)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.34            
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9106  0.33443  0.50146   0.0000  0.45807
## Specificity            0.6296  0.92046  0.78837   1.0000  0.99906
## Pos Pred Value         0.4942  0.50222  0.33358      NaN  0.99100
## Neg Pred Value         0.9466  0.85215  0.88215   0.8361  0.89112
## Prevalence             0.2843  0.19352  0.17441   0.1639  0.18383
## Detection Rate         0.2589  0.06472  0.08746   0.0000  0.08421
## Detection Prevalence   0.5240  0.12886  0.26218   0.0000  0.08497
## Balanced Accuracy      0.7701  0.62745  0.64491   0.5000  0.72857
```

### Random Forest
Looking at a random forest model we see staggering difference. After our first run with the training set we an in sample error of 0.39%. A number this close to 100% is almost impossible to believe. While our in sample error rate is high, this may be due to over fitting. To help validate our model, we rerun our prediction algorithm on our test data set. We find our out of sample error rate is 0.43%. Having a higher out of sample error rate is expected as the model will naturally be slighly tuned to the training data set. 

```r
modFitRF = randomForest(classe ~ ., data = training) 
modFitRF
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.39%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4461    2    0    0    1 0.000672043
## B   11 3025    2    0    0 0.004279131
## C    0   10 2723    5    0 0.005478451
## D    0    0   21 2549    3 0.009327633
## E    0    0    0    7 2879 0.002425502
```

```r
predRF = predict(modFitRF, testing)
confusionMatrix(predRF, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    2    0    0    0
##          B    1  756    3    0    0
##          C    0    1  681   10    0
##          D    0    0    0  633    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9931, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9945          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9960   0.9956   0.9844   1.0000
## Specificity            0.9993   0.9987   0.9966   1.0000   1.0000
## Pos Pred Value         0.9982   0.9947   0.9841   1.0000   1.0000
## Neg Pred Value         0.9996   0.9991   0.9991   0.9970   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1927   0.1736   0.1614   0.1838
## Detection Prevalence   0.2847   0.1937   0.1764   0.1614   0.1838
## Balanced Accuracy      0.9992   0.9974   0.9961   0.9922   1.0000
```

# Conclusion
After creating a tidy data set, using a Random Forest model yielded a high quality prediction algorithm on our training and test data set. The final setp is performing a prediction on the 20 with held data points. A quick prediction run yields us 20 predictions to the quality of exercise for each observation with 100% accuracy. 

```r
predTest = predict(modFitRF, testData)
```
