setwd("/home/hadoop/workspace/hackerdojo/ml201/homework")
#setwd("C:\\Documents and Settings\\nbeladia\\Desktop\\rwork")

rm(list=ls())

set.seed(0)

# Read the Wine data and shuffle observations to avoid any observation ordering
#  bias during cross-validation
wine <- read.table("winequality-red.txt",sep=";",header=TRUE)
shuffle.wine.index <- sample(1:nrow(wine),replace=FALSE)
nwine <- wine[shuffle.wine.index,]

wine <- nwine

head(wine)


# Forward Stepwise Selection using cross-validation
library(leaps)

testsize = floor(nrow(wine)/10)
epsilon <- 0.5


overall_min_err <- 999999
#10-fold cross-validation
for (i in 1:10){
  test.index <- seq((testsize*(i-1))+1, testsize * i)
  test.set <- wine[test.index,]
  train.set <- wine[-c(test.index),]

  old_bucket <- c()
  new_bucket <- c()
  varlst <- seq(1, ncol(train.set) - 1, 1)
  #find the best variable, the variable that improves the error the most on the given test data
  err_old <- 999999
  err_new <- 9999
  
  min_err_new <- 999999
  while(err_old > err_new){
    err_old <- err_new
    old_bucket <- new_bucket
    if (err_new != 9999)
      old_lm <- lm_new
      
    for (j in varlst){
      tmp_bucket <- c(old_bucket, j)
      tmp_lm_new <- lm(quality~., data=train.set[,c(tmp_bucket,ncol(train.set))])
      tmp_err_new <- sum((predict(tmp_lm_new, test.set)-test.set[,12])^2)/nrow(test.set)
      
      if (min_err_new > tmp_err_new){
        lm_new <- tmp_lm_new
        err_new <- tmp_err_new
        new_bucket <- tmp_bucket
        min_err_new <- tmp_err_new
      }
    }
    
    varlst <- setdiff(varlst, new_bucket)
  }
  
  if (overall_min_err > err_old){
    overall_min_err <- err_old
    overall_bucket <- old_bucket
    overall_lm <- old_lm
  }
}
overall_bucket <- sort(overall_bucket)
print(paste(c("Forward Step-wise Selection :", paste(names(test.set[,c(overall_bucket)]), collapse=", ")), collapse=" "))
print(paste(c("  Least Squared Error :", overall_min_err), collapse=" "))
fwd_error <- overall_min_err
fwd_lm <- overall_lm
fwd_cols <- overall_bucket


# Backward Stepwise Selection using cross-validation
overall_min_err <- 999999
#10-fold cross-validation
for (i in 1:10){
  test.index <- seq((testsize*(i-1))+1, testsize * i)
  test.set <- wine[test.index,]
  train.set <- wine[-c(test.index),]

  varlst <- seq(1, ncol(train.set) - 1, 1)
  old_bucket <- varlst
  new_bucket <- varlst
  
  #find the worst variable, the variable when removed from bucket, improves the error the most on the given test data
  err_old <- 999999
  err_new <- 9999
  
  min_err_new <- 999999
  while(err_old > err_new){
    err_old <- err_new
    old_bucket <- new_bucket
    if (err_new != 9999)
      old_lm <- lm_new
    for (j in old_bucket){
      tmp_bucket <- setdiff(old_bucket, j)
      tmp_lm_new <- lm(quality~., data=train.set[,c(tmp_bucket,ncol(train.set))])
      tmp_err_new <- sum((predict(tmp_lm_new, test.set)-test.set[,12])^2)/nrow(test.set)
      
      if (min_err_new > tmp_err_new){
        lm_new <- tmp_lm_new
        err_new <- tmp_err_new
        new_bucket <- tmp_bucket
        min_err_new <- tmp_err_new
        min_fold <- i
      }
    }
    
    varlst <- setdiff(varlst, new_bucket)
  }
  
  if (overall_min_err > err_old){
    overall_min_err <- err_old
    overall_bucket <- old_bucket
    overall_lm <- old_lm
    overall_fold <- min_fold
  }
}
overall_bucket <- sort(overall_bucket)
print(paste(c("Backward Step-wise Selection :", paste(names(test.set[,c(overall_bucket)]), collapse=", ")), collapse=" "))
print(paste(c("  Least Squared Error :", overall_min_err), collapse=" "))
back_error <- overall_min_err
back_lm <- overall_lm
back_cols <- overall_bucket



# All subset Selection using cross-validation
#10-fold cross-validation

overall_err <- 99999
for (i in 1:10){
  test.index <- seq((testsize*(i-1))+1, testsize * i)
  test.set <- wine[test.index,]
  train.set <- wine[-c(test.index),]
  
  leaps.rslt <- leaps(x=train.set[,-ncol(train.set)], y=train.set$quality, nbest=20)
  which.matrix <- leaps.rslt$which
  min_err <- 99999
  for (row in 1:nrow(which.matrix)){
    cols <- which(which.matrix[row,])
      if (length(cols) > 0){
      tmp_lm <- lm(quality~.,data=train.set[,c(cols,ncol(train.set))])
      tmp_err <- sum((predict(tmp_lm, test.set)-test.set[,12])^2)/nrow(test.set)
    
      if (min_err > tmp_err){
        min_err <- tmp_err
        min_lm <- tmp_lm
        min_cols <- cols
        min_fold <- i
        min_which <- which.matrix
      }
    }
  }
  
  if (overall_err > min_err){
    overall_err <- min_err
    overall_lm <- min_lm
    overall_cols <- min_cols
    overall_fold <- min_fold
    overall_which <- min_which
  }  
}

overall_cols <- sort(overall_cols)
print(paste(c("All Subset Selection :", paste(names(test.set[,c(overall_cols)]), collapse=", ")), collapse=" "))
print(paste(c("  Least Squared Error :", overall_err), collapse=" "))
all_error <- overall_err
all_lm <- overall_lm
all_cols <- overall_cols


# Lasso with cross-validation
i <- 1
test.index <- seq((testsize*(i-1))+1, testsize * i)
test.set <- wine[test.index,]
train.set <- wine[-c(test.index),]

library(lars)
#use lasso to fit the data and then plot
fit.lasso <- lars(as.matrix(train.set[,-12]),as.matrix(train.set[,12]), type="lasso")#,normalize = TRUE, intercept = TRUE)
cv.lasso <- cv.lars(as.matrix(train.set[,-12]),as.matrix(train.set[,12]), K=10, type='lasso', plot.it=TRUE) 

i.min <- which.min(cv.lasso$cv)
i.se <- which.min(abs(cv.lasso$cv-(cv.lasso$cv[i.min]+cv.lasso$cv.error[i.min])))
s.best <- cv.lasso$index[i.se]

s.best

#retrieve the optimal coefficients
predict.lars(fit.lasso, s = s.best, type="coefficients", mode = "fraction")

y.hat.test <- predict.lars(fit.lasso, as.matrix(test.set[,-12]), s=s.best, type = "fit", mode = "fraction")
print(paste(c("Best CV Lasso model error on Test Set", sum((y.hat.test$fit-test.set[,12])^2)/nrow(test.set)), collapse=" : "))

lasso_error <- sum((y.hat.test$fit-test.set[,12])^2)/nrow(test.set)
lasso_coeff <- predict.lars(fit.lasso, s = s.best, type="coefficients", mode = "fraction")$coefficients
lasso_coeff <- lasso_coeff[which(lasso_coeff != 0)]



# Chart

all.rslt <- list(fwd_error, fwd_lm$coefficients, back_error, back_lm$coefficients, all_error, all_lm$coefficients, lasso_error, lasso_coeff)
names(all.rslt) = c("Forward - LSE", "Forward - Coefficients",
                    "Backward - LSE","Backward - Coefficients", 
                    "All-Subset - LSE", "All-Subset - Coefficients",
                    "Lasso - LSE", "Lasso - Coefficients")
all.rslt


# Plot
fwd_size <- length(fwd_cols)
back_size <- length(back_cols)
all_size <- length(all_cols)
lasso_size <- length(lasso_coeff)

rslt <- matrix(c("Forward", fwd_error, fwd_size, "Backward", back_error, back_size,
          "All Subset", all_error, all_size, "Lasso", lasso_error, lasso_size), ncol=3, byrow=TRUE)
rslt <- data.frame(rslt)          
names(rslt) <- c("Method", "Error", "Size")


plot(as.numeric(levels(rslt$Size)[rslt$Size]), as.numeric(levels(rslt$Error)[rslt$Error]), main="Error vs. Subset Size",
   xlab="Subset Size", ylab="L.S. Error ")