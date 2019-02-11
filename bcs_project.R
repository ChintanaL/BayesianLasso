dataset<-read.csv('bcs_dataset.csv')
x<-model.matrix(bmi~., dataset)[,-1]
y<-dataset$bmi
lambda<-10^seq(10,-2,length=100)
library(glmnet)
set.seed(489)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
ytest=y[test]

#OLS (Ordinary least squares regression)
dataset_lm<-lm(bmi~.,data=dataset)
coef(dataset_lm)

#ridge
#ridge.mod<-glmnet(x,y,alpha=0,lambda=lambda)
#predict(ridge.mod, s=0, exact = T,type = 'coefficients')[1:18, ]

dataset_lm<-lm(bmi~.,data=dataset,subset=train)
ridge.mod<-glmnet(x[train,],y[train],alpha = 0,lambda=lambda)
plot(cv.glmnet(x[train,],y[train],alpha = 0,lambda=lambda))
#Using cross validation to pick the best value for lambda, the resulting plot indicates that the unregularized full model does pretty well in this case.

cv.out<-cv.glmnet(x[train,],y[train],alpha=0)
cv.out
bestlam<-cv.out$lambda.min
plot(ridge.mod,xvar="lambda",label=TRUE)
# Ridge keeps all the variables and shrinks the coefficients to 0.

#make predictions
ridge.pred<-predict(ridge.mod,s=bestlam,newx=x[test,])
s.pred<-predict(dataset_lm,newdata = dataset[test,])

#check MSE
mean((s.pred-ytest)^2)
# mean = 31.63803
mean((ridge.pred-ytest)^2)
# mean = 31.62602
# we see that ridge performs better than MSE

# Let's have a look at the coefficients
out=glmnet(x[train,],y[train],alpha=0)
predict(ridge.mod,type = "coefficients", s= bestlam)[1:18,]
# most of the coefficient estimates are more conservative, i.e lower than most other estimates


#LASSO
lasso.mod<-glmnet(x[train,],y[train],alpha=1,lambda=lambda)
plot(cv.glmnet(x[train,],y[train],alpha=1,lambda=lambda))
#Cross validation will indicate which variables to include and picks the coefficients from the best model.

lasso.pred<-predict(lasso.mod,s=bestlam,newx = x[test,])
mean((lasso.pred-ytest)^2)
# mean = 32.06512
# The MSE is bit higher for the lasso estimate than the ols and ridge MSE's.

# The coefficients are
lasso.coef<-predict(lasso.mod,type='coefficients',s=bestlam)[1:18,]
lasso.coef
# cholchk, age, educa, strength and weight might be of high importance to bmi variable.
plot(lasso.mod)
#lasso does both shrinkage and variable selection

# boxplot for ols,ridge and lasso
leastsq<-mean((s.pred-ytest)^2)
ridge<-mean((ridge.pred-ytest)^2)
lasso<-mean((lasso.pred-ytest)^2)
pdf("new_plot.pdf")
boxplot(sqrt(leastsq), sqrt(ridge), sqrt(lasso), names=c("OLS", "RIDGE","LASSO"), ylab="RMSE")
dev.off()

#BLASSO = bayesian lasso
library(monomvn)
reg.blas<-blasso(x,y)

## summarize the beta (regression coefficients) estimates
plot(reg.blas, burnin=200)
points(drop(lasso.mod$b), col=2, pch=20)
points(drop(dataset_lm$b), col=3, pch=18)
legend("topleft", c("blasso-map", "lasso", "lsr"),col=c(2,2,3), pch=c(21,20,18))
# lsr = ordinary least squares regression

## get the summary
s <- summary(reg.blas, burnin=200)

## calculate the probability that each beta coef != zero
s$bn0

## summarize s2 (initial variance parameter)
plot(reg.blas, burnin=200, which="s2")
s$s2

## summarize lambda2 (square of the initial lasso penalty parameter)
plot(reg.blas, burnin=200, which="lambda2")
s$lambda2

## bmonomvn implementation
## bmonomvn = Bayesian Estimation for Multivariate Normal Data with Monotone Missingness

data(dataset)
out <- bmonomvn(dataset)
out
out$mu
out$S
plot(out)
plot(out,"S")

##
## a Bayes/MLE comparison using least squares sparingly
##
## fit Bayesian and classical lasso
obls <- bmonomvn(dataset, p=0.25)
Ellik.norm(obls$mu, obls$S, out$mu, out$S)
omls <- monomvn(dataset, p=0.25, method="lasso")
Ellik.norm(omls$mu, omls$S, out$mu, out$S)
## compare to ridge regression
obrs <- bmonomvn(dataset, p=0.25, method="ridge")
Ellik.norm(obrs$mu, obrs$S, out$mu, out$S)
omrs <- monomvn(dataset, p=0.25, method="ridge")
Ellik.norm(omrs$mu, omrs$S, out$mu, out$S)
## using the maximum likelihood solution to initialize
## the Markov chain and avoid burn-in.
ob2s <- bmonomvn(dataset, p=0.25, B=0, start=omls, RJ="p")
Ellik.norm(ob2s$mu, ob2s$S, out$mu, out$S)




