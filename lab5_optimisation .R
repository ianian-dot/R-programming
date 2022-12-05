## STAT0030 LAB 5: OPTIMISATION

setwd("/Users/iantan/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Postgrad/Masters/MSc DS/Modules /STAT0030")

## We use numerical methods to find f'(x) = 0 for optimisation problems
# When it is hard to find f(x) or f'(x), when we can only obtain values of f at given values of x

## When we are working with many dimensions, we need a gradient vector (gradient of f with respect to each x1, x2...)
# to get the second derivative (to find out if the stationary pt is max, min, or inflection)
# We need the hessian function or the hessian matrix, which is the second derivative 
# We can get second derivative with respect to each x variable, but we also need 
# mixed derivatives (off diagonals)
# similar to the univariate case: we get a min value for positive definite matrix (much like how the second derivative is positive in univariate case)


## h(x) = x^4 + 4x + 12

h <- function(x) {return(x^4 + 4*x +12)}

dh_dx <- function(x) {4*x^3 + 4}

## gradient descent for this function 

grad_descent <- function(x_start, esp = 0.05, iter = 20){
  x_vec <- rep(NA, iter) ## make a placeholder vector of length iter
  x_vec[1] <- x_start ## x vector will capture journey of x values, starting from x_start
  ## first entry will be determined by user
    
  ## run gradient descent 
  for (i in (2:iter)) { ## for second entry to the last
    x_vec[i] <- x_vec[i-1] - esp*dh_dx(x_vec[i-1])
  }
  return(x_vec)
}

grad_descent(0)
grad_descent(5)

output <- grad_descent(0)
par(mfrow = c(1,2))
plot(1:20, output, type = 'b')
plot(output, h(output))


## 3: minimisation using nlm

quartic <- function(x) { 
  x^4 - 2*x^3 +3*x^2 - 4*x +5
}

quartic(1:10)

par(mfrow = c(1,1))
curve(quartic, -3, 3) ## use curve directly for functions, without having to simulate or make a sequence of input x values

d_quartic <- function(x) {4*x^3 - 6*x^2 + 6*x -4}


curve(quartic, -3,3 )
curve(d_quartic, -3,3)

## minimise using nlm 
nlm_min <- nlm(f = quartic,p = 1.5 )
nlm_min$estimate ## this is the estimate of the argmin of quartic
nlm_min$code ## 1 - converged, i.e. relative gradient is close to 0 

## Exercise: try starting points as 10, -10, and 100
nlm(f = quartic,p = 10 )
nlm(f = quartic,p = -10 )
nlm(f = quartic,p = 100 )
nlm(f = quartic,p = 1000) ## 29 interations needed for convergence
nlm(f = quartic,p = 100000 )



## --- Multivariate functions 
f_2d <- function(x){ ## since mvariate, x will be a vector
  x[1]^4*x[2]^2 + ((x[1]-4)^2)*((x[2]-4)^2) ## need to subset to get x1, x2... etc
} 

f_2d(x = c(1,2)) ## insert x as a vector

nlm(f_2d, p = c(0,0)) ## approximates to (0,4)
nlm(f_2d, p = c(-4,4)) ## approximates to (4,0)
#- different starting points bring us to different statinary points

## - Create log function of the f_2d
logf_2d <- function(x) {log(f_2d(x))}

## - Plot contour of the f_2d function on the log scale using contour function

n <- 50 ## number of points to evaluate function

## How to create data for 2D functions? -- need matrix
x1 <- seq(-6,6, length = n) ## input vectors
x2 <- seq(-6,6, length = n) ## input vectors
z <- matrix(nrow = n, ncol = n) ## placeholder first, then fill later 

## Entry i,j for function evaluated at x1[i], x2[j]
for (i in 1:n){
  for (j in 1:n){
    z[i,j] <- logf_2d(x = c(x1[i], x2[j]))
  }
}

## Contour
par(mfrow = c(1,1))
contour(x = x1, y = x2, z, 
        col = 'pink', nlevels = 20) ## contour lines require the output as a matrix

## ----- Optimisation exercises: MLE 

## We usually wnat to maximise the likelihood of data
# i.e. we want to find the argmax of loglikelihood of data (function of parameters)
# to get the most likely parameters that gave rise to the data
#L(parameters) = product(individual probs)

## - For truncated poisson: make a loglikelihood function
tpnegloglik <- function(theta, y) { ## function of both param and data
  ## negative log likelihood for truncated poisson
  
  return((theta - mean(y)*log(theta) +log(1-exp(-theta)))*length(y))
}

truncpois <- scan("truncpois.dat")

## recall that since this is a poisson, theta is the rate parameter
# theta has to be non negative 
tpoisfit <- nlm(f = tpnegloglik, y = truncpois, p = 2)
theta_mle <- tpoisfit$estimate

## plot values of tpnegloglik for this given dataset and for values of 
# theta from 0.1 to 10

y_tp <- tpnegloglik(theta = seq(0.1,10, length.out = 100), y = truncpois)
plot(x = seq(0.1,10, length.out = 100), 
     y = y_tp, 
     xlab = expression(paste(theta))) ## use expression and paste for greek symbols

### --- Variance of ML estimates -- how precise are our MLE estimates?
## - recall cramer rao lower bound for estimators 
## - i.e. the most efficient/lowest variance estimators 

## MLE theory gives us that the asymptotic variance (covariance) for ML estimators
# are the inverse of Fisher information 
## for mvariable: the asymptotic variance covariance will be the inverse of the fisher information matrix

## use Hessian = True to get the matrix which consists of the numerically
# estimated second derivatives of the function evaluated at the observed data
# this Hessian matrix will be AN ESTIMATE OF THE FISHER INFORMATION MATRIX  
fit3 <- nlm(tpnegloglik,3.7,y=truncpois,hessian=TRUE)
tphessian <- fit3$hessian

approx_var_estimate <- 1/tphessian
diag(solve(tphessian)) ## same, since we are only working in 1 Dim i.e. univariate 

## in parametric cases, we have a restricted range of values for theta
## e.g. in binomial case - p element of (0,1)
## e.g. in poisson case, lambda element of (0, inf)
## for some datasets, MLE is close to the boundaries of 
## the allowed parameter space -- this is a problem for the optimiser 

## Solution: trnasform the parameter and therefore its space
# e.g. for theta in poisson, log(theta) now can take both neg and pos values
# hence can use nlm on log(theta), then find estimate of log(theta) i.e. log(theta)_hat
# then find 

require(Bhat)


## ----- Non linear least squares 

## this is for non linear functions -- i.e. functions that are not linear with respect to the coefficients

## e.g. Yi = B0*(exp(Xi*B1)) + Ei 
# want to estimate B0 and B1 

## we cannot simply take logs due to the error structure (we want to fit models
# with additive errors! )

## define the sum of squares error (SSE) to be 
# S(B0, B1) = sum(i) (Yi = B0*(exp(Xi*B1)))^2

## - Make a function for this SSE for B0 and B1

sserrors <- function(beta, y, x){ ## beta is a vector of 2 parameters B0 B1
  sum = 0
  for (i in 1:length(y)) {
    sum = sum + (y[i] - beta[0]*exp(x[i]*beta[1]))^2
  }
  return(sum)
}

sserrors2 <- function(beta, y, x){ ## beta is a vector of 2 parameters B0 B1
  sum((y - beta[1]*exp(x*beta[2]))^2)
}
## get data to insert into function 
nonlindata <- read.table("nonlindata.dat")
view(nonlindata)
plot(x = nonlindata$explan, y = nonlindata$observed) ## evidently non linear, lol 

## debugging code
beta = c(1,1); y = nonlindata$observed; x = nonlindata$explan
vec_temp <- (y - beta[1]*exp(x*beta[2]))^2
sum(vec_temp)
sserrors2(c(1,1),nonlindata$observed, nonlindata$explan ) == sum(vec_temp)


## to use NLM, we need to know the function we want to minimise
## hence we made the SSE function

## first function doesnt work since not analytical 
nlm(sserrors, c(1,-1), y = nonlindata$observed, x = nonlindata$explan)

## second function works 
nlm(sserrors2, c(1,-1), y = nonlindata$observed, x = nonlindata$explan)

## Different starting point
fit4 <- nlm(sserrors2, c(1,0), y = nonlindata$observed, x = nonlindata$explan) ## no convergence (code 4)
fit5 <- nlm(sserrors2, c(1,0), y = nonlindata$observed, x = nonlindata$explan, 
            iterlim = 200) ## finally converges after the 159th step

## --- How to find standard errors -- recall: Hessian!
fit6 <- nlm(sserrors2, c(1,0), y = nonlindata$observed, x = nonlindata$explan, 
            iterlim = 200, hessian = T)
hessian_matrix <- fit6$hessian  
coeffs_fitted <- fit6$estimate
# to get standard errors, invert and then get diagonal
diag(solve(hessian_matrix)) ## Standard errors for B0, B1

## --- Estimate variance 
# from https://stats.stackexchange.com/questions/285023/compute-standard-errors-of-nonlinear-regression-parameters-with-maximum-likeliho
# back to sum of squares optimised -- SSE(B0*, B1*, data)
sse <- sserrors2(beta = coeffs_fitted, y = nonlindata$observed, x = nonlindata$explan)
df = length(y) - length(coeffs_fitted)
variance_estimate = sse / df

## --- Fit the fitted curve into the plot
plot(x = nonlindata$explan, y = nonlindata$observed, 
     pch = 3) 
lines(y = coeffs_fitted[1]*exp(coeffs_fitted[2]*x), x, 
      col = 'red')
legend(5, 1.5, legend = c("fitted line"), col = c("red"),
       lty = 1,
       title = "non-linear LS", 
       cex = .8)

## --- residuals: plots

res <- function(beta = coeffs_fitted, x, y){
  (y - beta[1]*exp(x*beta[2]))^2
}

resfit4 <- res(x = nonlindata$explan, y = nonlindata$observed)

plot(resfit4, nonlindata$explan)
boxplot(resfit4)
dotchart(resfit4)
qqnorm(resfit4)

## Compare with MLE method?
# Assume that errors are N(0, sigma^2)


