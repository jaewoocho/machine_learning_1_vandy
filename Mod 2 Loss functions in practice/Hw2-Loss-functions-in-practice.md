Homework 2 Loss functions in practice
================
Jaewoo Cho
January 29, 2022

# Questions

1.  Write functions that implement the L1 loss and tilted absolute loss
    functions.

2.  Create a figure that shows lpsa (x-axis) versus lcavol (y-axis). Add
    and label (using the ‘legend’ function) the linear model predictors
    associated with L2 loss, L1 loss, and tilted absolute value loss for
    tau = 0.25 and 0.75.

3.  Write functions to fit and predict from a simple nonlinear model
    with three parameters defined by ‘beta\[1\] +
    beta\[2\]*exp(-beta\[3\]*x)’. Hint: make copies of ‘fit_lin’ and
    ‘predict_lin’ and modify them to fit the nonlinear model. Use
    c(-1.0, 0.0, -0.3) as ‘beta_init’.

4.  Create a figure that shows lpsa (x-axis) versus lcavol (y-axis). Add
    and label (using the ‘legend’ function) the nonlinear model
    predictors associated with L2 loss, L1 loss, and tilted absolute
    value loss for tau = 0.25 and 0.75.

# Source Code

``` r
## load prostate data
prostate <- 
  read.table(url(
    'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data'))

## subset to training examples
prostate_train <- subset(prostate, train==TRUE)

## plot lcavol vs lpsa
plot_psa_data <- function(dat=prostate_train) {
  plot(dat$lpsa, dat$lcavol,
       xlab="log Prostate Screening Antigen (psa)",
       ylab="log Cancer Volume (lcavol)",
       pch = 20)
}
plot_psa_data()
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->
\# regular linear regression

## L1 Loss Function

``` r
L1_loss <- function(y, yhat)
  abs(y-yhat)

fit_lin_L1 <- function(y, x, loss=L1_loss, beta_init = c(-0.51, 0.75)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*x))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

predict_lin_L1 <- function(x, beta)
  beta[1] + beta[2]*x

lin_beta_L1 <- fit_lin_L1(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L1_loss)

x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred_L1 <- predict_lin_L1(x=x_grid, beta=lin_beta_L1$par)

## plot data
plot_psa_data()

## plot predictions
lines(x=x_grid, y=lin_pred_L1, col='darkgreen', lwd=2)

## do the same thing with 'lm'
lin_fit_lm <- lm(lcavol ~ lpsa, data=prostate_train)

## make predictions using 'lm' object
lin_pred_lm <- predict(lin_fit_lm, data.frame(lpsa=x_grid))

## plot predictions from 'lm'
lines(x=x_grid, y=lin_pred_lm, col='pink', lty=2, lwd=2)
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
############################
## regular linear regression
############################

## L2 loss function
L2_loss <- function(y, yhat)
  (y-yhat)^2

## fit simple linear model using numerical optimization
fit_lin <- function(y, x, loss=L2_loss, beta_init = c(-0.51, 0.75)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*x))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

## make predictions from linear model
predict_lin <- function(x, beta)
  beta[1] + beta[2]*x

## fit linear model
lin_beta <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L2_loss)

## compute predictions for a grid of inputs
x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
lin_pred_L2 <- predict_lin(x=x_grid, beta=lin_beta$par)

## plot data
plot_psa_data()

## plot predictions
lines(x=x_grid, y=lin_pred_L2, col='darkgreen', lwd=2)

## do the same thing with 'lm'
lin_fit_lm <- lm(lcavol ~ lpsa, data=prostate_train)

## make predictins using 'lm' object
lin_pred_lm <- predict(lin_fit_lm, data.frame(lpsa=x_grid))

## plot predictions from 'lm'
lines(x=x_grid, y=lin_pred_lm, col='pink', lty=2, lwd=2)
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
##################################
## try modifying the loss function
##################################

## custom loss function
custom_loss_25 <- function(y, yhat)
  qrnn::tilted.abs(y-yhat, tau = 0.25)

custom_loss_75 <- function(y, yhat)
  qrnn::tilted.abs(y-yhat, tau = 0.75)

## plot custom loss function
err_grd <- seq(-1,1,length.out=200)
plot(err_grd, custom_loss_25(err_grd,0), type='l',
     xlab='y-yhat', ylab='custom loss')
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

``` r
err_grd <- seq(-1,1,length.out=200)
plot(err_grd, custom_loss_75(err_grd,0), type='l',
     xlab='y-yhat', ylab='custom loss')
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-4-2.png)<!-- -->

``` r
## fit linear model with custom loss
lin_beta_custom_25 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=custom_loss_25)
lin_beta_custom_75 <- fit_lin(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=custom_loss_75)

lin_pred_custom_25 <- predict_lin(x=x_grid, beta=lin_beta_custom_25$par)
lin_pred_custom_75 <- predict_lin(x=x_grid, beta=lin_beta_custom_75$par)

## plot data
plot_psa_data()

## plot predictions from L1 loss
lines(x=x_grid, y=lin_pred_L1, col='pink', lwd=2)

## plot predictions from L2 loss
lines(x=x_grid, y=lin_pred_L2, col='darkgreen', lwd=2)

## plot predictions from custom loss 0.25
lines(x=x_grid, y=lin_pred_custom_25, col='red', lwd=2, lty=1)

## plot predictions from custom loss 0.75
lines(x=x_grid, y=lin_pred_custom_75, col='blue', lwd=2, lty=1)

# Create Legend
legend(0, 4, 
       legend=c("L1_loss", "L2_loss", "tau: 0.25", "tau: 0.75"),
       col=c("pink", "darkgreen", "red", "blue"),
       lty = 1)
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-4-3.png)<!-- -->
\# Non-linear model \# L1 loss function

``` r
fit_nonlin_L1 <- function(y, x, loss=L1_loss, beta_init = c(-1.0, 0.0, -0.3)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

predict_nonlin_L1 <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)

nonlin_beta_L1 <- fit_nonlin_L1(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L1_loss)

x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)
nonlin_pred_L1 <- predict_nonlin_L1(x=x_grid, beta=nonlin_beta_L1$par)
```

# L2 loss function

``` r
fit_nonlin_L2 <- function(y, x, loss=L2_loss, beta_init = c(-1.0, 0.0, -0.3)) {
  err <- function(beta)
    mean(loss(y,  beta[1] + beta[2]*exp(-beta[3]*x)))
  beta <- optim(par = beta_init, fn = err)
  return(beta)
}

predict_nonlin_L2 <- function(x, beta)
  beta[1] + beta[2]*exp(-beta[3]*x)

nonlin_beta_L2 <- fit_nonlin_L2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=L2_loss)

x_grid <- seq(min(prostate_train$lpsa),
              max(prostate_train$lpsa),
              length.out=100)

nonlin_pred_L2 <- predict_nonlin_L2(x=x_grid, beta=nonlin_beta_L2$par)

lin_fit_lm <- lm(lcavol ~ lpsa, data=prostate_train)

lin_pred_lm <- predict(lin_fit_lm, data.frame(lpsa=x_grid))
```

# Create graphs

``` r
nonlin_beta_custom_0.25 <- fit_nonlin_L2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=custom_loss_25)

nonlin_beta_custom_0.75 <- fit_nonlin_L2(y=prostate_train$lcavol,
                    x=prostate_train$lpsa,
                    loss=custom_loss_75)

nonlin_pred_custom_0.25 <- predict_nonlin_L2(x=x_grid, beta=nonlin_beta_custom_0.25$par)

nonlin_pred_custom_0.75 <- predict_nonlin_L2(x=x_grid, beta=nonlin_beta_custom_0.75$par)

plot_psa_data()

## plot from L1 loss
lines(x=x_grid, y=nonlin_pred_L1, col='blue', lwd=2)

## plot predictions from L2 loss
lines(x=x_grid, y=nonlin_pred_L2, col='red', lwd=2)

## plot predictions from custom loss, tau = 0.25
lines(x=x_grid, y=nonlin_pred_custom_0.25, col='pink', lwd=2, lty=1)

## plot predictions from custom loss, tau = 0.75
lines(x=x_grid, y=nonlin_pred_custom_0.75, col='darkgreen', lwd=2, lty=1)


## add legend
legend(-0, 4, 
       legend=c("L1_loss", "L2_loss", "tau: 0.25", "tau: 0.75"),
       col=c("blue", "red", "pink", "darkgreen"), 
       lty=1, cex=0.8)
```

![](Hw2-Loss-functions-in-practice_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->
