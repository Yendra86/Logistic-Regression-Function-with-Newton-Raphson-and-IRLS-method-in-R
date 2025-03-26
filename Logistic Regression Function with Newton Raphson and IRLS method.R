library(dplyr)

# Logistic Regression Function with Newton Raphson and IRLS method
logistic_regression <- function(X, y, method = c("NR", "IRLS"), 
                                tol = 1e-5, max.iter = 1000) {
  
  method <- match.arg(method)
  
  # Log Likelihood Function
  log_likelihood <- function(beta, X, y) {
    p <- 1 / (1 + exp(-X %*% beta))
    ll <- sum(y * log(p) + (1 - y) * log(1 - p))
    return(ll)
  }
  
  # Gradient Function
  gradient <- function(beta, X, y) {
    p <- 1 / (1 + exp(-X %*% beta))
    grad <- t(X) %*% (y - p)
    return(grad)
  }
  
  # Hessian Matrix Function
  hessian <- function(beta, X) {
    p <- 1 / (1 + exp(-X %*% beta))
    W <- diag(as.vector(p * (1 - p)))
    H <- -t(X) %*% W %*% X
    return(H)
  }
  
  # Newton-Raphson Method
  newton_raphson <- function(X, y) {
    beta <- rep(0, ncol(X))
    for (i in 1:max.iter) {
      grad <- gradient(beta, X, y)
      H <- hessian(beta, X)
      beta_new <- beta - solve(H) %*% grad
      
      if (sqrt(sum((beta_new - beta)^2)) < tol) {
        cat('Newton-Raphson Convergence reached in', i, 'Iterations\n')
        return(list(
          method = "Newton-Raphson",
          beta = beta_new,
          fit = 1 / (1 + exp(-X %*% beta_new))
        ))
      }
      beta <- beta_new
    }
    warning('Newton-Raphson: Maximum Iteration Reached Without Convergence')
    return(list(
      method = "Newton-Raphson",
      beta = beta,
      fit = 1 / (1 + exp(-X %*% beta))
    ))
  }
  
  # Iteratively Reweighted Least Squares (IRLS) Method
  irls <- function(X, y) {
    beta <- rep(0, ncol(X))
    for (i in 1:max.iter) {
      p <- 1 / (1 + exp(-X %*% beta))
      W <- diag(as.vector(p * (1 - p)))
      z <- X %*% beta + solve(W) %*% (y - p)
      xtw <- t(X) %*% W
      xtwx_inv <- solve(t(X) %*% W %*% X)
      beta_new <- xtwx_inv %*% (xtw %*% z)
      
      # Check convergence
      if (sqrt(sum((beta_new - beta)^2)) < tol) {
        cat('IRLS Converged in', i, 'iteration\n')
        return(list(
          method = "IRLS",
          beta = beta_new,
          fit = 1 / (1 + exp(-X %*% beta_new))
        ))
      }
      beta <- beta_new
    }
    
    warning('IRLS: Maximum Iteration Reached Without Convergence')
    return(list(
      method = "IRLS",
      beta = beta,
      fit = 1 / (1 + exp(-X %*% beta))
    ))
  }
  
  # Select method
  if (method == "NR") {
    return(newton_raphson(X, y))
  } else {
    return(irls(X, y))
  }
}

df <- read.csv("C:/Users/INFINIX/Downloads/M4.csv")
X <- cbind(1, as.matrix(df %>% select(-Purchased)))
y <- df$Purchased

# Newton-Raphson Method
nr_result <- logistic_regression(X, y, method = "NR")
nr_result
# IRLS Method
irls_result <- logistic_regression(X, y, method = "IRLS")
irls_result
