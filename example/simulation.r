library(tidyverse)

n <- 100
p <- 2

X <- rnorm(n * p) %>% matrix(ncol=2)

beta <- c(2,1)
sigmasq <- .5


y <- X %*% beta + sigmasq^.5 * rnorm(n)


M <- diag(p)
m <- rep(0, p)
a <- 2
b <- 1

N <- 5000

results <- gibbs_sampler(y, X, m, M, a, b, N)
