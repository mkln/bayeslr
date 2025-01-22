#include <RcppArmadillo.h>

using namespace std;


arma::vec full_cond_beta(const arma::mat& M,
               const arma::vec& m,
               const arma::mat& X,
               const arma::vec& y,
               double sigmasq){
  
  arma::mat fc_covariance = arma::inv_sympd(
    arma::inv_sympd(M) + 1.0/sigmasq * X.t() * X );
  arma::vec fc_mean = fc_covariance * (arma::inv_sympd(M)*m + 1.0/sigmasq * X.t() * y);
  
  // L L^T = C where C is covariance matrix
  // L is lower triangular matrix
  // cholesky = find L
  arma::mat fc_covariance_chol = arma::chol(fc_covariance, "lower");
  
  int p = X.n_cols;
  arma::vec u = arma::randn(p);
  
  // u ~ N(0, I)
  // x ~ N(m, C)
  // L L^T = C
  // m + L * u ~ N(m, C)
  // because cov(Lu) = L L^T = C
  arma::vec beta_fc = fc_mean + fc_covariance_chol * u;
  
  return beta_fc;
} 

double full_cond_sigmasq(double a, double b, const arma::mat& X,
                       const arma::vec& y, 
                       const arma::vec& beta){
  
  int n = y.n_elem;
  double a_sigma = n/2.0 + a;
  arma::vec yxb = y - X*beta;
  double b_sigma = 0.5 * arma::conv_to<double>::from(yxb.t()*yxb);
  // or 
  // double b_sigma = 0.5 * arma::accu(yxb % yxb);
  
  double sigma = 1.0/R::rgamma(a_sigma, 1.0/b_sigma);

  return sigma;
}


//[[Rcpp::export]]
Rcpp::List gibbs_sampler(const arma::vec& y,
                         const arma::mat& X,
                         const arma::vec& m,
                         const arma::mat& M,
                         double a=2, double b=1, 
                         int N=1000){
  
  int p = X.n_cols;
  int n = y.n_elem;
  
  arma::mat beta_samples = arma::mat(p, N);
  arma::vec sigmasq_samples = arma::vec(N);
  
  arma::vec beta_current = arma::zeros(p);
  double sigmasq_current = 2;
  
  
  for(int i=0; i<N; i++){
    //Rcpp::Rcout << "beta" << endl;
    beta_current = full_cond_beta(M, m, X, y, sigmasq_current);
    //Rcpp::Rcout << "sigmasq" << endl;
    sigmasq_current = full_cond_sigmasq(a, b, X, y, beta_current);
    
    beta_samples.col(i) = beta_current;
    sigmasq_samples(i) = sigmasq_current;
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_samples,
    Rcpp::Named("sigmasq") = sigmasq_samples
  );
  
  
  
  
}














