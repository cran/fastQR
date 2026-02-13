// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //
// ::::::::::::::::::::                 :::::::::::::::::::: //
// ::::::::::::::::::::    fastQR wrap  :::::::::::::::::::: //
// ::::::::::::::::::::                 :::::::::::::::::::: //
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

#include "fastQRutils.h"

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace Eigen;
using namespace arma;

//' @name qr_thin
//' @title Fast thin QR decomposition
//' @description qr_thin provides the thin QR factorization of the matrix \eqn{X\in\mathbb{R}^{n\times p}} with \eqn{n>p}. The thin QR factorization of the matrix \eqn{X} returns the matrices \eqn{Q\in\mathbb{R}^{n\times p}} and the upper triangular matrix \eqn{R\in\mathbb{R}^{p\times p}} such that \eqn{X=QR}. See Golub and Van Loan (2013) for further details on the method.
//' @param X a \eqn{n\times p} matrix with \eqn{n>p}.
//' @return A named list containing \describe{
//' \item{Q}{the Q matrix.}
//' \item{R}{the R matrix.}
//' }
//'
//' @examples
//' ## generate sample data
//' set.seed(1234)
//' n <- 12
//' p <- 5
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the thin QR factorization
//' output <- qr_thin(X = X)
//' Q      <- output$Q
//' R      <- output$R
//'
//' ## check
//' max(abs(Q %*% R - X))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Rcpp::List qr_thin (const Eigen::MatrixXd& X) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get dimensions                                        */
  int n = X.rows(), p = X.cols();
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  List output;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   checks                                           */
  if (p > n) {
    Rcpp::stop("* qr_thin: if n is smaller than p the QR decomposition can not be computed!\n");
  } else {
    output = thinqr(X);
  }
  
  /* return output          */
  return output;
}

//' @name qr_fast
//' @title Fast full QR decomposition
//' @description qr_fast provides the fast QR factorization of the matrix \eqn{X\in\mathbb{R}^{n\times p}} with \eqn{n>p}. The full QR factorization of the matrix \eqn{X} returns the matrices \eqn{Q\in\mathbb{R}^{n\times p}} and the upper triangular matrix \eqn{R\in\mathbb{R}^{p\times p}} such that \eqn{X=QR}. See Golub and Van Loan (2013) for further details on the method.
//' @param X a \eqn{n\times p} matrix with \eqn{n>p}.
//' @param tol the tolerance for detecting linear dependencies in the columns of \eqn{X}.
//' @param pivot a logical value indicating whether to pivot the columns of \eqn{X}. Defaults to FALSE, meaning no pivoting is performed.
//' @return A named list containing \describe{
//' \item{qr}{a matrix with the same dimensions as \eqn{X}. The upper triangle contains the \eqn{R} of the decomposition and the lower triangle contains information on the \eqn{Q} of the decomposition (stored in compact form).}
//' \item{qraux}{a vector of length ncol(x) which contains additional information on \eqn{Q}.}
//' \item{rank}{the rank of \eqn{X} as computed by the decomposition.}
//' \item{pivot}{information on the pivoting strategy used during the decomposition.}
//' \item{pivoted}{a boolean variable returning one if the pivoting has been performed and zero otherwise.}
//' }
//' @details The QR decomposition plays an important role in many statistical techniques. In particular it can be used to solve the equation \eqn{Ax=b} for given matrix \eqn{A\in\mathbb{R}^{n\times p}} and vectors \eqn{x\in\mathbb{R}^{p}} and \eqn{b\in\mathbb{R}^{n}}. It is useful for computing regression coefficients and in applying the Newton-Raphson algorithm.
//'
//' @examples
//' ## generate sample data
//' set.seed(1234)
//' n <- 12
//' p <- 5
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## reconstruct the reduced Q and R matrices
//' ## reduced Q matrix
//' Q1 <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux,
//'            rank = qr_res$rank, complete = FALSE)
//' Q1
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q1)-diag(1, p)))
//'
//' ## complete Q matrix
//' Q2 <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux,
//'            rank = NULL, complete = TRUE)
//' Q2
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q2)-diag(1, n)))
//'
//' ## reduced R matrix
//' R1 <- qr_R(qr = qr_res$qr,
//'            rank = NULL,
//'            complete = FALSE)
//'
//' ## check that X^TX = R^TR
//' ## get the permutation matrix
//' P <- qr_pivot2perm(pivot = qr_res$pivot)
//' max(abs(crossprod(R1 %*% P) - crossprod(X)))
//' max(abs(crossprod(R1) - crossprod(X %*% t(P))))
//'
//' ## complete R matrix
//' R2 <- qr_R(qr = qr_res$qr,
//'            rank = NULL,
//'            complete = TRUE)
//'
//' ## check that X^TX = R^TR
//' ## get the permutation matrix
//' P <- qr_pivot2perm(pivot = qr_res$pivot)
//' max(abs(crossprod(R2 %*% P) - crossprod(X)))
//' max(abs(crossprod(R2) - crossprod(X %*% t(P))))
//'
//' ## check that X = Q %*% R
//' max(abs(Q2 %*% R2 %*% P - X))
//' max(abs(Q1 %*% R1 %*% P - X))
//'
//' ## create data: n > p
//' set.seed(1234)
//' n <- 120
//' p <- 75
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X, pivot = FALSE)
//'
//' ## reconstruct the reduced Q and R matrices
//' ## reduced Q matrix
//' Q1 <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux,
//'            rank = p,
//'            complete = FALSE)
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q1)-diag(1, p)))
//'
//' ## complete Q matrix
//' Q2 <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux,
//'            rank = NULL, complete = TRUE)
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q2)-diag(1, n)))
//'
//' ## reduced R matrix
//' R1 <- qr_R(qr = qr_res$qr,
//'            rank = NULL,
//'            complete = FALSE)
//'
//'
//' ## check that X^TX = R^TR
//' max(abs(crossprod(R1) - crossprod(X)))
//'
//' ## complete R matrix
//' R2 <- qr_R(qr = qr_res$qr,
//'            rank = NULL,
//'            complete = TRUE)
//'
//' ## check that X^TX = R^TR
//' max(abs(crossprod(R2) - crossprod(X)))
//'
//' ## check that X^TX = R^TR
//' max(abs(crossprod(R2) - crossprod(X)))
//' max(abs(crossprod(R2) - crossprod(X)))
//' max(abs(crossprod(R1) - crossprod(X)))
//'
//' # check that X = Q %*% R
//' max(abs(Q2 %*% R2 - X))
//' max(abs(Q1 %*% R1 - X))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Rcpp::List qr_fast (const Eigen::MatrixXd& X,
                    Rcpp::Nullable<double> tol = R_NilValue,
                    Rcpp::Nullable<bool> pivot = R_NilValue) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  double tol_ = 0.0;
  bool pivot_ = false;
  Rcpp::List output;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   check for NULL:                                    */
  if (tol.isNotNull()) {
    tol_ = Rcpp::as<double>(tol);
  } else {
    tol_ = SQRT_DOUBLE_EPS;
  }
  if (pivot.isNotNull()) {
    pivot_ = Rcpp::as<bool>(pivot);
  } else {
    pivot_ = false;
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
    perform full QR decomposition                      */
  /*if (p > n) {
    Rcpp::stop("* qr_fast: if n is smaller than p the QR decomposition can not be computed!\n");
  } else {
    if (pivot_) {
      output = fastqr_pivot(X, tol_);
    } else {
      output = fastqr_nopivot(X, tol_);
    }
  }*/
  if (pivot_) {
    output = fastqr_pivot(X, tol_);
  } else {
    output = fastqr_nopivot(X, tol_);
  }
  
  /* return output          */
  return output;
}

//' @name qr_Q
//' @title Reconstruct the Q, matrix from a QR object.
//' @description returns the \eqn{Q} matrix of the full QR decomposition. If \eqn{r = \mathrm{rank}(X) < p}, then only the reduced \eqn{Q \in \mathbb{R}^{n \times r}} matrix is returned.
//' @param qr object representing a QR decomposition. This will typically have come from a previous call to qr.
//' @param tau a vector of length \eqn{ncol(X)} which contains additional information on \eqn{Q}. It corresponds to qraux from a previous call to qr.
//' @param rank the rank of x as computed by the decomposition.
//' @param complete logical flag (length 1). Indicates whether to compute the full \eqn{Q \in \bold{R}^{n \times n}} or the thin \eqn{Q \in \bold{R}^{n \times p}}. If \eqn{r = \mathrm{rank}(X) < p}, then only the reduced \eqn{Q \in \mathbb{R}^{n \times r}} matrix is returned.
//' @return returns part or all of \eqn{Q}, the order-\eqn{n} orthogonal (unitary) transformation represented by qr. If complete is TRUE, \eqn{Q} has \eqn{n} columns. If complete is FALSE, \eqn{Q} has \eqn{p} columns.
//'
//' @examples
//' ## generate sample data
//' set.seed(1234)
//' n <- 12
//' p <- 5
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## get the full Q matrix
//' Q1 <- qr_Q(qr_res$qr, qr_res$qraux, complete = TRUE)
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q1)-diag(1, n)))
//'
//' ## get the reduced Q matrix
//' Q2 <- qr_Q(qr_res$qr, qr_res$qraux, qr_res$rank, complete = FALSE)
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q2)-diag(1, p)))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Eigen::MatrixXd qr_Q (const Eigen::MatrixXd& qr,
                      const Eigen::VectorXd& tau,
                      Rcpp::Nullable<int> rank = R_NilValue,
                      Rcpp::Nullable<bool> complete = R_NilValue) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  int rank_ = 0, p = qr.cols();
  bool complete_ = false;
  Rcpp::List output;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   check for NULL:                                    */
  if (rank.isNotNull()) {
    rank_ = Rcpp::as<int>(rank);
    // check the rank
    if ((rank_ > p) || (rank_ > tau.size())) {
      Rcpp::stop("* qr_Q: invalid rank. It exceeds QR dimensions or tau length.");
    }
    if (rank_ < p) {
      Rcpp::warning("* qr_Q: 'complete' has been set to FALSE because the rank of X is less than the number of columns (p)!");
      complete_ = false;
    }
  } else {
    rank_ = p;
  }
  if (complete.isNotNull()) {
    complete_ = Rcpp::as<bool>(complete);
  } else {
    complete_ = false;
  }
  if (!rank.isNotNull() && (complete_ == false)) {
    Rcpp::stop("* qr_Q: the 'rank' input is missing; setting 'complete' to TRUE by default is therefore required.");
  }
  if ((!rank.isNotNull()) && (complete_ == true)) {
    Rcpp::warning("* qr_Q: the 'rank' input is missing and it has been set equal to p.");
    rank_ = p;
  }

  /* ::::::::::::::::::::::::::::::::::::::::::::::
    get the Q matrix                                */
  Eigen::MatrixXd Q = qrQ(qr, tau, rank_, complete_);
  
  /* return output          */
  return Q;
}

//' @name qr_R
//' @title Reconstruct the R, matrix from a QR object.
//' @description returns the \eqn{R} matrix of the full QR decomposition. If \eqn{r = \mathrm{rank}(X) < p}, then only the reduced \eqn{R \in \mathbb{R}^{r \times p}} matrix is returned.
//' @param qr object representing a QR decomposition. This will typically have come from a previous call to qr.
//' @param rank the rank of x as computed by the decomposition.
//' @param pivot a logical value indicating whether to pivot the columns of \eqn{X}. Defaults to FALSE, meaning no pivoting is performed.
//' @param complete logical flag (length 1). Indicates whether the \eqn{R} matrix is to be completed by binding zero-value rows beneath the square upper triangle. If \eqn{r = \mathrm{rank}(X) < p}, then only the reduced \eqn{R \in \mathbb{R}^{r \times p}} matrix is returned.
//' @param pivot a vector of length \eqn{p}, specifying the permutation of the columns of \eqn{X} applied during the QR decomposition process. The default is NULL if no pivoting has been applied.
//' @return returns part or all of \eqn{R}. If complete is TRUE, \eqn{R} has \eqn{n} rows. If complete is FALSE, \eqn{R} has \eqn{p} rows.
//'
//' @examples
//' ## generate sample data
//' set.seed(1234)
//' n <- 12
//' p <- 5
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## get the full R matrix
//' R1 <- qr_R(qr_res$qr, complete = TRUE)
//'
//' ## check that X^TX = R^TR
//' ## get the permutation matrix
//' P <- qr_pivot2perm(pivot = qr_res$pivot)
//' max(abs(crossprod(R1 %*% P) - crossprod(X)))
//' max(abs(crossprod(R1) - crossprod(X %*% t(P))))
//'
//' ## get the reduced R matrix
//' R2 <- qr_R(qr_res$qr, qr_res$rank, complete = FALSE)
//'
//' ## check that X^TX = R^TR
//' ## get the permutation matrix
//' P <- qr_pivot2perm(pivot = qr_res$pivot)
//' max(abs(crossprod(R2 %*% P) - crossprod(X)))
//' max(abs(crossprod(R2) - crossprod(X %*% t(P))))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Eigen::MatrixXd qr_R (const Eigen::MatrixXd& qr,
                      Rcpp::Nullable<int> rank = R_NilValue,
                      Rcpp::Nullable<Rcpp::IntegerVector> pivot = R_NilValue,
                      Rcpp::Nullable<bool> complete = R_NilValue) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  int rank_ = 0, n = qr.rows(), p = qr.cols();
  bool complete_ = false;
  Rcpp::List output;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   check for NULL:                                    */
  if (rank.isNotNull()) {
    rank_ = Rcpp::as<int>(rank);
    if (rank_ < p) {
      Rcpp::warning("* qr_R: 'complete' has been set to FALSE because the rank of X is less than the number of columns (p)!");
      complete_ = false;
    }
  } else {
    rank_ = p;
  }
  if (complete.isNotNull()) {
    complete_ = Rcpp::as<bool>(complete);
  } else {
    complete_ = false;
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
    get the R matrix                                */
  if (pivot.isNotNull()) {
    /* Nullable output declaration */
    Rcpp::IntegerVector pivot_tmp(pivot);
    
    // Transform Rcpp vector "vec_rcpp" into an Armadillo vector
    Eigen::VectorXi pivot_ = Rcpp::as<Eigen::VectorXi>(wrap(pivot_tmp));
    
    // get the R matrix
    Eigen::MatrixXd R_ = qrR(qr, rank_, complete_);
    
    // Get the permutation as a PermutationMatrix
    Eigen::MatrixXd P  = Eigen::MatrixXd::Zero(n, n);
    P                  = qr_piv2permmat(pivot_);
    
    // create the R matrix with permutation
    Rcpp::warning("* qr_R: the R matrix has been permuted according to the provided pivots elements!");
    Eigen::MatrixXd R = R_ * P;
    return R;
  } else {
    // create the R matrix without permutation
    Eigen::MatrixXd R = qrR(qr, rank_, complete_);
    return R;
  }
}

//' @name qr_pivot2perm
//' @title Reconstruct the permutation matrix from the pivot vector.
//' @description returns the permutation matrix for the QR decomposition.
//' @param pivot a vector of dimension \eqn{n} of pivot elements from the QR factorization.
//' @return the perumutation matrix \eqn{P} of dimension \eqn{n \times n}.
//'
//' @examples
//' ## generate sample data
//' set.seed(1234)
//' n <- 12
//' p <- 5
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## get the pivot matrix
//' P <- qr_pivot2perm(qr_res$pivot)
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Eigen::MatrixXd qr_pivot2perm(const Eigen::VectorXi& pivot) {
  
  // Get the size
  int n = pivot.size();

  // Get the permutation as a PermutationMatrix
  Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n);

  // create the permutation matrix
  for (int i=0; i<n; i++) {
    P(i, pivot[i]-1) = 1.0;
  }
  
  // return output          
  return P;
}

//' @name qr_X
//' @title Reconstruct the original matrix from which the object was constructed \eqn{X\in\mathbb{R}^{n\times p}} from the Q and R matrices of the QR decomposition.
//' @description returns the \eqn{X\in\mathbb{R}^{n\times p}} matrix.
//' @param Q either the reduced \eqn{Q\in\mathbb{R}^{n\times p}} of full \eqn{Q\in\mathbb{R}^{n\times n}}, Q matrix obtained from the QR decomposition.
//' @param R either the reduced \eqn{R\in\mathbb{R}^{p\times p}} of full \eqn{R\in\mathbb{R}^{n\times p}}, R matrix obtained from the QR decomposition.
//' @param pivot a vector of length \eqn{p}, specifying the permutation of the columns of \eqn{X} applied during the QR decomposition process. The default is NULL if no pivoting has been applied.
//' @return returns the matrix \eqn{X}.
//'
//' @examples
//' ## generate sample data
//' set.seed(1234)
//' n <- 12
//' p <- 5
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X, pivot = TRUE)
//'
//' ## get the Q and R matrices
//' Q  <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux, rank = qr_res$rank, complete = TRUE)
//' R  <- qr_R(qr = qr_res$qr, rank = qr_res$rank, complete = TRUE)
//' X1 <- qr_X(Q = Q, R = R, pivot = qr_res$pivot)
//' max(abs(X1 - X))
//'
//' ## get the full QR decomposition without pivot
//' qr_res <- fastQR::qr_fast(X = X, pivot = FALSE)
//'
//' ## get the Q and R matrices
//' Q  <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux, rank = p, complete = FALSE)
//' R  <- qr_R(qr = qr_res$qr, rank = NULL, complete = FALSE)
//' X1 <- qr_X(Q = Q, R = R, pivot = NULL)
//' max(abs(X1 - X))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Eigen::MatrixXd qr_X(const Eigen::MatrixXd& Q,
                     const Eigen::MatrixXd& R,
                     Rcpp::Nullable<Rcpp::IntegerVector> pivot = R_NilValue) {
  
  // get dimensions
  int n = Q.rows(), p = R.cols();
  
  // check the dimensions
  if (Q.cols() != R.rows()) {
    Rcpp::stop("* qr_X: the provided matrices are not compatible!");
  }
  
  // variable declaration
  Eigen::MatrixXd R_ = R.triangularView<Upper>();
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, p);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   check for NULL:                                    */
  if (pivot.isNotNull()) {
    /* Nullable output declaration */
    Rcpp::IntegerVector pivot_tmp(pivot);
    
    // Transform Rcpp vector "vec_rcpp" into an Armadillo vector
    Eigen::VectorXi pivot_ = Rcpp::as<Eigen::VectorXi>(wrap(pivot_tmp));
    
    // Get the permutation as a PermutationMatrix
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n);
    P                 = qr_pivot2perm(pivot_);
    
    // create the X matrix
    X = Q * R_ * P;
  } else {
    // create the X matrix
    X = Q * R_;
  }
  
  // return output
  return X;
}

//' @name qr_Q_full
//' @title Reconstruct the full Q matrix from the qr object.
//' @description returns the full \eqn{Q\in\mathbb{R}^{n\times n}} matrix.
//' @param qr object representing a QR decomposition. This will typically have come from a previous call to qr.
//' @param tau a vector of length \eqn{ncol(X)} which contains additional information on \eqn{Q}. It corresponds to qraux from a previous call to qr.
//' @return returns the matrix \eqn{Q\in\mathbb{R}^{n\times n}}.
//'
//' @examples
//' ## create data: n > p
//' set.seed(1234)
//' n <- 12
//' p <- 7
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## complete the reduced Q matrix
//' Q <- fastQR::qr_Q_full(qr  = qr_res$qr,
//'                        tau = qr_res$qraux)
//'
//' ## check the Q matrix (orthogonality)
//' max(abs(crossprod(Q)-diag(1, n)))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Eigen::MatrixXd qr_Q_full(const Eigen::MatrixXd& qr,
                          const Eigen::VectorXd& tau) {
  
  // get dimensions
  int n = qr.rows();
  
  // variable declaration
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(n, n);
  
  // build the complete Q matrix
  Q = qr_Q_raw2full(qr, tau);

  // return output
  return Q;
}

//' @name qr_Q_reduced2full
//' @title Reconstruct the full Q matrix from the reduced Q matrix.
//' @description returns the full \eqn{Q\in\mathbb{R}^{n\times n}} matrix.
//' @param Q a \eqn{n\times p} reduced Q matrix from the QR decomposition (with \eqn{n>p}).
//' @return a \eqn{n\times n} orthogonal matrix \eqn{Q}.
//'
//' @examples
//' ## create data: n > p
//' set.seed(1234)
//' n <- 12
//' p <- 7
//' X <- matrix(rnorm(n * p), n, p)
//'
//' ## get the full QR decomposition with pivot
//' qr_res <- fastQR::qr_fast(X = X,
//'                           tol = sqrt(.Machine$double.eps),
//'                           pivot = TRUE)
//'
//' ## reconstruct the reduced Q matrix
//' Q1 <- qr_Q(qr = qr_res$qr, tau = qr_res$qraux,
//'            rank = qr_res$rank, complete = FALSE)
//'
//' ## complete the reduced Q matrix
//' Q2 <- fastQR::qr_Q_reduced2full(Q = Q1)
//' R  <- fastQR::qr_R(qr = qr_res$qr, rank = NULL, complete = TRUE)
//'
//' X1 <- qr_X(Q = Q2, R = R, pivot = qr_res$pivot)
//' max(abs(X - X1))
//'
//' @references
//' \insertRef{golub_van_loan.2013}{fastQR}
//'
//' \insertRef{bjorck.2015}{fastQR}
//'
//' \insertRef{bjorck.2024}{fastQR}
//'
//' \insertRef{bernardi_etal.2024}{fastQR}
//'
// [[Rcpp::export]]
Eigen::MatrixXd qr_Q_reduced2full(const Eigen::MatrixXd& Q) {
  
  // get dimensions
  int n = Q.rows(), p = Q.cols();
  
  // check the dimensions
  if (n == p) {
    Rcpp::stop("* qr_Q_reduced2full: the provided Q matrix is n x n!");
  }
  
  // variable declaration
  Eigen::MatrixXd Q_full = Eigen::MatrixXd::Zero(n, n);
  
  // build the complete Q matrix
  Q_full = qr_Q_complete(Q);

  // return output
  return Q_full;
}

/* This function computes the ols estimates for the linear regression model.   */

//' @name qr_lm
//' @title Ordinary least squares for the linear regression model
//' @description qr_lm, or LS for linear regression models, solves the following optimization problem
//' \deqn{\textrm{min}_\beta ~ \frac{1}{2}\|y-X\beta\|_2^2,}
//' for \eqn{y\in\mathbb{R}^n} and \eqn{X\in\mathbb{R}^{n\times p}} witn \eqn{n>p}, to obtain a coefficient vector \eqn{\widehat{\beta}\in\mathbb{R}^p}. The design matrix \eqn{X\in\mathbb{R}^{n\times p}}
//' contains the observations for each regressor.
//' @param y a vector of length-\eqn{n} response vector.
//' @param X an \eqn{(n\times p)} full column rank matrix of predictors.
//' @param X_test an \eqn{(q\times p)} full column rank matrix. Test set. By default it set to NULL.
//' @return A named list containing \describe{
//' \item{coeff}{a length-\eqn{p} vector containing the solution for the parameters \eqn{\beta}.}
//' \item{coeff.se}{a length-\eqn{p} vector containing the standard errors for the estimated regression parameters \eqn{\beta}.}
//' \item{fitted}{a length-\eqn{n} vector of fitted values, \eqn{\widehat{y}=X\widehat{\beta}}.}
//' \item{residuals}{a length-\eqn{n} vector of residuals, \eqn{\varepsilon=y-\widehat{y}}.}
//' \item{residuals_norm2}{the squared L2-norm of the residuals, \eqn{\Vert\varepsilon\Vert_2^2.}}
//' \item{y_norm2}{the squared L2-norm of the response variable, \eqn{\Vert y\Vert_2^2.}}
//' \item{R}{the \eqn{R\in\mathbb{R}^{p\times p}} upper triangular matrix of the QR decomposition.}
//' \item{L}{the inverse of the \eqn{R\in\mathbb{R}^{p\times p}} upper triangular matrix of the QR decomposition \eqn{L = R^{-1}}.}
//' \item{XTX}{the Gram matrix \eqn{X^\top X\in\mathbb{R}^{p\times p}} of the least squares problem.}
//' \item{XTX_INV}{the inverse of the Gram matrix \eqn{X^\top X\in\mathbb{R}^{p\times p}} of the least squares problem \eqn{(X^\top X)^{-1}}.}
//' \item{XTy}{A vector equal to \eqn{X^\top y}, the cross-product of the design matrix \eqn{X} with the response vector \eqn{y}.}
//' \item{sigma2_hat}{An estimate of the error variance \eqn{\sigma^2}, computed as the residual sum of squares divided by the residual degrees of freedom \eqn{\widehat{\sigma}^2 = \frac{\|y - X\hat{\beta}\|_2^2}{df}}}
//' \item{df}{The residual degrees of freedom, given by \eqn{n - p}, where \eqn{n} is the number of observations and \eqn{p} is the number of estimated parameters.}
//' \item{R2}{\eqn{R^2}, coefficient of determination, measure of goodness-of-fit of the model.}
//' \item{predicted}{predicted values for the test set, \eqn{X_{\text{test}}\widehat{\beta}}. It is only available if X_test is not NULL.}
//' }
//' @examples
//'
//' ## generate sample data
//' ## create data: n > p
//' set.seed(1234)
//' n    <- 12
//' n0   <- 3
//' p    <- 7
//' X    <- matrix(rnorm(n * p), n, p)
//' b    <- rep(1, p)
//' sig2 <- 0.25
//' y    <- X %*% b + sqrt(sig2) * rnorm(n)
//' summary(lm(y~X))
//'
//' ## test
//' X_test <- matrix(rnorm(n0 * p), n0, p)
//'
//' ## lm
//' qr_lm(y = y, X = X, X_test = X_test)
//' qr_lm(y = y, X = X)
//'
// [[Rcpp::export]]
Rcpp::List qr_lm(const Eigen::VectorXd& y,
                 const Eigen::MatrixXd& X,
                 Rcpp::Nullable<Rcpp::NumericMatrix> X_test = R_NilValue) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  Rcpp::List out;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get output                                           */
  if (X_test.isNotNull()) {
    /* Nullable output declaration */
    Rcpp::NumericMatrix X_test_tmp(X_test);
    
    // Transform Rcpp vector "vec_rcpp" into an Armadillo vector
    Eigen::MatrixXd X_test_ = Rcpp::as<Eigen::MatrixXd>(wrap(X_test_tmp));
    
    // run OLS with prediction
    out = qr_lm_pred(y, X, X_test_);
  } else {
    // run OLS
    out = qr_lm_nopred(y, X);
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return out;
}
