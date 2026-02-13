// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //
// ::::::::::::::::::::                                         :::::::::::::::::::: //
// ::::::::::::::::::::    fast QR utils functions              :::::::::::::::::::: //
// ::::::::::::::::::::                                         :::::::::::::::::::: //
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: //

// Utility routines

// Authors:
//           Bernardi Mauro, University of Padova
//           Last update: october 7, 2024

// [[Rcpp::depends(RcppArmadillo)]]
#include "fastQRutils.h"

// 1. Thin QR decomposition (no pivoting)
List thinqr(const Eigen::MatrixXd& X) {
  
  // get dimensions
  int n = X.rows(), p = X.cols(), min_np = std::min(n, p);
  
  // perform the thin QR decomposition of X
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(X);
  Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(n, min_np);
  Eigen::MatrixXd R = qr.matrixQR().topRows(min_np).triangularView<Upper>();
  
  // return output
  return List::create(Named("Q") = Q,
                      Named("R") = R);
}

// 2. Full QR with column pivoting
List fastqr_pivot(const Eigen::MatrixXd& X,
                  const double toler) {
  
  // declare variables
  int rank = 0;

  // QR decomposition with column pivoting
  ColPivHouseholderQR<MatrixXd> qr(X);
  
  // Optional threshold override
  if (toler >= 0) {
    qr.setThreshold(toler);
  }
  rank = qr.rank();
  
  // 1-based pivot indices
  VectorXi piv = qr.colsPermutation().indices();
  IntegerVector pivot_(piv.size());
  for (int i = 0; i < piv.size(); ++i) pivot_[i] = piv[i] + 1;
  
  // Check if pivoting actually occurred
  LogicalVector pivoted = !std::is_sorted(pivot_.begin(), pivot_.end());
  
  // return output
  return List::create(Named("qr")      = qr.matrixQR(),
                      Named("qraux")   = qr.hCoeffs(),
                      Named("pivot")   = pivot_,
                      Named("pivoted") = pivoted,
                      Named("rank")    = rank);
}

// 3. Full QR without column pivoting
List fastqr_nopivot(const Eigen::MatrixXd& X,
                    const double toler) {
  
  // QR decomposition without column pivoting
  HouseholderQR<MatrixXd> qr(X);
  
  // return output
  return List::create(Named("qr")      = qr.matrixQR(),
                      Named("qraux")   = qr.hCoeffs(),
                      Named("pivoted") = 0);
}

// 4. Get the Q matrix from the qr efficient storing
Eigen::MatrixXd qrQ(const Eigen::MatrixXd& qr,
                    const Eigen::VectorXd& tau,
                    const int rank,
                    const bool complete) {
  
  // get dimensions
  int n = qr.rows(), p = qr.cols(), k = 0;
  
  // check for k
  if (complete) {
    k = n;
  } else {
    k = rank;
  }
  if (k <= 0) {
    Rcpp::stop("* qrQ: invalid k. The resulting Q matrix would have zero columns.");
  }
  
  Eigen::MatrixXd V(n, rank);                 V.setZero();
  Eigen::VectorXd tau_trunc(rank);            tau_trunc.setZero();
  Eigen::MatrixXd Q(n, k);                    Q.setZero();
  
  // get the Q matrix
  V         = qr.leftCols(rank);
  tau_trunc = tau.head(rank);
  Eigen::HouseholderSequence<Eigen::MatrixXd, Eigen::VectorXd> Qseq(V, tau_trunc);

  // apply it to the identity to extract Q
  Q = Qseq.setLength(p) * Eigen::MatrixXd::Identity(n, k);
  
  // return output
  return Q;
}

// 4. Get the R matrix from the qr efficient storing
Eigen::MatrixXd qrR(const Eigen::MatrixXd& qr,
                    const int rank,
                    const bool complete) {
  
  // get dimensions
  int n = qr.rows(), p = qr.cols(), min_np = std::min(n, p);

  // get the R matrix
  Eigen::MatrixXd R = qr.topLeftCorner(min_np, p);
  R.triangularView<Eigen::StrictlyLower>().setZero();
  if ((rank > 0) && (rank < p)) {
    R.conservativeResize(rank, p);
  }
  Eigen::MatrixXd out(n, R.cols());
  if (complete) {
    if (rank < p) {
      out.block(0, 0, rank, R.cols()) = R;
      out.block(rank, 0, n-rank, R.cols()).setZero();
    } else {
      out.block(0, 0, R.cols(), R.cols()) = R;
      out.block(R.cols(), 0, n-R.cols(), R.cols()).setZero();
    }
  } else {
    if (rank < p) {
      out.resize(rank, R.cols());
      out = R;
    } else {
      out.resize(R.cols(), R.cols());
      out = R;
    }
  }
  
  // return output
  return out;
}

Eigen::MatrixXd qr_piv2permmat(const Eigen::VectorXi& pivot) {
  
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

// get the complete Q matrix from the efficient qr storing
Eigen::MatrixXd qr_Q_raw2full(const Eigen::MatrixXd& qr,
                              const Eigen::VectorXd& hcoeffs) {
  
  int n = qr.rows(), p = qr.cols();
  using HouseholderSeq = HouseholderSequence<MatrixXd, VectorXd>;

  // Costruisce Q_full applicando i riflettori a una matrice identità n x n
  HouseholderSeq householder = HouseholderSeq(qr, hcoeffs).setLength(p);
  MatrixXd Q                 = householder * MatrixXd::Identity(n, n);

  // return output
  return Q;
}

// get the complete Q matrix from the reduced Q matrix
Eigen::MatrixXd qr_Q_complete(const Eigen::MatrixXd& Q) {
  
  // variables declaration
  const Index n = Q.rows();
  const Index k = Q.cols();
  double norm_v = 0.0;
  
  // matrices and vectors declaration
  MatrixXd Q_full(n, n);                      Q_full.setZero();
  
  // starting point
  Q_full.leftCols(k) = Q;

  // Fill the rest with an identity basis, then orthonormalize
  for (Index j=k; j<n; j++) {
    VectorXd v = VectorXd::Zero(n);
    v(j)       = 1.0;

    // Modified Gram-Schmidt orthogonalization
    for (Index i=0; i<j; i++) {
      v -= Q_full.col(i).dot(v) * Q_full.col(i);
    }
    norm_v = v.norm();
    if (norm_v < 1e-10) Rcpp::stop("qr_Q_complete: linearly dependent or near-zero vector encountered");
    Q_full.col(j) = v / norm_v;
  }

  // return output
  return Q_full;
}

/* linear model                 */

// Questa funzione calcola Q^Ty a partire dalla seguenza di Householder
// usando anche il rank di X calcolato da qr_fast(X). La funzione qundi
// si adatta anche al caso in cui X su cui è calcolata la QR sia rank deficient.
Eigen::VectorXd qr_Qty_rank(const Eigen::MatrixXd& qr,
                            const Eigen::VectorXd& tau,
                            const Eigen::VectorXd& y) {

  using Eigen::VectorXd;
  using Eigen::MatrixXd;
  using Eigen::HouseholderSequence;

  const int n = qr.rows();
  const int p = qr.cols();
  int k = 0;
  if (y.size() != n) {
    Rcpp::stop("nrow(qr) must match length(y).");
  }
  k = std::min(std::min(n, p),
               static_cast<int>(tau.size()));
  if (k <= 0) {
    Rcpp::stop("Invalid QR dimensions.");
  }
  
  Eigen::MatrixXd V(n, k);                 V.setZero();
  Eigen::VectorXd tau_trunc(k);            tau_trunc.setZero();
  
  // get the Qseq matrix
  V         = qr.leftCols(k);
  tau_trunc = tau.head(k);
  Eigen::HouseholderSequence<Eigen::MatrixXd, Eigen::VectorXd> Qseq(V, tau_trunc);

  // compute
  VectorXd Qty_full = Qseq.setLength(k).transpose() * y;  // length n
  
  // return output
  return (Qty_full);
}

// Questa funzione calcola Qy a partire dalla seguenza di Householder
// usando anche il rank di X calcolato da qr_fast(X). La funzione qundi
// si adatta anche al caso in cui X su cui è calcolata la QR sia rank deficient.
Eigen::VectorXd qr_Qy_rank(const Eigen::MatrixXd& qr,
                           const Eigen::VectorXd& tau,
                           const Eigen::VectorXd& y) {
  
  using Eigen::VectorXd;
  using Eigen::MatrixXd;
  using Eigen::HouseholderSequence;

  const int n = qr.rows();
  const int p = qr.cols();
  int k = 0;
  if (y.size() != n) {
    Rcpp::stop("nrow(qr) must match length(y).");
  }
  k = std::min(std::min(n, p),
               static_cast<int>(tau.size()));
  if (k <= 0) {
    Rcpp::stop("Invalid QR dimensions.");
  }
  
  Eigen::MatrixXd V(n, k);                 V.setZero();
  Eigen::VectorXd tau_trunc(k);            tau_trunc.setZero();
  
  // get the Qseq matrix
  V         = qr.leftCols(k);
  tau_trunc = tau.head(k);
  Eigen::HouseholderSequence<Eigen::MatrixXd, Eigen::VectorXd> Qseq(V, tau_trunc);

  // compute
  VectorXd Qy_full = Qseq.setLength(k) * y;
  
  // return output
  return (Qy_full);
}

Eigen::VectorXd qr_coef_rank(const Eigen::MatrixXd& qr,
                             const Eigen::VectorXd& tau,
                             const Eigen::VectorXd& y,
                             const int rank) {
  
  using Eigen::VectorXd;
  using Eigen::MatrixXd;
  using Eigen::HouseholderSequence;

  const int n = qr.rows();
  const int p = qr.cols();
  int k = 0;
  if (y.size() != n) {
    Rcpp::stop("nrow(qr) must match length(y).");
  }
  k = std::min(std::min(n, p),
               static_cast<int>(tau.size()));
  if (k <= 0) {
    Rcpp::stop("Invalid QR dimensions.");
  }
  if (rank < 0 || rank > k) {
    Rcpp::stop("'rank' out of range.");
  }

  Eigen::MatrixXd V(n, k);                 V.setZero();
  Eigen::VectorXd tau_trunc(k);            tau_trunc.setZero();
  
  // get the Qseq matrix
  V         = qr.leftCols(k);
  tau_trunc = tau.head(k);
  Eigen::HouseholderSequence<Eigen::MatrixXd, Eigen::VectorXd> Qseq(V, tau_trunc);

  // compute
  VectorXd Qty = Qseq.setLength(k).transpose() * y;
  
  // Solve R11 * b = Qty1  (R11 is rank×rank)
  MatrixXd R11 = qr.topLeftCorner(rank, rank).template triangularView<Eigen::Upper>();
  VectorXd b   = R11.template triangularView<Eigen::Upper>().solve(Qty.head(rank));

  // padded in pivoted order
  VectorXd beta_piv   = VectorXd::Zero(p);
  beta_piv.head(rank) = b;

  // return output
  return beta_piv;
}

Eigen::VectorXd qr_fitted_rank(const Eigen::MatrixXd& qr,
                               const Eigen::VectorXd& tau,
                               const Eigen::VectorXd& y) {
  using Eigen::VectorXd;
  using Eigen::MatrixXd;
  using Eigen::HouseholderSequence;

  const int n = qr.rows();
  const int p = qr.cols();
  int k = 0;
  if (y.size() != n) {
    Rcpp::stop("nrow(qr) must match length(y).");
  }
  k = std::min(std::min(n, p),
               static_cast<int>(tau.size()));
  if (k <= 0) {
    Rcpp::stop("Invalid QR dimensions.");
  }
  
  Eigen::MatrixXd V(n, k);                 V.setZero();
  Eigen::VectorXd tau_trunc(k);            tau_trunc.setZero();
  
  // get the Qseq matrix
  V         = qr.leftCols(k);
  tau_trunc = tau.head(k);
  Eigen::HouseholderSequence<Eigen::MatrixXd, Eigen::VectorXd> Qseq(V, tau_trunc);
  
  // thin: Q1^T y is first k components of full Q^T y
  VectorXd Qty = Qseq.setLength(k).transpose() * y;
  
  // zero out complement and apply Q back: yhat = Q * [Qty; 0]
  VectorXd tmp  = VectorXd::Zero(n);
  tmp.head(k)   = Qty.head(k);
  VectorXd yhat = Qseq.setLength(k) * tmp;
  
  // return output
  return yhat;
}

Rcpp::List qr_lm1(const Eigen::VectorXd& y,
                  const Eigen::MatrixXd& X) {
  
  // get dimensions
  int n = X.rows(), p = X.cols();

  // QR decomposition
  const HouseholderQR<MatrixXd> QR(X);
  const VectorXd betahat(QR.solve(y));
  const VectorXd fitted(X * betahat);
  const VectorXd resid(y - fitted);
  const int df(n - p);
  const VectorXd se(QR.matrixQR().topRows(p).triangularView<Upper>().solve(MatrixXd::Identity(p,p)).rowwise().norm());
  
  // return output
  return List::create(Named("beta")   = betahat,
                      Named("fitted") = fitted,
                      Named("resid")  = resid,
                      Named("df")     = df,
                      Named("se")     = se);
}

// [[Rcpp::export]]
Rcpp::List qr_lm_pred (const Eigen::VectorXd& y,
                       const Eigen::MatrixXd& X,
                       const Eigen::MatrixXd& X_test) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  unsigned int n = 0, p = 0, q = 0, n0 = 0, p0 = 0;
  double res2 = 0.0, y_norm2 = 0.0, y_mean = 0.0, df = 0.0;
  Rcpp::List output;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get dimensions                  */
  n  = X.rows();
  p  = X.cols();
  q  = y.size();
  n0 = X_test.rows();
  p0 = X_test.cols();
  
  // checks
  if (n < p) {
    Rcpp::warning("* qr_lm_pred : the number of rows of X is less than the number of columns of X!\n");
  }
  if (n != q) {
    Rcpp::stop("* qr_lm_pred : the number of rows of X is not equal to the number of elements of y!\n");
  }
  if (p0 != p) {
    Rcpp::stop("* qr_lm_pred : dimension of X and X_test not conformable!\n");
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  Eigen::MatrixXd XTX(p, p);                    XTX.setZero();
  Eigen::VectorXd XTy(p);                       XTy.setZero();
  Eigen::VectorXd regp(p);                      regp.setZero();
  Eigen::VectorXd se(p);                        se.setZero();
  Eigen::VectorXd resid(p);                     resid.setZero();
  Eigen::VectorXd fitted(n);                    fitted.setZero();
  Eigen::VectorXd predicted(n0);                predicted.setZero();
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get the Householder QR decomposition without pivoting    */
  const HouseholderQR<MatrixXd> QR(X);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get output                                                */
  regp      = QR.solve(y);
  fitted    = X * regp;
  resid     = y - fitted;
  df        = n - p;
  se        = QR.matrixQR().topRows(p).triangularView<Upper>().solve(MatrixXd::Identity(p,p)).rowwise().norm();
  res2      = resid.norm();
  y_mean    = y.mean();
  y_norm2   = y.transpose() * y - n * std::pow(y_mean, 2);
  predicted = X_test * regp;
  XTX       = X.transpose() * X;
  XTy       = X.transpose() * y;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get output                                           */
  output = Rcpp::List::create(Rcpp::Named("coeff")           = regp,
                              Rcpp::Named("coeff.se")        = se,
                              Rcpp::Named("fitted")          = fitted,
                              Rcpp::Named("residuals")       = resid,
                              Rcpp::Named("residuals_norm2") = res2,
                              Rcpp::Named("y_norm2")         = y_norm2,
                              Rcpp::Named("XTX")             = XTX,
                              Rcpp::Named("XTy")             = XTy,
                              Rcpp::Named("sigma2_hat")      = res2 / (n - p),
                              Rcpp::Named("df")              = df,
                              Rcpp::Named("R2")              = 1.0 - (res2 / y_norm2),
                              Rcpp::Named("predicted")       = predicted);
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return output;
}

Rcpp::List qr_lm_nopred (const Eigen::VectorXd& y,
                         const Eigen::MatrixXd& X) {
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   variable declaration                                  */
  unsigned int n = 0, p = 0, q = 0;
  double res2 = 0.0, y_norm2 = 0.0, y_mean = 0.0, df = 0.0, sig2_hat = 0.0;
  Rcpp::List output;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get dimensions                  */
  n  = X.rows();
  p  = X.cols();
  q  = y.size();
  
  // checks
  if (n < p) {
    Rcpp::warning("* qr_lm_nopred : the number of rows of X is less than the number of columns of X!\n");
  }
  if (n != q) {
    Rcpp::stop("* qr_lm_pred : the number of rows of X is not equal to the number of elements of y!\n");
  }
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   vector and matrices declaration                       */
  Eigen::MatrixXd XTX(p, p);                    XTX.setZero();
  Eigen::MatrixXd R(p, p);                      R.setZero();
  Eigen::MatrixXd L(p, p);                      L.setZero();
  Eigen::VectorXd XTy(p);                       XTy.setZero();
  Eigen::VectorXd regp(p);                      regp.setZero();
  Eigen::VectorXd se(p);                        se.setZero();
  Eigen::VectorXd resid(p);                     resid.setZero();
  Eigen::VectorXd fitted(n);                    fitted.setZero();
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get the Householder QR decomposition without pivoting    */
  const HouseholderQR<MatrixXd> QR(X);
  
  /* :::::::::::::::::::::::::::::::::::::::::
   Get output                                                */
  regp      = QR.solve(y);
  fitted    = X * regp;
  resid     = y - fitted;
  df        = n - p;
  R         = QR.matrixQR().topRows(p).triangularView<Upper>();
  L         = QR.matrixQR().topRows(p).triangularView<Upper>().solve(MatrixXd::Identity(p,p));
  se        = L.rowwise().norm();
  res2      = pow(resid.norm(), 2);
  y_mean    = y.mean();
  y_norm2   = y.transpose() * y - n * std::pow(y_mean, 2);
  XTX       = R.transpose() * R;
  XTy       = X.transpose() * y;
  sig2_hat  = res2 / (double)df;
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   get output                                           */
  output = Rcpp::List::create(Rcpp::Named("coeff")           = regp,
                              Rcpp::Named("coeff.se")        = se * std::sqrt(sig2_hat),
                              Rcpp::Named("fitted")          = fitted,
                              Rcpp::Named("residuals")       = resid,
                              Rcpp::Named("residuals_norm2") = res2,
                              Rcpp::Named("y_norm2")         = y_norm2,
                              Rcpp::Named("R")               = R,
                              Rcpp::Named("L")               = L,
                              Rcpp::Named("XTX")             = XTX,
                              Rcpp::Named("XTX_INV")         = L * L.transpose(),
                              Rcpp::Named("XTy")             = XTy,
                              Rcpp::Named("sigma2_hat")      = sig2_hat,
                              Rcpp::Named("df")              = df,
                              Rcpp::Named("R2")              = 1.0 - (res2 / y_norm2));
  
  /* ::::::::::::::::::::::::::::::::::::::::::::::
   return output                                        */
  return output;
}




// end of file
