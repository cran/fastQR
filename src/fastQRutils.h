#ifndef FASTQRUTILS_H
#define FASTQRUTILS_H

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Eigen/Core>
#include <Eigen/Householder>

#include "QRdecomposition.h"
#include "support_functions.h"
#include "QRupdate.h"
#include "thinQRupdate.h"

#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#define DOUBLE_EPS 2.220446e-16
#define SAFE_LOG(a) (((a) <= 0.0) ? log(DOUBLE_EPS) : log(a))
#define SAFE_ZERO(a) ((a) == 0 ? DOUBLE_EPS : (a))
#define SQRT_DOUBLE_EPS sqrt(DOUBLE_EPS)

// [[Rcpp::plugins("cpp11")]]
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;
//using namespace std;
using Eigen::HouseholderQR;


List thinqr(const Eigen::MatrixXd& X);
List fastqr_pivot(const Eigen::MatrixXd& X,
                  const double toler);
List fastqr_nopivot(const Eigen::MatrixXd& X,
                    const double toler);
Eigen::MatrixXd qrQ(const Eigen::MatrixXd& qr,
                    const Eigen::VectorXd& tau,
                    const int rank,
                    const bool complete);
Eigen::MatrixXd qrR(const Eigen::MatrixXd& qr,
                    const int rank,
                    const bool complete);
Eigen::MatrixXd qr_piv2permmat(const Eigen::VectorXi& pivot);

Eigen::MatrixXd qr_Q_raw2full(const Eigen::MatrixXd& qr,
                              const Eigen::VectorXd& hcoeffs);
Eigen::MatrixXd qr_Q_complete(const Eigen::MatrixXd& Q);

/* linear model                 */
Eigen::VectorXd qr_Qty_rank(const Eigen::MatrixXd& qr,
                            const Eigen::VectorXd& tau,
                            const Eigen::VectorXd& y);
Eigen::VectorXd qr_Qy_rank(const Eigen::MatrixXd& qr,
                           const Eigen::VectorXd& tau,
                           const Eigen::VectorXd& y);
Eigen::VectorXd qr_coef_rank(const Eigen::MatrixXd& qr,
                             const Eigen::VectorXd& tau,
                             const Eigen::VectorXd& y,
                             const int rank);
Eigen::VectorXd qr_fitted_rank(const Eigen::MatrixXd& qr,
                               const Eigen::VectorXd& tau,
                               const Eigen::VectorXd& y);

Rcpp::List qr_lm1(const Eigen::MatrixXd& X,
                  const Eigen::VectorXd& y);
Rcpp::List qr_lm_pred (const Eigen::VectorXd& y,
                       const Eigen::MatrixXd& X,
                       const Eigen::MatrixXd& X_test);
Rcpp::List qr_lm_nopred (const Eigen::VectorXd& y,
                         const Eigen::MatrixXd& X);







#endif
