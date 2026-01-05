#include <iostream> 
#include <Eigen/Dense> 
#include <Eigen/IterativeLinearSolvers> 
 
using namespace Eigen; 
using namespace std; 
 
typedef float DType; 
using Matrix = Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic>; 
using Vector = Eigen::Matrix<DType, Eigen::Dynamic, 1>; 
 
void generate_data(Matrix& X, Vector& y, int n) { 
    X.resize(n, 2); 
    y.resize(n, 1); 
     
    X.col(1) = Vector::Random(n, 1).array() * 5.0f + 5.0f;  
     
    X.col(0).setOnes(); 
 
    y = X.col(0) * 1.0f + X.col(1) * 2.0f + Vector::Random(n, 1) * 0.1f; 
} 
 
int main() { 
    int n_samples = 10000; 
    Matrix X_train, X_new; 
    Vector y_train; 
     
    generate_data(X_train, y_train, n_samples); 
 
    Matrix XtX = X_train.transpose() * X_train; 
    Vector Xty = X_train.transpose() * y_train; 
 
    Vector beta_ols = XtX.ldlt().solve(Xty); 
 
    cout << "--- Giải pháp Giải tích (OLS) ---" << endl; 
    cout << "Hệ số ước tính (beta_0, beta_1):" << endl << beta_ols.transpose() << endl; 
 
    Eigen::LeastSquaresConjugateGradient<Matrix> gd; 
     
    gd.setMaxIterations(1000);
    gd.setTolerance(0.001);
     
    gd.compute(X_train); 
     
    Vector beta_gd = gd.solve(y_train); 
 
    cout << "\n--- Giải pháp Lặp (Conjugate Gradient) ---" << endl; 
    cout << "Hệ số ước tính (beta_0, beta_1):" << endl << beta_gd.transpose() << endl; 
 
    X_new.resize(5, 2); 
    X_new << 1, 0.5,
             1, 3.0,
             1, 5.0,
             1, 8.5,
             1, 10.0;
    
    Vector y_pred = X_new * beta_ols; 
    cout << "\n--- Dự đoán trên dữ liệu mới ---" << endl; 
    cout << "Dữ liệu mới X (cột 1: bias, cột 2: x):\n" << X_new.transpose() << endl; 
    cout << "Dự đoán Y_pred:\n" << y_pred.transpose() << endl; 
    return 0; 
} 
