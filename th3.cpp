#include <iostream> 
#include <mlpack/core.hpp> 
#include <mlpack/methods/linear_regression/linear_regression.hpp> 
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/core/hpt/hyper_parameter_tuner.hpp> 
#include <mlpack/core/hpt/simple_cv.hpp> 
using namespace std; 
using namespace mlpack; 
using namespace mlpack::metric; 
using namespace mlpack::hpt; 

using arma::mat; 
using arma::rowvec; 
using arma::vec; 

pair<mat, rowvec> generate_data(size_t num_samples) { 
    mat samples = arma::randn<mat>(1, num_samples);  
    rowvec labels = samples.row(0) + arma::randn<rowvec>(num_samples, arma::distr_param(0.0, 0.5)); 
    return {samples, labels}; 
} 
 
int main() { 
    size_t num_samples = 200; 
    auto [raw_samples, raw_labels] = generate_data(num_samples); 
 
    data::StandardScaler sample_scaler; 
    sample_scaler.Fit(raw_samples); 
    mat samples; 
    sample_scaler.Transform(raw_samples, samples); 
 
    data::StandardScaler label_scaler; 
    label_scaler.Fit(raw_labels); 
    rowvec labels; 
    label_scaler.Transform(raw_labels, labels); 
 
    double validation_size = 0.2;  
 
    HyperParameterTuner< 
        regression::LinearRegression,  
        MSE,  
        SimpleCV 
    > parameters_tuner(validation_size, samples, labels); 
 
    vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0, 10.0};  
 
    double best_lambda; 
    std::tie(best_lambda) = parameters_tuner.Optimize(lambdas); 
 
    cout << "--- Kết quả Grid Search ---" << endl; 
    cout << "Các giá trị Lambda đã thử:" << endl << lambdas.t() << endl; 
    cout << "Giá trị Lambda tối ưu tìm được: " << best_lambda << endl; 
 
    auto& best_model = parameters_tuner.BestModel(); 
    mat new_sample = { { 0.5 } };  
    rowvec prediction; 
    best_model.Predict(new_sample, prediction); 
 
    double final_prediction = label_scaler.InverseTransform(prediction(0)); 
    cout << "\nDự đoán (Inverse Transform) cho x=0.5 (Standardized): " << final_prediction << endl; 
 
    return 0; 
} 
