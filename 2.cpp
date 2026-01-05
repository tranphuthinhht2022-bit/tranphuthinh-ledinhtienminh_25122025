#include <iostream> 
#include <vector> 
#include <algorithm> 
#include <Eigen/Dense> 
#include <armadillo> 
using namespace std; 

void setup_data(vector<double>& values, size_t rows, size_t cols) { 
    values.resize(rows * cols); 
    for (size_t i = 0; i < values.size(); ++i) { 
        values[i] = static_cast<double>(i + 1.0); 
    } 
} 

int main() { 
    size_t rows = 3; 
    size_t cols = 3; 
    vector<double> raw_data; 
    setup_data(raw_data, rows, cols); 

    cout << "Dữ liệu nguồn C++ (std::vector<double>):" << endl; 
    for (size_t i = 0; i < raw_data.size(); ++i) { 
        cout << raw_data[i] << (i % cols == cols - 1? "\n" : ", "); 
    } 
    cout << "---------------------------------------" << endl; 

    Eigen::Map<Eigen::Matrix<double,  
                            Eigen::Dynamic,  
                            Eigen::Dynamic,  
                            Eigen::RowMajor>> 
        eigen_matrix_view(raw_data.data(), rows, cols); 

    cout << "Ma trận Eigen (Row-Major View):" << endl; 
    cout << eigen_matrix_view << endl; 

    eigen_matrix_view(0, 0) = 99.0; 
    cout << "\nThay đổi  của Eigen::Map thành 99.0..." << endl; 
    cout << "Dữ liệu nguồn raw_data hiện tại: " << raw_data << endl; 

    arma::mat armadillo_matrix_view(raw_data.data(), rows, cols, false, true);  

    cout << "\nMa trận Armadillo (Column-Major View):" << endl; 
    armadillo_matrix_view.print(); 

    armadillo_matrix_view(2, 2) = 11.0; 
    cout << "Thay đổi [2,2] của Armadillo::Mat thành 11.0..." << endl; 
    cout << "Dữ liệu nguồn raw_data hiện tại: " << raw_data << endl;  

    return 0; 
} 
