#include <iostream> 
#include <vector> 
#include <cmath> 
#include <iomanip> 
#include <numeric> 
#include <algorithm> 

using namespace std; 

struct DiagnosisResult { 
    int epoch; 
    double training_loss; 
    double validation_loss; 
}; 

double calculate_mse(const vector<double>& true_values, const vector<double>& predicted_values) { 
    if (true_values.size() != predicted_values.size() || true_values.empty()) return 0.0; 
     
    double sum_squared_error = 0.0; 
    for (size_t i = 0; i < true_values.size(); ++i) { 
        double error = true_values[i] - predicted_values[i]; 
        sum_squared_error += error * error; 
    } 
    return sum_squared_error / true_values.size(); 
} 

vector<DiagnosisResult> simulate_overfitting_diagnosis(int num_epochs) { 
    vector<DiagnosisResult> history; 

    for (int epoch = 1; epoch <= num_epochs; ++epoch) { 
        DiagnosisResult result; 
        result.epoch = epoch; 
         
        result.training_loss = 0.5 * exp(-0.02 * epoch) + 0.01;  

        if (epoch <= 75) { 
            result.validation_loss = 0.5 * exp(-0.01 * epoch) + 0.05; 
        } else { 
            result.validation_loss = 0.5 * exp(-0.01 * 75) + 0.05 + 0.0005 * (epoch - 75); 
        } 

        history.push_back(result); 
    } 
    return history; 
} 

int main() { 
    int num_epochs = 150; 
     
    vector<DiagnosisResult> results = simulate_overfitting_diagnosis(num_epochs); 

    cout << "--- Ket qua Chan doan Mo hinh (Mo phong Overfitting) ---" << endl; 
    cout << fixed << setprecision(6); 
     
    cout << "| Epoch | Training Loss (MSE) | Validation Loss (MSE) | Ket Luan |" << endl; 
    cout << "|---|---|---|---|" << endl; 
     
    for (const auto& r : results) { 
        string comment = ""; 
        if (r.epoch <= 75) { 
            comment = "Giam (Hoc tap Binh thuong)"; 
        } else if (r.epoch == 76) { 
            comment = "Diem uon: Validation Loss bat dau TANG (Overfitting)"; 
        } else { 
            comment = "Tang (Overfitting)"; 
        } 

        if (r.epoch % 25 == 0 || r.epoch == 76) { 
            cout << "| " << setw(5) << r.epoch << " | " 
                 << setw(20) << r.training_loss << " | " 
                 << setw(20) << r.validation_loss << " | " 
                 << comment << " |" << endl; 
        } 
    } 
    return 0; 
} 
