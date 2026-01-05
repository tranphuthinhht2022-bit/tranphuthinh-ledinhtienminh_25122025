#include <iostream> 
#include <vector> 
#include <cmath> 
#include <mlpack/core.hpp> 
#include <mlpack/methods/kmeans/kmeans.hpp> 
#include <mlpack/methods/dbscan/dbscan.hpp> 
#include <armadillo> 
 
using namespace std; 
using namespace mlpack; 
using namespace mlpack::kmeans; 
using namespace mlpack::dbscan; 
using namespace arma; 
 
mat generate_two_moons(size_t num_samples, double noise_std) { 
    mat data(2, num_samples); 
     
    for (size_t i = 0; i < num_samples; ++i) { 
        double angle = i * M_PI / (num_samples / 2.0); 
        double radius = 1.0; 
         
        if (i < num_samples / 2) { 
            data(0, i) = radius * cos(angle); 
            data(1, i) = radius * sin(angle); 
        } else { 
            angle = (i - num_samples / 2) * M_PI / (num_samples / 2.0); 
            data(0, i) = radius * cos(angle) + 1.0; 
            data(1, i) = radius * sin(angle) - 0.5; 
        } 
    } 
 
    data += arma::randn<mat>(2, num_samples) * noise_std; 
    return data; 
} 
 
void print_cluster_distribution(const arma::Row<size_t>& assignments) { 
    std::map<int, size_t> counts; 
    for (size_t id : assignments) { 
        counts[(int)id]++; 
    } 
     
    for (const auto& pair : counts) { 
        if (pair.first == -1) { 
            cout << "  - Cum NHIEU (Outliers): " << pair.second << " mau" << endl; 
        } else { 
            cout << "  - Cum " << pair.first << ": " << pair.second << " mau" << endl; 
        } 
    } 
} 
 
int main() { 
    size_t num_samples = 2000; 
    size_t num_clusters = 2; 
    double noise_std = 0.1;  
     
    mat dataset = generate_two_moons(num_samples, noise_std); 
    cout << "Dataset Two Moons da duoc tao: " << dataset.n_cols << " mau, "  
         << dataset.n_rows << " features." << endl; 
    cout << "---------------------------------------" << endl; 
 
    cout << "I. Thuc hien K-Means Clustering (k=2)..." << endl; 
     
    KMeans<> kmeans(num_clusters); 
    arma::Row<size_t> kmeans_assignments; 
    arma::mat kmeans_centroids; 
     
    kmeans.Cluster(dataset, num_clusters, kmeans_assignments, kmeans_centroids); 
     
    cout << "Ket qua K-Means:" << endl; 
    print_cluster_distribution(kmeans_assignments); 
     
    cout << "\nII. Thuc hien DBSCAN Clustering (Density-Based)..." << endl; 
     
    double epsilon = 0.15; 
    size_t min_pts = 5;    
     
    DBSCAN<> dbscan_alg(epsilon, min_pts); 
    arma::Row<size_t> dbscan_assignments; 
     
    size_t actual_clusters = dbscan_alg.Cluster(dataset, dbscan_assignments); 
     
    cout << "Ket qua DBSCAN (so cum tim duoc: " << actual_clusters << "):" << endl; 
    print_cluster_distribution(dbscan_assignments); 
     
    return 0; 
} 
