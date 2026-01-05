#include <iostream> 
#include <mlpack/core.hpp> 
#include <mlpack/methods/kmeans/kmeans.hpp> 
#include <armadillo>
#include <map> 
using namespace std; 
using namespace mlpack; 
using namespace mlpack::kmeans; 
using namespace arma; 

using Clusters = std::map<size_t, std::pair<std::vector<double>, 
std::vector<double>>>; 

mat generate_data(size_t num_samples) { 
    size_t s_per_cluster = num_samples / 3; 
    mat cluster1 = arma::randn<mat>(2, s_per_cluster) * 0.5 + 1.0;  
    mat cluster2 = arma::randn<mat>(2, s_per_cluster) * 0.5 + 5.0;  
    mat cluster3 = arma::randn<mat>(2, s_per_cluster) * 0.5 + 9.0; 
    mat dataset = arma::join_rows(cluster1, cluster2); 
    dataset = arma::join_rows(dataset, cluster3); 
    return dataset; 
} 
 
int main() { 
    size_t num_samples = 3000; 
    size_t num_clusters = 3; 
     
    mat dataset = generate_data(num_samples); 
     
    KMeans<metric::EuclideanDistance,  
           KMeansPlusPlusInitialization,  
           EmptyClusterPolicy::AllowEmptyClusters, 
           HamerlyUpdate> 
        kmeans_algorithm; 

    arma::Row<size_t> assignments;
    arma::mat centroids;           
     
    kmeans_algorithm.Cluster(dataset, num_clusters, assignments, centroids); 

    cout << "--- K-Means Clustering voi mlpack ---" << endl; 
    cout << "So luong mau: " << dataset.n_cols << endl; 
    cout << "So luong cum yeu cau (k): " << num_clusters << endl; 
     
    cout << "\nToa do tam cum (Centroids) tim duoc:" << endl; 
    centroids.print(cout); 
 
    Clusters cluster_counts; 
    for(size_t i = 0; i < num_samples; ++i) { 
        size_t cluster_id = assignments(i); 
        cluster_counts[cluster_id]++; 
    } 
 
    cout << "Phan phoi so luong mau trong moi cum:" << endl; 
    for (const auto& pair : cluster_counts) { 
        cout << "Cum " << pair.first << ": " << pair.second << " mau" << endl; 
    } 
 
    cout << "\nChi muc cum cua mau dau tien: " << assignments(0) << endl; 
     
    return 0; 
} 
