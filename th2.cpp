#include <iostream> 
#include <opencv2/opencv.hpp> 
#include <vector> 
#include <random> 
#include <algorithm> 
#include <cmath> 
#include <iomanip> 
 
using namespace cv; 
using namespace std; 
 
const int NN_INPUT_SIZE = 224; 
 
Mat load_dummy_image() { 
    Mat img(300, 400, CV_8UC3, Scalar(100, 150, 200));  
    putText(img, "Original Image", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, 
Scalar(255, 255, 255), 2); 
    return img; 
} 
 
Mat perform_augmentation(const Mat& src) { 
    Mat dst = src.clone(); 
     
    random_device rd; 
    mt19937 gen(rd()); 
 
    uniform_real_distribution<> rot_dist(-15.0, 15.0); 
    double angle = rot_dist(gen); 
     
    Point2f center(dst.cols / 2.0, dst.rows / 2.0); 
    Mat rot_mat = getRotationMatrix2D(center, angle, 1.0); 
    warpAffine(dst, dst, rot_mat, dst.size()); 
 
    uniform_int_distribution<> flip_dist(0, 1); 
    if (flip_dist(gen) == 1) { 
        flip(dst, dst, 1); 
    } 
 
    resize(dst, dst, Size(NN_INPUT_SIZE, NN_INPUT_SIZE), 0, 0, INTER_LINEAR); 
     
    return dst; 
} 
 
vector<float> deinterleave_and_normalize(const Mat& src) { 
    Mat rgb_img; 
    cvtColor(src, rgb_img, COLOR_BGR2RGB); 
 
    Mat float_img; 
    rgb_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0); 
 
    vector<Mat> channels; 
    split(float_img, channels); 
 
    size_t channel_size = (size_t)channels.total() * channels.elemSize1(); 
     
    vector<float> data_flat; 
    data_flat.reserve(3 * channel_size / sizeof(float)); 
 
    for (const auto& channel : channels) { 
        if (channel.isContinuous()) { 
            const float* begin = (const float*)channel.data; 
            const float* end = begin + channel.total(); 
            data_flat.insert(data_flat.end(), begin, end); 
        } else { 
             cerr << "Error: Channel data is not contiguous!" << endl; 
        } 
    } 
     
    return data_flat; 
} 
