#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


int main()
{
    Mat image1 = imread("hah.png");
    Mat image2 = imread("res.png");
    Mat image3 = imread("res2.png");
    cout<<"size1:"<<image1.cols<<" "<<image1.rows<<endl;
    cout<<"size2:"<<image2.cols<<" "<<image2.rows<<endl;
    cout<<"size3:"<<image3.cols<<" "<<image3.rows<<endl;

    Mat origin_img = imread("origin_img.png");
    cout<<"size4:"<<origin_img.cols<<" "<<origin_img.rows<<endl;
    if (image3.empty() || origin_img.empty()) {
        cout << "Could not open or find the images!" << endl;
        return -1;
    }

    Mat gray_image3, gray_origin_img;
    cvtColor(image3, gray_image3, COLOR_BGR2GRAY);
    cvtColor(origin_img, gray_origin_img, COLOR_BGR2GRAY);
    int num = countNonZero(gray_image3 != gray_origin_img);
    if (num == 0) {
        cout<< num <<endl;
        cout << "res2 and origin_img are the same." << endl;
    } 
    else {
        cout << "res2 and origin_img are different." << endl;
        cout<< num <<endl;
    }
    return 0;
}