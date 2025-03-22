#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;

// 计算能量图
Mat calculateEnergyMap(const Mat& image) {
    Mat gray, gradX, gradY, energy;

    // 转换为灰度图
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 计算图像在 X 和 Y 方向的梯度
    Sobel(gray, gradX, CV_64F, 1, 0, 3);// 3*3d的卷积核的大小
    Sobel(gray, gradY, CV_64F, 0, 1, 3);

    magnitude(gradX, gradY, energy);

    return energy;
}

// 动态规划计算累积能量图
Mat calculateCumulativeEnergy(const Mat& energy) {
    // 初始化累积能量矩阵
    Mat cumulativeEnergy = energy.clone();
    int rows = energy.rows, cols = energy.cols;

    for (int i = 1; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double left = (j > 0) ? cumulativeEnergy.at<double>(i - 1, j - 1) : DBL_MAX;
            double up = cumulativeEnergy.at<double>(i - 1, j);
            double right = (j < cols - 1) ? cumulativeEnergy.at<double>(i - 1, j + 1) : DBL_MAX;

            cumulativeEnergy.at<double>(i, j) += std::min({left, up, right});
        }
    }

    return cumulativeEnergy;
}

// 回溯找到最低能量路径（seam）
vector<int> findSeam(const Mat& cumulativeEnergy) {
    int rows = cumulativeEnergy.rows, cols = cumulativeEnergy.cols;
    vector<int> seam(rows);
    double minVal;
    Point minLoc;

    // 找到最后一行的最小值位置
    minMaxLoc(cumulativeEnergy.row(rows - 1), &minVal, nullptr, &minLoc, nullptr);
    seam[rows - 1] = minLoc.x;

    // 从最后一行回溯
    for (int i = rows - 2; i >= 0; --i) {
        int prevCol = seam[i + 1];
        int bestCol = prevCol;
        double minEnergy = cumulativeEnergy.at<double>(i, prevCol);

        if (prevCol > 0 && cumulativeEnergy.at<double>(i, prevCol - 1) < minEnergy) {
            bestCol = prevCol - 1;
            minEnergy = cumulativeEnergy.at<double>(i, prevCol - 1);
        }
        if (prevCol < cols - 1 && cumulativeEnergy.at<double>(i, prevCol + 1) < minEnergy) {
            bestCol = prevCol + 1;
        }

        seam[i] = bestCol;
    }

    return seam;
}

// 从图像中移除路径
Mat removeSeam(const Mat& image, const vector<int>& seam) {
    int rows = image.rows, cols = image.cols;
    Mat newImage(rows, cols - 1, image.type());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < seam[i]; ++j) {
            newImage.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = seam[i] + 1; j < cols; ++j) {
            newImage.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        }
    }

    return newImage;
}

// 主函数：实现图像缩放
Mat seamCarving(const Mat& image, int newWidth) {
    Mat currentImage = image.clone();

    while (currentImage.cols > newWidth) {
        // 第一步：计算能量图
        Mat energy = calculateEnergyMap(currentImage);

        // 第二步：计算累积能量图
        Mat cumulativeEnergy = calculateCumulativeEnergy(energy);

        // 第三步：找到最低能量路径
        vector<int> seam = findSeam(cumulativeEnergy);

        // 第四步：移除路径
        currentImage = removeSeam(currentImage, seam);
    }

    return currentImage;
}

int main() {
    // 读取图像
    Mat image = imread("hah.png");
    if (image.empty()) {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    // 显示原始图像
    //imshow("Original Image", image);
    Mat energy = calculateEnergyMap(image);

    normalize(energy, energy, 0, 255, NORM_MINMAX, CV_8U);// 展示归一化的能量图
    //imshow("Energy Map", energy);
    //imwrite("energy.png", energy);// 也展示最开始的能量图
    // 调整图像宽度
    int newWidth = image.cols - 300; // 缩小 100 列
    auto start = chrono::high_resolution_clock::now();
    Mat resizedImage = seamCarving(image, newWidth);
    auto end = chrono::high_resolution_clock::now();
    std::cout << "运行时间:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // 显示缩放后的图像
    //imshow("Resized Image", resizedImage);

    // 保存结果
    imwrite("origin_img.png", resizedImage);

    //waitKey(0);
    return 0;
}