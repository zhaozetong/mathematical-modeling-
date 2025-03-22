#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;

// 计算能量图
Mat calculateEnergyMap(const Mat &gray)
{
    Mat gradX, gradY, energy;
    Sobel(gray, gradX, CV_64F, 1, 0, 3); // 3*3d的卷积核的大小
    Sobel(gray, gradY, CV_64F, 0, 1, 3);
    magnitude(gradX, gradY, energy);
    return energy;
}

double localSoble(const Mat &gray, int i, int j) // 用的是处理后的gray
{
    int rows = gray.rows, cols = gray.cols;
    double gx = 0, gy = 0;
    const uchar* prev_row = (i > 0) ? gray.ptr<uchar>(i-1) : nullptr;
    const uchar* curr_row = gray.ptr<uchar>(i);
    const uchar* next_row = (i < rows-1) ? gray.ptr<uchar>(i+1) : nullptr;    
    // 使用镜像填充：假设边界外的像素是边界像素的镜像
    // 处理四个角落的特殊情况
    if ((i == 0||i == rows - 1) && (j == 0||j == cols-1)) {  // 左上角
        
        gx = 0;
        gy = 0;
    }
    // 处理边缘情况
    else if (i == 0) {  // 上边缘
        gx = 2*(curr_row[j+1] - curr_row[j-1]+next_row[j+1] - next_row[j-1]);
        gy = 0;
    }
    else if (i == rows-1) {  // 下边缘
        gx = 2*(curr_row[j+1] - curr_row[j-1]+ prev_row[j+1] - prev_row[j-1]);
        gy = 0;
    }
    else if (j == 0) {  // 左边缘
        gx = 0;
        gy = 2*(next_row[0] - prev_row[0]+next_row[1] - prev_row[1]);
    }
    else if (j == cols-1) {  // 右边缘
        gx = 0;
        gy = 2*(next_row[j] - prev_row[j]+next_row[j-1] - prev_row[j-1]);
    }
    else {  // 非边缘区域，使用标准 Sobel 算子
        gx = (prev_row[j+1] + 2*curr_row[j+1] + next_row[j+1]) - // 水平
             (prev_row[j-1] + 2*curr_row[j-1] + next_row[j-1]);
        gy = (next_row[j-1] + 2*next_row[j] + next_row[j+1]) - 
             (prev_row[j-1] + 2*prev_row[j] + prev_row[j+1]);
    }

    return sqrt(gx*gx + gy*gy);
}

// 更新能量图, 传入修改后的gray
void updateEnergyMap(Mat &energy, const Mat &gray, const vector<int> &seam)
{
    if (seam.size() == 0) // 第一次调用
    {
        return;
    }
    // energy 比 gray 多一列,将不同模块解耦
    int rows = energy.rows, cols = energy.cols;

    // Mat newEnergy(rows, cols - 1, CV_64F);
    for (int i = 0; i < rows; ++i)
    {
        int j = seam[i];// 对j,左右进行energy计算
        double * energyRow = energy.ptr<double>(i);
        // double * newEnergyRow = newEnergy.ptr<double>(i);
        if(j==0)
        {
            energyRow[j] = localSoble(gray, i, j);
            energyRow[j+1] = localSoble(gray, i, j+1);// 这里的gray 和newEnergy是对应的
            for (int k = j+2; k < cols - 1; ++k)
            {
                energyRow[k] = energyRow[k+1];
            }
        }
        else if(j==cols-1)
        {
            // for (int k = 0; k < j - 1; ++k)
            // {
            //     energyRow[k] = energyRow[k];
            // }
            energyRow[j-1] = localSoble(gray, i, j-1);
        }
        else
        {
            // for (int k = 0; k < j-1; ++k)
            // {
            //     energyRow[k] = energyRow[k];
            // }
            energyRow[j-1] = localSoble(gray, i, j-1);
            energyRow[j] = localSoble(gray, i, j);
            for (int k = j+1; k < cols - 1; ++k)
            {
                energyRow[k] = energyRow[k+1];
            }
        }

    }
    // energy = newEnergy;
    energy = energy.colRange(0, cols - 1);
}

// 动态规划计算累积能量图
Mat calculateCumulativeEnergy(const Mat &energy)
{
    // 初始化累积能量矩阵
    Mat cumulativeEnergy = energy.clone();
    int rows = energy.rows, cols = energy.cols;

    for (int i = 1; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double left = (j > 0) ? cumulativeEnergy.at<double>(i - 1, j - 1) : DBL_MAX;
            double up = cumulativeEnergy.at<double>(i - 1, j);
            double right = (j < cols - 1) ? cumulativeEnergy.at<double>(i - 1, j + 1) : DBL_MAX;

            cumulativeEnergy.at<double>(i, j) += std::min({left, up, right});
        }
    }

    return cumulativeEnergy;
}

// 回溯找到最低能量路径
vector<int> findSeam(const Mat &cumulativeEnergy)
{
    int rows = cumulativeEnergy.rows, cols = cumulativeEnergy.cols;
    vector<int> seam(rows);
    double minVal;
    Point minLoc;

    // 找到最后一行的最小值位置
    minMaxLoc(cumulativeEnergy.row(rows - 1), &minVal, nullptr, &minLoc, nullptr);
    seam[rows - 1] = minLoc.x;

    // 从最后一行回溯
    for (int i = rows - 2; i >= 0; --i)
    {
        int prevCol = seam[i + 1];
        int bestCol = prevCol;
        double minEnergy = cumulativeEnergy.at<double>(i, prevCol);

        if (prevCol > 0 && cumulativeEnergy.at<double>(i, prevCol - 1) < minEnergy)
        {
            bestCol = prevCol - 1;
            minEnergy = cumulativeEnergy.at<double>(i, prevCol - 1);
        }
        if (prevCol < cols - 1 && cumulativeEnergy.at<double>(i, prevCol + 1) < minEnergy)
        {
            bestCol = prevCol + 1;
        }

        seam[i] = bestCol;
    }

    return seam;
}

// 从图像中移除路径
void removeSeam(Mat &image, const vector<int> &seam)
{
    int rows = image.rows, cols = image.cols;
    // Mat newImage(rows, cols - 1, image.type());

    for (int i = 0; i < rows; ++i)
    {
        // for (int j = 0; j < seam[i]; ++j)
        // {
        //     newImage.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        // }
        for (int j = seam[i] + 1; j < cols; ++j)
        {
            image.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        } // 直接进行拼接？
    }
    image =  image.colRange(0,cols-1);
    // return newImage;
}

// 从灰度图中移除路径
void removeSeamFromGray(Mat &gray, const vector<int> &seam)
{
    int rows = gray.rows, cols = gray.cols;
    // Mat newGray(rows, cols - 1, CV_8UC1);

    // 利用指针
    for (int i = 0; i < rows; ++i)                       
    {                             
        uchar *energyRow = gray.ptr<uchar>(i); // 指向原矩阵的行
        // uchar *newEnergyRow = newGray.ptr<uchar>(i); // 指向新矩阵的行

        // 复制前半部分
        // for (int j = 0; j < seam[i]; ++j)
        // {
        //     energyRow[j] = energyRow[j];
        // }

        // 复制后半部分（跳过 seam[i] 处的像素）
        for (int j = seam[i] + 1; j < cols; ++j)
        {
            energyRow[j - 1] = energyRow[j];
        }
    }
    gray = gray.colRange(0,cols-1);
}

// 主函数：实现图像缩放
Mat seamCarving(const Mat &image, int newWidth)
{
    Mat currentImage = image.clone();
    vector<int> seam = {};
    Mat gray;
    cvtColor(currentImage,gray, COLOR_BGR2GRAY);

    Mat energy = calculateEnergyMap(gray);
    while (gray.cols > newWidth)
    {
        // 第一步：计算能量图
        updateEnergyMap(energy, gray, seam);

        // 第二步：计算累积能量图
        Mat cumulativeEnergy = calculateCumulativeEnergy(energy);

        // 第三步：找到最低能量路径
        seam = findSeam(cumulativeEnergy);
        // 更新gray
        removeSeamFromGray(gray, seam);    
        // 第四步：移除路径
        removeSeam(currentImage, seam);
    }

    return currentImage;
}

int main()
{
    // 读取图像
    Mat image = imread("hah.png");
    if (image.empty())
    {
        cerr << "Error: Unable to load image." << endl;
        return -1;
    }

    // 显示原始图像
    // imshow("Original Image", image);
    Mat energy = calculateEnergyMap(image);

    normalize(energy, energy, 0, 255, NORM_MINMAX, CV_8U); // 展示归一化的能量图
    // imshow("Energy Map", energy);
    imwrite("energy.png", energy); // 也展示最开始的能量图
    //  调整图像宽度
    int newWidth = image.cols - 300; // 缩小 100 列
    auto start = chrono::high_resolution_clock::now();

    Mat resizedImage = seamCarving(image, newWidth);
    // Mat resizedImage = seamCarving_v0(image,newWidth);
    auto end = chrono::high_resolution_clock::now();
    std::cout << "运行时间:" << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
    // 显示缩放后的图像
    // imshow("Resized Image", resizedImage);

    // 保存结果
    imwrite("res2.png", resizedImage);

    cv::waitKey(0);
    return 0;
}