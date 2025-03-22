#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace cv;
using namespace std;
void print_energy(const Mat &energy);
void print_gray(const Mat& gray);
// 计算能量图
Mat calculateEnergyMap(const Mat &gray)
{
    Mat gradX, gradY, energy;

    // // 转换为灰度图
    // cvtColor(image, gray, COLOR_BGR2GRAY); // 0-255 

    // 计算图像在 X 和 Y 方向的梯度
    Sobel(gray, gradX, CV_64F, 1, 0, 3); // 3*3d的卷积核的大小
    Sobel(gray, gradY, CV_64F, 0, 1, 3);
    // cout<<"gradX"<<endl;
    // for(int i=0;i<4;++i)
    // {
    //     for(int j=0;j<4;++j)
    //     {
    //         cout<<gradX.at<double>(i,j)<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<"gradY"<<endl;
    // for(int i=0;i<4;++i)
    // {
    //     for(int j=0;j<4;++j)
    //     {
    //         cout<<gradY.at<double>(i,j)<<" ";
    //     }
    //     cout<<endl;
    // }
    magnitude(gradX, gradY, energy);
    // cout<<"gradXY"<<endl;
    // for(int i=0;i<4;++i)
    // {
    //     for(int j=0;j<4;++j)
    //     {
    //         cout<<energy.at<double>(i,j)<<" ";
    //     }
    //     cout<<endl;
    // }

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
// 更新能量图
void updateEnergyMap(Mat &energy, const Mat &gray, const vector<int> &seam)
{
    if (seam.size() == 0) // 第一次调用
    {
        return;
    }
    // energy 比 gray 多一列,将不同模块解耦
    int rows = energy.rows, cols = energy.cols;

    Mat newEnergy(rows, cols - 1, CV_64F);
    for (int i = 0; i < rows; ++i)
    {
        int j = seam[i];// 对j,左右进行energy计算
        double * energyRow = energy.ptr<double>(i);
        double * newEnergyRow = newEnergy.ptr<double>(i);
        if(j==0)
        {
            newEnergyRow[j] = localSoble(gray, i, j);
            newEnergyRow[j+1] = localSoble(gray, i, j+1);// 这里的gray 和newEnergy是对应的
            for (int k = j+2; k < cols - 1; ++k)
            {
                newEnergyRow[k] = energyRow[k+1];
            }
        }
        else if(j==cols-1)
        {
            for (int k = 0; k < j - 1; ++k)
            {
                newEnergyRow[k] = energyRow[k];
            }
            newEnergyRow[j-1] = localSoble(gray, i, j-1);
        }
        else
        {
            for (int k = 0; k < j-1; ++k)
            {
                newEnergyRow[k] = energyRow[k];
            }
            newEnergyRow[j-1] = localSoble(gray, i, j-1);
            newEnergyRow[j] = localSoble(gray, i, j);
            for (int k = j+1; k < cols - 1; ++k)
            {
                newEnergyRow[k] = energyRow[k+1];
            }
        }

    }
    energy = newEnergy;
}

// 从图像中移除路径
Mat removeSeam(const Mat &image, const vector<int> &seam)

{
    int rows = image.rows, cols = image.cols;
    Mat newImage(rows, cols - 1, image.type());

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < seam[i]; ++j)
        {
            newImage.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
        }
        for (int j = seam[i] + 1; j < cols; ++j)
        {
            newImage.at<Vec3b>(i, j - 1) = image.at<Vec3b>(i, j);
        } // 直接进行拼接？
    }

    return newImage;
}

// 从灰度图中移除路径
Mat removeSeamFromGray(const Mat &gray, const vector<int> &seam)
{
    int rows = gray.rows, cols = gray.cols;
    Mat newGray(rows, cols - 1, CV_8UC1);

    // 利用指针
    for (int i = 0; i < rows; ++i)                       
    {                             
        const uchar *energyRow = gray.ptr<uchar>(i); // 指向原矩阵的行
        uchar *newEnergyRow = newGray.ptr<uchar>(i); // 指向新矩阵的行

        // 复制前半部分
        for (int j = 0; j < seam[i]; ++j)
        {
            newEnergyRow[j] = energyRow[j];
        }

        // 复制后半部分（跳过 seam[i] 处的像素）
        for (int j = seam[i] + 1; j < cols; ++j)
        {
            newEnergyRow[j - 1] = energyRow[j];
        }
    }
    return newGray;
}

// 验证能量图更新的正确性
void validateEnergyMap(const string& imagePath) 
{
    // 1. 读取图像
    Mat gray(10, 10,CV_8UC1 );// 注意数据类型

    // 2. 填充随机灰度值 (0-255)
    RNG rng(123); // OpenCV 随机数生成器
    rng.fill(gray, RNG::UNIFORM, 0, 256);
    Mat gray1 = gray.clone();
    // 3. 计算初始能量图
    Mat energy1 = calculateEnergyMap(gray);
    cout<<"energy1"<<endl;
    print_energy(energy1);
    Mat energy2 = energy1.clone();

    // 4. 模拟移除一个seam并比较结果
    vector<int> seam(gray.rows);
    // 创建一个测试用的seam（例如，图像中间的一列）
    for (int i = 0; i < gray.rows; ++i) {
        seam[i] = gray.cols -1;
    }

    // 5. 方法1：移除seam后重新计算整个能量图
    Mat newGray = removeSeamFromGray(gray1, seam);
    Mat newEnergy1 = calculateEnergyMap(newGray);
    cout<<"gray"<<endl;
    for(int i=0;i<gray1.rows;++i)
    {
        for(int j=0;j<gray1.cols;++j)
        {
            //cout<<gray1.at<uchar>(i,j)<<" ";
            cout << static_cast<int>(gray1.at<uchar>(i,j)) << " ";
        }
        cout<<endl;
    }  
    cout << gray.cols<<endl;
    if(gray.cols != gray1.cols)
    {
        cout<<"not "<< endl;
    }
    // 6. 方法2：使用updateEnergyMap更新能量图
    updateEnergyMap(energy2, newGray, seam);

    // 7. 比较两种方法的结果
    Mat diff;
    absdiff(newEnergy1, energy2, diff);
    
    // 计算差异统计
    double minVal, maxVal, avgDiff;
    minMaxLoc(diff, &minVal, &maxVal);
    avgDiff = mean(diff)[0];

    cout << "能量图比较结果：" << endl;
    cout << "最大差异：" << maxVal << endl;
    cout << "最小差异：" << minVal << endl;
    cout << "平均差异：" << avgDiff << endl;

    for(int i=0;i<newEnergy1.rows;++i)
    {
        for(int j=0;j<newEnergy1.cols;++j)
        {
            cout<<newEnergy1.at<double>(i,j)<<" ";
        }
        cout<<endl;
    }
    cout<<"newEnergy2"<<endl;
    for(int i=0;i<energy2.rows;++i)
    {
        for(int j=0;j<energy2.cols;++j)
        {
            cout<<energy2.at<double>(i,j)<<" ";
        }
        cout<<endl;
    }
}

void print_energy(const Mat &energy)
{
    
    for(int i=0;i<energy.rows;++i)
    {
        for(int j=0;j<energy.cols;++j)
        {
            cout<<energy.at<double>(i,j)<<" ";
        }
        cout<<endl;
    }
}
void print_gray(const Mat& gray)
{
    for(int i=0;i<gray.rows;++i)
    {
        for(int j=0;j<gray.cols;++j)
        {
            cout << static_cast<int>(gray.at<uchar>(i,j)) << " ";
        }
        cout<<endl;
    }
}

Mat caculate_with_local(const Mat& gray)
{
    Mat energy(gray.rows, gray.cols, CV_64F);
    for(int i=0;i<gray.rows;++i)
    {
        for(int j=0;j<gray.cols;++j)
        {
            energy.at<double>(i,j) = localSoble(gray, i, j);
        }
    }
    return energy;
}

void validateEnergyMap1()
{
    Mat gray(10, 10,CV_8UC1 );// 注意数据类型
    // 2. 填充随机灰度值 (0-255)
    RNG rng(123); // OpenCV 随机数生成器
    rng.fill(gray, RNG::UNIFORM, 0, 256);
    cout<<"gray"<<endl;
    print_gray(gray);
    Mat energy1 = calculateEnergyMap(gray);
    cout<<"energy1"<<endl;
    print_energy(energy1);
    Mat energy2 = caculate_with_local(gray);
    cout<<"energy2"<<endl;
    print_energy(energy2);
    
}
int main() 
{
    validateEnergyMap("hah.png");
    //validateEnergyMap1();
    return 0;
}