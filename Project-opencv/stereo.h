#ifndef STEREO_H
#define STEREO_H

#include <iostream>
#include <string>

#include "monocular.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class stereo
{
public:
    stereo(int boardWidth, int boardHeight, int squareSize, string Lfilespath, string Rfilespath);
    ~stereo();

    //����궨
    void stereo_calibrate();

private:
    //����궨����
    void outputCameraParam();

    //����У��ͼ��
    void drawRectifyImage();

private:
    int mBoardWidth;            //����Ľǵ���Ŀ
    int mBoardHeight;           //����Ľǵ�����
    int mSquareSize;            //�궨��ڰ׸��ӵĴ�С ��λmm
    int mBoardCorner;           //�ܵĽǵ�����

    int mWidth, mHeight;         //����ͼ��Ĵ�С
    cv::Size mBoardSize;        //�궨��ĳߴ�

    cv::Mat R, T, E, F;                 //R ��תʸ�� Tƽ��ʸ�� E�������� F��������
    cv::Mat Rl, Rr, Pl, Pr, Q;          //У����ת����R��ͶӰ����P ��ͶӰ����Q
    cv::Mat mapLx, mapLy, mapRx, mapRy; //ӳ���

    cv::Mat rectifyImageL;              //У�����ͼ��
    cv::Mat rectifyImageR;

    cv::Mat mCameraMatrixL, mCameraMatrixR;  //�ڲ�����
    cv::Mat mDistCoeffL, mDistCoeffR;        //����

    cv::Rect validROIL, validROIR;           //ͼ��У��֮�󣬻��ͼ����вü��������validROI����ָ�ü�֮�������

    vector<Point2f> mvCornerL;               //��������ĳһ��Ƭ�ǵ����꼯��
    vector<Point2f> mvCornerR;               //�ұ������ĳһ��Ƭ�ǵ����꼯��

    vector<vector<Point2f>> mvImagePointL;   //��������������Ƭ�ǵ�����꼯��
    vector<vector<Point2f>> mvImagePointR;   //�ұ������������Ƭ�ǵ�����꼯��

    string mLfilespath, mRfilespath;         //���ұ궨ͼ���·��
    double mdStereo_RMS;                     //�����ܵľ�������ͶӰ���
};

#endif