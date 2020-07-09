#ifndef MONOCULAR_H
#define MONOCULAR_H

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class monocular
{
public:
	/*���캯��*/
	monocular(int boardWidth, int boardHeight, int squareSize, string filespath);

	/*ʵ������궨*/
	void calibrate();

	/*��������*/
	~monocular();

private:
	/*����궨����ģ���ʵ����������*/
	void calRealPoint(vector<vector<Point3f> >& obj, int boardwidth, int boardheight, int imgNumber, int squaresize);

	/*��������ĳ�ʼ���� Ҳ���Բ�����*/
	void guessCameraParam();

	/*����궨����*/
	void saveCameraParam();

	/*������ͶӰ���*/
	void reproject_error();

	/*ԭʼͼ��Ļ������*/
	void distort_correct();

	/*��ȡ����ڲ����������������*/
	void ReadCameraParam();

public:
	Mat mIntrinsic;                               //����ڲ���
	Mat mDistortion_coeff;                        //����������
	vector<Mat> mvRvecs;                          //��ת����
	vector<Mat> mvTvecs;                          //ƽ������
	vector<vector<Point2f> > mvCorners;           //����ͼ���ҵ��Ľǵ�ļ��� ��objRealPoint һһ��Ӧ
	vector<vector<Point3f> > mvObjRealPoint;      //����ͼ��Ľǵ��ʵ���������꼯��
	vector<Point2f> mvCorner;                     //ĳһ��ͼ���ҵ��Ľǵ�

	vector<String> mFiles;//���б궨ͼ���·��
	vector<Mat> mImages;  //���б궨ͼ��

	int mWidth, mHeight;  //�궨ͼ��Ĵ�С

	int mBoardWidth;      //����Ľǵ���Ŀ �0�2
	int mBoardHeight;     //����Ľǵ����� �0�2
	int mBoardCorner;     //�ܵĽǵ����� �0�2�0�2
	int mSquareSize;      //�궨��ڰ׸��ӵĴ�С ��λmm �0�2
	Size mBoardSize;      //�ܵ��ڽǵ�

	double mdRMS_error;   //�����ܵľ�������ͶӰ���

	double mdtotal_err;   //�����ܵ���ͶӰ���
	double mdave_error;   //����ƽ������ͶӰ���

	string mFilespath;    //����궨ͼ�����ڵ��ļ���·��
};
#endif