#ifndef __OPENCV_ALL_HPP__
#define __OPENCV_ALL_HPP__
#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>

#endif

#include <omp.h>

#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
//#include <opencv/cv.h>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

typedef complex < double > base;

#define PI			 3.14159265358979323846
#define theta		 31
#define Projection	 32
#define phi			 11.25	 // = 360 / 16
#define Level		 269
#define InputWidth   2352
#define InputHeight  2944
#define OutputWidth  2352
#define OutputHeight 2944
#define meshGridSize InputWidth*InputHeight
#define Resoultion 0.0495
#define TZa 61.461399078369141 //Tube Z axis
#define CZa -141.447109222  //Camera Z axis
#define Rt 36.929735859
#define Rc 84.989996044

struct params {
	//The real detector panel pixel density (number of pixels)
	int nu = OutputWidth;
	int nv = OutputHeight;

	// Detector setting (real size)
	double su = InputWidth * Resoultion;	// mm (real size)
	double sv = InputHeight * Resoultion;     // mm

						   // X-ray source and detector setting
	double DSD = TZa- CZa;    //  Distance source to camera
	double DSO = TZa;	//  X-ray source to object axis distance

	double dir = -1;   // gantry rotating direction (clock wise/ counter clockwise)
	double dang = 360/Projection; // angular step size (deg) 0:2.81:357.19
						//double deg[222]; = 0:dang:357.19;
	int nProj = 0;

	double du = (double)su / (double)nu;
	double dv = (double)sv / (double)nv;

	double us[OutputWidth];
	double vs[OutputHeight];
}param;

double meshGrid[meshGridSize];
int cvimage[Projection*InputWidth*InputHeight];
float imageData[InputHeight*InputWidth];

int main()
{
	double start, end;
	//計時
	start = (double)getTickCount();

	int cnt1 = 0;
	for (int i = -(param.nu - 1) / 2; i <= (param.nu - 1) / 2; i++) {
		param.us[cnt1++] = (double)i*param.du;
	}
	int cnt2 = 0;
	for (int i = -(param.nv - 1) / 2; i <= (param.nv - 1) / 2; i++) {
		param.vs[cnt2++] = (double)i*param.dv;
	}
	//cout << "cnt1: " << cnt1 << endl;
	//cout << "cnt2: "<<cnt2 << endl;
	for (int i = 0; i <= cnt2; i++) {
		int meshGridIndex = i * (cnt1+1);
		for (int j = 0; j <= cnt1; j++) {
			meshGrid[meshGridIndex++] = param.DSD / sqrt((param.DSD)*(param.DSD) + (param.us[j] * param.us[j]) + (param.vs[i] * param.vs[i]));
		}
		//cout << meshGridIndex << endl;
	}

	/* Read 16 Projection Images
	***************************************/
	CvMat *ProjectionImage[Projection];
	CvMat *OriImage[Projection];

	for (int k = 0; k < Projection; ++k)
	{
		// Open file.
		char filename[200] = { '\0' };
		sprintf(filename, "./Projection_image/%d_2352_2944_16bit.raw", k);
		//sprintf(filename, "./Projection_image_ori/16Projection%d.raw", 95+k);
		FILE *fp = NULL;
		fp = fopen(filename, "rb");

		// Memory allocation for bayer image data buffer.
		int framesize = InputHeight * InputWidth * 2;
		unsigned int *imagedata = NULL;
		imagedata = (unsigned int*)malloc(sizeof(unsigned int) * framesize);

		// Read image data and store in buffer.
		fread(imagedata, sizeof(unsigned int), framesize, fp);

		Mat img;
		img.create(InputHeight, InputWidth, CV_16UC1);
		memcpy(img.data, imagedata, framesize);

		CvMat *Image = cvCreateMat(InputHeight, InputWidth, CV_16UC1);
		CvMat temp = img;
		cvCopy(&temp, Image);

		ProjectionImage[k] = cvCreateMat(InputHeight, InputWidth, CV_16UC1);
		cvResize(Image, ProjectionImage[k]);
	}
	

	printf("Read 16 Projection Images Done.\n");
	
	// Create Filter
	int ll = OutputWidth;
	if (OutputHeight < OutputWidth) {
		ll = OutputHeight;
	}
	cout << ll << endl;
	int filt_len = ll;//getOptimalDFTSize(ll);

	vector < double > kernel, W, filt;
	vector < base > f_kernel(3 * filt_len, 0.0);

	int FilterType = 1;
	for (int i = -(filt_len / 2); i < (filt_len / 2); i++)
	{
		if (i % 2 == 0) {
			kernel.push_back(0);
			continue;
		}
		kernel.push_back(-1.0 / (PI * PI * i * i));
	}
	kernel[filt_len / 2] = 0.25;
	dft(kernel, f_kernel, DFT_COMPLEX_OUTPUT);
	for (int i = 0; i <= filt_len / 2; i++) filt.push_back(abs(f_kernel[i].real())*2.0);
	switch (FilterType) {
	case 0: //ramp
		break;
	case 1: //hann
		for (int i = 0; i < filt.size(); i++)
		{
			W.push_back(2.0*PI*i / filt_len);
			if (i >= 1)
				filt[i] = filt[i] * (1.0 + cos(W[i])) / 2.0;
		}
		break;
	case 2: //hamming
		for (int i = 0; i < filt.size(); i++)
		{
			W.push_back(2.0*PI*i / filt_len);
			if (i >= 1)
				filt[i] = filt[i] * (0.54 + 0.46 * cos(W[i]));
		}
		break;
	case 3: //cosine
		for (int i = 0; i < filt.size(); i++)
		{
			W.push_back(2 * PI *i / filt_len);
			if (i >= 1)
				filt[i] = filt[i] * cos(W[i] / 2);
		}
		break;
	case 4: // shepp-logan
		for (int i = 0; i < filt.size(); i++)
		{
			W.push_back(2.0*PI*i / filt_len);
			if (i >= 1)
				filt[i] = filt[i] * (sin(W[i] / 2) / (W[i] / 2));
		}
		break;
	case 5: // not do
		for (int i = 0; i < filt.size(); i++)
		{
			filt[i] = 1;
		}
		break;
	case 6: // gaussian
		for (int i = 0; i < filt.size(); i++)
		{
			if (i < 500) {
				float mu = 250;
				float sigma = 100.0;
				//filt[i] = filt[i] *1600* sqrt(1/(2 * PI * sigma * sigma)) * exp(-1*(i-mu)*(i-mu)/(2*sigma*sigma));
				filt[i] = 800 * sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu) * (i - mu) / (2 * sigma * sigma));
			}
			else {
				float mu = 750;
				float sigma = 100.0;
				//filt[i] = filt[i] *1600* sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu)*(i - mu) / (2 * sigma*sigma));
				filt[i] = 800 * sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu) * (i - mu) / (2 * sigma * sigma));
			}
		}
		break;
	case 7: // gaussian
		for (int i = 0; i < filt.size(); i++)
		{
			float mu = filt.size() / 4.0;
			float sigma = 200.0;
			filt[i] = filt[i] * 100 * sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu) * (i - mu) / (2 * sigma * sigma));
		}
		break;
	case 8: // gaussian
		for (int i = 0; i < filt.size(); i++)
		{
			if (i < 333) {
				float mu = 162;
				float sigma = 80.0;
				filt[i] = filt[i] * 400 * sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu) * (i - mu) / (2 * sigma * sigma));
			}
			else if (i >= 333 && i < 666) {
				float mu = 500;
				float sigma = 80.0;
				filt[i] = filt[i] * 400 * sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu) * (i - mu) / (2 * sigma * sigma));
			}
			else {
				float mu = 838;
				float sigma = 80.0;
				filt[i] = filt[i] * 400 * sqrt(1 / (2 * PI * sigma * sigma)) * exp(-1 * (i - mu) * (i - mu) / (2 * sigma * sigma));
			}

		}
		break;
	}
	for (int i = (int)filt.size() - 1 - 1; i > 0; i--) filt.push_back(filt[i]);
	for (int i = 0; i < filt.size(); i++) if (filt[i] > PI) filt[i] = 0.0;

	// Convert 1D filter to 2D filter.
	
	int filter_len = filt.size();
	int center = filter_len * 0.5;
	int distance;
	vector < vector < double > > filter2d(filter_len, vector < double >(filter_len));
	//改
	for (int i = 0; i < ll; i++)
	{
		for (int j = 0; j < ll; j++)
		{
			distance = (int)(sqrt((i - center) * (i - center) + (j - center) * (j - center)));
			if (distance > center || distance < 10)
				//filter2d[i][j] = 0;
				filter2d[i][j] = (filt[i] + filt[j]) / 2;
			else
				filter2d[i][j] = filt[center - distance];
			//filter2d[i][j] = 0;
		}
	}
	cv::Mat Original_Mat_Filter_2D(filter2d.size(), filter2d.at(0).size(), CV_64FC1);

	for (int i = 0; i < Original_Mat_Filter_2D.rows; ++i)
		for (int j = 0; j < Original_Mat_Filter_2D.cols; ++j)
			Original_Mat_Filter_2D.at<double>(i, j) = filter2d.at(i).at(j);

	cv::Mat Resize_Filter_2D = Mat(OutputHeight, OutputWidth, CV_64FC1);
	cv::resize(Original_Mat_Filter_2D, Resize_Filter_2D, Size(OutputWidth, OutputHeight), INTER_LINEAR);
	cv::Mat show_Filter_2D = Mat(OutputHeight, OutputWidth, CV_64FC1);
	show_Filter_2D = Resize_Filter_2D * 255;
	imwrite("2d_filter.bmp", show_Filter_2D);
	cout << Resize_Filter_2D.size[0] << endl;
	cout << Resize_Filter_2D.size[1] << endl;
	//exit(0);
	

	printf("Create Filter Done.\n");
	
	/* Fourier Transform
	***************************************/
	float max00 = -1000, min00 = 1000;
	#pragma omp parallel for num_threads(16) schedule(dynamic)
	for (int k = 0; k < Projection; k++)
	{
		CvMat* ori_image = cvCreateMat(OutputHeight, OutputWidth, CV_32FC2);
		CvMat* dft_image = cvCreateMat(OutputHeight, OutputWidth, CV_32FC2);
		cvZero(ori_image);
		cvZero(dft_image);
		float pixel_value = 0.0;
		//*meshgrid 
		for (int i = 0; i < OutputHeight; i++) {
			int tempIndex = i * OutputWidth;
			
			for (int j = 0; j < OutputWidth; j++)
			{	
				pixel_value = ProjectionImage[k]->data.s[tempIndex + j]*(meshGrid[tempIndex + j]);
				CvScalar pixel;
				pixel.val[0] = pixel_value;
				pixel.val[1] = 0.0;
				cvSet2D(ori_image, i, j, pixel);
			}
		}
		// DFT
		cvDFT(ori_image, dft_image, CV_DXT_FORWARD);
		double f_realNum = 0, f_imagNum = 0;
		for (int i = 0; i < OutputHeight; i++)
		{
			for (int j = 0; j < OutputWidth; j++)
			{
				f_realNum = cvGet2D(dft_image, i, j).val[0];
				f_imagNum = cvGet2D(dft_image, i, j).val[1];
				CvScalar dft_pixel;
				dft_pixel.val[0] = f_realNum * Resize_Filter_2D.at<double>(i, j);
				dft_pixel.val[1] = f_imagNum * Resize_Filter_2D.at<double>(i, j);
				cvSet2D(dft_image, i, j, dft_pixel);
			}
		}
		// IDFT
		cvZero(ori_image);
		cvDFT(dft_image, ori_image, CV_DXT_INVERSE);
		float realNum = 0, imageNum = 0;
		
		for (int i = 0; i < OutputHeight; i++)
		{
			for (int j = 0; j < OutputWidth; j++)
			{
				realNum = cvGet2D(ori_image, i, j).val[0] / (OutputHeight * OutputWidth);
				imageNum = cvGet2D(ori_image, i, j).val[1] / (OutputHeight * OutputWidth);
				ProjectionImage[k]->data.s[i * OutputWidth + j] = realNum;
				
				if(k==0){
					if (max00 < realNum) {
						max00 = realNum;
					}
					if (min00 > realNum) {
						min00 = realNum;
					}
				}
				
			}
		}
		printf("NO.%d IDFT Done.\n", k);

	}
	printf("Fourier Transform Done.\n");
	
	
	/**************************************************
	//normalization 0-255
	
	for (int k = 0; k < Projection; k++) {
		for (int i = 0; i < OutputHeight; i++)
			{
				for (int j = 0; j < OutputWidth; j++)
				{
					float a = ProjectionImage[k]->data.s[i * OutputWidth + j];
					float b = 255.0 * (a - min00) / (max00 - min00);
					if (b > 255) {
						b = 255;
					}
					if (b < 0) {
						b = 0;
					}
					ProjectionImage[k]->data.s[i * OutputWidth + j] = b;
				}
			}
		char filename[200] = { '\0' };
		sprintf(filename, "./test/ft/k%d.bmp", k);
		cvSaveImage(filename, ProjectionImage[k]);
	}
	*/



	/* Back Projection
	***************************************/
	CvMat *Reconstruction[Level];
	#pragma omp parallel for num_threads(16) schedule(dynamic)
	for (int z = 0; z < Level; z++) 
	{
		Reconstruction[z] = cvCreateMat(OutputWidth, OutputHeight, CV_32FC1);

		for (int i = 0; i < Reconstruction[z]->rows; ++i)
		{
			for (int j = 0; j < Reconstruction[z]->cols; ++j)
			{
				Reconstruction[z]->data.fl[i * Reconstruction[z]->cols + j] = 0;
			}
		}
	}
	

//shift base
	float sXt = Rt * cos(PI), sYt = Rt * sin(PI), sZt = TZa;//mm
	float sXc = Rc * cos(0), sYc = Rc * sin(0), sZc = CZa;//mm
	float sXXc = sXc + (InputWidth / 2 ) * Resoultion;//coord mm
	float sYYc = sYc + (InputHeight / -2) * Resoultion;//coord mm
	float x0 = (sXt - ((sXt - sXXc) * (sZt)) / (sZt - sZc)) ;//coord mm
	float y0 = (sYt - ((sYt - sYYc) * (sZt)) / (sZt - sZc));
	sXXc = sXc + (InputWidth / 2 - InputWidth) * Resoultion;//coord mm
	sYYc = sYc + (InputHeight - InputHeight / 2) * Resoultion;//coord mm
	float x1 = (sXt - ((sXt - sXXc) * (sZt - 0)) / (sZt - sZc));//coord mm
	float y1 = (sYt - ((sYt - sYYc) * (sZt - 0)) / (sZt - sZc));
	float tx = (sZt - sZc) / (sZt);
	cout << x0 << " " << y0 << endl;
	cout << x1 << " " << y1 << endl;
	cout << tx << endl;
#pragma omp parallel for num_threads(16) schedule(dynamic)
	for (int z = 0; z < Level; z++)
	{
		printf("z = %d\n", z);
		//printf("Execute by thread %d\n", omp_get_thread_num());
		//Mat temp[Projection];
		float Zp = (Level / 2 - z) * 0.015;
		for (int i = 0; i < OutputHeight; ++i)
		{
			for (int j = 0; j < OutputWidth; ++j)
			{
				//vector<float> v;
				float mean = 0.0;
				float ZZ = (CZa - TZa) / (Zp - TZa);//(Zc - Zt) / (Zp - Zt)
				for (int k = 0; k < Projection; k = k+1)
				{
					//float angleT = (k+16) * PI * 2 / Projection;
					float angleC = k * PI * 2 / Projection;
					//float Xt = Rt * cos(angleT), Yt = Rt * sin(angleT);
					float Xt = -Rt * cos(angleC), Yt = -Rt * sin(angleC);//mm
					//float Xc = Rc * cos(angleC), Yc = Rc * sin(angleC);//mm
					float xx = Resoultion * (float)(OutputWidth/2 - j) / tx;
					//((Level / 2) - z) * 0.015;
					int X = (float)(InputWidth / 2) - ((xx - Xt) * ZZ - (Rt + Rc) * cos(angleC)) / Resoultion;
					//int X = (InputWidth / 2) - ((xx - Xt) * ZZ + Xt-Xc)/Resoultion;//coord mm
					if (X >= 0 && X < InputWidth) {
						float yy = Resoultion * (float)(i - OutputHeight/2 ) / tx ;
						int Y = (float)(InputHeight / 2) + ((yy - Yt) * ZZ - (Rt + Rc) * sin(angleC)) / Resoultion;
						//int Y = (InputHeight / 2)+((yy - Yt) * ZZ + Yt-Yc)/Resoultion;//coord mm
						if (Y >= 0 && Y < InputHeight)
						{
							float projectionValue = ProjectionImage[k]->data.s[Y * InputWidth + X];
							//cout << projectionValue << endl;
							Reconstruction[z]->data.fl[j * InputHeight + i]
								+= (float)projectionValue;
							//v.push_back(projectionValue);
							//mean += projectionValue;
							
						}
					}
					
				}
				/*
				float vari = 0.0;
				mean /= (float)v.size();
				for (int vi = 0; vi<v.size(); vi++) {
					vari += (v[vi] - mean) * (v[vi] - mean);
				}
				float sd = sqrt(vari / (float)v.size());
				float final_value = 0.0;
				int nn = 0;
				for (int vi = 0; vi < v.size(); vi++) {
					if (fabs(v[vi] - mean) <= 2.0 * sd) {
						final_value += v[vi];
						nn++;
					}
				}
				Reconstruction[z]->data.fl[j * InputHeight + i] = final_value / (float)nn;
				*/
			}
		}
	}
	printf("Back Projection Done.\n");

	// Mat release
	//cvReleaseMat(ProjectionImage);
	
	FILE* fp = NULL;
	fp = fopen("./123.raw", "w");
	// Memory allocation for bayer image data buffer.
	long long framesize = InputHeight * InputWidth * Level;
	uchar* data0 = NULL;
	data0 = (uchar*)malloc(sizeof(uchar) * framesize);
	
	

	/* 16-bit to 8-bit
	***************************************/
	double min = 0, max = 0;
	cvMinMaxLoc(Reconstruction[0], &min, &max);
	cout << min << " " << max << endl;
	//cvMinMaxLoc(Reconstruction[Level / 2], &min, &max);
	//min /= 2;
	//max /= 2;
	#pragma omp parallel for num_threads(16) schedule(dynamic)
	for (int z = 0; z < Level; z++)
	{

		CvMat* Reconstruction8U;
		Reconstruction8U = cvCreateMat(OutputHeight, OutputWidth, CV_32FC1);
		printf("save z = %d\n", z);
		int average = 0;
		// Normalize.
		for (int i = 0; i < OutputHeight; i++)
		{
			int tempIndex = i * OutputWidth;

			for (int j = 0; j < OutputWidth; j++)
			{
				float a = (Reconstruction[z]->data.fl[j * OutputHeight + i] - min) * 255 / (max - min);

				if (a < 0) a = 0;
				if (a > 255) a = 255;
				//255-xx
				Reconstruction8U->data.fl[tempIndex + j] = 255.0 - a;

				data0[z * InputHeight * InputWidth + tempIndex + j] = 255-a;
				if (data0[z * InputHeight * InputWidth + tempIndex + j] == 10)
					data0[z * InputHeight * InputWidth + tempIndex + j] = 11;

				//if (z==8) {

				//	if (j == 9 && i == 1555) {
				//		cout << a << " "  <<255-a << endl;
				//		data0[z * InputHeight * InputWidth + tempIndex + j] = 10;
				//	}
				//	/*if (9 <= j && j <= 2351 && j != 16 && j != 12 && j != 8) {
				//		data0[z * InputHeight * InputWidth + tempIndex + j] = 255 - a;
				//	}*//*if (601 <= j && j <= 900) data0[z * InputHeight * InputWidth + tempIndex + j] = a / 2;*/

				//}
				//data0[z * InputHeight * InputWidth + tempIndex + j] = 255 - data0[z * InputHeight * InputWidth + tempIndex + j];
				/*
				if (int(Reconstruction8U->data.fl[tempIndex + j]) != int(data0[z * InputHeight * InputWidth + tempIndex + j])) {
					cout << Reconstruction8U->data.fl[tempIndex + j] << endl;
					cout << int(data0[z * InputHeight * InputWidth + tempIndex + j]) << endl;
				}
				*/
			}
		}

		char filename[200] = { '\0' };
		sprintf(filename, "./test/8bit/Reconstruction_16p_z%d.bmp", z);

		CvMat* OutputImage = cvCreateMat(OutputHeight, OutputWidth, CV_32FC1);

		cvResize(Reconstruction8U, OutputImage); \

			cvSaveImage(filename, OutputImage);

		//cvReleaseMat(Reconstruction8U);
		//cout << z << endl;
	}
	
	// Read image data and store in buffer.
	fwrite(data0, sizeof(uchar), framesize, fp);
	
	end = ((double)getTickCount() - start) / getTickFrequency();
	printf("Time: %lf\n", end);

	return 0;
}