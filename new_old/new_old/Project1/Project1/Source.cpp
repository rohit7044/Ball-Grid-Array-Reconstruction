#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp> 
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <opencv/highgui.h>
using namespace std;
using namespace cv;
typedef complex < double > base;
#define PI			 3.14159
#define theta		 25
#define phi			 22.5	// = 360 / 16
#define Projection	 16
#define Level		 200	// 200
#define OriginWidth  1000
#define OriginHeight 1124
#define InputWidth   1024
#define InputHeight  1024
#define OutputWidth  1200
#define OutputHeight 1200
/*
*  3D Reconstruction
*
*  2017.12.21  FBP v5
*
*/
typedef complex < double > base;

#define PI			 3.14159
#define theta		 35
#define phi			 22.5	 // = 360 / 16
#define Projection	 16
#define Level		 200
#define InputWidth   1000
#define InputHeight  1000
#define OutputWidth  1000
#define OutputHeight 1000
#define meshGridSize InputWidth*InputHeight

struct params {
	//The real detector panel pixel density (number of pixels)
	int nu = 1000;
	int nv = 1000;

	// Detector setting (real size)
	double su = 5.984;	// mm (real size)
	double sv = 5.984;     // mm

						   // X-ray source and detector setting
	double DSD = 237.57;    //  Distance source to detector
	double DSO = 26.475;	//  X-ray source to object axis distance

	double dir = -1;   // gantry rotating direction (clock wise/ counter clockwise)
	double dang = 22.5; // angular step size (deg) 0:2.81:357.19
						//double deg[222]; = 0:dang:357.19;
	int nProj = 0;

	double du = (double)su / (double)nu;
	double dv = (double)sv / (double)nv;

	double us[1000];
	double vs[1000];


}param;

double meshGrid[meshGridSize];
int cvimage[Projection*InputWidth*InputHeight];
float imageData[InputHeight*InputWidth];

int main()
{
	double start, end;
	//­p®É
	start = (double)getTickCount();

	int cnt1 = 0;
	for (int i = -(param.nu - 1) / 2; i <= (param.nu - 1) / 2; i++) {
		param.us[cnt1++] = (double)i*param.du;
	}
	int cnt2 = 0;
	for (int i = -(param.nv - 1) / 2; i <= (param.nv - 1) / 2; i++) {
		param.vs[cnt2++] = (double)i*param.dv;
	}
	for (int i = 0; i < cnt2; i++) {
		int meshGridIndex = i * cnt2;
		for (int j = 0; j < cnt1; j++) {
			meshGrid[meshGridIndex++] = param.DSD / sqrt((param.DSD)*(param.DSD) + (param.us[j] * param.us[j]) + (param.vs[i] * param.vs[i]));
		}
	}

	/* Read 16 Projection Images
	***************************************/
	CvMat *ProjectionImage[Projection];

	for (int k = 0; k < Projection; ++k)
	{
		// Open file.
		char filename[200] = { '\0' };
		sprintf(filename, "./Projection_image/16Projection%d.raw", 95 + k);

		FILE *fp = NULL;
		fp = fopen(filename, "rb");

		// Memory allocation for bayer image data buffer.
		int framesize = 1124 * 1000 * 2;
		unsigned int *imagedata = NULL;
		imagedata = (unsigned int*)malloc(sizeof(unsigned int) * framesize);

		// Read image data and store in buffer.
		fread(imagedata, sizeof(unsigned int), framesize, fp);

		Mat img;
		img.create(1124, 1000, CV_16UC1);
		memcpy(img.data, imagedata, framesize);

		CvMat *Image = cvCreateMat(1124, 1000, CV_16UC1);
		CvMat temp = img;
		cvCopy(&temp, Image);

		ProjectionImage[k] = cvCreateMat(InputWidth, InputHeight, CV_16UC1);
		cvResize(Image, ProjectionImage[k]);
		filename[k] = { '\0' };
		sprintf(filename, "./test/Projection_16p_%d.bmp", k);
		cvSaveImage(filename, ProjectionImage[k]);
	}

	printf("Read 16 Projection Images Done.\n");


	/* Create Filter
	***************************************/
	int filt_len = getOptimalDFTSize(InputHeight);

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
				filt[i] = filt[i] * (1.0 + cos(W[i])) / 3.0;
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
	}
	for (int i = (int)filt.size() - 1 - 1; i > 0; i--) filt.push_back(filt[i]);
	for (int i = 0; i < filt.size(); i++) if (filt[i] > PI) filt[i] = 0.0;

	// Convert 1D filter to 2D filter.
	int filter_len = filt.size();

	vector < vector < double > > filter2d(filter_len, vector < double >(filter_len));
	int center = filter_len * 0.5;
	int distance;

	for (int i = 0; i < filter_len; i++)
	{
		for (int j = 0; j < filter_len; j++)
		{
			distance = (int)(sqrt((i - center) * (i - center) + (j - center) * (j - center)));
			if (distance > center)
				//filter2d[i][j] = 0;
				filter2d[i][j] = (filt[i] + filt[j]) / 2;
			else
				filter2d[i][j] = filt[center - distance];
			//filter2d[i][j] = 0;
		}
	}

	/*write the hann filter value*/
	/*
	FILE *fp;
	fp = fopen("./test.txt", "w");
	if (fp == NULL) {
		printf("no file exist!\n");
		fp = fopen("./test.txt", "a");
		fp = fopen("./test.txt", "w");
	}
	else {
		for (int i = 0; i < filt.size(); i++) {
			float kernel_value = filt[i];
			fprintf(fp, "%f\n", kernel_value);
		}
	}
	fclose(fp);
	*/
	printf("Create Filter Done.\n");

	/* Fourier Transform
	***************************************/
	for (int k = 0; k < Projection; k++)
	{

		CvMat* ori_image = cvCreateMat(InputWidth, InputHeight, CV_32FC2);
		CvMat* dft_image = cvCreateMat(InputWidth, InputHeight, CV_32FC2);
		cvZero(ori_image);
		cvZero(dft_image);

		float pixel_value = 0.0;

		//*meshgrid 
		for (int i = 0; i < InputWidth; i++) {
			int tempIndex = i * InputHeight;
			for (int j = 0; j < InputHeight; j++)
			{
				pixel_value = ProjectionImage[k]->data.s[tempIndex + j] * meshGrid[tempIndex + j];
				ProjectionImage[k]->data.s[tempIndex + j] = pixel_value;
				CvScalar pixel;
				pixel.val[0] = pixel_value;
				pixel.val[1] = 0.0;
				//height , width
				cvSet2D(ori_image, j, i, pixel);
			}
		}

		// DFT
		cvDFT(ori_image, dft_image, CV_DXT_FORWARD);

		double f_realNum = 0, f_imagNum = 0;
		for (int i = 0; i < InputHeight; i++)
		{
			for (int j = 0; j < InputWidth; j++)
			{
				f_realNum = cvGet2D(dft_image, i, j).val[0];
				f_imagNum = cvGet2D(dft_image, i, j).val[1];
				CvScalar dft_pixel;
				dft_pixel.val[0] = f_realNum * filter2d[i][j];
				dft_pixel.val[1] = f_imagNum * filter2d[i][j];
				cvSet2D(dft_image, i, j, dft_pixel);

			}
		}
		// IDFT
		cvZero(ori_image);
		cvDFT(dft_image, ori_image, CV_DXT_INVERSE);

		float realNum = 0;
		for (int i = 0; i < InputHeight; i++)
		{
			for (int j = 0; j < InputWidth; j++)
			{
				realNum = cvGet2D(ori_image, i, j).val[0] / (InputHeight * InputWidth);
				ProjectionImage[k]->data.s[j*InputWidth + i] = realNum;
			}
		}


	}

	printf("Fourier Transform Done.\n");


	/* Back Projection
	***************************************/
	CvMat *Reconstruction[Level];
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

	float angleUnit = PI * 0.005556;  // 0.005556 = 1 / 180
	float value = sin(theta * angleUnit) * Level;
	float reciprocalLevel = 1 / (float)Level;

	float RadiusUnit = 360.0 * 0.0625 * PI * 0.005556;
	for (int z = 0; z < Level; z++)
	{
		//printf("z = %d\n", z);
		float zValue = (float)(Level - z) * reciprocalLevel;
		for (int k = 0; k < Projection; k++)
		{
			float angle = k * phi * angleUnit;
			float sinAngle = sin(angle);
			float cosAngle = cos(angle);

			int x, y;

			for (int i = 0; i < InputHeight; ++i)
			{
				y = (i + value) - zValue * value * sinAngle + 0.5;
				if (y >= 0 && y < OutputHeight)
				{
					for (int j = 0; j < InputWidth; ++j)
					{
						x = (j + value) + zValue * value * cosAngle + 0.5;
						if (x >= 0 && x < OutputWidth)
						{
							int projectionValue = ProjectionImage[k]->data.s[i * ProjectionImage[k]->cols + j];
							Reconstruction[z]->data.fl[y * Reconstruction[z]->cols + x]
								+= ((float)projectionValue * reciprocalLevel);
						}
					}
				}
			}
		}
	}
	/*
	printf("Back Projection Done.\n");

	for (int z = 0; z < Level; z++)
	{
	char filename[200] = { '\0' };
	sprintf(filename, "./ReconstructionImage/noconstrast_Reconstruction_16p_z%d.bmp", z);
	cvSaveImage(filename, Reconstruction[z]);
	}
	*/

	/* 16-bit to 8-bit
	***************************************/

	for (int z = 0; z < Level; z++)
	{
		// Get min and max.
		double min, max;
		cvMinMaxLoc(Reconstruction[z], &min, &max);

		min /= 2;
		max /= 128;

		CvMat *Reconstruction8U;
		Reconstruction8U = cvCreateMat(OutputWidth, OutputHeight, CV_32FC1);
		int average = 0;
		// Normalize.
		for (int i = 0; i < OutputHeight; i++)
		{
			int tempIndex = i * OutputWidth;

			for (int j = 0; j < OutputWidth; j++)
			{
				//(pixel value - min)/(max-min)*255
				Reconstruction8U->data.fl[tempIndex + j]
					= (Reconstruction[z]->data.fl[tempIndex + j] - min) * 255 / (max - min);

				if (Reconstruction8U->data.fl[tempIndex + j] < 0)
					Reconstruction8U->data.fl[tempIndex + j] = 0;

				if (Reconstruction8U->data.fl[tempIndex + j] > 255)
					Reconstruction8U->data.fl[tempIndex + j] = 255;
				//255-xx
				Reconstruction8U->data.fl[tempIndex + j]
					= 255.0 - Reconstruction8U->data.fl[tempIndex + j];
			}
		}

		char filename[200] = { '\0' };
		sprintf(filename, "./Reconstruction_image/FBP_16p_200/Reconstruction_16p_z%d.bmp", z);

		CvMat *OutputImage = cvCreateMat(1124, 1000, CV_32FC1);

		cvResize(Reconstruction8U, OutputImage);
		cvSaveImage(filename, OutputImage);
	}

	end = ((double)getTickCount() - start) / getTickFrequency();
	printf("Time: %lf\n", end);

	return 0;
}
