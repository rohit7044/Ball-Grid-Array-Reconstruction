#define _USE_MATH_DEFINES

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>


#include <iostream>
#include <cstdio>
#include <cmath>
#include <vector>



#define theta		 35
//#define phi			 22.5	// = 360 / 16
#define Level        100


using namespace cv;

using namespace std;

typedef complex< double > base;
void  post_processing_sharpen(Mat& img);
Mat rotate(Mat& src, double angle) {
	Mat dst;
	Point2f pt(src.cols / 2., src.rows / 2.);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(src, dst, r, Size(src.cols, src.rows));
	return dst;
}

Mat art(Mat& sinogram)
{
	Mat results(171, 164, CV_32FC1, Scalar(1));

	double step = 4;
	for (int z = 0; z < 100; z++) {
		for (double k = 0; k < 180; k += step) {
			for (int i = 0; i < results.cols; i++) {
				float colSum = sinogram.at<float>(k, i) * sinogram.rows;
				if (colSum == 0) {
					colSum = 1;
				}
				float newColSum = 0;
				// measured Data for projection

				for (int j = 0; j < results.rows; j++) {
					
					newColSum += results.at<float>(j, i);
				}
				for (int j = 0; j < results.rows; j++) {
					results.at<float>(j, i) += ((colSum - newColSum) / 171); //ART
				}				
			}
			results = rotate(results, step);
			post_processing_sharpen(results);
		}
		results = rotate(results, 180);

	}
	return results;
}

Mat mart(Mat& sinogram)
{
	Mat results(171, 164, CV_32FC1, Scalar(1));

	double step = 1;
	
	for (int z = 0; z < 100; z++) 
	{
		for (double k = 0; k < 180; k += step) 
		{
			for (int i = 0; i < results.cols; i++) 
			{
				float colSum = sinogram.at<float>(k, i) * sinogram.rows;
				if (colSum == 0) 
				{
					colSum = 1;
				}
				float newColSum = 0;
				// measured Data for projection

				for (int j = 0; j < results.rows; j++) 
				{
					if (fabs(results.at<float>(j, i)) < 1e-9) 
					{ //MART
						results.at<float>(j, i) = 1;
					}
					newColSum += results.at<float>(j, i);
				}
				for (int j = 0; j < results.rows; j++) 
				{
					results.at<float>(j, i) = colSum / newColSum * results.at<float>(j, i); //MART
				}

			}
			results = rotate(results, step);
		}
		results = rotate(results, 250);

	}
	return results;
}

Mat radonTransform(Mat& original)
{
	Mat rotatedImage;
	Mat sinogram(180, 164, CV_32FC1);
	for (int k = 0; k < 180; k++) {

		rotatedImage = rotate(original, k);
		//imshow("rti", rotatedImage);
		//waitKey(0);

		float originalColSum;

		for (int i = 0; i < rotatedImage.cols; i++) {
			originalColSum = 0;
			for (int j = 0; j < rotatedImage.rows; j++) {
				originalColSum += rotatedImage.at<uchar>(j, i);
			}
			sinogram.at<float>(k, i) = originalColSum / rotatedImage.rows;
		}
	}
	normalize(sinogram, sinogram, 0, 1, NORM_MINMAX, CV_32FC1);
	return sinogram;
}

void post_processing_fourier(Mat&img) {
	Mat sinogram = img;
	//Fourier Tansform
	Mat forwarddft = Mat::zeros(sinogram.size().height, sinogram.size().width, CV_32F);
	dft(sinogram, forwarddft, DFT_ROWS | DFT_COMPLEX_OUTPUT);
	//dft(sinogram, forwarddft, DFT_ROWS | DFT_REAL_OUTPUT);
	//imshow("DFT image", forwarddft);
	//waitKey(0);
	//Applying hanning Filter
	Mat window, filtered_sinogram;
	filtered_sinogram = Mat::zeros(sinogram.size().height, sinogram.size().width, CV_32F);
	createHanningWindow(window, sinogram.size(), CV_32F);
	for (int i = 0; i < sinogram.size().height; i++)
	{
		for (int j = 0; j < sinogram.size().width; j++)
		{
			filtered_sinogram.at<float>(i, j) = forwarddft.at<float>(i, j) * window.at<float>(i, j);
		}
	}
	//imshow("Hanning Filter image", filtered_sinogram);
	//waitKey(0);
	// Inverse FFT
	Mat inversefft = Mat::zeros(sinogram.size(), CV_32F);
	dft(filtered_sinogram, inversefft, DFT_INVERSE | DFT_ROWS | DFT_REAL_OUTPUT);
	normalize(inversefft, inversefft, 0, 1, NORM_MINMAX, CV_32F);
	imshow("Filtered Sinogram After Fourier Transform", inversefft);
	waitKey(0);
}

void post_processing_morphology(Mat& img) {
	Mat kernel = getStructuringElement(MORPH_RECT,Size(3,3));
	Mat eroded;
	erode(img, eroded, kernel,Point(-1,-1),1);
	Mat opening;
	morphologyEx(img,opening,MORPH_OPEN,kernel);
	imshow("Eroded Image", eroded);
	imshow("Opening Image", opening);
	waitKey(0);


}

void post_processing_sharpen(Mat& img) {
	Mat sharp;
	Mat sharpening_kernel = (Mat_<double>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	filter2D(img, sharp, -1, sharpening_kernel);
	//imshow("Sharp Image",sharp);
	//waitKey(0);

}



int main() {
	string image_path = samples::findFile("D:/Lab Works/Computed Tomography/BGA_Reconstruction/1.png");
	Mat original = imread(image_path,0);

	if (original.empty()) {
		cout << "Could not read the image: " << image_path << endl;
		return 1;
	}
	// Radon Transform
	Mat sinogram = radonTransform(original);
	//imshow("Sinogram", sinogram);	
	//waitKey(0);

	//  Algebraic Reconstruction Technique
	//Algebraic Reconstruction Technique(Additive)
	Mat artRconstructedAdditive = art(sinogram);
	normalize(artRconstructedAdditive, artRconstructedAdditive, 0, 1, NORM_MINMAX, CV_32FC1);
	imshow("ART Reconstruction (Additive)", artRconstructedAdditive);
	waitKey(0);
	//post_processing_fourier(artRconstructedAdditive);
	//post_processing_morphology(artRconstructedAdditive);
	//post_processing_sharpen(artRconstructedAdditive);

	// Algebraic Reconstruction Technique (Multiplicative)
	/*Mat artRconstructedMultiplicative = mart(sinogram);
	normalize(artRconstructedMultiplicative, artRconstructedMultiplicative, 0, 1, NORM_MINMAX, CV_32FC1);
	imshow("ART Reconstruction (Multiplicative)", artRconstructedMultiplicative);
	waitKey(0);*/

	return 0;

}