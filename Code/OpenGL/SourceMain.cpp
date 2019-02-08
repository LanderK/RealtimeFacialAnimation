/*
	Real-Time Facial Animation for Untrained Users 

	Final Year Project By Lander King 


*/

#define GLEW_STATIC
#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/optimization.h>
#include <dlib/statistics.h>

#include <vector>
#include "Shader.h"
#include "Mesh.h"
#include "Blendshape.h"
#include "Model.h"
#include "Camera.h"
#include "Function.h"

#define	FACE_DOWNSAMPLE_RATIO 2
using namespace dlib;
typedef matrix<double, 0, 1> column_vector;

GLuint screenWidth = 1800, screenHeight = 900;
const float  PI_F = 3.14159265358979f;

//Storage for current face shape, Example Faces, and Estimated Face Shapes
full_object_detection currentFaceShape;
std::vector<point> shape2;
std::vector<dlib::point> regionSolve;
std::vector<full_object_detection> exampleShapes;

//Storage for Transformation matrices
double Transform[4] = {1,1,0,0};
matrix<double> Rotation(2, 2);

//Storage for PCA data
cv::PCA pca;
cv::Mat PCAEigenVectors, PCAMean;
std::vector<column_vector> EigModels;


#ifndef BIGVISION_RENDER_FACE_H_
#define BIGVISION_RENDER_FACE_H_

void printColumnVector(const column_vector& m) {


	std::cout << "Vector = ";

	for (int i = 0; i < m.size(); i++) {
		std::cout << m(i);
		std::cout << ", ";
	}

	std::cout << std::endl;

}

point calcWeightedShape(const column_vector& m, int i) {

	assert(m.size() == (exampleShapes.size()));

	point currentPoint;
	currentPoint.x() = (Transform[0] * exampleShapes[0].part(i).x());
	currentPoint.y() = (Transform[1] * exampleShapes[0].part(i).y());
	
	for (unsigned int j=0; j < exampleShapes.size() ; j++) {

		currentPoint.x() += m(j) * (Transform[0] * (exampleShapes[j].part(i).x() - exampleShapes[0].part(i).x()));
		currentPoint.y() += m(j) * (Transform[1] * (exampleShapes[j].part(i).y() - exampleShapes[0].part(i).y()));
	}

	//Apply Rotation and translation
	double temp = (currentPoint.x() * Rotation(0, 0)) + (currentPoint.y() * Rotation(0, 1)) + Transform[2];
	currentPoint.y() = (currentPoint.x() * Rotation(1, 0)) + (currentPoint.y() * Rotation(1, 1)) + Transform[3];
	currentPoint.x() = temp;

	return currentPoint;
}

point calcRotatedShape(const column_vector& m, int i) {

	assert(m.size() == 4);

	point currentPoint;
	currentPoint.x() = (Transform[0] * exampleShapes[0].part(i).x());
	currentPoint.y() = (Transform[1] * exampleShapes[0].part(i).y());

	int limit = exampleShapes.size();
	int j = limit - m.size();
	//Check if  the current point is one associated with the jawline or nose region
	
	for (int k=0; k < m.size(); j++,k++) {
		currentPoint.x() += m(k) * (Transform[0] * (exampleShapes[j].part(i).x() - exampleShapes[0].part(i).x()));
		currentPoint.y() += m(k) * (Transform[1] * (exampleShapes[j].part(i).y() - exampleShapes[0].part(i).y()));
	}
	

	//Apply Rotation and translation
	double temp = (currentPoint.x() * Rotation(0, 0)) + (currentPoint.y() * Rotation(0, 1)) + Transform[2];
	currentPoint.y() = (currentPoint.x() * Rotation(1, 0)) + (currentPoint.y() * Rotation(1, 1)) + Transform[3];
	currentPoint.x() = temp;

	return currentPoint;
}

point calcRegionShape(const column_vector& m, int i) {

	assert(m.size() == (exampleShapes.size())-4);

	point currentPoint;
	currentPoint.x() = regionSolve[i].x();
	currentPoint.y() = regionSolve[i].y();

	//Check if  the current point is one associated with the eyebrow region
	if (i > 17 && i<= 26)
	{
		for (int j = 0; j < 6; j++) {
			currentPoint.x() += m(j) * (Transform[0] * (exampleShapes[j].part(i).x() - exampleShapes[0].part(i).x()));
			currentPoint.y() += m(j) * (Transform[1] * (exampleShapes[j].part(i).y() - exampleShapes[0].part(i).y()));
		}
	}
	//Check if the current Point is associated with the mouth region
	else if (i<=16 ||(i >= 48 && i <= 68 )) {

		for (int j = 6; j < m.size(); j++) {
			currentPoint.x() += m(j) * (Transform[0] * (exampleShapes[j].part(i).x() - exampleShapes[0].part(i).x()));
			currentPoint.y() += m(j) * (Transform[1] * (exampleShapes[j].part(i).y() - exampleShapes[0].part(i).y()));
		}
	}
	//Apply Rotation and translation
	double temp = (currentPoint.x() * Rotation(0, 0)) + (currentPoint.y() * Rotation(0, 1));
	currentPoint.y() = (currentPoint.x() * Rotation(1, 0)) + (currentPoint.y() * Rotation(1, 1));
	currentPoint.x() = temp;

	return currentPoint;
}

double calcDiffL2NormSquared(point l, point m) {

	double xdif = l.x() - m.x();
	double ydif = l.y() - m.y();

	ydif = ydif*ydif;
	xdif = xdif*xdif;

	double difLength = sqrt(ydif + xdif);

	return difLength;

}

double calcL1Norm(const column_vector &m) {
    
	double l1 = 0;
	for (int i = 0; i < m.size(); i++) {
		l1 += abs(m(i));
	}
	return l1;
}

double minFunction(const column_vector& m) {

	double result = 0.0;
	//Total L2 Norm of Difference between Estimated Points and Observed Points 
	for (unsigned int i = 0; i < currentFaceShape.num_parts(); i++) {
		point weightedPoint = calcWeightedShape(m, i);
		result += calcDiffL2NormSquared(currentFaceShape.part(i), weightedPoint);
	}
	//Add L1 Norm to normallaize result(make sparse)
	result += calcL1Norm(m);
	return result;
}

double rotationMinFunction(const column_vector& m) {

	double result = 0.0;
	//Total L2 Norm of Difference between Estimated Points and Observed Points 
	for (unsigned int i = 0; i < currentFaceShape.num_parts(); i++) {
		point weightedPoint = calcRotatedShape(m, i);
		result += calcDiffL2NormSquared(currentFaceShape.part(i), weightedPoint);
	}
	//Add L1 Norm to normallaize result(make sparse)
	result += calcL1Norm(m);
	return result;
}

double regionMinFunction(const column_vector& m) {

	double result = 0.0;
	//Total L2 Norm of Difference between Estimated Points and Observed Points 
	for (unsigned int i = 0; i < currentFaceShape.num_parts(); i++) {
		point weightedPoint = calcRegionShape(m, i);
		result += calcDiffL2NormSquared(currentFaceShape.part(i), weightedPoint);
	}
	//Add L1 Norm to normallaize result(make sparse)
	result += calcL1Norm(m);
	return result;
}

double PCAsolvefunction(const column_vector& m) {

	double result = 0.0;
	int eigenId = EigModels.size();
	double xdif, ydif;

	//Solve for the Total of the L2 Norm of Difference between Estimated Points and Observed Points 

	//Solve for mean
	if (eigenId == 0) {
		for (int i = 0; i < PCAEigenVectors.cols / 2; i++) {
			point weightedPoint = calcWeightedShape(m, i);
			xdif = PCAMean.at<double>(2 * i) - weightedPoint.x();
			ydif = PCAMean.at<double>((2 * i) + 1) - weightedPoint.y();

			ydif = ydif*ydif;
			xdif = xdif*xdif;

			result += sqrt(ydif + xdif);
		}
	}
	//Solve for Eigen Vectors
	else {
		
		for (int i = 0; i < PCAEigenVectors.cols / 2; i++) {
			point weightedPoint = calcWeightedShape(m, i);
			xdif = PCAEigenVectors.at<double>(eigenId, 2 * i) + PCAMean.at<double>(2 * i) - weightedPoint.x();
			ydif = PCAEigenVectors.at<double>(eigenId, (2 * i) + 1) + PCAMean.at<double>((2 * i) + 1) - weightedPoint.y();

			ydif = ydif*ydif;
			xdif = xdif*xdif;

			result += sqrt(ydif + xdif);
		}
	}


	//Add L1 Norm to normallaize result(make sparse)
	result += calcL1Norm(m);
	return result;


}


full_object_detection moveToTop(full_object_detection shape, rectangle r) {

	//Re-align the detected face to the Top Corner
	for (unsigned long i = 0; i < shape.num_parts(); ++i) {
		shape.part(i).x() = shape.part(i).x() - r.left();
		shape.part(i).y() = shape.part(i).y() - r.top();
	}

	return shape;
}

std::vector<point> getFinalShape(const column_vector& m) {

	std::vector<point> FinalShape;
	point Point;
	for (unsigned long i = 0; i < exampleShapes[0].num_parts(); i++) {
		Point.x() =(Transform[0] * exampleShapes[0].part(i).x());
		Point.y() = (Transform[1] * exampleShapes[0].part(i).y());
		for (unsigned int j = 0; j < exampleShapes.size(); j++) {
			Point.x() = Point.x() + (m(j) * (Transform[0] * (exampleShapes[j].part(i).x()-exampleShapes[0].part(i).x())));
			Point.y() = Point.y() + (m(j) * (Transform[1] * (exampleShapes[j].part(i).y()-exampleShapes[0].part(i).y())));
		}

		double temp = (Point.x() * Rotation(0, 0)) + (Point.y() * Rotation(0, 1));
		Point.y() = (Point.x() * Rotation(1, 0)) + (Point.y() * Rotation(1, 1)) + Transform[3];
		Point.x() = temp + Transform[2];

		FinalShape.push_back(Point);
	}

	return FinalShape;
}

std::vector<point> getFinalPcaShape(const cv::Mat &m) {
	
	std::vector<point> FinalShape;
	point Point;
	for (unsigned long i = 0; i < exampleShapes[0].num_parts(); i++) {
		Point.x() = (Transform[0] * PCAMean.at<double>(2 * i));
		Point.y() = (Transform[1] * PCAMean.at<double>((2 * i) + 1));
		for (int j = 0; j < PCAEigenVectors.rows; j++) {
			Point.x() = Point.x() + (m.at<double>(j) * PCAEigenVectors.at<double>(j,2*i));
			Point.y() = Point.y() + (m.at<double>(j) * PCAEigenVectors.at<double>(j, (2 * i)+1));
		}

		double temp = (Point.x() * Rotation(0, 0)) + (Point.y() * Rotation(0, 1));
		Point.y() = (Point.x() * Rotation(1, 0)) + (Point.y() * Rotation(1, 1)) + Transform[3];
		Point.x() = temp + Transform[2];

		FinalShape.push_back(Point);
	}

	return FinalShape;
}

std::vector<full_object_detection> initial_example_face_matrix(shape_predictor &sp, frontal_face_detector &detector) {

	//Load Example Facial Expressions
	const int nImages = 16;
	array2d<rgb_pixel> images[nImages];
	load_image(images[0], "images/Neutral.png");
	load_image(images[1], "images/BrowLL.png"); //Lowered Left Eyebrow
	load_image(images[2], "images/BrowRL.png"); //Raised Left Eyebrow
	load_image(images[3], "images/BrowLR.png"); //Lowered Right Eyebrow
	load_image(images[4], "images/BrowRR.png"); //Raised Right Eyebrow
	load_image(images[5], "images/Suck.png");
	load_image(images[6], "images/PullOpen.png");
    load_image(images[7], "images/PullL.png");
	load_image(images[8], "images/PullR.png");
	load_image(images[9], "images/Depress.png");
	load_image(images[10], "images/Pucker.png");
	load_image(images[11], "images/Stretch.png");
	load_image(images[12], "images/RotationL.png");
	load_image(images[13], "images/RotationR.png");
	load_image(images[14], "images/tiltL.png");
	load_image(images[15], "images/tiltR.png");

	// Make the image larger so we can detect small faces.
	std::vector<full_object_detection> shapes;
	for (unsigned long i = 0; i < nImages; i++) {
		//pyramid_up(images[i]);
		// Now tell the face detector to give us a list of bounding boxes
		// around all the faces in the image.
		std::vector<rectangle> dets = detector(images[i]);
		if (dets.size() == 0) {
			pyramid_up(images[i]);
			dets = detector(images[i]);
		}
		// Now we will go ask the shape_predictor to tell us the pose of
		// each face we detected.
		for (unsigned long j = 0; j < dets.size(); ++j)
		{
			full_object_detection shape = sp(images[i], dets[j]);
			shapes.push_back(shape);
		}
	}

	std::cout << "Shapes Size = " + shapes.size() << std::endl;

	
	return shapes;


}


void draw_polyline(cv::Mat &img, const dlib::full_object_detection& d, const int start, const int end, bool isClosed = false)
{
	std::vector <cv::Point> points;
	for (int i = start; i <= end; ++i)
	{
		points.push_back(cv::Point(d.part(i).x(), d.part(i).y()));
	}
	cv::polylines(img, points, isClosed, cv::Scalar(255, 0, 0), 2, 16);

}

void render_face(cv::Mat &img, const dlib::full_object_detection& d, const std::vector<point> e, const std::vector<point> f,int j)
{
	//Draw the landmarks ontop of the image
	for (unsigned int i = 0; i < d.num_parts(); i++) {
		if(j==0)cv::circle(img, cv::Point(d.part(i).x(), d.part(i).y()), 1, cv::Scalar(255, 0, 0), 2, 16); // Tracked Landmarks
		if(j==1)cv::circle(img, cv::Point(e[i].x(), e[i].y()), 1, cv::Scalar(0, 255, 0), 2, 16); // All-in-One Landmarks
		if(j==2)cv::circle(img, cv::Point(f[i].x(), f[i].y()), 1, cv::Scalar(255, 0, 255), 2, 16); // Region Landmarks
	}
	

}

#endif // BIGVISION_RENDER_FACE_H_

rectangle findLandmarkRectangle(full_object_detection &d) {

	//Finds the Rectangle that bounds the detected landmarks exluding the points representing the Eyebrows

	float minX = d.part(0).x(), minY = d.part(0).y(), maxX = d.part(0).x(), maxY = d.part(0).y();
	for (unsigned long j = 1; j < d.num_parts(); ++j) {
		if (!(j >= 16 && j <= 27)) {
			if (d.part(j).x() <= minX) minX = d.part(j).x();
			if (d.part(j).x() >= maxX) maxX = d.part(j).x();
			if (d.part(j).y() <= minY) minY = d.part(j).y();
			if (d.part(j).y() >= maxY) maxY = d.part(j).y();
		}
	}

	rectangle r(minX, maxY, maxX, minY);

	return r;
}

rectangle findLandmarkRectangle(std::vector<point> &d) {

	//Finds the Rectangle that bounds the detected landmarks exluding the points representing the Eyebrows

	float minX = d[0].x(), minY = d[0].y(), maxX = d[0].x(), maxY = d[0].y();
	for (unsigned long j = 1; j < d.size(); ++j) {
		if (!(j >= 16 && j <= 27)) {
			if (d[j].x() <= minX) minX = d[j].x();
			if (d[j].x() >= maxX) maxX = d[j].x();
			if (d[j].y() <= minY) minY = d[j].y();
			if (d[j].y() >= maxY) maxY = d[j].y();
		}
	}

	rectangle r(minX, maxY, maxX, minY);

	return r;
}

void findTransform(dlib::rectangle &currentRec, dlib::rectangle &baseRec) {

	//Finds the scale/ transform and rotation needed to map the estimated face onto the current face

	double currentWidth = (currentRec.right() - currentRec.left());
	double currentHeight = (currentRec.top() - currentRec.bottom());
	double baseWidth = (baseRec.right() - baseRec.left());
	double baseHeight = (baseRec.top() - baseRec.bottom());

	double currentMidx = currentRec.left() + (currentRec.width() / 2);
	double currentMidy = currentRec.bottom() + (currentRec.height() / 2);
	double baseMidx = (currentWidth / baseWidth) * (baseRec.left() + (baseRec.width() / 2));
	double baseMidy = (currentHeight / baseHeight) * (baseRec.bottom() + (baseRec.height() / 2));

	Transform[0] = currentWidth / baseWidth;  // X Scale
	Transform[1] = currentHeight / baseHeight; // Y Scale
	Transform[2] = currentMidx - baseMidx; // X translation
	Transform[3] = currentMidy - baseMidy; // Y translation

}

void findRotation(full_object_detection &face) {

	//Find the Rotation needed to match the Scaled Estimatated Face Shape onto the Viewed Face Shape Use Singular Value Decompostion
	//This is needed to to get a better estimation of the shape and to be able to add rotation in the xy-plane for the final model

	column_vector estPoint(2), currPoint(2);

	column_vector centerC(2), centerE(2);
	
	for (unsigned int i = 0; i < face.num_parts(); i++) {
		currPoint = face.part(i).x(), face.part(i).y();
		estPoint = (Transform[0] * exampleShapes[0].part(i).x()), (Transform[1] * exampleShapes[0].part(i).y());
		centerC += currPoint;
		centerE += estPoint;
	}

	centerC = centerC / face.num_parts();
	centerE = centerE / face.num_parts();

	matrix<double> H, U, W, V,rotNew;

	for (unsigned int i = 0; i < face.num_parts(); i++) {
		currPoint = face.part(i).x(), face.part(i).y();
		estPoint = (Transform[0] * exampleShapes[0].part(i).x()), (Transform[1] * exampleShapes[0].part(i).y());
		H += (estPoint - centerE) * trans(currPoint - centerC);
	}

	svd(H, U, W, V);
	
	rotNew = V * trans(U);

	if (!(det(rotNew) < 0) & isfinite(rotNew)) {
		Rotation = rotNew;
	}

	matrix<double> T = (-Rotation * centerE) + centerC;

	Transform[2] = T(0);
	Transform[3] = T(1);

	std::cout << Rotation << std::endl;


}

cv::Mat objectDetectionToVector(full_object_detection &d) {
	//Reformat Face Detection to a matrix so we can run PCA on it 
	//Change ((x1,y1),(x2,y2),....) to (x1,y1,x2,y2,....)

	cv::Mat vec = (cv::Mat_<double>(1, 2) << d.part(0).x(), d.part(0).y());
	cv::Mat x;
#
	for (unsigned int i = 1; i < d.num_parts(); i++) {
		x = (cv::Mat_<double>(1, 2) << d.part(i).x(), d.part(i).y());
		cv::hconcat(vec, x, vec);
	}

	//std::cout << vec.size() << std::endl;
	return vec;

}

cv::Mat pcaFaceSolve(full_object_detection &face) {
	
	//Solve the following 

	//Use SVD to find pseudo invderse for B as it is not square/invertible
	

	cv::Mat b = objectDetectionToVector(face);

	//Need to Scale Mean and EigenVectors to match current face shape
	cv::Mat scaledNorm = b;
	for (int i = 0; i < PCAMean.cols; i++) {
		if (i % 2 == 0) {
			scaledNorm.at<double>(i) -= (Transform[0] * PCAMean.at<double>(i)) +Transform[2];
		}
		else {
			scaledNorm.at<double>(i) -= (Transform[1] * PCAMean.at<double>(i)) +Transform[3];
		}
	}

	cv::Mat EigenTranspose;
	cv::transpose(PCAEigenVectors,EigenTranspose);

	//Project the scaled norm into 
	cv::Mat p = scaledNorm * EigenTranspose;

	//std::cout << p.size() << std::endl;
	//std::cout << p << std::endl;
	return p;

}



bool doesFileExist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}


void PCAsetup() {

	
	cv::Mat pcaMat = objectDetectionToVector(exampleShapes[0]), colVec;
	

	for (unsigned int i = 1; i < exampleShapes.size(); i++) {
		colVec = objectDetectionToVector(exampleShapes[i]);
		vconcat(pcaMat, colVec, pcaMat);
	}

	
	pca = pca(pcaMat, cv::Mat(), CV_PCA_DATA_AS_ROW);

	cv::Mat eigenValues = pca.eigenvalues.clone();
	cv::Mat eigenVectors = pca.eigenvectors.clone();
	PCAMean = pca.mean.clone();


	double TotalEnergy = cv::sum(eigenValues)[0];
	double currentEnergy = 0;

	//Find number of EigenVectors to keep to have 99.9999% Energy
	int EigenVecsToKeep = 0;
	while (currentEnergy / TotalEnergy <= 0.9999) {
		currentEnergy += abs(eigenValues.at<double>(EigenVecsToKeep));
		EigenVecsToKeep++;
	}

	//Store the desired EigenVectors

	PCAEigenVectors = cv::Mat(EigenVecsToKeep, eigenVectors.cols, CV_64FC1, eigenVectors.data);
	eigenValues = cv::Mat(EigenVecsToKeep, eigenValues.cols, CV_64FC1, eigenValues.data);
	

	//Solve for new eigenVector Models
	for (int i = 0; i <= PCAEigenVectors.rows; i++) {
		column_vector sol(16);
		sol = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		//find optimized solution for Each EigenVector
		find_min_bobyqa(PCAsolvefunction, sol, 34, uniform_matrix<double>(16, 1, 0.0), uniform_matrix<double>(16, 1, 1.0), 0.075, 0.01, 30000);
		sol(0) = 1.0;
		//Repeat Optimization with reseted first element and smaller stoping criteria
		find_min_bobyqa(PCAsolvefunction, sol, 34, uniform_matrix<double>(16, 1, 0.0), uniform_matrix<double>(16, 1, 1.0), 0.05, 0.005, 30000);
		EigModels.push_back(sol);
	}

	//Save to file
	cv::FileStorage fs("EigenVectors.xml", cv::FileStorage::WRITE);
	//fs << "pca" << pca;
	fs << "eigenMatrix" << PCAEigenVectors;
	fs << "PCAmean" << PCAMean;
	//fs << "PCAeigVals" << eigenValues;
	fs << "PCAMODEL" << EigModels;
	fs.release();

	serialize("PCAEigModels.dat") << EigModels;


}

void roundColumnVector(column_vector &m) {

	for (int i = 0; i < m.size(); i++) {
		m(i) = floor((m(i) *1000.0) + 0.5) / 1000.0;
	}

}

std::vector<Vertex> blendshapeSolve(Blendshape &blend,const column_vector &w) {

	std::vector<Vertex> blendModel;

	for (unsigned int i = 0; i < blend.baseModel.size(); i++) {
		Vertex newV = {blend.baseModel[i].Position , blend.baseModel[i].Normal, blend.baseModel[i].TexCoords };
		for (int j = 1; j < w.size(); j++) {
			newV.Position += (blend.blendshapes[j-1][i].Position * (float)w(j));
		}
		blendModel.push_back(newV);
	}
	return blendModel;

}

std::vector<Vertex> blendshapeSolvePca(Blendshape &blend, const column_vector &w) {

	std::vector<Vertex> blendModel;
	

	for (unsigned int i = 0; i < blend.baseModel.size(); i++) {
		Vertex newV = { blend.baseModel[i].Position , blend.baseModel[i].Normal, blend.baseModel[i].TexCoords };
		for (int j = 1; j < w.size(); j++) {
			newV.Position += (blend.blendshapes[j-1][i].Position * (float)w(j));
		}
		blendModel.push_back(newV);
	}
	return blendModel;

}


column_vector matToColumn(const cv::Mat &m) {

	assert(m.rows == 1);

	column_vector vec(m.cols);

	for (int i = 0; i < m.cols; i++) {
		vec(i) = m.at<double>(i);
	}

	return vec;

}

column_vector concatColVec(const column_vector &a, const column_vector &b) {

	column_vector concatVec(a.size() + b.size());

	for (int i = 0; i < concatVec.size(); i++) {
		if (i >= a.size()) {
			concatVec(i) = b(i - a.size());
		}
		else {
			concatVec(i) = a(i);
		}
	}

	return concatVec;
}

double pcaError(std::vector<dlib::point> &f) {

	double result = 0;

	for (unsigned int i = 0; i < currentFaceShape.num_parts(); i++) {
		result += calcDiffL2NormSquared(currentFaceShape.part(i), f[i]);
	}

	return result;
}

int main(){

	
	try {

		Rotation = 1.0, 0.0, 0.0, 1.0;

		//Setup of OPenCv/Dlib Facial Feature Tracking
		cv::VideoCapture cap(0); //Change to 0 if 1 isn't working
		//cv::VideoCapture cap("C:/Users/Lander/OneDrive/Work/Final Year Project/OpenGL/OpenGL/testing/Test1.avi"); //Load Test Video
		if(!cap.isOpened())
		{
			std::cerr << "Unable to connect to camera" << std::endl;
			return 1;
		}
		

		cap.set(CV_CAP_PROP_FRAME_WIDTH, 600);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;

		//load landmark detector
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		//check if example landmark shapes have already been saved, else load it.
		if (pose_model.num_parts() == 68) {
			if (doesFileExist("exampleShapes68.dat")) {
				std::cout << "read exprssion data from file" << std::endl;
				deserialize("exampleShapes68.dat") >> exampleShapes;
			}
			else {
				std::cout << "loaded images" << std::endl;
				exampleShapes = initial_example_face_matrix(pose_model, detector);
				serialize("exampleShapes68.dat") << exampleShapes;
			}
		}
		//check if PCA has been done before.
		if (doesFileExist("EigenVectors.xml") && doesFileExist("PCAEigModels.dat")) {
			std::cout << "read ePCA data from file" << std::endl;
			cv::FileStorage fs("EigenVectors.xml", cv::FileStorage::READ);
			fs["eigenMatrix"] >> PCAEigenVectors;
			fs["PCAmean"] >> PCAMean;
			fs.release();
			
			deserialize("PCAEigModels.dat") >> EigModels;
		}
		else {
			//Run PCA on Example Shapes and find solutions for the Orthogonal basis shapes
			PCAsetup();
		}
		

		//OpenGL SetUP
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

		GLFWwindow* window = glfwCreateWindow(screenWidth, screenHeight, "Facial Animation Project", nullptr, nullptr);
		if (window == nullptr)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return -1;
		}
		glfwMakeContextCurrent(window);

		glewExperimental = GL_TRUE;
		if (glewInit() != GLEW_OK)
		{
			std::cout << "Failed to initialize GLEW" << std::endl;
			return -1;
		}

		glViewport(0, 0, screenWidth, screenHeight);

		glEnable(GL_DEPTH_TEST);

		//Load OpenGL Shaders
		Shader modelShader("model.vs", "model.frag");
		Shader basicShader("default.vs", "default.frag");
		Shader lightShader("light.vs", "light.frag");

		Model ourModel("models/Emily.obj");
		//Mesh to draw in 2nd View Port
		Mesh outputModel = ourModel.getMesh(0);
		//Mesh to draw in 3rd view Port
		Mesh outputModel2 = outputModel;

		//Load Models for Blendshape Model
		Blendshape blend("models/Brow_lower_l.obj", ourModel.getVertices(0));
		blend.addShape("models/Brow_raise_l.obj");
		blend.addShape("models/Brow_lower_r.obj");
		blend.addShape("models/Brow_raise_r.obj");
		blend.addShape("models/suck.obj");
		blend.addShape("models/mouth_PullOpen.obj");
		blend.addShape("models/mouth_Pull_l.obj");
		blend.addShape("models/mouth_Pull_r.obj");
		blend.addShape("models/mouth_LowerLipDepress.obj");
		blend.addShape("models/pucker.obj");
		blend.addShape("models/stretch.obj");
		blend.addShape("models/rotateL.obj");
		blend.addShape("models/rotateR.obj");
		blend.addShape("models/tiltL.obj");
		blend.addShape("models/tiltR.obj");

		//Define PCA Blendshapes
		Blendshape pcaBlend(blendshapeSolve(blend, EigModels[0]));

		std::vector<Vertex> pcaBlendModel;
		for (unsigned int j = 1; j < EigModels.size(); j++) {
			pcaBlendModel = blendshapeSolve(blend, EigModels[j]);
			pcaBlend.addShape(pcaBlendModel);
		}

		//Find Bounding Sphere of Model for the camera lookat vector and Position of the light Source
		glm::vec3 min = ourModel.getMinValues();
		glm::vec3 max = ourModel.getMaxValues();
		glm::vec3 center = glm::vec3((max.x + min.x) / 2, (max.y + min.y) / 2, (max.z + min.z) / 2);
		float rad = glm::length(max - center);
		glm::vec3 lightPos(center.x, center.y, center.z + 3 * rad);

		//Define Rectangle for the Webcam to be mapped onto
		GLfloat vertices[] = {
			// Positions          // Colors           // Texture Coords
			1.0f,  1.0f, -1.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // Top Right
			1.0f, -1.0f, -1.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // Bottom Right
			-1.0f, -1.0f, -1.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // Bottom Left
			-1.0f,  1.0f, -1.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // Top Left 
		};
		GLuint indices[] = {
			0, 1, 3, // First Triangle
			1, 2, 3  // Second Triangle
		};
		//Buffer
		GLuint VBO, VAO, EBO;
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);


		//put vertices in the buffer
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
		//Position att
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(0);
		//Color Att
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
		glEnableVertexAttribArray(1);
		//Text Att
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
		glEnableVertexAttribArray(2);

		glBindVertexArray(0);

		GLuint texture1;

		glGenTextures(1, &texture1);
		glBindTexture(GL_TEXTURE_2D, texture1); 
												
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		// Set texture filtering
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Load, create texture and generate mipmaps
		glBindTexture(GL_TEXTURE_2D, 0);

		//Cube used for the light
		GLfloat cube[] = {
			-0.5f, -0.5f, -0.5f,
			0.5f, -0.5f, -0.5f,
			0.5f,  0.5f, -0.5f,
			0.5f,  0.5f, -0.5f,
			-0.5f,  0.5f, -0.5f,
			-0.5f, -0.5f, -0.5f,

			-0.5f, -0.5f,  0.5f,
			0.5f, -0.5f,  0.5f,
			0.5f,  0.5f,  0.5f,
			0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f,  0.5f,
			-0.5f, -0.5f,  0.5f,

			-0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f, -0.5f,
			-0.5f, -0.5f, -0.5f,
			-0.5f, -0.5f, -0.5f,
			-0.5f, -0.5f,  0.5f,
			-0.5f,  0.5f,  0.5f,

			0.5f,  0.5f,  0.5f,
			0.5f,  0.5f, -0.5f,
			0.5f, -0.5f, -0.5f,
			0.5f, -0.5f, -0.5f,
			0.5f, -0.5f,  0.5f,
			0.5f,  0.5f,  0.5f,

			-0.5f, -0.5f, -0.5f,
			0.5f, -0.5f, -0.5f,
			0.5f, -0.5f,  0.5f,
			0.5f, -0.5f,  0.5f,
			-0.5f, -0.5f,  0.5f,
			-0.5f, -0.5f, -0.5f,

			-0.5f,  0.5f, -0.5f,
			0.5f,  0.5f, -0.5f,
			0.5f,  0.5f,  0.5f,
			0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f,  0.5f,
			-0.5f,  0.5f, -0.5f,
		};

		// First, set the container's VAO (and VBO)
		GLuint VBO2, lightVAO;
		glGenVertexArrays(1, &lightVAO);
		glGenBuffers(1, &VBO2);

		glBindBuffer(GL_ARRAY_BUFFER, VBO2);
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube), cube, GL_STATIC_DRAW);

		glBindVertexArray(lightVAO);
		// Position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(0);
		glBindVertexArray(0);

		float fov = 45.0f;
		
		//Storage for the Output of the Blendshape Sum
		std::vector<Vertex> BlendedPosition;
		std::vector<Vertex> OldPosition;
		float g_interp = 0;

		//Get current time used for time analysis of the program
		double lastTime = glfwGetTime(), startTime, endTime;
		int nbFrames = -1;
		column_vector expressionVector(16);
		column_vector rotationVector(4);
		column_vector expressVector(12);
		expressionVector = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
		rotationVector = 0.0, 0.0, 0.0, 0.0;
		expressVector = 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

		//Rotation = 1.0, 0.0, 0.0, 1.0;
		
		
		while (!glfwWindowShouldClose(window))
		{
			//Calculate the time per frame
			double currentTime = glfwGetTime();
			nbFrames++;
			if (currentTime - lastTime >= 1.0) {
				printf("%f ms/frame\n", 1000.0 / double(nbFrames));
				nbFrames = -1;
				lastTime += 10;
			}

			glfwPollEvents();
			glClearColor(0.05f, 0.05f, 0.05f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			
			BlendedPosition.clear();
			//Solve for the Final Model
			BlendedPosition = blendshapeSolve(blend, expressionVector);
			outputModel.reshapeMesh(BlendedPosition);

			for (int viewp = 0; viewp < 3; viewp++) {

				//Split the Screen into 3 Sections
				//1st displays the Webcam
				//2nd the Final Blendshape Sum
				//3rd the PCA Blendshape Sum

				if (viewp == 0) {

					glViewport(0, 0, screenWidth / 3, screenHeight);

				}

				if (viewp == 1) {

					glViewport(screenWidth / 3, 0, screenWidth / 3, screenHeight);

				}


				if (viewp == 2) {

					glViewport(screenWidth* 2/3, 0, screenWidth /3, screenHeight);

				}

				glClear(GL_DEPTH_BUFFER_BIT);

				cv::Mat temp;
				cv::Mat temp_small;

				if (viewp == 0) {

					//Matrices To store current frame from the webcam
					startTime = glfwGetTime();
					
					cap >> temp;
					
					//Storage for the faces and face shapes detected
					std::vector<rectangle> faces;
					std::vector<full_object_detection> shapes;
					rectangle r;

					//resize webcam image to be small so the facial detection can be computed quicker
					cv::resize(temp, temp_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO, 1.0 / FACE_DOWNSAMPLE_RATIO);

					cv_image<bgr_pixel> cimg_small(temp_small);
					cv_image<bgr_pixel> cimg(temp);

					//Detect Faces every Frame
					if(nbFrames%1==0 ) faces = detector(cimg_small);

					//Find the Pose for a detected faces
					for (unsigned long i = 0; i < faces.size(); ++i) {
						if (i == 1) {
							break;
						}
						r = rectangle(
							(long)(faces[i].left() * FACE_DOWNSAMPLE_RATIO) - 5,
							(long)(faces[i].top() * FACE_DOWNSAMPLE_RATIO) + 5,
							(long)(faces[i].right() * FACE_DOWNSAMPLE_RATIO) + 5,
							(long)(faces[i].bottom() * FACE_DOWNSAMPLE_RATIO) - 5
						);

						currentFaceShape = pose_model(cimg, r);
						findTransform(findLandmarkRectangle(currentFaceShape), findLandmarkRectangle(exampleShapes[0]));
						endTime = glfwGetTime();
						//std::cout << "   Time Taken for Landmark tracking = " << endTime - startTime << std::endl;
						
						startTime = glfwGetTime();
						//Solve for expression Vector
						expressionVector(0) = 1.0;
						find_min_bobyqa(minFunction, expressionVector, 33, uniform_matrix<double>(16, 1, 0.0), uniform_matrix<double>(16, 1, 1.0), 0.075, 0.01, 30000);
						expressionVector(0) = 1.0;
						//Repeat Optimization with reseted first element and smaller stoping criteria
						double error = find_min_bobyqa(minFunction, expressionVector, 33, uniform_matrix<double>(16, 1, 0.0), uniform_matrix<double>(16, 1, 1.0), 0.05, 0.005, 30000);
						//Testing 
						endTime = glfwGetTime();
						//std::cout << "All-in-One method" << std::endl;
						//std::cout << "Average Error = " << error / exampleShapes[0].num_parts();
						//std::cout << "   Time Taken = " <<  endTime-startTime << std::endl;
						

						roundColumnVector(expressionVector);
						shape2 = getFinalShape(expressionVector);

						
						/*Solve for PCA exprssion Vectors
						startTime = glfwGetTime();
						cv::Mat pcaSolve = pcaFaceSolve(currentFaceShape);
						std::vector<dlib::point> shapePCA= getFinalPcaShape(pcaSolve);
						endTime = glfwGetTime();
						//Testing
						//std::cout << "PCA method" << std::endl;
						error = pcaError(shapePCA);
						//std::cout << "Average Error = " << error / exampleShapes[0].num_parts();
						//std::cout << "   Time Taken = " << endTime - startTime << std::endl;
						*/

						//Solve for Region based expression Vector
						startTime = glfwGetTime();
						column_vector regionExpressionVector(16);
						//Solve for Tilting and rotation
						find_min_bobyqa(rotationMinFunction, rotationVector, 9, uniform_matrix<double>(4, 1, 0.0), uniform_matrix<double>(4, 1, 1.0), 0.075, 0.01, 30000);
						regionExpressionVector = concatColVec(expressVector, rotationVector);
						regionSolve = getFinalShape(regionExpressionVector);
						//Solve for facial expression
						error = find_min_bobyqa(regionMinFunction, expressVector, 15, uniform_matrix<double>(12, 1, 0.0), uniform_matrix<double>(12, 1, 1.0), 0.075, 0.01, 30000);
						regionExpressionVector = concatColVec(expressVector, rotationVector);

						//Testing
						endTime = glfwGetTime();
						//std::cout << "Region method" << std::endl;
						//std::cout << "Average Error = " << error / exampleShapes[0].num_parts();
						//std::cout << "   Time Taken = " << endTime - startTime << std::endl;
						
						
						regionSolve = getFinalShape(regionExpressionVector);

						//render_face(temp, currentFaceShape, shape2,	shapePCA,0); // Show the Tracked Landmarks, All in One Landmarks and PCA landmarks
						// Show the Tracked Landmarks, All in One Landmarks and Region Landmarks
						//change 0 to 1,or 2 to view the other marked postions
						render_face(temp, currentFaceShape, shape2,regionSolve,0); 
					
						shapes.push_back(currentFaceShape);

						/*
						//Solve for the PCA BlendShapes
						BlendedPosition.clear();
						BlendedPosition = blendshapeSolvePca(pcaBlend, matToColumn(pcaSolve));
						outputModel2.reshapeMesh(BlendedPosition);
						*/

						//Solve for the Region Based Blendshapes
						BlendedPosition.clear();
						BlendedPosition = blendshapeSolve(blend, regionExpressionVector);
						outputModel2.reshapeMesh(BlendedPosition);
					}
					//Apply current frame as a texture for the Rectangle in the first viewport

					cv::flip(temp, temp, 0);

					glBindTexture(GL_TEXTURE_2D, texture1);

					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, temp.cols, temp.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, temp.ptr());        
					glGenerateMipmap(GL_TEXTURE_2D);

					basicShader.Use();
					glActiveTexture(GL_TEXTURE0);
					glUniform1i(glGetUniformLocation(basicShader.Program, "ourTexture1"), 0);
					glBindVertexArray(VAO);
					glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
					glBindVertexArray(0);
					glBindTexture(GL_TEXTURE_2D, 0);
				}

				if (viewp == 1) {

					//Draw the Final Blendshape Sum

					modelShader.Use();
					GLint objectColorLoc = glGetUniformLocation(modelShader.Program, "objectColor");
					GLint lightColorLoc = glGetUniformLocation(modelShader.Program, "lightColor");
					GLint lightPosLoc = glGetUniformLocation(modelShader.Program, "lightPos");
					GLint viewPosLoc = glGetUniformLocation(modelShader.Program, "viewPos");
					glUniform3f(objectColorLoc, 1.0f, 0.6f, 0.31f);
					glUniform3f(lightColorLoc, 1.0f, 1.0f, 1.0f);
					glUniform3f(lightPosLoc, lightPos.x, lightPos.y, lightPos.z);
					glUniform3f(viewPosLoc, center.x, center.y, center.z + 2 * rad);

					float aspect = (float)screenWidth / (float)(2 * screenHeight);
					glm::mat4 view = glm::lookAt(glm::vec3(center.x, center.x, center.x + 2 * rad), center, glm::vec3(0.0f, 1.0f, 0.0f));
					glm::mat4 projection = glm::perspective(fov, aspect, 0.1f, 100.0f);

					glUniformMatrix4fv(glGetUniformLocation(modelShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
					glUniformMatrix4fv(glGetUniformLocation(modelShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));

					glm::mat4 model;
					//model = glm::translate(model, glm::vec3(0.0f, -1.75f, 0.0f)); // Translate it down a bit so it's at the center of the scene
					model = glm::rotate(model, -0.15f, glm::vec3(1.0f, 0.0f, 0.0f));
					model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));	// It's a bit too big for our scene, so scale it down
					glUniformMatrix4fv(glGetUniformLocation(modelShader.Program, "model"), 1, GL_FALSE, glm::value_ptr(model));
					outputModel.Draw(modelShader);

					lightShader.Use();
					glUniformMatrix4fv(glGetUniformLocation(lightShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
					glUniformMatrix4fv(glGetUniformLocation(lightShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));
					model = glm::scale(model, glm::vec3(0.2f));	//small cube light
					glUniformMatrix4fv(glGetUniformLocation(lightShader.Program, "model"), 1, GL_FALSE, glm::value_ptr(model));

					glBindVertexArray(lightVAO);
					glDrawArrays(GL_TRIANGLES, 0, 36);
					glBindVertexArray(0);
				}

				if (viewp == 2) {

					//Draw the Final PCA Blendshape sum

					modelShader.Use();
					GLint objectColorLoc = glGetUniformLocation(modelShader.Program, "objectColor");
					GLint lightColorLoc = glGetUniformLocation(modelShader.Program, "lightColor");
					GLint lightPosLoc = glGetUniformLocation(modelShader.Program, "lightPos");
					GLint viewPosLoc = glGetUniformLocation(modelShader.Program, "viewPos");
					glUniform3f(objectColorLoc, 1.0f, 1.0f, 0.31f);
					glUniform3f(lightColorLoc, 1.0f, 1.0f, 1.0f);
					glUniform3f(lightPosLoc, lightPos.x, lightPos.y, lightPos.z);
					glUniform3f(viewPosLoc, center.x, center.y, center.z + 2 * rad);

					float aspect = (float)screenWidth / (float)(2 * screenHeight);
					glm::mat4 view = glm::lookAt(glm::vec3(center.x, center.x, center.x + 2 * rad), center, glm::vec3(0.0f, 1.0f, 0.0f));
					glm::mat4 projection = glm::perspective(fov, aspect, 0.1f, 100.0f);

					glUniformMatrix4fv(glGetUniformLocation(modelShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
					glUniformMatrix4fv(glGetUniformLocation(modelShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));

					glm::mat4 model;
					model = glm::rotate(model, -0.15f, glm::vec3(1.0f, 0.0f, 0.0f));
					model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));	// It's a bit too big for our scene, so scale it down
					glUniformMatrix4fv(glGetUniformLocation(modelShader.Program, "model"), 1, GL_FALSE, glm::value_ptr(model));
					outputModel2.Draw(modelShader);
			

					lightShader.Use();
					glUniformMatrix4fv(glGetUniformLocation(lightShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
					glUniformMatrix4fv(glGetUniformLocation(lightShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));
					model = glm::scale(model, glm::vec3(0.2f));	//scale the cube light to be vsmall
					glUniformMatrix4fv(glGetUniformLocation(lightShader.Program, "model"), 1, GL_FALSE, glm::value_ptr(model));

					glBindVertexArray(lightVAO);
					glDrawArrays(GL_TRIANGLES, 0, 36);
					glBindVertexArray(0);
				}
			}
			glfwSwapBuffers(window);
		}

		//Clean Up
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);

		glfwTerminate();

		return 0;
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}