#pragma once
#include <dlib/optimization.h>
#include <dlib/optimization/find_optimal_parameters.h>
#include <dlib/image_processing/frontal_face_detector.h>

/*
using namespace dlib;
typedef matrix<double, 0, 1> column_vector;

class Function {
	public:
		//Variables
		
		//Functions
		Function(std::vector<full_object_detection> exampleShapes) {
			this->exampleShapes = exampleShapes;
			this->startingPoint = column_vector(exampleShapes.size()-1);
			this->startingPoint = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			max.set_size(startingPoint.size());
			max = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
			min.set_size(startingPoint.size());
			min = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
			setUpBaseShape();
		}
		double minFunction(const column_vector& m) {

			double result;
			
			for (unsigned int i = 0; i < currentFaceShape.num_parts(); i++) {
				point weightedPoint = calcWeightedShape(m,i);
				result =+ (currentFaceShape.part(i)- weightedPoint).length_squared();	
			}
			return result;
		}
		void setCurrentFace(full_object_detection currentFaceShape) {
			this->currentFaceShape = currentFaceShape;
		}
		std::vector<point> getFinalShape() {
			std::vector<point> FinalShape;
			point Point;
			for (unsigned long i = 0; i < baseShape.size(); i++) {
				Point = baseShape[i];
				for (unsigned int j = 1; j < exampleShapes.size(); j++) {
					Point = +startingPoint * exampleShapes[j].part(i);
				}
			}
		}
		void runOptimization() {
		
			//find_min_box_constrained(bfgs_search_strategy(),objective_delta_stop_strategy(1e-9), 
				//this->minFunction, derivative(this->minFunction), this->startingPoint, 0.0, 1.0);
			find_optimal_parameters(1.0, 0.01, 100, startingPoint, min, max, minFunction);
		}
    private:
		//Varaibles
		column_vector startingPoint,min,max;
		std::vector<full_object_detection> exampleShapes;
		full_object_detection currentFaceShape;
		std::vector<point> baseShape;
		//Functions
		point calcWeightedShape(const column_vector& m,int i){
			assert(m.size() == (exampleShapes.size() - 1));
			
			point currentPoint = baseShape[i];

			for (unsigned int j = 0; j < exampleShapes.size()-1; j++) {
				currentPoint += m(j) * exampleShapes[j + 1].part(i);
			}

			return currentPoint;
		}
		void setUpBaseShape() {
			
			for (unsigned int i = 0; i < exampleShapes[0].num_parts(); i++) {
				baseShape.push_back(exampleShapes[0].part(i));
			}
		}
};
*/