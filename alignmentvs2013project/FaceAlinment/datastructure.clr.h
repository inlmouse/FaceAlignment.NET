#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect_c.h>
#include <opencv2/objdetect.hpp>
#include <lbf/lbf.hpp>
#include <lbf/rf.hpp>
#include <lbf/common.hpp>

using namespace cv;
//using namespace std;

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

//CascadeClassifier faceDec("C:\\Research\\face-alignment\\model\\haarcascade_frontalface_alt.xml");

namespace alinment
{
	public ref class ConfigCLI {
	//public:
	//	static inline Config& GetInstance() {
	//		static Config c;
	//		return c;
	//	}

	public:
		int stages_n;
		int tree_n;
		int tree_depth;

		System::String^ dataset;
		System::String^ saved_file_name;
		int landmark_n;
		int initShape_n;
		array<int>^ feats_m;
		array<double>^ radius_m;
		double bagging_overlap;
		array<int,2>^ pupils;

	public:
		//ConfigCLI();
		ConfigCLI(System::String^ _dataset, System::String^ _modelfile);
		//~ConfigCLI();
	//private:
	//	Config();
	//	~Config() {}
	//	Config(const Config &other);
	//	Config &operator=(const Config &other);
	};

	public ref class ABox {
	public:
		//BBox();
		//~BBox();
		//BBox(const BBox &other);
		//BBox &operator=(const BBox &other);
		ABox(double x, double y, double w, double h);

	public:
		cv::Mat Project(const cv::Mat &shape);
		cv::Mat ReProject(const cv::Mat &shape);
		

	public:
		double x, y;
		double x_center, y_center;
		double x_scale, y_scale;
		double width, height;
	};

	public ref class LbfCascade {
	public:
		LbfCascade(System::String ^path);
		/*~LbfCascador();*/
		//LbfCascador(const LbfCascador &other);
		//LbfCascador &operator=(const LbfCascador &other);

	public:
		/*void Init(int stages_n);
		void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
			std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
			cv::Mat &mean_shape, int start_from = 0);
		void Test(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<BBox> &bboxes);
		void GlobalRegressionTrain(std::vector<cv::Mat> &lbfs, std::vector<cv::Mat> &deltashapes, int stage);
		cv::Mat GlobalRegressionPredict(const cv::Mat &lbf, int stage);*/
		array<double, 2>^ Predict(cv::Mat &img, ABox^ bbox);
		array<double, 2>^ Predict(System::String ^imgpath, ABox^ bbox);
		/*void DumpTrainModel(int stage);
		void ResumeTrainModel(int start_from = 0);*/

		//void Read(FILE *fd);
		//void Write(FILE *fd);

	public:
		int stages_n;
		int landmark_n;
		lbf::LbfCascador &aliemnt = lbf::LbfCascador();
		/*cv::Mat mean_shape;
		std::vector<RandomForest> random_forests;
		std::vector<cv::Mat> gl_regression_weights;*/
	};
}