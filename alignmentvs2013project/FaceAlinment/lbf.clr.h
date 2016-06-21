#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include <string>
#include <opencv2/core/core.hpp>
#include "lbf/rf.hpp"

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

namespace lbf {

	public class Aline_LbfCascador{
	public:
		Aline_LbfCascador();
		~Aline_LbfCascador();
		//LbfCascador(const LbfCascador &other);
		//LbfCascador &operator=(const LbfCascador &other);

	public:
		void Init(int stages_n);
		void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
			std::vector<cv::Mat> &current_shapes, std::vector<Aline_BBox> &bboxes, \
			cv::Mat &mean_shape, int start_from = 0);
		void Test(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<Aline_BBox> &bboxes);
		void GlobalRegressionTrain(std::vector<cv::Mat> &lbfs, std::vector<cv::Mat> &deltashapes, int stage);
		cv::Mat GlobalRegressionPredict(const cv::Mat &lbf, int stage);
		cv::Mat Predict(cv::Mat &img, Aline_BBox bbox);
		void DumpTrainModel(int stage);
		void ResumeTrainModel(int start_from = 0);

		void Read(FILE *fd);
		void Write(FILE *fd);

	public:
		int stages_n;
		int landmark_n;
		cv::Mat mean_shape;
		std::vector<Aline_RandomForest> random_forests;
		std::vector<cv::Mat> gl_regression_weights;
	};

}