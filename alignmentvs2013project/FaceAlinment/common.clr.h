#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include <string>
#include <opencv2/core/core.hpp>
#include "lbf/common.hpp"

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

namespace lbf
{
	public class Aline_Config
	{
	public:
		static inline Aline_Config& GetInstance() {
			static Aline_Config c;
			return c;
		}
	public:
		int stages_n;
		int tree_n;
		int tree_depth;

		std::string dataset;
		std::string saved_file_name;
		int landmark_n;
		int initShape_n;
		std::vector<int> feats_m;
		std::vector<double> radius_m;
		double bagging_overlap;
		std::vector<int> pupils[2];

	private:
		Aline_Config();
		~Aline_Config() {}
		Aline_Config(const Aline_Config &other);
		Aline_Config &operator=(const Aline_Config &other);
	};

	public ref class Aline_BBox{
	public:
		Aline_BBox(){};
		~Aline_BBox(){};
		//BBox(const BBox &other);
		//BBox &operator=(const BBox &other);
		Aline_BBox(double x, double y, double w, double h);

	public:
		cv::Mat Project(const cv::Mat &shape);
		cv::Mat ReProject(const cv::Mat &shape);

	public:
		double x, y;
		double x_center, y_center;
		double x_scale, y_scale;
		double width, height;
	};

	public class Aline_RandomTree{
	public:
		Aline_RandomTree();
		~Aline_RandomTree();
		//RandomTree(const RandomTree &other);
		//RandomTree &operator=(const RandomTree &other);

	public:
		void Init(int landmark_id, int depth);
		void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<Aline_BBox> &bboxes, \
			std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, std::vector<int> &index, int stage);
		void SplitNode(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<Aline_BBox> &bboxes, \
			cv::Mat &delta_shapes, cv::Mat &mean_shape, std::vector<int> &root, int idx, int stage);

		void Read(FILE *fd);
		void Write(FILE *fd);

	public:
		int depth;
		int nodes_n;
		int landmark_id;
		cv::Mat_<double> feats;
		std::vector<int> thresholds;
	};

	public class Aline_RandomForest{
	public:
		Aline_RandomForest();
		~Aline_RandomForest();
		//RandomForest(const RandomForest &other);
		//RandomForest &operator=(const RandomForest &other);

	public:
		void Init(int landmark_n, int trees_n, int tree_depth);
		void Train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes, \
			std::vector<Aline_BBox> &bboxes, std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, int stage);
		cv::Mat GenerateLBF(cv::Mat &img, cv::Mat &current_shape, Aline_BBox bbox, cv::Mat &mean_shape);

		void Read(FILE *fd);
		void Write(FILE *fd);

	public:
		int landmark_n;
		int trees_n, tree_depth;
		std::vector<std::vector<Aline_RandomTree> > random_trees;
	};

	void calcSimilarityTransform(const cv::Mat &shape1, const cv::Mat &shape2, double &scale, cv::Mat &rotate);

	double calcVariance(const cv::Mat &vec);
	double calcVariance(const std::vector<double> &vec);
	double calcMeanError(std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes);

	cv::Mat getMeanShape(std::vector<cv::Mat> &gt_shapes, std::vector<Aline_BBox> &bboxes);
	std::vector<cv::Mat> getDeltaShapes(std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes, \
		std::vector<Aline_BBox> &bboxes, cv::Mat &mean_shape);

	cv::Mat drawShapeInImage(const cv::Mat &img, const cv::Mat &shape, const Aline_BBox bbox);

	void LOG(const char *fmt, ...);
}