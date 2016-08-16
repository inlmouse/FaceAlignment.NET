#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <msclr/marshal.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect_c.h>
//#include <opencv2/objdetect.hpp>
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

	public value struct randomtree
	{
		int depth;
		int nodes_n;
		int landmark_id;
		array<array<double>^>^ feats;
		array<int>^ thresholds;
	};

	public value struct randomforest
	{
		int landmark_n;
		int trees_n, tree_depth;
		array<array<randomtree>^>^ random_trees;
	};

	public value struct LbfCascade {
	

	public:
		int stages_n;
		int landmark_n;
		//lbf::LbfCascador &aliemnt = lbf::LbfCascador();
		array<array<double>^>^ mean_shape;
		array<randomforest>^ random_forests;
		array<array<array<double>^>^>^ gl_regression_weights;
		
	};

	

	public ref class ImageConvert
	{
	public:
		static System::Drawing::Bitmap^ Mat2Bitmap(cv::Mat& img);

		static cv::Mat Bitmap2Mat(System::Drawing::Bitmap^ bitmap);

		static cv::Mat Double2Mat(array<array<double>^>^ bitmap);

		static array<array<double>^>^ Mat2Double(Mat mat);
	};

	public ref class InterReaction 
	{
	public:

		static lbf::RandomTree Managed2Native(randomtree^ rt)
		{
			lbf::RandomTree RT = lbf::RandomTree();
			RT.depth = rt->depth;
			RT.nodes_n = rt->nodes_n;
			RT.landmark_id = rt->landmark_id;
			int l = rt->thresholds->Length;
			RT.thresholds = std::vector<int>(l);
			for (int i = 0; i < l; i++)
			{
				RT.thresholds[i] = rt->thresholds[i];
			}
			RT.feats = (cv::Mat_<double>)ImageConvert::Double2Mat(rt->feats);
			return RT;
		}

		static lbf::RandomForest Managed2Native(randomforest^ rf)
		{
			lbf::RandomForest RF = lbf::RandomForest();
			RF.landmark_n = rf->landmark_n;
			RF.trees_n = rf->trees_n;
			RF.tree_depth = rf->tree_depth;
			int l = rf->random_trees->Length;
			RF.random_trees = std::vector<std::vector<lbf::RandomTree>>(l);
			for (int i = 0; i < l; i++)
			{
				int l2 = rf->random_trees[i]->Length;
				RF.random_trees[i] = std::vector<lbf::RandomTree>(l2);
				for (int j = 0; j < l2; j++)
				{
					RF.random_trees[i][j] = Managed2Native(rf->random_trees[i][j]);
				}
			}
			return RF;
		}

		static lbf::LbfCascador Managed2Native(LbfCascade^ cas)
		{
			lbf::LbfCascador CAS = lbf::LbfCascador();
			CAS.stages_n = cas->stages_n;
			CAS.landmark_n = cas->landmark_n;
			CAS.mean_shape = ImageConvert::Double2Mat(cas->mean_shape);
			int l = cas->random_forests->Length;
			CAS.random_forests = std::vector<lbf::RandomForest>(l);
			for (int i = 0; i < l; i++)
			{
				CAS.random_forests[i] = Managed2Native(cas->random_forests[i]);
			}
			int g1 = cas->gl_regression_weights->Length;
			CAS.gl_regression_weights = std::vector<cv::Mat>(g1);
			for (int i = 0; i < g1; i++)
			{
				CAS.gl_regression_weights[i] = ImageConvert::Double2Mat(cas->gl_regression_weights[i]);
			}
			return CAS;
		}

		static randomtree  Native2Managed(lbf::RandomTree RT)
		{
			randomtree^ rt = gcnew randomtree();
			rt->depth = RT.depth;
			rt->landmark_id = RT.landmark_id;
			rt->nodes_n = RT.nodes_n;
			int l = RT.thresholds.size();
			rt->thresholds = gcnew array<int>(l);
			for (int i = 0; i < l; i++)
			{
				rt->thresholds[i] = RT.thresholds[i];
			}
			rt->feats = ImageConvert::Mat2Double((Mat)RT.feats);
			return *rt;
		}

		static randomforest  Native2Managed(lbf::RandomForest RF)
		{
			randomforest^ rf = gcnew randomforest();
			rf->landmark_n = RF.landmark_n;
			rf->trees_n = RF.trees_n;
			rf->tree_depth = RF.tree_depth;
			int l1 = RF.random_trees.size();
			rf->random_trees = gcnew array<array<randomtree>^>(l1);
			for (int i = 0; i < l1; i++)
			{
				int l2 = RF.random_trees[i].size();
				rf->random_trees[i] = gcnew array<randomtree>(l2);
				for (int j = 0; j < l2; j++)
				{
					rf->random_trees[i][j] = Native2Managed(RF.random_trees[i][j]);
				}
			}
			return *rf;
		}

		static LbfCascade  Native2Managed(lbf::LbfCascador *CAS)
		{
			LbfCascade^ cas = gcnew LbfCascade();
			cas->stages_n = CAS->stages_n;
			cas->landmark_n = CAS->landmark_n;
			cas->mean_shape = ImageConvert::Mat2Double(CAS->mean_shape);
			int l = CAS->random_forests.size();
			cas->random_forests = gcnew array<randomforest>(l);
			for (int i = 0; i < l; i++)
			{
				cas->random_forests[i] = Native2Managed(CAS->random_forests[i]);
			}
			int m = CAS->gl_regression_weights.size();
			cas->gl_regression_weights = gcnew array<array<array<double>^>^>(m);
			for (int i = 0; i < m; i++)
			{
				cas->gl_regression_weights[i] = ImageConvert::Mat2Double(CAS->gl_regression_weights[i]);
			}
			return *cas;
		}

	};

	public ref class Landmarks
	{
	private:
		LbfCascade landmarkcascade;
		
	public:
		
		Landmarks(System::String ^path)
		{
			lbf::LbfCascador &CAS = lbf::LbfCascador();
			pin_ptr<const wchar_t> wch = PtrToStringChars(path);
			//printf_s("%S\n", wch);
			size_t convertedChars = 0;
			size_t sizeInBytes = ((path->Length) * 2);
			errno_t err = 0;
			char *ch = (char *)malloc(sizeInBytes);
			err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);
			FILE *model = fopen(ch, "rb");
			CAS.Read(model);
			fclose(model);
			landmarkcascade = InterReaction::Native2Managed(&CAS);
			//free(&CAS);
		}

		array<double, 2>^ Pridect(Mat mat, lbf::BBox &bbox)
		{
			lbf::LbfCascador CAS = InterReaction::Managed2Native(landmarkcascade);//lbf::LbfCascador
			//Mat inmat = ImageConvert::Bitmap2Mat(bitmap);
			//ABox^ abox
			//lbf::BBox bbox = lbf::BBox(abox->x, abox->y, abox->width, abox->height); 
			//lbf::BBox bbox = lbf::BBox(0, 0, 256, 256);
			std::vector<std::vector<double>>x = CAS.Predict(mat, bbox, true);
			array<double, 2>^ output = gcnew array<double, 2>(68, 2);
			for (int i = 0; i < 68; i++)
			{
				output[i, 0] = x[i][0];
				output[i, 1] = x[i][1];
			}
			free(&CAS);
			return output;
		}
	};
}