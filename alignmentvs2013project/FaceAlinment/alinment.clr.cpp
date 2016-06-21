#include "datastructure.clr.h"
//#include "lbf.clr.h"
//#include "rf.clr.h"

using namespace cv;

namespace alinment
{
	ConfigCLI::ConfigCLI(System::String^ _dataset, System::String^ _modelfile)
	{
		dataset = gcnew System::String(_dataset);
		saved_file_name = gcnew System::String(_modelfile);
		stages_n = 5;
		tree_n = 6;
		tree_depth = 5;
		landmark_n = 68;
		initShape_n = 10;
		bagging_overlap = 0.4;
		pupils = gcnew array<int,2>{ { 36, 37, 38, 39, 40, 41 }, { 42, 43, 44, 45, 46, 47 } };
		feats_m = gcnew array<int>{ 500, 500, 500, 300, 300, 300, 200, 200, 200, 100 };
		radius_m = gcnew array<double>{ 0.3, 0.2, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.05 };
	}

	ABox::ABox(double x, double y, double w, double h)
	{
		this->x = x; this->y = y;
		this->width = w; this->height = h;
		this->x_center = x + w / 2.;
		this->y_center = y + h / 2.;
		this->x_scale = w / 2.;
		this->y_scale = h / 2.;
	}


	cv::Mat ABox::Project(const cv::Mat &shape)
	{
		Mat_<double> res(shape.rows, shape.cols);
		const Mat_<double> &shape_ = (Mat_<double>)shape;
		for (int i = 0; i < shape.rows; i++) {
			res(i, 0) = (shape_(i, 0) - x_center) / x_scale;
			res(i, 1) = (shape_(i, 1) - y_center) / y_scale;
		}
		return res;
	}

	cv::Mat ABox::ReProject(const cv::Mat &shape)
	{
		Mat_<double> res(shape.rows, shape.cols);
		const Mat_<double> &shape_ = (Mat_<double>)shape;
		for (int i = 0; i < shape.rows; i++) {
			res(i, 0) = shape_(i, 0)*x_scale + x_center;
			res(i, 1) = shape_(i, 1)*y_scale + y_center;
		}
		return res;
	}

	array<double, 2>^ LbfCascade::Predict(cv::Mat &img, ABox^ abox)
	{
		lbf::BBox temp(abox->x,abox->y,abox->width,abox->height);
		//lbf::LbfCascador _aliemnt=aliemnt;
		array<double, 2>^ output = gcnew array<double, 2>(68, 2);
		std::vector<std::vector<double>> landmarks= aliemnt.Predict(img, temp, true);
		for (int i = 0; i < 68; i++)
		{
			output[i, 0] = landmarks[i][0];
			output[i, 1] = landmarks[i][1];
		}
		return output;
	}

	array<double, 2>^ LbfCascade::Predict(System::String ^imgpath, ABox^ abox)
	{
		pin_ptr<const wchar_t> wch = PtrToStringChars(imgpath);
		//printf_s("%S\n", wch);
		size_t convertedChars = 0;
		size_t sizeInBytes = ((imgpath->Length) * 2);
		errno_t err = 0;
		char *ch = (char *)malloc(sizeInBytes);
		err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);

		lbf::BBox temp(abox->x, abox->y, abox->width, abox->height);
		//lbf::LbfCascador _aliemnt = aliemnt;
		array<double, 2>^ output = gcnew array<double, 2>(68, 2);
		Mat img = cv::imread(ch);
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);
		std::vector<std::vector<double>> landmarks = aliemnt.Predict(gray, temp, true);
		for (int i = 0; i < 68; i++)
		{
			output[i, 0] = landmarks[i][0];
			output[i, 1] = landmarks[i][1];
		}
		return output;
	}

	LbfCascade::LbfCascade(System::String ^path)
	{
		//lbf::LbfCascador _aliemnt = lbf::LbfCascador();
		//System::String ^path = x;
		pin_ptr<const wchar_t> wch = PtrToStringChars(path);
		//printf_s("%S\n", wch);
		size_t convertedChars = 0;
		size_t sizeInBytes = ((path->Length) * 2);
		errno_t err = 0;
		char *ch = (char *)malloc(sizeInBytes);
		err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);
		//printf_s("%s\n", ch);
		//convert System::String to char*
		
		/*lbf::LbfCascador &lbf_cascador = lbf::LbfCascador();
		FILE *model = fopen(ch, "rb");
		lbf_cascador.Read(model);
		fclose(model);
		aliemnt = lbf_cascador;*/
		FILE *model = fopen(ch, "rb");
		aliemnt.Read(model);
		fclose(model);

	}
	
}
