#include "datastructure.clr.h"
#include <iostream>
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

	//array<double, 2>^ LbfCascade::Predict(cv::Mat &img, ABox^ abox)
	//{
	//	lbf::BBox temp(abox->x,abox->y,abox->width,abox->height);
	//	//lbf::LbfCascador _aliemnt=aliemnt;
	//	array<double, 2>^ output = gcnew array<double, 2>(68, 2);
	//	std::vector<std::vector<double>> landmarks= aliemnt.Predict(img, temp, true);
	//	for (int i = 0; i < 68; i++)
	//	{
	//		output[i, 0] = landmarks[i][0];
	//		output[i, 1] = landmarks[i][1];
	//	}
	//	return output;
	//}

	//array<double, 2>^ LbfCascade::Predict(System::String ^imgpath, ABox^ abox)
	//{
	//	pin_ptr<const wchar_t> wch = PtrToStringChars(imgpath);
	//	//printf_s("%S\n", wch);
	//	size_t convertedChars = 0;
	//	size_t sizeInBytes = ((imgpath->Length) * 2);
	//	errno_t err = 0;
	//	char *ch = (char *)malloc(sizeInBytes);
	//	err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);

	//	lbf::BBox temp(abox->x, abox->y, abox->width, abox->height);
	//	//lbf::LbfCascador _aliemnt = aliemnt;
	//	array<double, 2>^ output = gcnew array<double, 2>(68, 2);
	//	Mat img = cv::imread(ch); 
	//	Mat gray;
	//	cvtColor(img, gray, CV_BGR2GRAY);
	//	std::vector<std::vector<double>> landmarks = aliemnt.Predict(gray, temp, true);
	//	for (int i = 0; i < 68; i++)
	//	{
	//		output[i, 0] = landmarks[i][0];
	//		output[i, 1] = landmarks[i][1];
	//	}
	//	return output;
	//}

	//array<double, 2>^ LbfCascade::Predict(System::Drawing::Bitmap^ bitmap, ABox^ abox)
	//{
	//	lbf::BBox temp(abox->x, abox->y, abox->width, abox->height);
	//	array<double, 2>^ output = gcnew array<double, 2>(68, 2);
	//	Mat img = ImageConvert::Bitmap2Mat(bitmap);
	//	Mat gray;
	//	cvtColor(img, gray, CV_BGR2GRAY);
	//	std::vector<std::vector<double>> landmarks = aliemnt.Predict(gray, temp, true);
	//	for (int i = 0; i < 68; i++)
	//	{
	//		output[i, 0] = landmarks[i][0];
	//		output[i, 1] = landmarks[i][1];
	//	}
	//	return output;
	//}

	//LbfCascade::LbfCascade(System::String ^path)
	//{
	//	//lbf::LbfCascador _aliemnt = lbf::LbfCascador();
	//	//System::String ^path = x;
	//	pin_ptr<const wchar_t> wch = PtrToStringChars(path);
	//	//printf_s("%S\n", wch);
	//	size_t convertedChars = 0;
	//	size_t sizeInBytes = ((path->Length) * 2);
	//	errno_t err = 0;
	//	char *ch = (char *)malloc(sizeInBytes);
	//	err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);
	//	//printf_s("%s\n", ch);
	//	//convert System::String to char*
	//	
	//	/*lbf::LbfCascador &lbf_cascador = lbf::LbfCascador();
	//	FILE *model = fopen(ch, "rb");
	//	lbf_cascador.Read(model);
	//	fclose(model);
	//	aliemnt = lbf_cascador;*/
	//	FILE *model = fopen(ch, "rb");
	//	aliemnt.Read(model);
	//	fclose(model);

	//}
	
	System::Drawing::Bitmap^ ImageConvert::Mat2Bitmap(cv::Mat& img)
	{
	//	int stride = img.size().width * img.channels();//calc the srtide
	//	int hDataCount = img.size().height;

	//	System::Drawing::Bitmap^ retImg;

	//	System::IntPtr ptr(img.data);

	//	//create a pointer with Stride
	//	if (stride % 4 != 0){//is not stride a multiple of 4?
	//		//make it a multiple of 4 by fiiling an offset to the end of each row

	//		//to hold processed data
	//		uchar *dataPro = new uchar[((img.size().width * img.channels() + 3) & -4) * hDataCount];

	//		uchar *data = img.ptr();

	//		//current position on the data array
	//		int curPosition = 0;
	//		//current offset
	//		int curOffset = 0;

	//		int offsetCounter = 0;

	//		//itterate through all the bytes on the structure
	//		for (int r = 0; r < hDataCount; r++){
	//			//fill the data
	//			for (int c = 0; c < stride; c++){
	//				curPosition = (r * stride) + c;

	//				dataPro[curPosition + curOffset] = data[curPosition];
	//			}

	//			//reset offset counter
	//			offsetCounter = stride;

	//			//fill the offset
	//			do{
	//				curOffset += 1;
	//				dataPro[curPosition + curOffset] = 0;

	//				offsetCounter += 1;
	//			} while (offsetCounter % 4 != 0);
	//		}

	//		ptr = (System::IntPtr)dataPro;//set the data pointer to new/modified data array

	//		//calc the stride to nearest number which is a multiply of 4
	//		stride = (img.size().width * img.channels() + 3) & -4;

	//		retImg = gcnew System::Drawing::Bitmap(img.size().width, img.size().height,
	//			stride,
	//			System::Drawing::Imaging::PixelFormat::Format32bppArgb,
	//			ptr);
	//	}
	//	else{

	//		//no need to add a padding or recalculate the stride
	//		retImg = gcnew System::Drawing::Bitmap(img.size().width, img.size().height,
	//			stride,
	//			System::Drawing::Imaging::PixelFormat::Format32bppArgb,
	//			ptr);
	//	}

	//	array<byte>^ imageData;
	//	System::Drawing::Bitmap^ output;

	//	// Create the byte array.
	//	{
	//		System::IO::MemoryStream^ ms = gcnew System::IO::MemoryStream();
	//		retImg->Save(ms, System::Drawing::Imaging::ImageFormat::Jpeg);
	//		imageData = ms->ToArray();
	//		delete ms;
	//	}

	//	// Convert back to bitmap
	//{
	//	System::IO::MemoryStream^ ms = gcnew System::IO::MemoryStream(imageData);
	//	output = (System::Drawing::Bitmap^)System::Drawing::Bitmap::FromStream(ms);
	//}

	//return output;
		if (img.channels()==4)
		{
			uchar *c = new uchar[img.step*img.rows];
			System::IntPtr ip(c);
			memcpy(ip.ToPointer(), img.data, img.step*img.rows);

			System::Drawing::Bitmap^ bmp = gcnew System::Drawing::Bitmap(
				img.cols, img.rows, img.step,
				System::Drawing::Imaging::PixelFormat::Format32bppArgb,
				ip);
			// delete¤·¤Ê¤¯¤Æ¤¤¤¤£¿
			//delete c;
			return bmp;
		}
		else if (img.channels() == 1)
		{
			uchar *c = new uchar[img.step*img.rows];
			System::IntPtr ip(c);
			memcpy(ip.ToPointer(), img.data, img.step*img.rows);

			System::Drawing::Bitmap^ bmp = gcnew System::Drawing::Bitmap(
				img.cols, img.rows, img.step,
				System::Drawing::Imaging::PixelFormat::Format8bppIndexed,ip);
			// delete¤·¤Ê¤¯¤Æ¤¤¤¤£¿
			//delete c;
			return bmp;
		}
		else if (img.channels() == 3)
		{
			uchar *c = new uchar[img.step*img.rows];
			System::IntPtr ip(c);
			memcpy(ip.ToPointer(), img.data, img.step*img.rows);

			System::Drawing::Bitmap^ bmp = gcnew System::Drawing::Bitmap(
				img.cols, img.rows, img.step,
				System::Drawing::Imaging::PixelFormat::Format24bppRgb, ip);
			// delete¤·¤Ê¤¯¤Æ¤¤¤¤£¿
			//delete c;
			return bmp;
		}
		else
		{
			std::cout << "!!!!!" <<std::endl;
		}
	}

	cv::Mat ImageConvert::Bitmap2Mat(System::Drawing::Bitmap^ bitmap)
	{
		IplImage* tmp;

		System::Drawing::Imaging::BitmapData^ bmData = bitmap->LockBits(System::Drawing::Rectangle(0, 0, bitmap->Width, bitmap->Height), System::Drawing::Imaging::ImageLockMode::ReadWrite, bitmap->PixelFormat);
		if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format8bppIndexed)
		{
			tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 1);
			tmp->imageData = (char*)bmData->Scan0.ToPointer();
		}

		else if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format24bppRgb)
		{
			tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 3);
			tmp->imageData = (char*)bmData->Scan0.ToPointer();
		}

		else if (bitmap->PixelFormat == System::Drawing::Imaging::PixelFormat::Format32bppArgb)
		{
			tmp = cvCreateImage(cvSize(bitmap->Width, bitmap->Height), IPL_DEPTH_8U, 4);
			tmp->imageData = (char*)bmData->Scan0.ToPointer();
		}
		
		bitmap->UnlockBits(bmData);

		return cv::cvarrToMat(tmp,true);
	}

	cv::Mat ImageConvert::Double2Mat(array<array<double>^>^ DoubleArray)
	{
		Mat H(DoubleArray->Length, DoubleArray[0]->Length, CV_64F);
		for (int i = 0; i < DoubleArray->Length; i++)
		{
			for (int j = 0; j < DoubleArray[0]->Length; j++)
			{
				H.at<double>(i, j) = DoubleArray[i][j];
			}
		}
		return H;
	}

	array<array<double>^>^ ImageConvert::Mat2Double(Mat mat)
	{
		array<array<double>^>^ out = gcnew array<array<double>^> (mat.rows);
		for (int i = 0; i < mat.rows; i++)
		{
			out[i] = gcnew array<double>(mat.cols);
			for (int j = 0; j < mat.cols; j++)
				out[i][j] = mat.at<double>(i, j);
		}
		return out;
	}
}
