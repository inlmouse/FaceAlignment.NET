#pragma once

#include <msclr\marshal_cppstd.h>
#include "lbf/lbf.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect_c.h>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

using namespace FastDetection;

namespace FacePipeline
{
	public value struct FrameInfo
	{
		int count;
		array<array<int, 2>^>^ landmarks;
		List<System::Drawing::Rectangle> ^r;
		FrameInfo(int Count) :count(Count){}
		FrameInfo(int Count, List<System::Drawing::Rectangle>^ R, array<array<int, 2>^>^ Landmarks) :count(Count), r(R), landmarks(Landmarks){}
	};

	public ref class Pipeline
	{
	public:
		Pipeline(System::String^ modelFile);
		~Pipeline();

		FrameInfo GetFrameInfo(System::Drawing::Bitmap^ bmpImg);

	private:
		lbf::LbfCascador *Cascador;
		
		FastFace^ fastdetection;
		int ConvertBitmapToMat(System::Drawing::Bitmap^ bmpImg, cv::Mat& cvImg);
		Rect enlarge(Rect r, double s);
	};

}