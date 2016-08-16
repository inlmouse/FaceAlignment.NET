#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "lbf/lbf.hpp"
#include <msclr/marshal.h>
#include "datastructure.clr.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
//using namespace std;

using namespace System;
using namespace System::Collections;
using namespace System::Collections::Generic;
using namespace System::Globalization;
using namespace System::IO;
using namespace System::Runtime::InteropServices;
using namespace msclr::interop;

namespace FastDetection
{
	public ref class FrameFaceInfo
	{
	private:
		int width;
		int height;
		
		//void DrawShowBmp(System::Drawing::Color facecolor, System::Drawing::Color landmarkcolor);
	public:
		int count;
		array<int, 2>^ facedetails;
		array<array<double, 2>^>^ landmarks;
		array<array<int, 2>^>^ correctedlandmarks;
		//FrameInfo(int Count, array<int, 2>^ Facedetails, array<array<double, 2>^>^ Landmarks) :count(Count), facedetails(Facedetails),landmarks(Landmarks){}
		System::Drawing::Bitmap^ ShowBitmap;
		System::Drawing::Bitmap^ OriBitmp;
		FrameFaceInfo(System::Drawing::Bitmap^ Oribmp);
		void Correction();
	};

	public value struct FaceInfo
	{
		int count;
		array<int, 2>^ facedetails;
		FaceInfo(int Count, array<int, 2>^ Facedetails) :count(Count), facedetails(Facedetails){}
	};

	public ref class FastFace
	{
	private:
		float scale;//1.2f
		int min_neighbors;//3
		int min_object_width;//24
		lbf::LbfCascador *lbf_cascador;
		alinment::Landmarks^ ldmks;
		FaceInfo Facedetect_Frontal(System::String ^path);
		FaceInfo Facedetect_Multiview(System::String ^path);
		FaceInfo Facedetect_Multiview_Reinforce(System::String ^path);
		FaceInfo Facedetect_Frontal_Tmp(System::String ^path);
	public:
		FastFace(float _scale, int _min_neighbors, int _min_object_width, System::String ^modelpath);
		~FastFace();
		FrameFaceInfo^ Detect_Frontal(System::Drawing::Bitmap^ Oribmp);
	};
}