#include "wapper.clr.h"
#include "datastructure.clr.h"
#include "DetectionCore.h"

using namespace cv;

namespace FastDetection
{
	FastFace::FastFace(float _scale, int _min_neighbors, int _min_object_width, System::String ^modelpath)
	{
		lbf_cascador = new lbf::LbfCascador(); 
		scale = _scale;
		min_neighbors = _min_neighbors;
		min_object_width = _min_object_width;
		ldmks =gcnew alinment::Landmarks(modelpath);
	}

	FastFace::~FastFace()
	{
		delete lbf_cascador;
	}

	FaceInfo FastFace::Facedetect_Frontal(System::String ^path)
	{
		
		pin_ptr<const wchar_t> wch = PtrToStringChars(path);
		//printf_s("%S\n", wch);
		size_t convertedChars = 0;
		size_t sizeInBytes = ((path->Length) * 2);
		errno_t err = 0;
		char *ch = (char *)malloc(sizeInBytes);
		err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);

		Mat gray = imread(ch, CV_LOAD_IMAGE_GRAYSCALE);
		if (gray.empty())
		{
			fprintf(stderr, "Can not load the image file.\n");
			array<int, 2>^ output = gcnew array<int, 2>(0, 5);
			return FaceInfo(0, output);
		}

		int * pResults = NULL;

		///////////////////////////////////////////
		// frontal face detection 
		// it's fast, but cannot detect side view faces
		//////////////////////////////////////////
		//!!! The input image must be a gray one (single-channel)
		//!!! DO NOT RELEASE pResults !!!
		
		pResults = facedetect_frontal((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
			scale, min_neighbors, min_object_width);

		//printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));
		array<int, 2>^ output = gcnew array<int, 2>((pResults ? *pResults : 0),5);
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 6 * i;
			output[i,0] = p[0];//x
			output[i,1] = p[1];//y
			output[i,2] = p[2];//w
			output[i,3] = p[3];//h
			output[i,4] = p[4];//neighbors
			//printf("face_rect=[%d, %d, %d, %d], neighbors=%d\n", x, y, w, h, neighbors);
		}
		return FaceInfo((pResults ? *pResults : 0),output);
	}

	FaceInfo FastFace::Facedetect_Multiview(System::String ^path)
	{
		pin_ptr<const wchar_t> wch = PtrToStringChars(path);
		//printf_s("%S\n", wch);
		size_t convertedChars = 0;
		size_t sizeInBytes = ((path->Length) * 2);
		errno_t err = 0;
		char *ch = (char *)malloc(sizeInBytes);
		err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);

		Mat gray = imread(ch, CV_LOAD_IMAGE_GRAYSCALE);
		if (gray.empty())
		{
			fprintf(stderr, "Can not load the image file.\n");
			array<int, 2> ^ output = gcnew array<int, 2>(0,6);
			return FaceInfo(0, output);
		}
		int * pResults = NULL;
		pResults = facedetect_multiview((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
			scale, min_neighbors, min_object_width);
		//printf("%d faces detected.\n", (pResults ? *pResults : 0));
		array<int, 2> ^ output = gcnew array<int, 2>((pResults ? *pResults : 0), 6);
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 6 * i;
			output[i,0] = p[0];//x
			output[i,1] = p[1];//y
			output[i,2] = p[2];//w
			output[i,3] = p[3];//h
			output[i,4] = p[4];//neighbors
			output[i,5] = p[5];//angle
		}
		return FaceInfo((pResults ? *pResults : 0), output);
	}

	FaceInfo FastFace::Facedetect_Multiview_Reinforce(System::String ^path)
	{
		pin_ptr<const wchar_t> wch = PtrToStringChars(path);
		//printf_s("%S\n", wch);
		size_t convertedChars = 0;
		size_t sizeInBytes = ((path->Length) * 2);
		errno_t err = 0;
		char *ch = (char *)malloc(sizeInBytes);
		err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);

		Mat gray = imread(ch, CV_LOAD_IMAGE_GRAYSCALE);
		if (gray.empty())
		{
			fprintf(stderr, "Can not load the image file.\n");
			array<int, 2> ^ output = gcnew array<int, 2>(0,6);
			return FaceInfo(0, output);
		}
		int * pResults = NULL;
		pResults = facedetect_multiview_reinforce((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
			scale, min_neighbors, min_object_width);
		//printf("%d faces detected.\n", (pResults ? *pResults : 0));
		array<int, 2> ^ output = gcnew array<int, 2>((pResults ? *pResults : 0),6);
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 6 * i;
			output[i,0] = p[0];//x
			output[i,1] = p[1];//y
			output[i,2] = p[2];//w
			output[i,3] = p[3];//h
			output[i,4] = p[4];//neighbors
			output[i,5] = p[5];//angle
		}
		return FaceInfo((pResults ? *pResults : 0), output);
	}

	FaceInfo FastFace::Facedetect_Frontal_Tmp(System::String ^path)
	{
		pin_ptr<const wchar_t> wch = PtrToStringChars(path);
		//printf_s("%S\n", wch);
		size_t convertedChars = 0;
		size_t sizeInBytes = ((path->Length) * 2);
		errno_t err = 0;
		char *ch = (char *)malloc(sizeInBytes);
		err = wcstombs_s(&convertedChars, ch, sizeInBytes, wch, sizeInBytes);

		Mat gray = imread(ch, CV_LOAD_IMAGE_GRAYSCALE);
		if (gray.empty())
		{
			fprintf(stderr, "Can not load the image file.\n");
			array<int, 2> ^ output = gcnew array<int, 2>(0,5);
			return FaceInfo(0, output);
		}

		int * pResults = NULL;


		pResults = facedetect_frontal((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
			scale, min_neighbors, min_object_width);

		//printf("%d frontal faces detected.\n", (pResults ? *pResults : 0));
		array<int, 2> ^ output = gcnew array<int, 2>((pResults ? *pResults : 0),5);
		//print the detection results
		for (int i = 0; i < (pResults ? *pResults : 0); i++)
		{
			short * p = ((short*)(pResults + 1)) + 6 * i;
			output[i,0] = p[0];//x
			output[i,1] = p[1];//y
			output[i,2] = p[2];//w
			output[i,3] = p[3];//h
			output[i,4] = p[4];//neighbors
			//printf("face_rect=[%d, %d, %d, %d], neighbors=%d\n", x, y, w, h, neighbors);
		}
		return FaceInfo((pResults ? *pResults : 0), output);
	}

	FrameFaceInfo^ FastFace::Detect_Frontal(System::Drawing::Bitmap^ Oribmp)
	{
		FrameFaceInfo^ info = gcnew FrameFaceInfo(Oribmp);
		Mat img = alinment::ImageConvert::Bitmap2Mat(Oribmp);
		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);
		if (gray.empty())
		{
			fprintf(stderr, "Can not load the image.\n");
			array<int, 2>^ output = gcnew array<int, 2>(0, 5);
			info->count = -1;
			return info;
		}

		int * pResults = NULL;
		
		pResults = facedetect_frontal((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step,
			scale, min_neighbors, min_object_width);
		info->count = (pResults ? *pResults : 0);//face count cout
		info->facedetails = gcnew array<int, 2>(info->count, 5);
		info->landmarks = gcnew array<array<double, 2>^>(info->count);
		info->correctedlandmarks = gcnew array<array<int, 2>^>(info->count);

		for (int i = 0; i < info->count; i++)//face details cout
		{
			short * p = ((short*)(pResults + 1)) + 6 * i;
			info->facedetails[i, 0] = p[0];//x
			info->facedetails[i, 1] = p[1];//y
			info->facedetails[i, 2] = p[2];//w
			info->facedetails[i, 3] = p[3];//h
			info->facedetails[i, 4] = p[4];//neighbors
			///landmarks cout
			Rect r;
			r.x = p[0]; r.y = p[1]; r.width = p[2]; r.height = p[3];//
			//resize and bourder check!!
			r.x -= Convert::ToInt32(r.width*0.1); r.x = r.x < 0 ? 0 : r.x;
			r.y -= Convert::ToInt32(r.height*0.1); r.y = r.y < 0 ? 0 : r.y;
			r.width += Convert::ToInt32(r.width*0.2); r.width = r.width + r.x >= Oribmp->Width ? Oribmp->Width - 1 - r.x : r.width;
			r.height += Convert::ToInt32(r.height*0.2); r.height = r.height + r.y >= Oribmp->Height ? Oribmp->Height - 1 - r.y : r.height;
			//
			Mat tempgray;
			gray(r).copyTo(tempgray);
			//Mat input;
			//cvtColor(input, tempgray, CV_BGR2GRAY);
			lbf::BBox bbox_(r.x, r.y, r.width, r.height);

			info->landmarks[i] = ldmks->Pridect(tempgray, bbox_);
			info->correctedlandmarks[i] = gcnew array<int, 2>(68, 2);
		}
		info->Correction();
		return info;
	}

	FrameFaceInfo::FrameFaceInfo(System::Drawing::Bitmap^ Oribmp)
	{
		width = Oribmp->Width;
		height = Oribmp->Height;
		//OriBitmp = Oribmp;
	}

	void FrameFaceInfo::Correction()
	{
		for (int i = 0; i < count; i++)
		{
			for (int j = 0; j < 68; j++)
			{
				//x÷·Ω√’˝&±ﬂ‘µ“Ï≥£ºÏ≤‚
				int temp = Convert::ToInt32(landmarks[i][j, 0]);// +facedetails[i, 0];
				if (temp<0)
				{
					correctedlandmarks[i][j, 0] = 0;
				}
				else if (temp>=width)
				{
					correctedlandmarks[i][j, 0] = width-1;
				}
				else
				{
					correctedlandmarks[i][j, 0] = temp;
				}
				//y÷·Ω√’˝&±ﬂ‘µ“Ï≥£ºÏ≤‚
				temp = Convert::ToInt32(landmarks[i][j, 1]) + facedetails[i, 1];
				if (temp<0)
				{
					correctedlandmarks[i][j, 1] = 0;
				}
				else if (temp>=height)
				{
					correctedlandmarks[i][j, 1] = height-1;
				}
				else
				{
					correctedlandmarks[i][j, 1] = temp;
				}
			}
		}
	}

	/*void FrameFaceInfo::DrawShowBmp(System::Drawing::Color facecolor, System::Drawing::Color landmarkcolor)
	{
		for (int i = 0; i < count; i++)
		{
			System::Drawing::Rectangle^ rect = gcnew System::Drawing::Rectangle(facedetails[i, 0], facedetails[i, 1], facedetails[i, 2], facedetails[i, 3]);

		}
	}*/
}