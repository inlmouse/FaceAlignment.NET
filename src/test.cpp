#include "lbf/lbf.hpp"

#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect_c.h>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace lbf;

CascadeClassifier faceDec("C:\\Research\\face-alignment\\model\\haarcascade_frontalface_alt.xml");

// dirty but works
void parseTxt(string &txt, vector<Mat> &imgs, vector<Mat> &gt_shapes, vector<BBox> &bboxes);

Rect enlarge(Rect r, double s);

int test(void) {
    Config &config = Config::GetInstance();

    LbfCascador lbf_cascador;
    FILE *fd = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(fd);
    fclose(fd);

    LOG("Load test data from %s", config.dataset.c_str());
    string txt = config.dataset + "/test.txt";
    vector<Mat> imgs, gt_shapes;
    vector<BBox> bboxes;
    parseTxt(txt, imgs, gt_shapes, bboxes);

    int N = imgs.size();
    lbf_cascador.Test(imgs, gt_shapes, bboxes);

    return 0;
}

int run(void) {
    Config &config = Config::GetInstance();
    FILE *fd = fopen((config.dataset + "/test.txt").c_str(), "r");
    assert(fd);
    int N;
    int landmark_n = config.landmark_n;
    fscanf(fd, "%d", &N);
    char img_path[256];
    double bbox[4];
    vector<double> x(landmark_n), y(landmark_n);

    LbfCascador lbf_cascador;
    FILE *model = fopen(config.saved_file_name.c_str(), "rb");
    lbf_cascador.Read(model);
    fclose(model);

    for (int i = 0; i < N; i++) {
        fscanf(fd, "%s", img_path);
        for (int j = 0; j < 4; j++) {
            fscanf(fd, "%lf", &bbox[j]);
        }
        for (int j = 0; j < landmark_n; j++) {
            fscanf(fd, "%lf%lf", &x[j], &y[j]);
        }
        Mat img = imread(img_path);
        // crop img
        double x_min, y_min, x_max, y_max;
        x_min = *min_element(x.begin(), x.end());
        x_max = *max_element(x.begin(), x.end());
        y_min = *min_element(y.begin(), y.end());
        y_max = *max_element(y.begin(), y.end());
        x_min = max(0., x_min - bbox[2] / 2);
        x_max = min(img.cols - 1., x_max + bbox[2] / 2);
        y_min = max(0., y_min - bbox[3] / 2);
        y_max = min(img.rows - 1., y_max + bbox[3] / 2);
        double x_, y_, w_, h_;
        x_ = x_min; y_ = y_min;
        w_ = x_max - x_min; h_ = y_max - y_min;
        BBox bbox_(bbox[0] - x_, bbox[1] - y_, bbox[2], bbox[3]);
        Rect roi(x_, y_, w_, h_);
        img = img(roi).clone();

        Mat gray;
        cvtColor(img, gray, CV_BGR2GRAY);
        LOG("Run %s", img_path);
        Mat shape = lbf_cascador.Predict(gray, bbox_);
        img = drawShapeInImage(img, shape, bbox_);
        imshow("landmark", img);
        waitKey(0);
    }
    fclose(fd);
    return 0;
}

/* used to detect a single image */
int detect(char * filename){
	// get the configure file
	Config &config = Config::GetInstance();
	int landmark_n = config.landmark_n;
	double bbox[4]; // the bounding box
	vector<double> x(landmark_n), y(landmark_n); //used to strore the shape
	// regressors
	LbfCascador lbf_cascador;
	// load the trained model
	TIMER_BEGIN
	FILE *model = fopen(config.saved_file_name.c_str(), "rb");
	lbf_cascador.Read(model);
	fclose(model);
	cout <<"load time: "<< TIMER_NOW << endl;
	TIMER_END
	// read the image based on filename
	assert(filename != NULL);
	Mat img = imread(filename);
	// detect the face and get the bounding box
	// based on CascadedClassifier provided
	// by OpenCV
	vector<Rect> rects;
	TIMER_BEGIN
	faceDec.detectMultiScale(img, rects, 1.05, 3, CV_HAAR_SCALE_IMAGE, Size(30, 30));
	printf("Detection Down, cost %.4lf s\n", TIMER_NOW);
	TIMER_END
	if (rects.size() == 0) return -1;

	for (int i = 0; i < rects.size(); i++){
		Rect r = rects[i];
		printf("the bounding box: %d %d %d %d\n", r.x, r.y, r.width, r.height);
		//show the bounding box:
		rectangle(img, r, Scalar(255, 255, 0));
		/*imshow("bounding box", img);
		waitKey(0);*/

		// enlarge the bounding box
		Rect rec = enlarge(r, 1.2);
		// double check to guarentee that
		// the bounding box is contained by the img.
		rec.x = max(0, rec.x);
		rec.y = max(0, rec.y);
		rec.width = min(img.cols - rec.x, rec.width);
		rec.height = min(img.rows - rec.y, rec.height);

		Mat img_t = img(rec).clone();
		BBox bbox_(abs(rec.x - r.x), abs(rec.y - r.y), r.width, r.height);

		Mat gray;
		cvtColor(img_t, gray, CV_BGR2GRAY);
		LOG("Detection--- %s", filename);
		TIMER_BEGIN
		vector<vector<double>>x=lbf_cascador.Predict(gray, bbox_, true);//
		Mat shape = lbf_cascador.Predict(gray, bbox_);
		printf("Detection Down, cost %.4lf s\n", TIMER_NOW);
		//cout <<"Lamdmark[55][0]: "<< x[55][0] << endl;
		img_t = drawShapeInImage(img_t, shape, bbox_);
		TIMER_END
		
		imshow("landmark", img_t);
		waitKey(0);
	}
	return 0;
}


/*
* enlarge the bounding box by scaler s
*/
Rect enlarge(Rect r, double s){
	Rect rec;
	assert(s > 1.0);
	rec.width = r.width * s;
	rec.height = r.height * s;
	rec.x = r.x - (rec.width - r.width) / 2;
	rec.y = r.y - (rec.height - r.height) / 2;

	return rec;
}