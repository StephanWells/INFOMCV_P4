#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	VideoCapture vid("data/ucf-101/BrushingTeeth/v_BrushingTeeth_g01_c01.avi");

	if (!vid.isOpened()) cout << "Cannot open video file" << endl;

	double count = vid.get(CV_CAP_PROP_FRAME_COUNT);
	vid.set(CV_CAP_PROP_POS_FRAMES, count / 2);

	Mat frame;
	vid.read(frame);

	namedWindow("test", CV_WINDOW_AUTOSIZE);
	imshow("test", frame);
	waitKey(0);

	return 0;
}