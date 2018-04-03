#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <random>

using namespace cv;
using namespace std;

enum class Label {BRUSH_TEETH, CUT_VEG, JUMP_JACK, LUNGE, WALL_PUSHUP};

struct labelMat
{
	Label label;
	Mat frame;
};

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

vector<labelMat> getLabelledFrames()
{
	vector<labelMat> labelledFrames;

	return labelledFrames;
}

float neuralNetworkClassifier(vector<labelMat> training, vector<labelMat> testing)
{
	float accuracy = 0;
	// Classify here.
	return accuracy;
}

float crossValidation(int k)
{
	vector<labelMat> labelledFrames = getLabelledFrames();

	// Shuffle the labelledFrames.
	random_device rd; // Seed.
	auto rngShuffle = default_random_engine{rd()};
	shuffle(begin(labelledFrames), end(labelledFrames), rngShuffle);
	float accuracy = 0;
	int foldSize = labelledFrames.size() / k;
	vector<labelMat>::const_iterator first = labelledFrames.begin();
	vector<labelMat>::const_iterator last = labelledFrames.end();

	for (int i = 0; i < k; i++)
	{
		// Take all labelled frames to the left and right of one of the folds.
		int endOffset = i < k - 1 ? 1 : 0;
		vector<labelMat> left(first, first + foldSize * i);
		vector<labelMat> right(first + foldSize * (i + 1) + endOffset, last);
		vector<labelMat> trainingFrames;
		trainingFrames.insert(trainingFrames.end(), left.begin(), left.end());
		trainingFrames.insert(trainingFrames.end(), right.begin(), right.end());

		// Take the fold and use it for validation.
		vector<labelMat> validationFrames;
		validationFrames.insert(validationFrames.end(), first + foldSize * i, first + foldSize * (i + 1));

		accuracy += neuralNetworkClassifier(trainingFrames, validationFrames);
	}

	accuracy /= k;

	return accuracy;
}