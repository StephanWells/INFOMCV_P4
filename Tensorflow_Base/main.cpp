#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <map>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <Windows.h>
#include <iomanip>
#include <sstream>

#define ACTION_NUM 5

using namespace cv;
using namespace std;

enum Action { BrushingTeeth, CuttingInKitchen, JumpingJack, Lunges, WallPushups };
map<Action, String>  ActionMap;

struct LabelMat
{
	Action label;
	Mat mat;
};

string floatToString(float input)
{
	stringstream stream;
	stream << fixed << setprecision(2) << input;

	return stream.str();
}

Mat getFrameMat(String path)
{
	VideoCapture vid(path);

	if (!vid.isOpened()) { cout << "Cannot open video file." << endl; return Mat(); }

	double count = vid.get(CV_CAP_PROP_FRAME_COUNT);
	vid.set(CV_CAP_PROP_POS_FRAMES, count / 2);

	Mat frame;
	vid.read(frame);

	namedWindow("test", CV_WINDOW_AUTOSIZE);
	imshow("test", frame);
	waitKey(0);

	return frame;
}

vector<LabelMat> getFrameMats()
{
	vector<LabelMat> frames;

	String datapath = "data/ucf-101/";

	HANDLE dir;
	WIN32_FIND_DATA file_data;

	map<Action, String>::iterator it;

	for (it = ActionMap.begin(); it != ActionMap.end(); it++)
	{
		if ((dir = FindFirstFile((datapath + it->second + "/*").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
			return frames; /* No files found */

		do
		{
			const string file_name = file_data.cFileName;
			const string full_file_name = datapath + it->second + "/" + file_name;
			const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

			if (file_name[0] == '.')
				continue;

			if (is_directory)
				continue;
			
			LabelMat lm;
			lm.label = it->first;

			Size size(90, 90);
			Mat frame = getFrameMat(full_file_name);
			resize(frame, lm.mat, size);

			frames.push_back(lm);
		} while (FindNextFile(dir, &file_data));

		FindClose(dir);
	}	

	return frames;
}

float neuralNetworkClassifier(vector<LabelMat> training, vector<LabelMat> testing)
{
	float accuracy = 0;
	// Classify here.
	return accuracy;
}

void normaliseConfusionMatrix(Mat &confMat)
{
	for (int i = 0; i < confMat.rows; i++)
	{
		float tempTotal = 0;

		for (int j = 0; j < confMat.cols; j++)
		{
			tempTotal += confMat.at<float>(i, j);
		}

		for (int j = 0; j < confMat.cols; j++)
		{
			confMat.at<float>(i, j) /= tempTotal;
		}
	}
}

void showConfusionMatrix(Mat confMat) // confMat should be of type CV_32FC1 with each value being the number of times that the actual label (row) was classified as a predicted label (column).
{
	if (confMat.rows != confMat.cols)
	{
		cout << "ERROR: Confusion matrix is not a square matrix" << endl;

		return;
	}

	normaliseConfusionMatrix(confMat);

	Size tileSize(100, 100);
	Mat confusionOutput(tileSize * confMat.rows, CV_8UC3);

	for (int i = 0; i < confMat.rows; i++)
	{
		for (int j = 0; j < confMat.cols; j++)
		{
			float confVal = confMat.at<float>(i, j);
			Vec3b colour(0, 0, 0);
			colour[0] =  confVal * 255;

			for (int k = 0; k < tileSize.width; k++)
			{
				for (int l = 0; l < tileSize.height; l++)
				{
					confusionOutput.at<Vec3b>(i * tileSize.width + k, j * tileSize.height + l) = colour;
				}
			}

			putText(confusionOutput, floatToString(confVal), Point2i(j * tileSize.width + (int)(tileSize.width / 3), i * tileSize.height + (int)(tileSize.height / 2)), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
		}
	}

	imshow("Confusion Matrix", confusionOutput);
	waitKey(0);
}

float crossValidation(int k)
{
	vector<LabelMat> labelledFrames = getFrameMats();

	// Shuffle the labelledFrames.
	random_device rd; // Seed.
	auto rngShuffle = default_random_engine{ rd() };
	shuffle(begin(labelledFrames), end(labelledFrames), rngShuffle);
	float accuracy = 0;
	int foldSize = labelledFrames.size() / k;
	vector<LabelMat>::const_iterator first = labelledFrames.begin();
	vector<LabelMat>::const_iterator last = labelledFrames.end();

	for (int i = 0; i < k; i++)
	{
		// Take all labelled frames to the left and right of one of the folds.
		int endOffset = i < k - 1 ? 1 : 0;
		vector<LabelMat> left(first, first + foldSize * i);
		vector<LabelMat> right(first + foldSize * (i + 1) + endOffset, last);
		vector<LabelMat> trainingFrames;
		trainingFrames.insert(trainingFrames.end(), left.begin(), left.end());
		trainingFrames.insert(trainingFrames.end(), right.begin(), right.end());

		// Take the fold and use it for validation.
		vector<LabelMat> validationFrames;
		validationFrames.insert(validationFrames.end(), first + foldSize * i, first + foldSize * (i + 1));

		accuracy += neuralNetworkClassifier(trainingFrames, validationFrames);
	}

	accuracy /= k;

	return accuracy;
}

int main()
{
	ActionMap[BrushingTeeth] = "BrushingTeeth";
	ActionMap[CuttingInKitchen] = "CuttingInKitchen";
	ActionMap[JumpingJack] = "JumpingJack";
	ActionMap[Lunges] = "Lunges";
	ActionMap[WallPushups] = "WallPushups";

	//vector<LabelMat> frameMats = getFrameMats();

	Mat test = (Mat_<float>(5, 5) << 
		20, 5, 4, 0, 2,
		4, 18, 2, 1, 6,
		0, 2, 28, 0, 1,
		6, 12, 1, 12, 0,
		0, 0, 0, 0, 31);

	showConfusionMatrix(test);

	return 0;
}