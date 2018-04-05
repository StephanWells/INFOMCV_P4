#include "tfplus.h"

#include <vector>
#include <map>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <Windows.h>
#include <iomanip>
#include <sstream>
#define BATCH_SIZE 64
#define EPOCHS 50

#define ACTION_NUM 5
#define INPUT_SQ_SIZE 90

enum Action { BrushingTeeth, CuttingInKitchen, JumpingJack, Lunges, WallPushups };
map<Action, String>  ActionMap;

struct LabelMat
{
	Action label;
	Mat mat;
	Tensor tensor;
};

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth)
	{
		case CV_8U:  r = "8U"; break;
		case CV_8S:  r = "8S"; break;
		case CV_16U: r = "16U"; break;
		case CV_16S: r = "16S"; break;
		case CV_32S: r = "32S"; break;
		case CV_32F: r = "32F"; break;
		case CV_64F: r = "64F"; break;
		default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

Tensor cvtCVMatToTensor(Mat input)
{
	Tensor input_tensor(DT_FLOAT, TensorShape({ 1, input.rows, input.cols, input.channels() }));

	float *p = input_tensor.flat<float>().data();

	Mat tensorMat(input.rows, input.cols, CV_32FC3, p);
	input.convertTo(tensorMat, CV_32FC3);
	
	cout << input_tensor.DebugString() << endl;

	return input_tensor;
}

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

			cv::Size size(INPUT_SQ_SIZE, INPUT_SQ_SIZE);
			Mat frame = getFrameMat(full_file_name);

			frame.convertTo(frame, CV_32FC3);
			resize(frame, lm.mat, size);

			cout << type2str(lm.mat.type()) << endl;
			cout << lm.mat.rows << endl;
			cout << lm.mat.cols << endl;
			cout << lm.mat.channels() << endl;

			lm.tensor = cvtCVMatToTensor(lm.mat);

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