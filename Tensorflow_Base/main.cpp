#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <map>
#include <math.h>
#include <iostream>
#include <Windows.h>

#define ACTION_NUM 5

using namespace cv;
using namespace std;

enum Action { BrushingTeeth, CuttingInKitchen, JumpingJack, Lunges, WallPushups };
map<Action, String>  ActionMap;

struct LabelMat
{
	String label;
	Mat mat;
};

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
			lm.label = it->second;

			Size size(90, 90);
			Mat frame = getFrameMat(full_file_name);
			resize(frame, lm.mat, size);

			frames.push_back(lm);
		} while (FindNextFile(dir, &file_data));

		FindClose(dir);
	}	

	return frames;
}

int main()
{
	ActionMap[BrushingTeeth] = "BrushingTeeth";
	ActionMap[CuttingInKitchen] = "CuttingInKitchen";
	ActionMap[JumpingJack] = "JumpingJack";
	ActionMap[Lunges] = "Lunges";
	ActionMap[WallPushups] = "WallPushups";

	vector<LabelMat> frameMats = getFrameMats();

	return 0;
}