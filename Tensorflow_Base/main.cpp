#include "tfcpp.h"
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

string typeToString(int type) {
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

string floatToString(float input)
{
	stringstream stream;
	stream << fixed << setprecision(2) << input;

	return stream.str();
}

Tensor cvMatToTensor(Mat input)
{
	Tensor input_tensor(DT_FLOAT, TensorShape({ 1, input.rows, input.cols, input.channels() }));

	float *p = input_tensor.flat<float>().data();

	Mat tensorMat(input.rows, input.cols, CV_32FC3, p);
	input.convertTo(tensorMat, CV_32FC3);
	
	cout << input_tensor.DebugString() << endl;

	return input_tensor;
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

			cout << typeToString(lm.mat.type()) << endl;
			cout << lm.mat.rows << endl;
			cout << lm.mat.cols << endl;
			cout << lm.mat.channels() << endl;

			lm.tensor = cvMatToTensor(lm.mat);

			frames.push_back(lm);

			return frames;
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

	cv::Size tileSize(100, 100);
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

void buildNetwork()
{
	Scope scope = Scope::NewRootScope();

	auto x = Placeholder(scope, DT_FLOAT);
	auto y = Placeholder(scope, DT_FLOAT);

	// weights init
	auto w1 = Variable(scope, { 3, 3 }, DT_FLOAT);
	auto assign_w1 = Assign(scope, w1, RandomNormal(scope, { 3, 3 }, DT_FLOAT));

	auto w2 = Variable(scope, { 3, 2 }, DT_FLOAT);
	auto assign_w2 = Assign(scope, w2, RandomNormal(scope, { 3, 2 }, DT_FLOAT));

	auto w3 = Variable(scope, { 2, 1 }, DT_FLOAT);
	auto assign_w3 = Assign(scope, w3, RandomNormal(scope, { 2, 1 }, DT_FLOAT));

	// bias init
	auto b1 = Variable(scope, { 1, 3 }, DT_FLOAT);
	auto assign_b1 = Assign(scope, b1, RandomNormal(scope, { 1, 3 }, DT_FLOAT));

	auto b2 = Variable(scope, { 1, 2 }, DT_FLOAT);
	auto assign_b2 = Assign(scope, b2, RandomNormal(scope, { 1, 2 }, DT_FLOAT));

	auto b3 = Variable(scope, { 1, 1 }, DT_FLOAT);
	auto assign_b3 = Assign(scope, b3, RandomNormal(scope, { 1, 1 }, DT_FLOAT));

	// layers
	auto layer_1 = Tanh(scope, Add(scope, MatMul(scope, x, w1), b1));
	auto layer_2 = Tanh(scope, Add(scope, MatMul(scope, layer_1, w2), b2));
	auto layer_3 = Tanh(scope, Add(scope, MatMul(scope, layer_2, w3), b3));

	// regularization
	auto regularization = AddN(scope, initializer_list<Input>
	{
		L2Loss(scope, w1),
		L2Loss(scope, w2),
		L2Loss(scope, w3)
	});

	// loss calculation
	auto loss = Add(scope,
		ReduceMean(scope, Square(scope, Sub(scope, layer_3, y)), { 0, 1 }),
		Mul(scope, Cast(scope, 0.01, DT_FLOAT), regularization));

	// add the gradients operations to the graph
	std::vector<Output> grad_outputs;
	TF_CHECK_OK(AddSymbolicGradients(scope, { loss }, { w1, w2, w3, b1, b2, b3 }, &grad_outputs));

	// update the weights and bias using gradient descent
	auto apply_w1 = ApplyGradientDescent(scope, w1, Cast(scope, 0.01, DT_FLOAT), { grad_outputs[0] });
	auto apply_w2 = ApplyGradientDescent(scope, w2, Cast(scope, 0.01, DT_FLOAT), { grad_outputs[1] });
	auto apply_w3 = ApplyGradientDescent(scope, w3, Cast(scope, 0.01, DT_FLOAT), { grad_outputs[2] });
	auto apply_b1 = ApplyGradientDescent(scope, b1, Cast(scope, 0.01, DT_FLOAT), { grad_outputs[3] });
	auto apply_b2 = ApplyGradientDescent(scope, b2, Cast(scope, 0.01, DT_FLOAT), { grad_outputs[4] });
	auto apply_b3 = ApplyGradientDescent(scope, b3, Cast(scope, 0.01, DT_FLOAT), { grad_outputs[5] });

	ClientSession session(scope);
	std::vector<Tensor> outputs;

	// init the weights and biases by running the assigns nodes once
	TF_CHECK_OK(session.Run({ assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3 }, nullptr));

}

void buildCNN(LabelMat input)
{
	Scope root = Scope::NewRootScope();

	Tensor input_layer = input.tensor;

	//Variable = tensor that persists across steps {width, height, input depth, output depth}
	//Assign = update ref (2nd para) by assigning value (3rd para)

	auto filter1 = Variable(root, { 5, 5, 3, 10 }, DT_FLOAT);
	Assign(root, filter1, TruncatedNormal(root, { 5, 5, 3, 3 }, DT_FLOAT));
	auto conv1 = Conv2D(root, input_layer, filter1, { 1, 1, 1, 1 }, "SAME");
	auto elu1 = Elu(root, conv1);
	auto maxpool1 = MaxPool(root, elu1, { 1, 2, 2, 1 }, { 1, 1, 1, 1 }, "SAME");

	// change 7
	auto flatten = Reshape(root, maxpool1, {-1, 7 * 7 * 10});

	vector<Tensor> outputs;
	ClientSession session(root);
	TF_CHECK_OK(session.Run({ flatten }, &outputs));

	cout << outputs[0].DebugString();
}

int main()
{
	ActionMap[BrushingTeeth] = "BrushingTeeth";
	ActionMap[CuttingInKitchen] = "CuttingInKitchen";
	ActionMap[JumpingJack] = "JumpingJack";
	ActionMap[Lunges] = "Lunges";
	ActionMap[WallPushups] = "WallPushups";

	vector<LabelMat> frameMats = getFrameMats();

	buildCNN(frameMats[0]);

	/*Mat test = (Mat_<float>(5, 5) << 
		20, 5, 4, 0, 2,
		4, 18, 2, 1, 6,
		0, 2, 28, 0, 1,
		6, 12, 1, 12, 0,
		0, 0, 0, 0, 31);
		 
	showConfusionMatrix(test);*/

	return 0;
}