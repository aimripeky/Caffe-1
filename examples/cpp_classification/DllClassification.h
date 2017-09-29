#pragma once
#include<vector>
#include<memory>
#include<opencv2/opencv.hpp>
#include<string>
class DllClassification
{
public:
	virtual char *Version();
	virtual void SetBatchSize(int batch_size);
	virtual void SetMeans(const std::shared_ptr<std::vector<float>> mean_vals);
	virtual void SetScale(float scale);
	virtual void Classify(const std::vector<cv::Mat> &inputs, std::vector<std::vector<float>> &results);
	virtual ~DllClassification(){};
};

extern "C" __declspec(dllexport) DllClassification *CaffeClassification(const std::string &model_file,const std::string &train_file);
  
