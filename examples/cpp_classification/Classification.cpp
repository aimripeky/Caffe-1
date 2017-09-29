#include "Classification.h"


Classification::Classification(const string& model_file, const string& trian_file) {
	Caffe::set_mode(Caffe::GPU);
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trian_file);
	auto input_blobs = net_->input_blobs()[0];
	scale_ = 1.0;
	channels_ = input_blobs->channels();
	size_ = cv::Size(input_blobs->width(), input_blobs->height());
	num_ = input_blobs->num();
	CHECK_GT(num_, 0);
	CHECK_GT(channels_, 0);
	CHECK_GT(size_.width, 0);
	CHECK_GT(size_.height, 0);
	this->Setup();
	means_.resize(channels_, cv::Mat(size_, CV_32FC1, cv::Scalar(0)));
}


void Classification::SetMeans(const std::shared_ptr<vector<float>> mean_vals){
	int mean_size = mean_vals->size();
	CHECK_EQ(mean_size, channels_);

	for (int i = 0; i < channels_; i++) {
		means_[i].setTo(cv::Scalar(mean_vals->at(i)));
	}
}

void Classification::Setup(){

	int offset = size_.width*size_.height;
	auto input_data = net_->input_blobs()[0]->mutable_cpu_data();

	inputs_.resize(num_);
	for (auto &input : inputs_){
		input.clear();
		for (int c = 0; c < channels_; c++) {
			input.push_back(cv::Mat(size_, CV_32FC1, input_data));
			input_data += offset;
		}
	}
}

void Classification::SetBatchSize(int batch_size){
	CHECK_GT(batch_size, 0);

	auto input_blobs = net_->input_blobs()[0];

	if (batch_size != num_){
		num_ = batch_size;
		net_->input_blobs()[0]->Reshape(num_, channels_, size_.height, size_.width);
		net_->Reshape();
		this->Setup();
	}
}


const std::vector<std::vector<float>> *Classification::Classify(const std::vector<cv::Mat> &inputs){
	if (results_.size() != inputs.size()) results_.resize(inputs.size());
	auto input_blobs = net_->input_blobs()[0];
	int total_size = inputs.size();
	int batch_size = num_;
//	auto input_size = cv::Size(input_blobs->width(),input_blobs->height());

	for (int n = 0; n < total_size; n = n + num_) {
		if (n + num_ > total_size) {
			Blob<float>* input_layer = net_->input_blobs()[0];
			this->SetBatchSize(total_size - n);
		}

#pragma omp parallel for
		for (int i = 0; i < num_; i++) {
			cv::Mat c_sample;
			int input_channels = inputs[n + i].channels();
			if (input_channels == 3 && channels_ == 1)
				cv::cvtColor(inputs[n + i], c_sample, cv::COLOR_BGR2GRAY);
			else if (input_channels == 4 && channels_ == 1)
				cv::cvtColor(inputs[n + i], c_sample, cv::COLOR_BGRA2GRAY);
			else if (input_channels == 4 && channels_ == 3)
				cv::cvtColor(inputs[n + i], c_sample, cv::COLOR_BGRA2BGR);
			else if (input_channels == 1 && channels_ == 3)
				cv::cvtColor(inputs[n + i], c_sample, cv::COLOR_GRAY2BGR);
			else
				c_sample = inputs[n + i];
			cv::Mat cs_sample;
			if (c_sample.size() != size_){
				cv::resize(c_sample, cs_sample, size_);
			}
			else
				cs_sample = c_sample;
			cv::Mat csd_sample;
			if (channels_ == 3){
				cs_sample.convertTo(csd_sample, CV_32FC3);
			}
			else{
				cs_sample.convertTo(csd_sample, CV_32FC1);
			}

			cv::Mat f_sample = scale_ * csd_sample;
			cv::split(f_sample, inputs_[i]);
			for (int c = 0; c < channels_; c++) {
				inputs_[i][c] -= means_[c];
			}
		}
		input_blobs->mutable_cpu_data();
		net_->Forward();
		Blob<float>* output_layer = net_->output_blobs()[0];
		const float* begin = output_layer->cpu_data();

		CHECK_EQ(num_, output_layer->num());
		int output_channel = output_layer->channels();
		for (int i = 0; i < num_; i++)
		{
			results_[n + i].resize(output_channel);
			for (int c = 0; c < output_channel; c++){
				results_[n + i][c] = begin[i*output_layer->channels() + c];
			}
		}
	}
	this->SetBatchSize(batch_size);
	return &results_;
}
