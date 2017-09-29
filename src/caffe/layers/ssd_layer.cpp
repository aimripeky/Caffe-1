#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/ssd_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

	template <typename Dtype>
	void CalDefaultBox(const std::vector<int> &input_range, const std::vector<int> &def_box_shape,
		const std::vector<Dtype> &box_size, const std::vector<Dtype> &variances, int start_idx, const int channel_offset,
		int dim, int prev_idx, std::vector<std::pair<Dtype, Dtype>> &cur_datas, Dtype *def_loc_data, Dtype *variance_data) {
		int cur_length = def_box_shape[dim];
		Dtype step = static_cast<Dtype>(input_range[dim]) / static_cast<Dtype>(cur_length);
		for (int len = 0; len < cur_length; ++len) {
			int cur_idx = prev_idx + len;
			std::pair<Dtype, Dtype> &cur_data = cur_datas[dim];
			Dtype center = step*(len + static_cast<Dtype>(0.5));
			cur_data.first = (center - box_size[dim] / static_cast<Dtype>(2)) / static_cast<Dtype>(input_range[dim]);
			cur_data.second = (center + box_size[dim] / static_cast<Dtype>(2)) / static_cast<Dtype>(input_range[dim]);
			int next_dim = dim + 1;
			if (next_dim < input_range.size()) {
				int next_length = def_box_shape[next_dim];
				int next_idx = cur_idx*next_length;
				CalDefaultBox(input_range, def_box_shape,
					box_size, variances, start_idx, channel_offset, 
					next_dim, next_idx, cur_datas, def_loc_data, variance_data);
			}
			else {
				const int num_axis = cur_datas.size();
				for (int i = 0; i < num_axis; ++i) {
					const int j = i + num_axis;
					int min_idx = start_idx + cur_idx + i*channel_offset;
					int max_idx = start_idx + cur_idx + j*channel_offset;
					def_loc_data[min_idx] = cur_datas[i].first;
					def_loc_data[max_idx] = cur_datas[i].second;
					variance_data[min_idx] = variances[i];
					variance_data[max_idx] = variances[j];
				}
			}
		}
	}

	template <typename Dtype> 
	void CalDefaultBox(const std::vector<int> &input_range, const std::vector<int> &def_box_shape,
		const std::vector<std::vector<Dtype>> &box_sizes, const std::vector<Dtype> &variances,
		Dtype *def_loc_data, Dtype *variance_data) {
		std::vector<std::pair<Dtype, Dtype>> cur_datas(input_range.size());
		CHECK_EQ(input_range.size(), def_box_shape.size());
		int channel_offset = 1;
		for (size_t j = 0; j < def_box_shape.size(); ++j) {
			channel_offset *= def_box_shape[j];
		}
		const int box_offset = 2 * box_sizes[0].size()*channel_offset;

		for (size_t i = 0; i < box_sizes.size(); ++i){
			int start_idx = i*box_offset;
			CalDefaultBox(input_range, def_box_shape,
				box_sizes[i], variances, start_idx, channel_offset,
				0, 0, cur_datas, def_loc_data, variance_data);
		}
	}


	template <typename Dtype>
	void SSDLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		box_shapes_.clear();
		const DetectionParameter &detection_param = this->public_param_.detection_param();
		const SSDParameter &ssd_param = this->layer_param_.ssd_param();
		const BlobShape &input_range = detection_param.range();
		box_dimensions_ = input_range.dim_size();
		input_range_.clear();
		for (int i = 0; i < box_dimensions_; ++i){
			input_range_.push_back(input_range.dim(i));
		}
		const int num_default_box = ssd_param.default_box_size();
		for (int i_default_box = 0; i_default_box < num_default_box; ++i_default_box){
			const DefaultBox &default_box = ssd_param.default_box(i_default_box);
			const Dtype box_size = static_cast<Dtype>(default_box.size());
			CHECK_GT(box_size, 0);
			//Set size of default box.
			const int side_ratio_size = default_box.side_ratios_size();
			CHECK_GT(side_ratio_size, 0);
			for (int i_side_ratio = 0; i_side_ratio < side_ratio_size; ++i_side_ratio) {
				const SideRatio &side_ratio = default_box.side_ratios(i_side_ratio);
				const int ratio_size = side_ratio.ratios_size();
				std::vector<Dtype> box_shape(ratio_size);
				CHECK_GT(ratio_size, 0);
				CHECK_EQ(box_dimensions_, ratio_size);
				Dtype product = 1;
				for (int i_ratio = 0; i_ratio < ratio_size; ++i_ratio){
					product *= side_ratio.ratios(i_ratio);
					box_shape[i_ratio] = side_ratio.ratios(i_ratio)*box_size;
				}
				CHECK_GT(product, 0);
				const Dtype solver = pow(product, static_cast<Dtype>(1.0) / static_cast<Dtype>(ratio_size));
				for (int i = 0; i < box_shape.size(); ++i) {
					box_shape[i] = box_shape[i] / solver;
				}
				box_shapes_.push_back(box_shape);
			}
		}
		

		variances_.resize(2 * box_dimensions_, 1.0f);
		const int variance_size = ssd_param.variances_size();
		if (variance_size > 1){
			CHECK_EQ(variance_size, 2 * box_dimensions_);
			for (int i_variance = 0; i_variance < variance_size; ++i_variance){
				const Dtype variances = ssd_param.variances(i_variance);
				CHECK_GT(variances, 0);
				variances_[i_variance] = variances;
			}
		}
		else if (variance_size == 1){
			const Dtype variances = ssd_param.variances(0);
			CHECK_GT(variances, 0);
			variances_.resize(box_dimensions_, variances);
		}

		//bottom[0]: input data.
		//bootom[1]: loc bottom.
		//bootom[2]: conf bottom.
		CHECK_GE(bottom.size(), 2);
		CHECK_LE(bottom.size(), 3);

		int num_classes = detection_param.num_classes();
		bool share_location = detection_param.share_location();
		const int num_def_output = 2 * static_cast<int>(box_dimensions_) * static_cast<int>(box_shapes_.size());
		const int num_loc_output = num_def_output * pow(num_classes, !share_location);
		const int num_conf_output = static_cast<int>(box_shapes_.size()) * num_classes * pow(num_classes, !share_location);

		const ConvolutionParameter &conv_param = this->layer_param_.convolution_param();
		const bool bias_term = conv_param.bias_term();
		channel_axis_ = bottom[1]->CanonicalAxisIndex(conv_param.axis());
		num_spatial_axes_ = bottom[1]->num_axes() - channel_axis_ - 1;
		CHECK_EQ(num_spatial_axes_, box_dimensions_);

		LayerParameter loc_layer_param(layer_param_);
		LayerParameter conf_layer_param(layer_param_);
		loc_layer_param.set_name("ssd_loc_conv_layer");
		conf_layer_param.set_name("ssd_conf_conv_layer");
		loc_layer_param.set_type("Convolution");
		conf_layer_param.set_type("Convolution");
		ConvolutionParameter *loc_conv_param = loc_layer_param.mutable_convolution_param();
		ConvolutionParameter *conf_conv_param = conf_layer_param.mutable_convolution_param();
		loc_conv_param->set_num_output(num_loc_output);
		conf_conv_param->set_num_output(num_conf_output);
		loc_conv_layer_ = LayerRegistry<Dtype>::CreateLayer(loc_layer_param);
		conf_conv_layer_ = LayerRegistry<Dtype>::CreateLayer(conf_layer_param);
		if (blobs_.size() > 0){
			LOG(INFO) << "Skipping parameter initialization.";
			CHECK_GE(blobs_.size(), 2);
			loc_conv_layer_->blobs().clear();
			conf_conv_layer_->blobs().clear();
			loc_conv_layer_->blobs().push_back(blobs_[0]);
			conf_conv_layer_->blobs().push_back(blobs_[1]);
			if (bias_term){
				CHECK_EQ(blobs_.size(), 4);
				loc_conv_layer_->blobs().push_back(blobs_[3]);
				conf_conv_layer_->blobs().push_back(blobs_[4]);
			}
			else{
				CHECK_EQ(blobs_.size(), 2);
			}
		}

		def_loc_bottom_.resize(1, bottom[0]);
		loc_bottom_.resize(1, bottom[1]);
		if (bottom.size() == 3)
			conf_bottom_.resize(1, bottom[2]);
		else
			conf_bottom_.resize(1, bottom[1]);
		//top[0]: def_loc_boxes.
		//top[1]: pred_loc_boxes.
		//top[2]: pred_conf_boxes.
		CHECK_EQ(top.size(), 3);
		def_loc_top_.resize(1, top[0]);
		loc_top_.resize(1, top[1]);
		conf_top_.resize(1, top[2]);
		loc_conv_layer_->LayerSetUp(loc_bottom_, loc_top_);
		conf_conv_layer_->LayerSetUp(conf_bottom_, conf_top_);
		if (blobs_.empty()){
			blobs_.push_back(loc_conv_layer_->blobs()[0]);
			blobs_.push_back(conf_conv_layer_->blobs()[0]);
			blobs_.push_back(loc_conv_layer_->blobs()[1]);
			blobs_.push_back(conf_conv_layer_->blobs()[1]);
		}
	}

	template <typename Dtype>
	void SSDLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
		const int  first_spatial_axis = channel_axis_ + 1;
		const int num_axis = def_loc_bottom_[0]->num_axes();
		for (int i = 1; i < bottom.size(); ++i) {
			CHECK_EQ(num_axis, bottom[i]->num_axes());
			for (int j = first_spatial_axis; j < num_axis;++j) {
				CHECK_EQ(bottom[1]->shape(j), bottom[i]->shape(j));
			}
		}
		for (int i = 0; i < bottom.size(); ++i) {
			for (int j = 0; j < num_axis; ++j) {
				CHECK_GT(bottom[i]->shape(j), 0);
			}
		}

		loc_conv_layer_->Reshape(loc_bottom_, loc_top_);
		conf_conv_layer_->Reshape(conf_bottom_, conf_top_);
		std::vector<int> shape;
		shape.push_back(2);
		shape.push_back(2 * box_dimensions_ * box_shapes_.size());
		for (int j = first_spatial_axis; j < num_axis; ++j) {
			shape.push_back(top[1]->shape(j));
		}
		def_loc_top_[0]->Reshape(shape);
		for (int i = 0; i < top.size(); ++i) {
			CHECK_EQ(num_axis, top[i]->num_axes());
			for (int j = first_spatial_axis; j < num_axis; ++j) {
				CHECK_EQ(top[0]->shape(j), top[i]->shape(j));
			}
		}
		for (int i = 0; i < top.size(); ++i) {
			for (int j = 0; j < num_axis; ++j) {
				CHECK_GT(top[i]->shape(j), 0);
			}
		}
		Dtype* def_loc_data = top[0]->mutable_cpu_data();
		Dtype* variance_data = def_loc_data + top[0]->offset(std::vector<int>(1, 1));
		CalDefaultBox(input_range_, def_loc_shape_, box_shapes_, variances_, def_loc_data, variance_data);
	}


	template <typename Dtype>
	void SSDLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		loc_conv_layer_->Forward(loc_bottom_, loc_top_);
		conf_conv_layer_->Forward(conf_bottom_, conf_top_);
	}

//	template <typename Dtype>
//	void SSDLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//		const vector<Blob<Dtype>*>& top){
//		Forward_box(loc_bottom_, loc_top_);
//		loc_conv_layer_->Forward(loc_bottom_, loc_top_);
//		conf_conv_layer_->Forward(conf_bottom_, conf_top_);
//	}
//
	template <typename Dtype>
	void SSDLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
		CHECK_EQ(propagate_down[0], false);
		const vector<bool> loc_propagate_down(1, propagate_down[1]);
		const vector<bool> conf_propagate_down(1, propagate_down[2]);
		loc_conv_layer_->Backward(loc_top_, loc_propagate_down, loc_bottom_);
		conf_conv_layer_->Backward(conf_top_, conf_propagate_down, conf_bottom_);
	}

//	template <typename Dtype>
//	void SSDLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
//		CHECK_EQ(propagate_down[0], false);
//		const vector<bool> loc_propagate_down(1, propagate_down[1]);
//		const vector<bool> conf_propagate_down(1, propagate_down[2]);
//		loc_conv_layer_->Backward(loc_bottom_, loc_propagate_down, loc_top_);
//		conf_conv_layer_->Backward(conf_bottom_, conf_propagate_down, conf_top_);
//	}


	INSTANTIATE_CLASS(SSDLayer);
	REGISTER_LAYER_CLASS(SSD);

}