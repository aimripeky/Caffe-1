#include "caffe/layers/multibox_loss_layer.hpp"
#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include <list>
#include "caffe/util/math_functions.hpp"

using std::list;

namespace caffe {

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// bottom[3*i+0] stores the default boxes.
		// bottom[3*i+1] stores the location predictions.
		// bottom[3*i+2] stores the confidence predictions.
		// bottom[3*n+1] stores the label truth bounding boxes.
		// bottom[3*n+2] stores the ground truth bounding boxes.
		const DetectionParameter &detection_param = this->public_param_.detection_param();
		box_dimensions_ = detection_param.range().dim_size();
		share_location_ = detection_param.share_location();
		num_classes_ = detection_param.num_classes();
		const MultiBoxLossParameter &multibox_loss_param = this->layer_param_.multibox_loss_param();
	    match_type_ = multibox_loss_param.match_type();
		overlap_threshold_ = multibox_loss_param.overlap_threshold();
		conf_loss_type_ = multibox_loss_param.conf_loss_type();
		background_label_ = multibox_loss_param.background_label();
		CHECK_GT(background_label_, 0);
	}

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		gt_boxes_.clear();
		pred_boxes_.clear();
		const int pyramid_box_size = bottom.size() - 2;
		ExtractGroundTruth(bottom[pyramid_box_size], bottom[pyramid_box_size + 1], box_dimensions_, gt_boxes_);
		CHECK_EQ(pyramid_box_size % 3, 0);
		num_pyramid_layer_ = pyramid_box_size / 3;

		for (int i = 0; i < num_pyramid_layer_; ++i){
			const int id = 3 * i;
			ExtractPredBoxes(bottom[id + 0], bottom[id + 1], bottom[id + 2], box_dimensions_,
				share_location_, num_classes_, num_, pred_boxes_);
		}
		int box_classes = share_location_ ? 1 : num_classes_;
		vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
		top[0]->Reshape(loss_shape);
	}

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		for (int i = 0; i < bottom.size();++i){
			//Copy data from GPU to CPU;
			bottom[i]->cpu_data();
		}
		//match_boxes[pred_box][gt_box]
		for (int n = 0; n < num_; ++n) {
			// overlaps[0][class_id][gt_box][pred_box]
			// overlaps[1][class_id][pred_box][gt_box]
			map<const Box<Dtype> *, const Box<Dtype> *> match_boxes;
			vector<map<int, map<const Box<Dtype> *, map<const Box<Dtype>*, Dtype>>>> overlaps;
			list<pair<const int, const Box<Dtype> *>> negtives;
			JaccardOverlaps(&gt_boxes_[n], &pred_boxes_[n], match_type_, share_location_,
				background_label_, overlaps);
			MathBoxes(&overlaps, match_type_, overlap_threshold_, match_boxes);
			ExtractNegtiveBox(&pred_boxes_[n], share_location_, background_label_, negtives);
			int num_negtive = negtives.size();
			int num_train_negtive =static_cast<int>(static_cast<float>(match_boxes.size())*neg_pos_ratio_ + 0.5);
			if (neg_pos_ratio_ > 0){
				num_train_negtive = std::min(num_train_negtive, num_negtive);
			}
			CHECK_LE(min_negtive_, num_negtive);
			num_train_negtive = std::max(num_negtive, min_negtive_);
			vector<Box<Dtype> *> train_negtives;
			
			for (const auto &negtive : negtives)
			for (int i = 0; i < num_negtive; i++){
				if (num_train_negtive <= 0)
					break;
				if (overlaps[0][negtive.first][negtive.second].begin()->second < overlap_threshold_) {
					train_negtives.push_back(overlap_threshold_);
					if (train_negtives.)
				}
			}
		}
		
	}

	template <typename Dtype>
	void MultiBoxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		

	}

	INSTANTIATE_CLASS(MultiBoxLossLayer);
	REGISTER_LAYER_CLASS(MultiBoxLoss);

}  // namespace caffe
