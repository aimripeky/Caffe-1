#ifndef CAFFE_MULTIBOX_LOSS_LAYER_HPP_
#define CAFFE_MULTIBOX_LOSS_LAYER_HPP_

#include <map>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/box.hpp"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	class MultiBoxLossLayer : public LossLayer <Dtype> {
	public:
		explicit MultiBoxLossLayer(const LayerParameter& param)
			: LossLayer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		// bottom[3*i+0] stores the default boxes.
		// bottom[3*i+1] stores the location predictions.
		// bottom[3*i+2] stores the confidence predictions.
		// bottom[3*n+1] stores the label truth bounding boxes.
		// bottom[3*n+2] stores the ground truth bounding boxes.
		virtual inline int MinBottomBlobs() const { return 5; }
		virtual inline int ExactNumTopBlobs() const { return 1; }
		virtual inline const char* type() const { return "MultiBoxLoss"; }
	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	private:
		bool share_location_;
		int num_pyramid_layer_;
		int box_dimensions_;
		int num_classes_;
		int num_;
		int background_label_;
		MultiBoxLossParameter_MatchType match_type_;
		MultiBoxLossParameter_ConfLossType conf_loss_type_;
		float overlap_threshold_;
		float neg_pos_ratio_;
		int min_negtive_;

		vector<DefBox<Dtype>> def_boxes_;
		vector<vector<GTBox<Dtype>>> gt_boxes_; //gt_boxes_[image_id][box_id]
		vector<vector<vector<Box<Dtype>>>> pred_boxes_; // pred_boxes_[image_id][class_id][box_id]
	};
}

#endif