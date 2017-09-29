#ifndef CAFFE_SSD_LAYER_HPP_
#define CAFFE_SSD_LAYER_HPP_
#include <opencv2/core/types.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/conv_layer.hpp"

namespace caffe {
/**
* @brief Single Shot MultiBox Detector.
*/
	template <typename Dtype>
	class SSDLayer : public Layer<Dtype>
	{
	public:
	/**
	*@param param provides SSDLayerParameter ssd_param
	*   with SSDLayer options:
	*   - share_location. very classes have their own boxes if this param is true.
	*   - num_classes. The numbel of each box's class.   
	*/
		explicit SSDLayer(const LayerParameter &param) 
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "SSD"; }

		virtual inline int MinBottomBlobs() const { return 1; }
		virtual inline int MaxBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 3; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
//		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	private:
		std::vector<std::vector<Dtype>> box_shapes_;
		std::vector<Dtype> variances_;
		vector<Blob<Dtype>*> def_loc_bottom_;
		vector<Blob<Dtype>*> loc_bottom_;
		vector<Blob<Dtype>*> conf_bottom_;
		vector<Blob<Dtype>*> loc_top_;
		vector<Blob<Dtype>*> conf_top_;
		vector<Blob<Dtype>*> def_loc_top_;
		shared_ptr<Layer<Dtype>> loc_conv_layer_;
		shared_ptr<Layer<Dtype>> conf_conv_layer_;
		int channel_axis_;
		int num_spatial_axes_;
		int box_dimensions_;
		std::vector<int> def_loc_shape_;
		std::vector<int> input_range_;
	};
}  // namespace caffe


#endif  // CAFFE_SSD_LAYER_HPP_