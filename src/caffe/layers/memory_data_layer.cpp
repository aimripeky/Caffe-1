#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/memory_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.memory_data_param().batch_size();
  channels_ = this->layer_param_.memory_data_param().channels();
  height_ = this->layer_param_.memory_data_param().height();
  width_ = this->layer_param_.memory_data_param().width();
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_ * size_, 0) <<
      "batch_size, channels, height, and width must be specified and"
      " positive in memory_data_param";
  vector<int> label_shape(1, batch_size_);
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(label_shape);
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(label_shape);
  data_ = NULL;
  labels_ = NULL;
  added_data_.cpu_data();
  added_label_.cpu_data();
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector) {
  CHECK(!has_new_data_) <<
      "Can't add data until current data has been consumed.";
  size_t num = datum_vector.size();
  CHECK_GT(num, 0) << "There is no datum to add.";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(datum_vector, &added_data_);
  // Copy Labels

  CHECK_GT(datum_vector.size(), 0) << "Datum vector is empty.";
  const DatumShape::DataDepth label_depth = datum_vector[0].label_shape().data_depth();
  const int label_channels = datum_vector[0].label_shape().channels();
  const int label_height = datum_vector[0].label_shape().height();
  const int label_width = datum_vector[0].label_shape().width();
  added_label_.Reshape(num, label_channels, label_height, label_width);

  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
	  CHECK_EQ(label_depth, datum_vector[item_id].label_shape().data_depth());
	  CHECK_EQ(label_channels, datum_vector[item_id].label_shape().channels());
	  CHECK_EQ(label_height, datum_vector[item_id].label_shape().height());
	  CHECK_EQ(label_width, datum_vector[item_id].label_shape().width());

	  int size = datum_vector[item_id].label().size();
	  int label_offset = item_id * size;
	  for (int id = 0; id < size; id++) {
		  if (label_depth == DatumShape::DEPTH_8S) {
			  if (sizeof(Dtype) < sizeof(char)) {
				  LOG(WARNING) << "Conversion from 'char' to 'Dtype', possible loss of data.";
			  }
			  const char *pRead = (const char *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else if (label_depth == DatumShape::DEPTH_8U) {
			  if (sizeof(Dtype) < sizeof(unsigned char)) {
				  LOG(WARNING) << "Conversion from 'unsigned char' to 'Dtype', possible loss of data.";
			  }
			  const unsigned char *pRead = (unsigned char *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else if (label_depth == DatumShape::DEPTH_16S) {
			  if (sizeof(Dtype) < sizeof(short)) {
				  LOG(WARNING) << "Conversion from 'short' to 'Dtype', possible loss of data.";
			  }
			  const short *pRead = (const short *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else if (label_depth == DatumShape::DEPTH_16U) {
			  if (sizeof(Dtype) < sizeof(unsigned short)) {
				  LOG(WARNING) << "Conversion from 'unsigned short' to 'Dtype', possible loss of data.";
			  }
			  const unsigned short *pRead = (const unsigned short *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else if (label_depth == DatumShape::DEPTH_32S) {
			  if (sizeof(Dtype) < sizeof(int)) {
				  LOG(WARNING) << "Conversion from 'int' to 'Dtype', possible loss of data.";
			  }
			  const int *pRead = (const int *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else if (label_depth == DatumShape::DEPTH_32F) {
			  if (sizeof(Dtype) < sizeof(float)) {
				  LOG(WARNING) << "Conversion from 'float' to 'Dtype', possible loss of data.";
			  }
			  const float *pRead = (const float *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else if (label_depth == DatumShape::DEPTH_64F) {
			  if (sizeof(Dtype) < sizeof(double)) {
				  LOG(WARNING) << "Conversion from 'double' to 'Dtype', possible loss of data.";
			  }
			  const double *pRead = (const double *)datum_vector[item_id].label().data();
			  top_label[label_offset + id] = pRead[id];
		  }
		  else LOG(FATAL) << "Unknow deth of label!";
	  }

//	  for (int i = 0; i < size; i++)
//		  top_label[item_id*size + i] = datum_vector[item_id].label_int_data(i);
//	  size = datum_vector[item_id].label_float_data_size();
//	  for (int i = 0; i < size; i++)
//		  top_label[item_id*size + i] = datum_vector[item_id].label_float_data(i);
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}

#ifdef USE_OPENCV
template <typename Dtype>
void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
    const vector<int>& labels) {
  size_t num = mat_vector.size();
  CHECK(!has_new_data_) <<
      "Can't add mat until current data has been consumed.";
  CHECK_GT(num, 0) << "There is no mat to add";
  CHECK_EQ(num % batch_size_, 0) <<
      "The added data must be a multiple of the batch size.";
  added_data_.Reshape(num, channels_, height_, width_);
  added_label_.Reshape(num, 1, 1, 1);
  // Apply data transformations (mirror, scale, crop...)
  this->data_transformer_->Transform(mat_vector, &added_data_);
  // Copy Labels
  Dtype* top_label = added_label_.mutable_cpu_data();
  for (int item_id = 0; item_id < num; ++item_id) {
    top_label[item_id] = labels[item_id];
  }
  // num_images == batch_size_
  Dtype* top_data = added_data_.mutable_cpu_data();
  Reset(top_data, top_label, num);
  has_new_data_ = true;
}
#endif  // USE_OPENCV

template <typename Dtype>
void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n) {
  CHECK(data);
  CHECK(labels);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  // Warn with transformation parameters since a memory array is meant to
  // be generic and no transformations are done with Reset().
  if (this->layer_param_.has_transform_param()) {
    LOG(WARNING) << this->type() << " does not transform array data on Reset()";
  }
  data_ = data;
  labels_ = labels;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::set_batch_size(int new_size) {
  CHECK(!has_new_data_) <<
      "Can't change batch_size until current data has been consumed.";
  batch_size_ = new_size;
  added_data_.Reshape(batch_size_, channels_, height_, width_);
  added_label_.Reshape(batch_size_, 1, 1, 1);
}

template <typename Dtype>
void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data_) << "MemoryDataLayer needs to be initialized by calling Reset";
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[1]->Reshape(batch_size_, 1, 1, 1);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  top[1]->set_cpu_data(labels_ + pos_);
  pos_ = (pos_ + batch_size_) % n_;
  if (pos_ == 0)
    has_new_data_ = false;
}

INSTANTIATE_CLASS(MemoryDataLayer);
REGISTER_LAYER_CLASS(MemoryData);

}  // namespace caffe
