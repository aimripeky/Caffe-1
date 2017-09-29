#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {

//    vector<int> label_shape(1, batch_size);
//    top[1]->Reshape(label_shape);
//    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
//      this->prefetch_[i].label_.Reshape(label_shape);
//      }

	  const int label_channels = datum.label_shape().channels();
	  const int label_height = datum.label_shape().height();
	  const int label_width = datum.label_shape().width();
	  vector<int> label_shape;
	  label_shape.push_back(batch_size);
	  label_shape.push_back(label_channels);
	  label_shape.push_back(label_height);
	  label_shape.push_back(label_width);
	  top[1]->Reshape(label_shape);
	  LOG(INFO) << "label size: " << batch_size << ","
		  << label_channels << "," << label_height << ","
		  << label_width;
	  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		  this->prefetch_[i].label_.Reshape(label_shape);
	  }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    Datum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));

    // Copy label.
    if (this->output_labels_) {
//     top_label[item_id] = datum.label();
		const DatumShape::DataDepth label_depth = datum.label_shape().data_depth();
		const int label_channels = datum.label_shape().channels();
		const int label_height = datum.label_shape().height();
		const int label_width = datum.label_shape().width();

		const int top_label_offset = item_id*label_channels*label_height*label_width;
			// ((n * channels() + c) * height() + h) * width() + w
		for (int c = 0; c < label_channels; c++) {
			int cs = c * label_height;
			for (int h = 0; h < label_height; h++) {
				int hs = (cs + h) * label_width;
				for (int w = 0; w < label_width; w++) {
					int id = hs + w;
					if (label_depth == DatumShape::DEPTH_8S) {
						if (sizeof(Dtype) < sizeof(char)) {
							LOG(WARNING) << "Conversion from 'char' to 'Dtype', possible loss of data.";
						}
						const char *pRead = (const char *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else if (label_depth == DatumShape::DEPTH_8U) {
						if (sizeof(Dtype) < sizeof(unsigned char)) {
							LOG(WARNING) << "Conversion from 'unsigned char' to 'Dtype', possible loss of data.";
						}
						const unsigned char *pRead = (unsigned char *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else if (label_depth == DatumShape::DEPTH_16S) {
						if (sizeof(Dtype) < sizeof(short)) {
							LOG(WARNING) << "Conversion from 'short' to 'Dtype', possible loss of data.";
						}
						const short *pRead = (const short *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else if (label_depth == DatumShape::DEPTH_16U) {
						if (sizeof(Dtype) < sizeof(unsigned short)) {
							LOG(WARNING) << "Conversion from 'unsigned short' to 'Dtype', possible loss of data.";
						}
						const unsigned short *pRead = (const unsigned short *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else if (label_depth == DatumShape::DEPTH_32S) {
						if (sizeof(Dtype) < sizeof(int)) {
							LOG(WARNING) << "Conversion from 'int' to 'Dtype', possible loss of data.";
						}
						const int *pRead = (const int *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else if (label_depth == DatumShape::DEPTH_32F) {
						if (sizeof(Dtype) < sizeof(float)) {
							LOG(WARNING) << "Conversion from 'float' to 'Dtype', possible loss of data.";
						}
						const float *pRead = (const float *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else if (label_depth == DatumShape::DEPTH_64F) {
						if (sizeof(Dtype) < sizeof(double)) {
							LOG(WARNING) << "Conversion from 'double' to 'Dtype', possible loss of data.";
						}
						const double *pRead = (const double *)datum.label().data();
						top_label[top_label_offset + id] = pRead[id];
					}
					else LOG(FATAL) << "Unknow deth of label!";
				}
			}
		}
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<Datum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
