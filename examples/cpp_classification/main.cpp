
#include <mutex>
#include <vector>
#include <Windows.h>
#include <caffe/layers/ssd_layer.hpp>
#include <caffe/layers/multibox_loss_layer.hpp>
#include <complex>
#include <opencv2/opencv.hpp>
//  layer {
//    name: "ssd1"
//    type : "SSD"
//    bottom : "data"
//    bottom : "pool1"
//    top : "ssd_def_box1"
//    top : "ssd_loc_box1"
//    top : "ssd_conf_box1"
//    ssd_param {
//  	default_box { 
//  	  size: 4
//  	  side_ratios{ ratios: 1 ratios : 1 }
//  	  side_ratios{ ratios: 1 ratios : 2 }
//  	  side_ratios{ ratios: 2 ratios : 1 }
//  	  side_ratios{ ratios: 1 ratios : 3 }
//  	  side_ratios{ ratios: 3 ratios : 1 }
//  	}
//  	default_box{
//  	  size: 5.656854249
//  	  side_ratios{ ratios: 1 ratios : 1 }
//  	}
//      variances : 1.0
//  	variances : 2.0
//  	variances : 3.0
//  	variances : 4.0
//  	share_location : true
//  	num_classes : 3
//    }
//    param {
//      lr_mult: 1
//      decay_mult : 1
//    }
//    param{
//  	lr_mult: 2
//      decay_mult : 0
//    }
//    convolution_param {
//      kernel_size: 3
//  	stride : 1
//      pad : 1
//      weight_filler{
//  	  type: "xavier"
//  	  std : 0.1
//  	}
//  	bias_filler {
//  	  type: "constant"
//  	  value : 0.2
//  	}
//    }
//  }

int main() {
//	caffe::LayerParameter layer_param;
//	caffe::SSDParameter *ssd_param = layer_param.mutable_ssd_param();
//	ssd_param->add_variances(1);
//	ssd_param->add_variances(2);
//	ssd_param->add_variances(3);
//	ssd_param->add_variances(4);
//
//	caffe::DefaultBox *defalut_box = ssd_param->add_default_box();
//	defalut_box->set_size(4);
//	caffe::SideRatio *side_ratio = defalut_box->add_side_ratios();
//	side_ratio->add_ratios(1); side_ratio->add_ratios(1);
//	side_ratio = defalut_box->add_side_ratios();
//	side_ratio->add_ratios(1); side_ratio->add_ratios(2);
//	side_ratio = defalut_box->add_side_ratios();
//	side_ratio->add_ratios(1); side_ratio->add_ratios(3);
//	side_ratio = defalut_box->add_side_ratios();
//	side_ratio->add_ratios(2); side_ratio->add_ratios(1);
//	side_ratio = defalut_box->add_side_ratios();
//	side_ratio->add_ratios(3); side_ratio->add_ratios(1);
//
//	defalut_box = ssd_param->add_default_box();
//	defalut_box->set_size(5.656854249);
//	side_ratio = defalut_box->add_side_ratios();
//	side_ratio->add_ratios(1); side_ratio->add_ratios(1);
//
//	ssd_param->add_variances(1.0);
//	ssd_param->add_variances(2.0);
//	ssd_param->add_variances(3.0);
//	ssd_param->add_variances(4.0);
//
//	caffe::ParamSpec *param = layer_param.add_param();
//	param->set_lr_mult(1);
//	param->set_decay_mult(1);
//	param = layer_param.add_param();
//	param->set_lr_mult(2);
//	param->set_decay_mult(0);
//
//	caffe::ConvolutionParameter *conv_param = layer_param.mutable_convolution_param();
//	conv_param->add_kernel_size(3); conv_param->add_kernel_size(3);
//	conv_param->add_pad(1); conv_param->add_pad(1);
//	conv_param->add_stride(1); conv_param->add_stride(1);
//	caffe::FillerParameter *weight_filler = conv_param->mutable_weight_filler();
//	weight_filler->set_type("xavier");
//	weight_filler->set_std(0.1);
//	caffe::FillerParameter *bias_filler = conv_param->mutable_bias_filler();
//	bias_filler->set_type("constant");
//	bias_filler->set_std(0.2);


//	caffe::SSDLayer<float> ssd_layer(layer_param);
//	ssd_layer.blobs();

	return 0;
}