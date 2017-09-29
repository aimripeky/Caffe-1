// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.
#ifndef CAFFE_TEST_TEST_CAFFE_MAIN_HPP_
#define CAFFE_TEST_TEST_CAFFE_MAIN_HPP_

#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/ssd_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
	std::string model_file = "B:\Caffe\ssd.prototxt";
	std::shared_ptr<caffe::Net<float>> net;
//	net.reset(new caffe::Net<float>(model_file, TEST));


}  // namespace caffe


#endif 
