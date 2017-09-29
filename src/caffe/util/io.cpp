#include <fcntl.h>

#if defined(_MSC_VER)
#include <io.h>
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::FileOutputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;
	using google::protobuf::io::ZeroCopyOutputStream;
	using google::protobuf::io::CodedOutputStream;
	using google::protobuf::Message;

	bool ReadProtoFromTextFile(const char* filename, Message* proto) {
		int fd = open(filename, O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << filename;
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		close(fd);
		return success;
	}

	void WriteProtoToTextFile(const Message& proto, const char* filename) {
		int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		FileOutputStream* output = new FileOutputStream(fd);
		CHECK(google::protobuf::TextFormat::Print(proto, output));
		delete output;
		close(fd);
	}

	bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
#if defined (_MSC_VER)  // for MSC compiler binary flag needs to be specified
		int fd = open(filename, O_RDONLY | O_BINARY);
#else
		int fd = open(filename, O_RDONLY);
#endif
		CHECK_NE(fd, -1) << "File not found: " << filename;
		ZeroCopyInputStream* raw_input = new FileInputStream(fd);
		CodedInputStream* coded_input = new CodedInputStream(raw_input);
		coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

		bool success = proto->ParseFromCodedStream(coded_input);

		delete coded_input;
		delete raw_input;
		close(fd);
		return success;
	}

	void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
		fstream output(filename, ios::out | ios::trunc | ios::binary);
		CHECK(proto.SerializeToOstream(&output));
	}

#ifdef USE_OPENCV
	cv::Mat ReadImageToCVMat(const string& filename,
		const int height, const int width, const bool is_color) {
		cv::Mat cv_img;
		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
			CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
		if (!cv_img_origin.data) {
			LOG(ERROR) << "Could not open or find file " << filename;
			return cv_img_origin;
		}
		if (height > 0 && width > 0) {
			cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
		}
		else {
			cv_img = cv_img_origin;
		}
		return cv_img;
	}

	cv::Mat ReadImageToCVMat(const string& filename,
		const int height, const int width) {
		return ReadImageToCVMat(filename, height, width, true);
	}

	cv::Mat ReadImageToCVMat(const string& filename,
		const bool is_color) {
		return ReadImageToCVMat(filename, 0, 0, is_color);
	}

	cv::Mat ReadImageToCVMat(const string& filename) {
		return ReadImageToCVMat(filename, 0, 0, true);
	}

	// Do the file extension and encoding match?
	static bool matchExt(const std::string & fn,
		std::string en) {
		size_t p = fn.rfind('.');
		std::string ext = p != fn.npos ? fn.substr(p) : fn;
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		std::transform(en.begin(), en.end(), en.begin(), ::tolower);
		if (ext == en)
			return true;
		if (en == "jpg" && ext == "jpeg")
			return true;
		return false;
	}

	bool ReadMatToDatum(const string& mat_name, const string& label_name, Datum* datum,
		bool is_img_file, bool is_img_label) {
		cv::Mat img, label;
		if (is_img_file) img = cv::imread(mat_name);
		else {
			CvMat *load = (CvMat *)cvLoad(mat_name.data());
			img = cv::cvarrToMat(load, true);
			cvReleaseMat(&load);
		}
		if (is_img_label) img = cv::imread(label_name);
		else {
			CvMat *load = (CvMat *)cvLoad(mat_name.data());
			label = cv::cvarrToMat(load, true);
			cvReleaseMat(&load);
		}
		if (img.empty() || label.empty())
			return false;
		CVMatToDatum(img, datum);
		CVMatToDatum(label, datum, true);
		return true;
	}

	bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
		const int height, const int width, const bool is_color,
		const std::string & encoding, Datum* datum,
		DatumShape::DataDepth label_depth) {

		int label_elem_size = 0;
		DatumShape *label_shape = new DatumShape;
		if (label_depth == DatumShape::DEPTH_8S) {
			label_shape->set_data_depth(DatumShape::DEPTH_8S);
			label_elem_size = sizeof(char);
		}
		else if (label_depth == DatumShape::DEPTH_8U) {
			label_shape->set_data_depth(DatumShape::DEPTH_8U);
			label_elem_size = sizeof(unsigned char);
		}
		else if (label_depth == DatumShape::DEPTH_16S) {
			label_shape->set_data_depth(DatumShape::DEPTH_16S);
			label_elem_size = sizeof(short);
		}
		else if (label_depth == DatumShape::DEPTH_16U) {
			label_shape->set_data_depth(DatumShape::DEPTH_16U);
			label_elem_size = sizeof(unsigned short);
		}
		else if (label_depth == DatumShape::DEPTH_32S) {
			label_shape->set_data_depth(DatumShape::DEPTH_32S);
			label_elem_size = sizeof(int);
		}
		else if (label_depth == DatumShape::DEPTH_32F) {
			label_shape->set_data_depth(DatumShape::DEPTH_32F);
			label_elem_size = sizeof(float);
		}
		else if (label_depth == DatumShape::DEPTH_64F) {
			label_shape->set_data_depth(DatumShape::DEPTH_64F);
			label_elem_size = sizeof(double);
		}
		else LOG(FATAL) << "Unknow deth of label!";

		label_shape->set_channels(label.size() / label_elem_size);
		label_shape->set_width(1);
		label_shape->set_height(1);

		cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
		if (cv_img.data) {
			if (encoding.size()) {
				if ((cv_img.channels() == 3) == is_color && !height && !width &&
					matchExt(filename, encoding)) {
					return ReadFileToDatum(filename, label, datum);
					delete label_shape;
				}
				std::vector<uchar> buf;
				cv::imencode("." + encoding, cv_img, buf);
				datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
					buf.size()));
				datum->set_encoded(true);
				datum->set_allocated_label_shape(label_shape);
				datum->set_label(label.data(), label.size());
				return true;
			}
			CVMatToDatum(cv_img, datum);
			datum->set_allocated_label_shape(label_shape);
			datum->set_label(label.data(), label.size());
			return true;
		}
		else {
			return false;
		}
	}

#endif  // USE_OPENCV
	//  
	bool ReadFileToDatum(const string& filename, const std::vector<char>& label, Datum* datum,
		DatumShape::DataDepth label_depth) {
		std::streampos size;
		fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
		if (file.is_open()) {
			size = file.tellg();
			std::string buffer(size, ' ');
			file.seekg(0, ios::beg);
			file.read(&buffer[0], size);
			file.close();
			datum->set_data(buffer);

			int label_elem_size = 0;
			DatumShape *label_shape = new DatumShape;
			if (label_depth == DatumShape::DEPTH_8S) {
				label_shape->set_data_depth(DatumShape::DEPTH_8S);
				label_elem_size = sizeof(char);
			}
			else if (label_depth == DatumShape::DEPTH_8U) {
				label_shape->set_data_depth(DatumShape::DEPTH_8U);
				label_elem_size = sizeof(unsigned char);
			}
			else if (label_depth == DatumShape::DEPTH_16S) {
				label_shape->set_data_depth(DatumShape::DEPTH_16S);
				label_elem_size = sizeof(short);
			}
			else if (label_depth == DatumShape::DEPTH_16U) {
				label_shape->set_data_depth(DatumShape::DEPTH_16U);
				label_elem_size = sizeof(unsigned short);
			}
			else if (label_depth == DatumShape::DEPTH_32S) {
				label_shape->set_data_depth(DatumShape::DEPTH_32S);
				label_elem_size = sizeof(int);
			}
			else if (label_depth == DatumShape::DEPTH_32F) {
				label_shape->set_data_depth(DatumShape::DEPTH_32F);
				label_elem_size = sizeof(float);
			}
			else if (label_depth == DatumShape::DEPTH_64F) {
				label_shape->set_data_depth(DatumShape::DEPTH_64F);
				label_elem_size = sizeof(double);
			}
			else LOG(FATAL) << "Unknow deth of label!";
			label_shape->set_channels(label.size() / label_elem_size);
			label_shape->set_width(1);
			label_shape->set_height(1);
			datum->set_allocated_label_shape(label_shape);
			datum->set_label(label.data(), label.size());
			datum->set_encoded(true);
			return true;
		}
		else {
			return false;
		}
	}
	//  
	//  bool ReadFileToDatum(const string& filename, const std::vector<int> labels,
	//  	Datum* datum) {
	//  	std::streampos size;
	//  
	//  	fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
	//  	if (file.is_open()) {
	//  		size = file.tellg();
	//  		std::string buffer(size, ' ');
	//  		file.seekg(0, ios::beg);
	//  		file.read(&buffer[0], size);
	//  		file.close();
	//  		datum->set_data(buffer);
	//  		datum->clear_label_int_data();
	//  		for (size_t i = 0; i < labels.size(); i++)
	//  			datum->add_label_int_data(labels[i]);
	//  		datum->set_encoded(true);
	//  		return true;
	//  	}
	//  	else {
	//  		return false;
	//  	}
	//  }
	//  
	//bool ReadFileToDatum(const string& filename, const std::vector<float> labels,
	//	Datum* datum) {
	//	std::streampos size;
	//
	//	fstream file(filename.c_str(), ios::in | ios::binary | ios::ate);
	//	if (file.is_open()) {
	//		size = file.tellg();
	//		std::string buffer(size, ' ');
	//		file.seekg(0, ios::beg);
	//		file.read(&buffer[0], size);
	//		file.close();
	//		datum->set_data(buffer);
	//		datum->clear_label_float_data();
	//		for (size_t i = 0; i < labels.size(); i++)
	//			datum->add_label_float_data(labels[i]);
	//		datum->set_encoded(true);
	//		return true;
	//	}
	//	else {
	//		return false;
	//	}
	//}

#ifdef USE_OPENCV
	cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
		cv::Mat cv_img;
		CHECK(datum.encoded()) << "Datum not encoded";
		const string& data = datum.data();
		std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
		cv_img = cv::imdecode(vec_data, -1);
		if (!cv_img.data) {
			LOG(ERROR) << "Could not decode datum ";
		}
		return cv_img;
	}
	cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
		cv::Mat cv_img;
		CHECK(datum.encoded()) << "Datum not encoded";
		const string& data = datum.data();
		std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
		int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
			CV_LOAD_IMAGE_GRAYSCALE);
		cv_img = cv::imdecode(vec_data, cv_read_flag);
		if (!cv_img.data) {
			LOG(ERROR) << "Could not decode datum ";
		}
		return cv_img;
	}

	// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
	// If Datum is not encoded will do nothing
	bool DecodeDatumNative(Datum* datum) {
		if (datum->encoded()) {
			cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
			CVMatToDatum(cv_img, datum);
			return true;
		}
		else {
			return false;
		}
	}
	bool DecodeDatum(Datum* datum, bool is_color) {
		if (datum->encoded()) {
			cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
			CVMatToDatum(cv_img, datum);
			return true;
		}
		else {
			return false;
		}
	}

	void CVMatToDatum(const cv::Mat& cv_img, Datum* datum, bool is_label) {
		//CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
		DatumShape *shape = new DatumShape;
		int elem_size = 0;
		if (cv_img.depth() == CV_8S) {
			shape->set_data_depth(DatumShape::DEPTH_8S);
			elem_size = sizeof(char);
		}
		else if (cv_img.depth() == CV_8U) {
			shape->set_data_depth(DatumShape::DEPTH_8U);
			elem_size = sizeof(unsigned char);
		}
		else if (cv_img.depth() == CV_16S) {
			shape->set_data_depth(DatumShape::DEPTH_16S);
			elem_size = sizeof(short);
		}
		else if (cv_img.depth() == CV_16U) {
			shape->set_data_depth(DatumShape::DEPTH_16U);
			elem_size = sizeof(unsigned short);
		}
		else if (cv_img.depth() == CV_32S) {
			shape->set_data_depth(DatumShape::DEPTH_32S);
			elem_size = sizeof(int);
		}
		else if (cv_img.depth() == CV_32F) {
			shape->set_data_depth(DatumShape::DEPTH_32F);
			elem_size = sizeof(float);
		}
		else if (cv_img.depth() == CV_64F) {
			shape->set_data_depth(DatumShape::DEPTH_64F);
			elem_size = sizeof(double);
		}
		else LOG(FATAL) << "Unknow deth of image!";

		shape->set_channels(cv_img.channels());
		shape->set_height(cv_img.rows);
		shape->set_width(cv_img.cols);
		if (is_label) datum->set_allocated_label_shape(shape);
		else datum->set_allocated_data_shape(shape);

		datum->set_encoded(false);
		int datum_channels = shape->channels();
		int datum_height = shape->height();
		int datum_width = shape->width();
		int datum_size = datum_channels * datum_height * datum_width * elem_size;
		std::string buffer(datum_size, ' ');
		for (int h = 0; h < datum_height; h++) {
			for (int w = 0; w < datum_width; w++) {
				for (int c = 0; c < datum_channels; c++) {
					for (int e = 0; e < elem_size; e++) {
						int img_index = ((h * datum_width + w) * datum_channels + c) * elem_size + e;
						int datum_index = ((c * datum_height + h) * datum_width + w) * elem_size + e;
						buffer[datum_index] = cv_img.data[img_index];
					}
				}
			}
		}
		if (is_label) datum->set_label(buffer);
		else datum->set_data(buffer);
	}
#endif  // USE_OPENCV
} // namespace caffe
