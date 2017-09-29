#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <boost/filesystem.hpp>
#include <iomanip>
#include <iostream>  // NOLINT(readability/streams)
#include <string>

#include "google/protobuf/message.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"

#ifndef CAFFE_TMP_DIR_RETRIES
#define CAFFE_TMP_DIR_RETRIES 100
#endif

namespace caffe {

using ::google::protobuf::Message;
using ::boost::filesystem::path;

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
  // Place all temp directories under temp_root, to be able to delete all of
  // them at once, without knowing their name.
  const path& temp_root =
    boost::filesystem::temp_directory_path() / "caffe_test";
  boost::filesystem::create_directory(temp_root);
  const path& model = temp_root / "%%%%-%%%%";
  for ( int i = 0; i < CAFFE_TMP_DIR_RETRIES; i++ ) {
    const path& dir = boost::filesystem::unique_path(model).string();
    bool done = boost::filesystem::create_directory(dir);
    if ( done ) {
      *temp_dirname = dir.string();
      return;
    }
  }
  LOG(FATAL) << "Failed to create a temporary directory.";
}

inline void MakeTempFilename(string* temp_filename) {
  path temp_files_subpath;
  static uint64_t next_temp_file = 0;
  temp_filename->clear();
  if ( temp_files_subpath.empty() ) {
    string path_string="";
    MakeTempDir(&path_string);
    temp_files_subpath = path_string;
  }
  *temp_filename =
    (temp_files_subpath/caffe::format_int(next_temp_file++, 9)).string();
}

#ifdef _MSC_VER

inline void RemoveCaffeTempDir() {
  boost::system::error_code err;
  boost::filesystem::remove_all(
    boost::filesystem::temp_directory_path() / "caffe_test", err);
}

#else

inline void RemoveCaffeTempDir() {
}

#endif

bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadFileToDatum(const string& filename, const std::vector<char>& label, Datum* datum, 
	DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
	std::vector<char> label;
	label.resize(sizeof(int));
	int *p_data = (int *)(&label[0]);
	*p_data = -1;
	return ReadFileToDatum(filename, label, datum);
}

bool ReadMatToDatum(const string& filename, const string& label_name, Datum* datum,
	bool is_img_file = true, bool is_img_label = true);

bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum,
	DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S);

inline bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
    const int height, const int width, const bool is_color, Datum* datum,
	DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S) {
	return ReadImageToDatum(filename, label, height, width, is_color,
		"", datum, label_depth);
}

inline bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
    const int height, const int width, Datum* datum,
	DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S) {
	return ReadImageToDatum(filename, label, height, width, true, datum, label_depth);
}

inline bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
    const bool is_color, Datum* datum,
	DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S) {
	return ReadImageToDatum(filename, label, 0, 0, is_color, datum, label_depth);
}

inline bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
    Datum* datum, DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S) {
	return ReadImageToDatum(filename, label, 0, 0, true, datum, label_depth);
}

inline bool ReadImageToDatum(const string& filename, const std::vector<char>& label,
    const std::string & encoding, Datum* datum,
	DatumShape::DataDepth label_depth = DatumShape::DEPTH_32S) {
	return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum, label_depth);
}

//bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
//	const int height, const int width, const bool is_color,
//	const std::string & encoding, Datum* datum);
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
//	const int height, const int width, const bool is_color, Datum* datum) {
//	return ReadImageToDatum(filename, labels, height, width, is_color,
//		"", datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
//	const int height, const int width, Datum* datum) {
//	return ReadImageToDatum(filename, labels, height, width, true, datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
//	const bool is_color, Datum* datum) {
//	return ReadImageToDatum(filename, labels, 0, 0, is_color, datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
//	Datum* datum) {
//	return ReadImageToDatum(filename, labels, 0, 0, true, datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
//	const std::string & encoding, Datum* datum) {
//	return ReadImageToDatum(filename, labels, 0, 0, true, encoding, datum);
//}
//
//bool ReadImageToDatum(const string& filename, const std::vector<float> labels,
//	const int height, const int width, const bool is_color,
//	const std::string & encoding, Datum* datum);
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<float> labels,
//	const int height, const int width, const bool is_color, Datum* datum) {
//	return ReadImageToDatum(filename, labels, height, width, is_color,
//		"", datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<float> labels,
//	const int height, const int width, Datum* datum) {
//	return ReadImageToDatum(filename, labels, height, width, true, datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<float> labels,
//	const bool is_color, Datum* datum) {
//	return ReadImageToDatum(filename, labels, 0, 0, is_color, datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<float> labels,
//	Datum* datum) {
//	return ReadImageToDatum(filename, labels, 0, 0, true, datum);
//}
//
//inline bool ReadImageToDatum(const string& filename, const std::vector<float> labels,
//	const std::string & encoding, Datum* datum) {
//	return ReadImageToDatum(filename, labels, 0, 0, true, encoding, datum);
//}

bool DecodeDatumNative(Datum* datum);

bool DecodeDatum(Datum* datum, bool is_color);

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

cv::Mat ReadImageToCVMat(const string& filename);

cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum, bool is_label = false);
#endif  // USE_OPENCV

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
