// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/tokenizer.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
//DEFINE_int32(label_channels, 1, "The channels of the label");
//DEFINE_int32(label_height, 1, "The height of the label");
//DEFINE_int32(label_width, 1, "The width of the label");
DEFINE_string(label_type, "int", "The data type of the label");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        "The ImageNet dataset for the training demo is at\n"
        "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
//int label_channels = FLAGS_label_channels;
//int label_width = FLAGS_label_width;
//int label_height = FLAGS_label_height;
  const string label_type = FLAGS_label_type;
  std::ifstream infile(argv[2]);
//std::vector<std::pair<std::string, int> > lines;
  std::vector<std::pair<std::string, std::vector<int>> > int_lines;
  std::vector<std::pair<std::string, std::vector<float>> > float_lines;
  std::string line;
//size_t pos;
  std::vector<int> int_labels;
  std::vector<float> float_labels;
  while (std::getline(infile, line)) {
//    pos = line.find_last_of(' ');
//    label = atoi(line.substr(pos + 1).c_str());
//    lines.push_back(std::make_pair(line.substr(0, pos), label));
	  std::vector<std::string> tokens;
	  boost::char_separator<char> sep(" ");
	  boost::tokenizer<boost::char_separator<char> > tok(line, sep);
	  tokens.clear();
	  std::copy(tok.begin(), tok.end(), std::back_inserter(tokens));

	  for (int i = 1; i < tokens.size(); ++i)
	  {
		  if (label_type == "int") int_labels.push_back(atoi(tokens.at(i).c_str()));
		  else if (label_type == "float") float_labels.push_back(atof(tokens.at(i).c_str()));
		  else  LOG(ERROR) << "Unknow type of label data: " << label_type;
	  }

	  if(!int_labels.empty()) int_lines.push_back(std::make_pair(tokens.at(0), int_labels));
	  else if (!float_labels.empty()) float_lines.push_back(std::make_pair(tokens.at(0), float_labels));
	  //###To clear the vector labels
	  int_labels.clear();
	  float_labels.clear();
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    if(label_type == "int") shuffle(int_lines.begin(), int_lines.end());
	else if(label_type == "float") shuffle(float_lines.begin(), float_lines.end());
  }
 
  if (label_type == "int")  LOG(INFO) << "A total of " << int_lines.size() << " images.";
  else if (label_type == "float")  LOG(INFO) << "A total of " << float_lines.size() << " images.";;

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;
  int lines_size = 0;
  if (label_type == "int") lines_size = int_lines.size();
  else if (label_type == "float") lines_size = float_lines.size();

  for (int line_id = 0; line_id < lines_size; ++line_id) {
    bool status;
    std::string enc = encode_type;
    if (encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn;
	  if (label_type == "int") fn = int_lines[line_id].first;
	  else if (label_type == "float") fn = float_lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
	if (label_type == "int")
		status = ReadImageToDatum(root_folder + int_lines[line_id].first,
			int_lines[line_id].second, resize_height, resize_width, is_color,
			enc, &datum);
	else if(label_type == "float")
		status = ReadImageToDatum(root_folder + int_lines[line_id].first,
			int_lines[line_id].second, resize_height, resize_width, is_color,
			enc, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      } else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
            << data.size();

      }
    }
    // sequential
	string key_str;
	if (label_type == "int") key_str = caffe::format_int(line_id, 8) + "_" + int_lines[line_id].first;
	else if(label_type == "float")  key_str = caffe::format_int(line_id, 8) + "_" + float_lines[line_id].first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
