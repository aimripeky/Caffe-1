#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <caffe/util/io.hpp>
#if defined(USE_LEVELDB) && defined(USE_LMDB)
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#endif

#if defined(_MSC_VER)
#include <direct.h>
#include <io.h>
#define mkdir(X, Y) _mkdir(X)
#endif

#include <stdint.h>
#include <sys/stat.h>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "boost/scoped_ptr.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"

#include <opencv2/opencv.hpp>

#if defined(USE_LEVELDB) && defined(USE_LMDB)

using namespace caffe;  // NOLINT(build/namespaces)
using boost::scoped_ptr;
using std::string;

DEFINE_string(backend, "lmdb", "The backend for storing the result");

uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}
namespace lx
{
	struct File
	{
		std::vector<std::string> Floders;
		std::vector<std::string> Names;
	};

	File FindFineName(std::string file_dir,std::string ext = std::string())
	{
		File reVal;
		std::string tmp_file_name;
		if (file_dir.empty()) return reVal;
		std::string find = file_dir;
		if (find[find.size() - 1] != '/' && find[find.size() - 1] != '\\')
			find += "/";
		find += '*';
		if (ext.empty())
			find += ".*";
		else find += ext;
		// to form search key string
		struct  _finddata_t  file_info;
		size_t hFile = (rsize_t)_findfirst(find.c_str(), &file_info);
		if (hFile == -1) return reVal;
		while (true)
		{
			if (file_info.attrib & _A_SUBDIR)
			{
				if (strcmp(file_info.name, ".") != 0 && strcmp(file_info.name, "..") != 0)
				{
					std::string tmp = file_info.name;
					reVal.Floders.push_back(tmp);
				}
			}
			else
			{
				if (strcmp(file_info.name, ".") != 0 && strcmp(file_info.name, "..") != 0)
				{
					std::string tmp = file_info.name;
					reVal.Names.push_back(tmp);
				}
			}
			if (_findnext(hFile, &file_info) != 0)
				break;
		}
		_findclose(hFile);
		return reVal;
	}
}



void convert_dataset(const char* positive_file_dir, const char* negative_file_dir,
	const char* db_path, const string& db_backend) {
	std::string pos_file_dir = positive_file_dir;
	std::string neg_file_dir = negative_file_dir;
	if (pos_file_dir[pos_file_dir.size() - 1] != '\\' && pos_file_dir[pos_file_dir.size() - 1] != '/')
		pos_file_dir += "/";
	if (neg_file_dir[neg_file_dir.size() - 1] != '\\' && neg_file_dir[pos_file_dir.size() - 1] != '/')
		neg_file_dir += "/";
	lx::File pos_file = lx::FindFineName(pos_file_dir);
	lx::File neg_file = lx::FindFineName(neg_file_dir);

	scoped_ptr<db::DB> db(db::GetDB(db_backend));
	db->Open(db_path, db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	int pos_count = 0, neg_count = 0;
	for (int i = 0; i < pos_file.Names.size(); i++)
	{
		Datum datum;
		std::vector<char> label;
		label.resize(sizeof(int));
		int *pRead = (int *)label.data();
		*pRead = 1;
		bool status = ReadImageToDatum(pos_file_dir + pos_file.Names[i], label, 40, 40, &datum);
		if (status == false) continue;
		string key_str = caffe::format_int(pos_count, 8);
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);
		pos_count++;
		if (pos_count % 200 == 0) {
			txn->Commit();
			LOG(INFO) << "Processed positive image: " << pos_count << "/" << pos_file.Names.size() << ".";
		}
	}
	if (pos_count % 200 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed positive image: " << pos_count << "/" << pos_file.Names.size() << ".";
	}

	for (int i = 0; i < neg_file.Names.size(); i++)
	{
		Datum datum;
		std::vector<char> label;
		label.resize(sizeof(int));
		int *pRead = (int *)label.data();
		*pRead = 0;
		bool status = ReadImageToDatum(neg_file_dir + neg_file.Names[i], label, 40, 40, &datum);
		if (status == false) continue;
		string key_str = caffe::format_int(neg_count + pos_count, 8);
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);
		neg_count++;
		if (neg_count % 200 == 0) {
			txn->Commit();
			LOG(INFO) << "Processed negative image: " << neg_count << "/" << neg_file.Names.size() << ".";
		}
	}
	if (neg_count % 200 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed negative image: " << neg_count << "/" << neg_file.Names.size() << ".";
	}
	db->Close();
}

int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	FLAGS_alsologtostderr = 1;

	gflags::SetUsageMessage("This script converts the bacillus dataset to\n"
		"the lmdb/leveldb format used by Caffe to load data.\n"
		"Usage:\n"
		"   convert_bacillus_data [FLAGS] input_positive_image_file input_negative_image_file "
		"output_db_file\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	const string& db_backend = FLAGS_backend;

	if (argc != 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0],
			"examples/physic_image/convert_bacillus_data");
	}
	else {
		google::InitGoogleLogging(argv[0]);
		convert_dataset(argv[1], argv[2], argv[3], db_backend);
	}
	return 0;
}
#else
	int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires LevelDB and LMDB; " <<
		"compile with USE_LEVELDB and USE_LMDB.";
}
#endif  // USE_LEVELDB and USE_LMDB
