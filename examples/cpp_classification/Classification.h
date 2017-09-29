#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
class    Classification{
public:
	void SetBatchSize(int batch_size);
	void SetMeans(const std::shared_ptr<vector<float>> mean_vals);
	void SetScale(float scale) {
		scale_ = scale;
	}
	const std::vector<std::vector<float>> *Classify(const std::vector<cv::Mat> &inputs);
	~Classification(){};
	Classification(const string& model_file, const string& trian_file);
private:
	void Setup();
	std::vector<cv::Mat> means_;
	std::shared_ptr<Net<float>> net_;
	std::vector<std::vector<cv::Mat>> inputs_;
	float scale_;
	cv::Size size_;
	int channels_;
	int num_;
	std::vector<std::vector<float>> results_;
};