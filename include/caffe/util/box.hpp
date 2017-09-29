#ifndef CAFFE_UTIL_BOX_H_
#define CAFFE_UTIL_BOX_H_
#include <map>
#include <list>
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
using std::list;
namespace caffe {
	template  <typename Dtype>
	class Box<Dtype>;
	template  <typename Dtype>
	class GTBox<Dtype>;
	template <typename Dtype>
	class DefBox<Dtype>;
	template <typename Dtype>
	class DRBox<Dtype>;

	template <typename Dtype>
	struct Box {
		vector<const Dtype*> min_location_;
		vector<const Dtype*> max_location_;
	};

	//Ground True Box
	template <typename Dtype>
	struct GTBox: public Box<Dtype>{
		const Dtype *lable_;
		map<const DefBox<Dtype> *, Dtype> def_boxes_;
	};

	//Default Box
	template <typename Dtype>
	struct DefBox : public Box<Dtype>{
		map<const GTBox<Dtype> *, Dtype> gt_boxes_;
		vector<const DRBox<Dtype>*> dr_boxes_;
		vector<const Dtype*> min_variances_;
		vector<const Dtype*> max_variances_;
	};

	//Decation Result Box
	template <typename Dtype>
	struct DRBox : public Box<Dtype>{
		const DefBox<Dtype> *def_box_;
		const Dtype* conf_;
	};



	template <typename Dtype>
	void ExtractGroundTruth(const Blob<Dtype> *label, const Blob<Dtype> *location,
		const int box_dimensions, vector<vector<GTBox<Dtype>>> &boxes);

	//@input data:
	//  if shared_location is true, the class_id = 0;
	//  def_location: K1 K2 ... KN
	//  pred_location: 
	//    share_location = true:  K1 K2 ... KN.
	//    share_location = false: K1C1 K1C2 ... K1CN K2C1 K2C2 ... K3CN ...... KMC1 KMC2 ... KMCM.
	//  pred_confidence: K1C1 K1C2 ... K1CN K2C1 K2C2 ... K3CN ...... KMC1 KMC2 ... KMCM.
	//@return data:
	// dr_boxes[image_id][class_id][box_id]
	template <typename Dtype>
	void ExtractPredBoxes(const Blob<Dtype> *def_location, const Blob<Dtype> *pred_location,
		const Blob<Dtype> *pred_confidence, const int box_dimensions,
		const bool share_location, const int num_classes, int &num,
		vector<vector<DefBox<Dtype>>> &def_boxes,
		vector<vector<vector<DRBox<Dtype>>>> &dr_boxes);

	template <typename Dtype>
	Dtype JaccardOverlap(const Box<Dtype> *box1, const Box<Dtype> *box2);


	template <typename Dtype>
	void JaccardOverlaps(const vector<Box<Dtype>> *gt_boxes, const vector<vector<Box<Dtype>>> *pred_boxes,
		const MultiBoxLossParameter_MatchType match_type, const bool share_location, const int background_label,
		vector<map<int, map<const Box<Dtype> *, map<const Box<Dtype>*, Dtype>>>> &overlaps);
	
	template <typename Dtype>
	void MathBoxes(const vector<map<int, map<const Box<Dtype> *, map<const Box<Dtype>*, Dtype>>>> *overlaps,
		const MultiBoxLossParameter_MatchType match_type, const float overlap_threshold,
		map<const Box<Dtype> *, const Box<Dtype> *> &match_boxes);

	template <typename Dtype>
	void ExtractNegtiveBox(const vector<vector<Box<Dtype>>> *pred_boxes,
		const bool share_location, const int background_label,
		list<pair<const int, const Box<Dtype> *>> &negatives);

}
#endif