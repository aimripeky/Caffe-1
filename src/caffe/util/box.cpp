#include "caffe/util/box.hpp"


namespace caffe {

	template <typename Dtype>
	void ExtractGroundTruth(const Blob<Dtype> *label, const Blob<Dtype> *location,
		const int box_dimensions, vector<vector<GTBox<Dtype>>> &boxes){
		const int num_axes = label->num_axes();
		CHECK_EQ(num_axes, location->num_axes());
		CHECK_GE(num_axes, 3);
		const int num = label->shape(0);
		CHECK_EQ(num, location->shape(0));
		int channel_offset = 1;
		CHECK_EQ(label->shape(1), 1);
		for (int i = 2; i < num_axes; ++i) {
			int shape = location->shape(i);
			CHECK_EQ(shape, label->shape(i));
			channel_offset *= shape;
		}
		if (boxes.size() != num)
			boxes.resize(num);

		const Dtype *label_data = label->cpu_data();
		for (int n = 0; n < num; ++n) {
			for (int i = 0; i < channel_offset; ++i) {
				GTBox<Dtype> box;
				box.lable_ = &(label_data[i]);
				boxes[n].push_back(box);
			}
		}

		const Dtype *location_data = location->cpu_data();
		CHECK_EQ(location->shape(1) % 2, 0);
		CHECK_EQ(box_dimensions, location->shape(1) / 2);
		for (int n = 0; n < num; ++n) {
			for (int i = 0; i < channel_offset; ++i) {
				GTBox<Dtype> &box = boxes[n][i];
				for (size_t d = 0; d < box_dimensions; ++d) {
					int index = d*channel_offset + i;
					box.min_location_.push_back(&(location_data[index]));
				}
				for (size_t d = box_dimensions; d < 2 * box_dimensions; ++d) {
					int index = d*channel_offset + i;
					box.max_location_.push_back(&(location_data[index]));
				}
			}
		}
	}

	template void ExtractGroundTruth(const Blob<float> *label, const Blob<float> *location,
		const int box_dimensions, vector<vector<GTBox<float>>> &boxes);
	template void ExtractGroundTruth(const Blob<double> *label, const Blob<double> *location,
		const int box_dimensions, vector<vector<GTBox<double>>> &boxes);

	template <typename Dtype>
	void ExtractPredBoxes(const Blob<Dtype> *def_location, const Blob<Dtype> *pred_location,
		const Blob<Dtype> *pred_confidence, const int box_dimensions,
		const bool share_location, const int num_classes, int &num,
		vector<vector<DefBox<Dtype>>> &def_boxes,
		vector<vector<vector<DRBox<Dtype>>>> &dr_boxes) {
		const int num_axes = def_location->num_axes();
		CHECK_EQ(num_axes, pred_location->num_axes());
		CHECK_EQ(num_axes, pred_confidence->num_axes());
		CHECK_GE(num_axes, 3);
		num = pred_location->shape(0);
		CHECK_EQ(def_location->shape(0), 2);
		CHECK_EQ(pred_confidence->shape(0), num);

		const int def_loc_channels = def_location->shape(1);
		const int pred_loc_channels = pred_location->shape(1);
		const int pred_conf_channels = pred_confidence->shape(1);
		CHECK_EQ(def_loc_channels % (box_dimensions * 2), 0);
		const int shape_kinds = def_loc_channels / (box_dimensions * 2);
		CHECK_EQ(share_location, (def_loc_channels == pred_loc_channels));
		CHECK_EQ(pred_loc_channels%shape_kinds, 0);
		CHECK_EQ(num_classes, pred_conf_channels / shape_kinds);
		if (!share_location){
			CHECK_EQ(0, pred_loc_channels % def_loc_channels);
			CHECK_EQ(num_classes, pred_loc_channels / def_loc_channels);
		}
		int channel_offset = 1;
		for (int i = 2; i < num_axes; ++i){
			const int shape = def_location->shape(i);
			CHECK_EQ(shape, pred_location->shape(i));
			CHECK_EQ(shape, pred_confidence->shape(i));
			channel_offset *= shape;
		}
		if (!def_boxes.empty()){
			def_boxes.resize(num);
		}
		else{
			CHECK_EQ(def_boxes.size(), num);
		}
		if (!dr_boxes.empty()){
			dr_boxes.resize(num);
		}
		else{
			CHECK_EQ(dr_boxes.size(), num);
		}
		//def_location: K1 K2 ... KN
		//pred_location: 
		//  share_location = true:  LK1 LK2 ... LKN.
		//  share_location = false: LK1C1 LK1C2 ... LK1CN LK2C1 LK2C2 ... LK2CN ...... LKMC1 LKMC2 ... LKMCM.
		//  note:
		//    LK: Location of each kinds of box shape. 
		//        Example of 2D: 
		//          Xmin Ymin Xmax Ymax;
		//     C: The classification of each kinds of box shape.
		//pred_confidence: K1C1 K1C2 ... K1CN K2C1 K2C2 ... K2CN ...... KMC1 KMC2 ... KMCM.
		const int loc_classes = share_location ? 1 : num_classes;
		// Ki -> Ki+1 offset

		const int loc_kinds_offset = 2 * box_dimensions * loc_classes * channel_offset;
		const int conf_kinds_offset = num_classes*channel_offset;
		for (int n = 0; n < num; ++n) {
			if (dr_boxes[n].empty()){
				dr_boxes[n].resize(num_classes);
			}
			else{
				CHECK_EQ(num_classes, dr_boxes[n].size());
			}
		}
		const Dtype *def_loc_data = def_location->cpu_data();
		const Dtype *pred_loc_data = pred_location->cpu_data();
		const Dtype *pred_conf_data = pred_confidence->cpu_data();
		const Dtype *variance_data = def_loc_data + def_location->offset(vector<int>(1, 1));
		for (int n = 0; n < num; ++n) {
			for (int i = 0; i < channel_offset; ++i) {
				for (int k = 0; k < shape_kinds; ++k) {
					DefBox<Dtype> def_box;
					const int start = 2 * k * box_dimensions;
					const int middle = (2 * k + 1)*box_dimensions;
					const int end = 2 * (k + 1)*box_dimensions;
					for (int d = start; d < middle; ++d) {
						int box_index = d*channel_offset + i;
						def_box.min_location_.push_back(&def_loc_data[box_index]);
						def_box.min_variances_.push_back(&variance_data[box_index]);
					}
					for (int d = middle; d < end; ++d) {
						int box_index = d*channel_offset + i;
						def_box.max_location_.push_back(&def_loc_data[box_index]);
						def_box.max_variances_.push_back(&variance_data[box_index]);
					}
					int last_def_box_id = def_boxes[num].size();
					def_boxes[num].push_back(def_box);
					for (int c = 0; c < num_classes; ++c) {
						DRBox<Dtype> dr_box;
						int loc_class_id = share_location ? 0 : c;
						for (int d = start; d < middle; ++d) {
							int box_index = d*channel_offset + i;
							int loc_index = k*loc_kinds_offset + loc_class_id*channel_offset + box_index;
							dr_box.min_location_.push_back(&pred_loc_data[loc_index]);
						}
						for (int d = middle; d < end; ++d) {
							int box_index = d*channel_offset + i;
							int loc_index = k*loc_kinds_offset + loc_class_id*channel_offset + box_index;
							dr_box.max_location_.push_back(&pred_loc_data[loc_index]);
						}
						const int conf_index = k*conf_kinds_offset + c*channel_offset + i;
						dr_box.conf_ = &pred_conf_data[conf_index];
						dr_box.def_box_ = &def_boxes[num][last_def_box_id];
						int last_dr_box_id = dr_boxes.size();
						dr_boxes[num][c].push_back(dr_box);
						def_boxes[num][last_def_box_id].dr_boxes_.push_back(&dr_boxes[num][c][last_dr_box_id]);
					}
				}
			}
		}
	}

	template void ExtractPredBoxes(const Blob<float> *def_location, const Blob<float> *pred_location,
		const Blob<float> *pred_confidence, const int box_dimensions,
		const bool share_location, const int num_classes, int &num,
		vector<vector<DefBox<float>>> &def_boxes,
		vector<vector<vector<DRBox<float>>>> &dr_boxes);

	template void ExtractPredBoxes(const Blob<double> *def_location, const Blob<double> *pred_location,
		const Blob<double> *pred_confidence, const int box_dimensions,
		const bool share_location, const int num_classes, int &num,
		vector<vector<DefBox<double>>> &def_boxes,
		vector<vector<vector<DRBox<double>>>> &dr_boxes);

	template <typename Dtype>
	Dtype JaccardOverlap(const Box<Dtype> *box1, const Box<Dtype> *box2){
		const auto box_dimensions_ = box1->min_location_.size();
		CHECK_EQ(box1->max_location_.size(), box_dimensions_);
		CHECK_EQ(box2->min_location_.size(), box_dimensions_);
		CHECK_EQ(box2->max_location_.size(), box_dimensions_);
		vector<Dtype> len_overlap(box_dimensions_, 0);
		vector<Dtype> len_gt_box(box_dimensions_, 0);
		vector<Dtype> len_def_box(box_dimensions_, 0);
		for (int i = 0; i < box_dimensions_; i++) {
			len_overlap[i] = min(*box1->max_location_[i], *box2->max_location_[i])
				- max(*box1->min_location_[i], *box2->min_location_[i]);
			if (len_overlap[i] <= 0) {
				return 0;
			}
			len_gt_box[i] = *box1->max_location_[i] - *box1->min_location_[i];
			len_def_box[i] = *box2->max_location_[i] - *box2->min_location_[i];
		}
		Dtype overlap_area = 1;
		Dtype gt_box_area = 1;
		Dtype def_box_area = 1;
		for (int i = 0; i < box_dimensions_; i++) {
			overlap_area *= len_overlap[i];
			gt_box_area *= len_gt_box[i];
			def_box_area *= len_def_box[i];
		}
		return overlap_area / (gt_box_area + def_box_area - overlap_area);
	}

	template float JaccardOverlap(const Box<float> *box1, const Box<float> *box2);
	template double JaccardOverlap(const Box<double> *box1, const Box<double> *box2);


	template <typename Dtype>
	void JaccardOverlaps(const vector<Box<Dtype>> *gt_boxes, const vector<vector<Box<Dtype>>> *pred_boxes,
		const MultiBoxLossParameter_MatchType match_type, const bool share_location, const int background_label,
		vector<map<int, map<const Box<Dtype> *, map<const Box<Dtype>*, Dtype>>>> &overlaps) {
		overlaps.resize(2);
		const int gt_size = gt_boxes->size();
		for (size_t i = 0; i < gt_size; i++) {
			const Box<Dtype> *gt_box = &gt_boxes->at(i);
			const int label = *gt_box->lable_;
			if (label >= 0){
				CHECK_NE(label, background_label);
				if (share_location){
					const int num_classes = pred_boxes->size();
					for (size_t c = 0; c < num_classes; c++) {
						const auto &loc_boxes = pred_boxes->at(c);
						const int loc_size = loc_boxes.size();
						for (size_t j = 0; j < loc_size; j++) {
							const Box<Dtype> *pred_box = &loc_boxes[i];
							Dtype overlap = JaccardOverlap(gt_box, pred_box);
							overlaps[0][c][gt_box][pred_box] = overlap;
							overlaps[1][c][pred_box][gt_box] = overlap;
						}
					}
				}

			}
		}
	}

	template void JaccardOverlaps(const vector<Box<float>> *gt_boxes, const vector<vector<Box<float>>> *pred_boxes,
		const MultiBoxLossParameter_MatchType match_type, const bool share_location, const int background_label,
		vector<map<int, map<const Box<float> *, map<const Box<float>*, float>>>> &overlaps);

	template void JaccardOverlaps(const vector<Box<double>> *gt_boxes, const vector<vector<Box<double>>> *pred_boxes,
		const MultiBoxLossParameter_MatchType match_type, const bool share_location, const int background_label,
		vector<map<int, map<const Box<double> *, map<const Box<double>*, double>>>> &overlaps);

	template <typename Dtype>
	void MathBoxes(const vector<map<int, map<const Box<Dtype> *, map<const Box<Dtype>*, Dtype>>>> *overlaps,
		const MultiBoxLossParameter_MatchType match_type, const float overlap_threshold,
		map<const Box<Dtype> *, const Box<Dtype> *> &match_boxes) {
		if (match_type == MultiBoxLossParameter_MatchType_BIPARTITE){
			//Find max overlap between gt_boxes and pred_boxes.
			//Each of gt_boxes only have one pred_boxes,and each of pred_boxes aslo have only one gt_boxes.
			for (const auto &overlap : overlaps->at(0)){
				list<const Box<Dtype> *> gt_boxes;
				for (auto gt_iter = overlap.second.begin(); gt_iter != overlap.second.end(); ++gt_iter) {
					gt_boxes.push_back(gt_iter->first);
				}
				while (!gt_boxes.empty()) {
					list<const Box<Dtype> *>::iterator max_gt_iter;
					const Box<Dtype> * max_pred_box = nullptr;
					Dtype max_overlap = 0;
					for (auto gt_iter = gt_boxes.begin(); gt_iter != gt_boxes.end(); ++gt_iter) {
						for (auto gt_box = gt_boxes.begin(); gt_box != gt_boxes.end(); ++gt_box) {
							const auto pred_boxes = overlap.second.find(*gt_box);
							if (pred_boxes == overlap.second.end()){
								LOG(FATAL) << "Caculate overlap function have error.";
								break;
							};
							for (const auto &pred_box : pred_boxes->second){
								if (match_boxes.find(pred_box.first) != match_boxes.end()){
									continue;
								}
								const Dtype value = overlap.second.at(*gt_box).at(pred_box.first);
								if (value >= max_overlap){
									max_overlap = value;
									max_gt_iter = gt_iter;
									max_pred_box = pred_box.first;
								}
							}
						}
					}
					CHECK_GE(max_overlap, static_cast<float>(overlap_threshold));
					match_boxes[max_pred_box] = *max_gt_iter;
					gt_boxes.erase(max_gt_iter);
				}
			}
		}
		else if (match_type == MultiBoxLossParameter_MatchType_PER_PREDICTION){
			//Find max overlap between pred_boxes and gt_boxes.
			//Each of gt_boxes only have mulit pred_boxes if overlap is greater than overlap threshold,
			//and each of pred_boxes have only one gt_boxes which the overlap is maximum.
			for (const auto &overlap : overlaps->at(1)){
				for (const auto &pred_box : overlap.second) {
					//find max_overlap
					const Box<Dtype> *max_gt_box = nullptr;
					Dtype max_overlap = 0;
					for (const auto &gt_box : pred_box.second) {
						if (gt_box.second > max_overlap) {
							max_overlap = gt_box.second;
							max_gt_box = gt_box.first;
						}
					}
					if (max_overlap >= static_cast<Dtype>(overlap_threshold)){
						match_boxes[pred_box.first] = max_gt_box;
					}
				}
			}

		}
		else {
			LOG(FATAL) << "Unknown matching type.";
		}
	}

	template void MathBoxes(const vector<map<int, map<const Box<float> *, map<const Box<float>*, float>>>> *overlaps,
		const MultiBoxLossParameter_MatchType match_type, const float overlap_threshold,
		map<const Box<float> *, const Box<float> *> &match_boxes);

	template void MathBoxes(const vector<map<int, map<const Box<double> *, map<const Box<double>*, double>>>> *overlaps,
		const MultiBoxLossParameter_MatchType match_type, const float overlap_threshold,
		map<const Box<double> *, const Box<double> *> &match_boxes);


	template <typename Dtype>
	void ExtractNegtiveBox(const vector<vector<Box<Dtype>>> *pred_boxes,
		const bool share_location, const int background_label,
		list<pair<const int, const Box<Dtype> *>> &negatives) {
		list<pair<const Box<Dtype> *, pair<const int, Dtype>>> max_conf_box;
		const int num_classes = static_cast<int>(pred_boxes->size());
		CHECK_GT(num_classes, 0);
		const int conf_box_size = static_cast<int>((pred_boxes->at(background_label).size()));
		for (int i = 0; i < conf_box_size; ++i) {
			Dtype max_conf = 0;
			int max_class = 0;
			for (int j = 0; j < num_classes; j++) {
				const auto &class_pred_box = pred_boxes->at(j);
				CHECK_EQ(static_cast<int>(class_pred_box.size()), conf_box_size);
				const Dtype *pred_conf = class_pred_box[i].conf_;

				if (*pred_conf > max_conf) {
					max_conf = *pred_conf;
					max_class = j;
				}
			}
			if (max_class != background_label){
				bool unsaved = true;
				for (auto iter = max_conf_box.begin(); iter != max_conf_box.end(); ++iter) {
					if (max_conf >= iter->second.second){
						max_conf_box.insert(iter, make_pair(&pred_boxes->at(max_class).at(i), make_pair(max_class, max_conf)));
						unsaved = false;
						break;
					}
				}
				if (unsaved)
					max_conf_box.push_back(make_pair(&pred_boxes->at(max_class).at(i), make_pair(max_class, max_conf)));
			}
		}

		for (const auto &box : max_conf_box) {
			negatives.push_front(make_pair(box.second.first, box.first));
		}
	}

	template void ExtractNegtiveBox(const vector<vector<Box<float>>> *pred_boxes,
		const bool share_location, const int background_label,
		list<pair<const int, const Box<float> *>> &negatives);

	template void ExtractNegtiveBox(const vector<vector<Box<double>>> *pred_boxes,
		const bool share_location, const int background_label,
		list<pair<const int, const Box<double> *>> &negatives);
}
