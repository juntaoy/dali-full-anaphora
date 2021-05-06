#include <map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


REGISTER_OP("OracleClusters")
.Input("mention_starts: int32")
.Input("mention_ends: int32")
.Input("mention_cluster_ids: int32")
.Input("max_cluster_size: int32")
.Output("oracle_clusters: int32")
.Output("oracle_cluster_size: int32");

class OracleClustersOp : public OpKernel{
public:
  explicit OracleClustersOp(OpKernelConstruction* context) : OpKernel(context){}

  void Compute(OpKernelContext* context) override{
    TTypes<int32>::ConstVec mention_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec mention_ends = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec mention_cluster_ids = context->input(2).vec<int32>();
    int max_cluster_size = context->input(3).scalar<int>()(0);

    CHECK_EQ(mention_starts.dimension(0), mention_ends.dimension(0));
    CHECK_EQ(mention_starts.dimension(0), mention_cluster_ids.dimension(0));

    int num_mentions = mention_starts.dimension(0);

    Tensor* oracle_clusters_tensor = nullptr;
    TensorShape oracle_clusters_shape({num_mentions, max_cluster_size});
    OP_REQUIRES_OK(context, context->allocate_output(0,oracle_clusters_shape, &oracle_clusters_tensor));
    TTypes<int32>::Matrix oracle_clusters = oracle_clusters_tensor->matrix<int32>();

    Tensor* oracle_cluster_size_tensor = nullptr;
    TensorShape oracle_cluster_size_shape({num_mentions});
    OP_REQUIRES_OK(context, context->allocate_output(1,oracle_cluster_size_shape, &oracle_cluster_size_tensor));
    TTypes<int32>::Vec oracle_cluster_size = oracle_cluster_size_tensor->vec<int32>();

    std::map<int, int> cl2last;
    for (int i =0; i<num_mentions;++i){
      int cid = mention_cluster_ids(i);
      if(cl2last.find(cid) == cl2last.end()){
        if(cid>0){//cid == 0 means not mention
          cl2last[cid] = i;
        }
        oracle_cluster_size(i) = 1;
        oracle_clusters(i,0) = i;
        for(int k =1; k<max_cluster_size;++k){
          oracle_clusters(i,k) = 0;//paddings
        }
      }else{
        int lasti = cl2last[cid];
        cl2last[cid] = i;
        int cl_index = std::min(max_cluster_size-1,oracle_cluster_size(lasti));
        oracle_cluster_size(i) = cl_index+1;
        for(int k =0;k<max_cluster_size;++k){
          oracle_clusters(i,k) = oracle_clusters(lasti, k);
        }
        oracle_clusters(i,cl_index) = i;//insert the current mention
      }
    }

  }

};
REGISTER_KERNEL_BUILDER(Name("OracleClusters").Device(DEVICE_CPU),OracleClustersOp);

REGISTER_OP("GoldScores")
.Input("mention_starts: int32")
.Input("mention_ends: int32")
.Input("mention_type_scores: float32")
.Input("gold_starts: int32")
.Input("gold_ends: int32")
.Input("gold_cluster_ids: int32")
.Input("gold_types: int32")
.Input("crac_doc:bool")
.Input("cluster_indices: int32")
.Input("cluster_size: int32")
.Input("n_types: int32")
.Output("gold_labels:bool");

class GoldScoresOp : public OpKernel {
public:
  explicit GoldScoresOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec mention_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec mention_ends = context->input(1).vec<int32>();
    TTypes<float>::ConstMatrix mention_type_scores = context->input(2).matrix<float>();
    TTypes<int32>::ConstVec gold_starts = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec gold_ends = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec gold_cluster_ids = context->input(5).vec<int32>();
    TTypes<int32>::ConstVec gold_types = context->input(6).vec<int32>();
    bool crac_doc = context->input(7).scalar<bool>()(0);
    TTypes<int32, 3>::ConstTensor cluster_indices = context->input(8).tensor<int32, 3>();

    TTypes<int32>::ConstMatrix cluster_size = context->input(9).matrix<int32>();

    int n_types = context->input(10).scalar<int>()(0);
    float negative_inf = std::log(0);

    CHECK_EQ(mention_starts.dimension(0), mention_ends.dimension(0));
    CHECK_EQ(gold_starts.dimension(0), gold_ends.dimension(0));

    int num_mentions = mention_starts.dimension(0);
    int num_gold = gold_starts.dimension(0);

    int max_cluster = cluster_indices.dimension(1);

    Tensor* gold_labels_tensor = nullptr;
    TensorShape gold_labels_shape({num_mentions, max_cluster + n_types});
    OP_REQUIRES_OK(context, context->allocate_output(0, gold_labels_shape, &gold_labels_tensor));
    TTypes<bool>::Matrix gold_labels = gold_labels_tensor->matrix<bool>();


    std::map<std::pair<int, int>, int> mention_indices;
    for (int i = 0; i < num_mentions; ++i) {
      mention_indices[std::pair<int, int>(mention_starts(i), mention_ends(i))] = i;
    }

    std::vector<int> mention_cluster_ids(num_mentions, -1);
    std::vector<int> mention_type_ids(num_mentions,-1);

    for (int i = 0; i < num_gold; ++i) {
      auto iter = mention_indices.find(std::pair<int, int>(gold_starts(i), gold_ends(i)));
      if (iter != mention_indices.end()) {
        mention_cluster_ids[iter->second] = gold_cluster_ids(i);
        mention_type_ids[iter->second] = gold_types(i);
      }
    }

    std::map<int, int> gold_cl_size;
    std::map<int, int> cl2steps;
    int curr_steps = 0;
    for (int i = 0; i <num_mentions; ++i){
      int cid = mention_cluster_ids[i];
      if(cid >= 0){
        if(cl2steps.find(cid)==cl2steps.end()){
          cl2steps[cid] = curr_steps;
          gold_cl_size[cid] = 1;
          curr_steps++;
        }else{
          gold_cl_size[cid]++;
        }
      }
    }

    int num_of_cl = gold_cl_size.size();
    if (num_of_cl == 0){
      for(int i = 0; i < num_mentions; ++i){
        gold_labels(i,0) = true;
        for(int j=1; j < max_cluster + n_types; ++j){
          gold_labels(i,j) = false;
        }
      }
    }else{
      int max_cl_size = std::max_element(gold_cl_size.begin(),gold_cl_size.end(),
                    [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
                     return p1.second < p2.second;})->second;

      std::vector<int> curr_gold_cl_size(num_of_cl,0);
      std::vector<std::vector<int>> gold_clusters;
      gold_clusters.resize(num_of_cl, std::vector<int>(max_cl_size,0));


      for (int i = 0; i < num_mentions; ++i) {
        int cid = mention_cluster_ids[i];
        if(cid >=0){
          cid = cl2steps[cid];
          int gsize = curr_gold_cl_size[cid];
          if(gsize==0){
            if(crac_doc){
                gold_labels(i,0) = false;
                for(int j=1; j< n_types-1;j++){
                    gold_labels(i,j) = mention_type_ids[i]==j;
                }
                gold_labels(i,n_types-1) = mention_type_ids[i]<=0;//DN
            }else{
                gold_labels(i,0) = true;
                for(int j=1;j<n_types;++j){
                    gold_labels(i,j) = false;
                }
            }
            for(int j=n_types; j < max_cluster + n_types; ++j){
              gold_labels(i,j) = false;
            }
          }else{
            bool allfalse = true;
            for(int j=0;j<max_cluster;++j){
              bool con_gold = false;
              int size = cluster_size(i,j);
              if(size > 0){
                for(int k=0;k<size;++k){
                  for(int n=0;n<gsize;++n){
                    if(cluster_indices(i,j,k) == gold_clusters[cid][n]){
                      con_gold = true;
                      allfalse = false;
                      break;
                    }
                  }
                  if(con_gold){
                    break;
                  }
                }
                if(con_gold){
                  gold_labels(i,j+n_types) = true;
                }else{
                  gold_labels(i,j+n_types) = false;
                }
              }else{
                gold_labels(i,j+n_types) = false;
              }
            }
            if(allfalse){
              for(int k=0;k<n_types;++k){
                gold_labels(i,k) = false;
              }
              if(crac_doc){
                gold_labels(i,n_types-1) = true;//DN
              }else{
                gold_labels(i,0) = true;
              }
            }else{
              for(int k=0;k<n_types;++k){
                gold_labels(i,k) = false;
              }
            }
          }

          gold_clusters[cid][curr_gold_cl_size[cid]] = i;
          curr_gold_cl_size[cid]++;
        }else{
          gold_labels(i,0) = true;
          for(int j=1; j < max_cluster + n_types; ++j){
            gold_labels(i,j) = false;
          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GoldScores").Device(DEVICE_CPU), GoldScoresOp);

REGISTER_OP("ExtractSpans")
.Input("span_scores: float32")
.Input("candidate_starts: int32")
.Input("candidate_ends: int32")
.Input("num_output_spans: int32")
.Input("max_sentence_length: int32")
.Attr("sort_spans: bool")
.Output("output_span_indices: int32");

class ExtractSpansOp : public OpKernel {
public:
  explicit ExtractSpansOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sort_spans", &_sort_spans));
  }

  void Compute(OpKernelContext* context) override {
    TTypes<float>::ConstMatrix span_scores = context->input(0).matrix<float>();
    TTypes<int32>::ConstMatrix candidate_starts = context->input(1).matrix<int32>();
    TTypes<int32>::ConstMatrix candidate_ends = context->input(2).matrix<int32>();
    TTypes<int32>::ConstVec num_output_spans = context->input(3).vec<int32>();
    int max_sentence_length = context->input(4).scalar<int32>()();

    int num_sentences = span_scores.dimension(0);
    int num_input_spans = span_scores.dimension(1);
    int max_num_output_spans = 0;
    for (int i = 0; i < num_sentences; i++) {
      if (num_output_spans(i) > max_num_output_spans) {
        max_num_output_spans = num_output_spans(i);
      }
    }

    Tensor* output_span_indices_tensor = nullptr;
    TensorShape output_span_indices_shape({num_sentences, max_num_output_spans});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_span_indices_shape,
                                                     &output_span_indices_tensor));
    TTypes<int32>::Matrix output_span_indices = output_span_indices_tensor->matrix<int32>();

    std::vector<std::vector<int>> sorted_input_span_indices(num_sentences,
                                                            std::vector<int>(num_input_spans));
    for (int i = 0; i < num_sentences; i++) {
      std::iota(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(), 0);
      std::sort(sorted_input_span_indices[i].begin(), sorted_input_span_indices[i].end(),
                [&span_scores, &i](int j1, int j2) {
                  return span_scores(i, j2) < span_scores(i, j1);
                });
    }

    for (int l = 0; l < num_sentences; l++) {
      std::vector<int> top_span_indices;
      std::unordered_map<int, int> end_to_earliest_start;
      std::unordered_map<int, int> start_to_latest_end;

      int current_span_index = 0,
          num_selected_spans = 0;
      while (num_selected_spans < num_output_spans(l) && current_span_index < num_input_spans) {
        int i = sorted_input_span_indices[l][current_span_index];
        bool any_crossing = false;
        const int start = candidate_starts(l, i);
        const int end = candidate_ends(l, i);
        for (int j = start; j <= end; ++j) {
          auto latest_end_iter = start_to_latest_end.find(j);
          if (latest_end_iter != start_to_latest_end.end() && j > start && latest_end_iter->second > end) {
            // Given (), exists [], such that ( [ ) ]
            any_crossing = true;
            break;
          }
          auto earliest_start_iter = end_to_earliest_start.find(j);
          if (earliest_start_iter != end_to_earliest_start.end() && j < end && earliest_start_iter->second < start) {
            // Given (), exists [], such that [ ( ] )
            any_crossing = true;
            break;
          }
        }
        if (!any_crossing) {
          if (_sort_spans) {
            top_span_indices.push_back(i);
          } else {
            output_span_indices(l, num_selected_spans) = i;
          }
          ++num_selected_spans;
          // Update data struct.
          auto latest_end_iter = start_to_latest_end.find(start);
          if (latest_end_iter == start_to_latest_end.end() || end > latest_end_iter->second) {
            start_to_latest_end[start] = end;
          }
          auto earliest_start_iter = end_to_earliest_start.find(end);
          if (earliest_start_iter == end_to_earliest_start.end() || start < earliest_start_iter->second) {
            end_to_earliest_start[end] = start;
          }
        }
        ++current_span_index;
      }
      // Sort and populate selected span indices.
      if (_sort_spans) {
        std::sort(top_span_indices.begin(), top_span_indices.end(),
                  [&candidate_starts, &candidate_ends, &l] (int i1, int i2) {
                    if (candidate_starts(l, i1) < candidate_starts(l, i2)) {
                      return true;
                    } else if (candidate_starts(l, i1) > candidate_starts(l, i2)) {
                      return false;
                    } else if (candidate_ends(l, i1) < candidate_ends(l, i2)) {
                      return true;
                    } else if (candidate_ends(l, i1) > candidate_ends(l, i2)) {
                      return false;
                    } else {
                      return i1 < i2;
                    }
                  });
        for (int i = 0; i < num_output_spans(l); ++i) {
          output_span_indices(l, i) = top_span_indices[i];
        }
      }
      // Pad with the first span index.
      for (int i = num_selected_spans; i < max_num_output_spans; ++i) {
        output_span_indices(l, i) = output_span_indices(l, 0);
      }
    }
  }
private:
  bool _sort_spans;
};

REGISTER_KERNEL_BUILDER(Name("ExtractSpans").Device(DEVICE_CPU), ExtractSpansOp);


REGISTER_OP("ClusterWidthBins")
.Input("widths: int32")
.Output("bins: int32");

class ClusterWidthBinsOp : public OpKernel {
public:
  explicit ClusterWidthBinsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec widths  = context->input(0).vec<int32>();

    int d0 = widths.dimension(0);

    Tensor* bins_tensor = nullptr;
    TensorShape bins_shape({d0});
    OP_REQUIRES_OK(context, context->allocate_output(0, bins_shape, &bins_tensor));
    TTypes<int32>::Vec bins = bins_tensor->vec<int32>();

    for (int i = 0; i < d0; ++i) {
       bins(i) = get_bin(widths(i));
    }
  }
private:
  int get_bin(int d) {
    if (d <= 0) {
      return 0;
    } else if (d == 1) {
      return 1;
    } else if (d == 2) {
      return 2;
    } else if (d == 3) {
      return 3;
    } else if (d == 4) {
      return 4;
    } else if (d < 8) {
      return 5;
    } else if (d < 12) {
      return 6;
    } else if (d < 20) {
      return 7;
    } else{
      return 8;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ClusterWidthBins").Device(DEVICE_CPU), ClusterWidthBinsOp);



REGISTER_OP("DistanceBins")
.Input("distances: int32")
.Output("bins: int32");

class DistanceBinsOp : public OpKernel {
public:
  explicit DistanceBinsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec distances  = context->input(0).vec<int32>();

    int d0 = distances.dimension(0);

    Tensor* bins_tensor = nullptr;
    TensorShape bins_shape({d0});
    OP_REQUIRES_OK(context, context->allocate_output(0, bins_shape, &bins_tensor));
    TTypes<int32>::Vec bins = bins_tensor->vec<int32>();

    for (int i = 0; i < d0; ++i) {
        bins(i) = get_bin(distances(i));
    }
  }
private:
  int get_bin(int d) {
    if (d <= 0) {
      return 0;
    } else if (d == 1) {
      return 1;
    } else if (d == 2) {
      return 2;
    } else if (d == 3) {
      return 3;
    } else if (d == 4) {
      return 4;
    } else if (d <= 7) {
      return 5;
    } else if (d <= 15) {
      return 6;
    } else if (d <= 31) {
      return 7;
    } else if (d <= 63) {
      return 8;
    } else {
      return 9;
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("DistanceBins").Device(DEVICE_CPU), DistanceBinsOp);

REGISTER_OP("ExtractAntecedentLabels")
.Input("mention_starts: int32")
.Input("mention_ends: int32")
.Input("gold_starts: int32")
.Input("gold_ends: int32")
.Input("gold_cluster_ids: int32")
.Input("gold_types: int32")
.Input("crac_doc:bool")
.Input("antecedent_indices:int32")
.Input("antecedent_masks:bool")
.Input("n_types: int32")
.Output("antecedent_labels: bool");

class ExtractAntecedentLabelsOp : public OpKernel {
public:
  explicit ExtractAntecedentLabelsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec mention_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec mention_ends = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec gold_starts = context->input(2).vec<int32>();
    TTypes<int32>::ConstVec gold_ends = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec gold_cluster_ids = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec gold_types = context->input(5).vec<int32>();
    bool crac_doc = context->input(6).scalar<bool>()(0);
    TTypes<int32>::ConstMatrix antecedent_indices = context->input(7).matrix<int32>();
    TTypes<bool>::ConstMatrix antecedent_masks = context->input(8).matrix<bool>();

    int n_types = context->input(9).scalar<int>()(0);

    CHECK_EQ(mention_starts.dimension(0), mention_ends.dimension(0));
    CHECK_EQ(gold_starts.dimension(0), gold_ends.dimension(0));

    int num_mentions = mention_starts.dimension(0);
    int num_gold = gold_starts.dimension(0);

    int max_antecedents = antecedent_indices.dimension(1);

    Tensor* labels_tensor = nullptr;
    TensorShape labels_shape({num_mentions, max_antecedents + n_types});
    OP_REQUIRES_OK(context, context->allocate_output(0, labels_shape, &labels_tensor));
    TTypes<bool>::Matrix labels = labels_tensor->matrix<bool>();


    std::map<std::pair<int, int>, int> mention_indices;
    for (int i = 0; i < num_mentions; ++i) {
      mention_indices[std::pair<int, int>(mention_starts(i), mention_ends(i))] = i;
    }

    std::vector<int> mention_cluster_ids(num_mentions, -1);
    std::vector<int> mention_type_ids(num_mentions, -1);
    for (int i = 0; i < num_gold; ++i) {
      auto iter = mention_indices.find(std::pair<int, int>(gold_starts(i), gold_ends(i)));
      if (iter != mention_indices.end()) {
        mention_cluster_ids[iter->second] = gold_cluster_ids(i);
        mention_type_ids[iter->second] = gold_types(i);
      }
    }

    for (int i = 0; i < num_mentions; ++i) {
      int antecedent_count = 0;
      bool null_label = true;
      if(crac_doc && mention_type_ids[i] >=0){
        for(int j=0;j<max_antecedents + n_types;++j){
            labels(i, j) = j == mention_type_ids[i];
        }
      }else if(mention_cluster_ids[i] < 0){
        for(int j=0;j<max_antecedents + n_types;++j){
            labels(i, j) = j == 0;
        }
      }else{
        for(int j=0;j<max_antecedents;++j){
          if(antecedent_masks(i,j) && mention_cluster_ids[i] == mention_cluster_ids[antecedent_indices(i,j)]){
            labels(i,j+n_types) = true;
            null_label = false;
          }else{
            labels(i, j+n_types) = false;
          }
        }

        if(null_label){
          if(crac_doc){//for crac doc
            for(int j=0;j<n_types;++j){
              labels(i,j) = j==n_types-1; //DN
            }
          }else{// for conll doc
            for(int j=0;j<n_types;++j){
              labels(i,j) = j==0;
            }
          }
        }else{
          for(int j=0;j<n_types;++j){
            labels(i, j) = false;
          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("ExtractAntecedentLabels").Device(DEVICE_CPU), ExtractAntecedentLabelsOp);

REGISTER_OP("GoldScoresWithSplitAntecedents")
.Input("mention_starts: int32")
.Input("mention_ends: int32")
.Input("gold_starts: int32")
.Input("gold_ends: int32")
.Input("gold_cluster_ids: int32")
.Input("gold_types: int32")
.Input("split_antecedent_cids: int32")
.Input("split_antecedent_size: int32")
.Input("crac_doc:bool")
.Input("cluster_indices: int32")
.Input("cluster_size: int32")
.Input("n_types: int32")
.Output("gold_labels:bool")
.Output("gold_split_antecedent_labels:bool");

class GoldScoresWithSplitAntecedentsOp : public OpKernel {
public:
  explicit GoldScoresWithSplitAntecedentsOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    TTypes<int32>::ConstVec mention_starts = context->input(0).vec<int32>();
    TTypes<int32>::ConstVec mention_ends = context->input(1).vec<int32>();
    TTypes<int32>::ConstVec gold_starts = context->input(2).vec<int32>();
    TTypes<int32>::ConstVec gold_ends = context->input(3).vec<int32>();
    TTypes<int32>::ConstVec gold_cluster_ids = context->input(4).vec<int32>();
    TTypes<int32>::ConstVec gold_types = context->input(5).vec<int32>();
    TTypes<int32>::ConstMatrix split_antecedent_cids = context->input(6).matrix<int32>();
    TTypes<int32>::ConstVec split_antecedent_size = context->input(7).vec<int32>();
    bool crac_doc = context->input(8).scalar<bool>()(0);
    TTypes<int32, 3>::ConstTensor cluster_indices = context->input(9).tensor<int32, 3>();
    TTypes<int32>::ConstMatrix cluster_size = context->input(10).matrix<int32>();

    int n_types = context->input(11).scalar<int>()(0);
    float negative_inf = std::log(0);

    CHECK_EQ(mention_starts.dimension(0), mention_ends.dimension(0));
    CHECK_EQ(gold_starts.dimension(0), gold_ends.dimension(0));

    int num_mentions = mention_starts.dimension(0);
    int num_gold = gold_starts.dimension(0);

    int max_cluster = cluster_indices.dimension(1);
    int max_split_antecedent = split_antecedent_cids.dimension(1);

    Tensor* gold_labels_tensor = nullptr;
    TensorShape gold_labels_shape({num_mentions, max_cluster + n_types});
    OP_REQUIRES_OK(context, context->allocate_output(0, gold_labels_shape, &gold_labels_tensor));
    TTypes<bool>::Matrix gold_labels = gold_labels_tensor->matrix<bool>();

    Tensor* gold_split_antecedent_labels_tensor = nullptr;
    TensorShape gold_split_antecedent_labels_shape({num_mentions, max_cluster + 1});
    OP_REQUIRES_OK(context, context->allocate_output(1, gold_split_antecedent_labels_shape, &gold_split_antecedent_labels_tensor));
    TTypes<bool>::Matrix gold_split_antecedent_labels = gold_split_antecedent_labels_tensor->matrix<bool>();

    std::map<std::pair<int, int>, int> mention_indices;
    for (int i = 0; i < num_mentions; ++i) {
      mention_indices[std::pair<int, int>(mention_starts(i), mention_ends(i))] = i;
    }

    std::vector<int> mention_cluster_ids(num_mentions, -1);
    std::vector<int> mention_type_ids(num_mentions,-1);
    std::vector<std::vector<int>> mention_split_antecedent_ante_ids;
    mention_split_antecedent_ante_ids.resize(num_mentions, std::vector<int>(max_split_antecedent,-1));
    std::vector<int> mention_split_antecedent_size(num_mentions,0);


    for (int i = 0; i < num_gold; ++i) {
      auto iter = mention_indices.find(std::pair<int, int>(gold_starts(i), gold_ends(i)));
      if (iter != mention_indices.end()) {
        mention_cluster_ids[iter->second] = gold_cluster_ids(i);
        mention_type_ids[iter->second] = gold_types(i);
        mention_split_antecedent_size[iter->second] = split_antecedent_size(i);
        for (int j=0; j<split_antecedent_size(i);++j){
          mention_split_antecedent_ante_ids[iter->second][j] = split_antecedent_cids(i,j);
        }
      }
    }

    std::map<int, int> gold_cl_size;
    std::map<int, int> cl2steps;
    int curr_steps = 0;
    for (int i = 0; i <num_mentions; ++i){
      int cid = mention_cluster_ids[i];
      if(cid >= 0){
        if(cl2steps.find(cid)==cl2steps.end()){
          cl2steps[cid] = curr_steps;
          gold_cl_size[cid] = 1;
          curr_steps++;
        }else{
          gold_cl_size[cid]++;
        }
      }
    }

    int num_of_cl = gold_cl_size.size();
    if (num_of_cl == 0){
      for(int i = 0; i < num_mentions; ++i){
        gold_labels(i,0) = true;
        for(int j=1; j < max_cluster + n_types; ++j){
          gold_labels(i,j) = false;
        }
        gold_split_antecedent_labels(i,0) = true;
        for(int j=1; j < max_cluster + 1; ++j){
          gold_split_antecedent_labels(i,j) = false;
        }
      }
    }else{
      int max_cl_size = std::max_element(gold_cl_size.begin(),gold_cl_size.end(),
                    [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
                     return p1.second < p2.second;})->second;

      std::vector<int> curr_gold_cl_size(num_of_cl,0);
      std::vector<std::vector<int>> gold_clusters;
      gold_clusters.resize(num_of_cl, std::vector<int>(max_cl_size,0));


      for (int i = 0; i < num_mentions; ++i) {
        //split_antecedent
        bool allfalse = true;
        for(int l=0;l<max_cluster;++l){
          bool con_gold = false;
          for(int k=0;k<cluster_size(i,l);++k){
            for(int j=0; j< mention_split_antecedent_size[i];++j){
              int pcid = mention_split_antecedent_ante_ids[i][j];
              pcid = cl2steps[pcid];
              int gsize = curr_gold_cl_size[pcid];
              for(int n=0;n<gsize;++n){
                if(cluster_indices(i,l,k) == gold_clusters[pcid][n]){
                  con_gold = true;
                  allfalse = false;
                  break;
                }
              }
              if(con_gold){
                break;
              }
            }
            if(con_gold){
              break;
            }
          }
          if(con_gold){
            gold_split_antecedent_labels(i,l+1) = true;
          }else{
            gold_split_antecedent_labels(i,l+1) = false;
          }
        }
        gold_split_antecedent_labels(i,0) = allfalse;

        if(allfalse){
          gold_split_antecedent_labels(i,0) = true;
          for(int j=1; j<max_cluster+1;++j){
            gold_split_antecedent_labels(i,j) = false;
          }
        }

        //Coreference
        int cid = mention_cluster_ids[i];
        if(cid >=0){
          cid = cl2steps[cid];
          int gsize = curr_gold_cl_size[cid];
          if(gsize==0){
            if(crac_doc){
                gold_labels(i,0) = false;
                for(int j=1; j< n_types-1;j++){
                    gold_labels(i,j) = mention_type_ids[i]==j;
                }
                gold_labels(i,n_types-1) = mention_type_ids[i]<=0;//DN
            }else{
                gold_labels(i,0) = true;
                for(int j=1;j<n_types;++j){
                    gold_labels(i,j) = false;
                }
            }
            for(int j=n_types; j < max_cluster + n_types; ++j){
              gold_labels(i,j) = false;
            }
          }else{
            bool allfalse = true;
            for(int j=0;j<max_cluster;++j){
              bool con_gold = false;
              int size = cluster_size(i,j);
              if(size > 0){
                for(int k=0;k<size;++k){
                  for(int n=0;n<gsize;++n){
                    if(cluster_indices(i,j,k) == gold_clusters[cid][n]){
                      con_gold = true;
                      allfalse = false;
                      break;
                    }
                  }
                  if(con_gold){
                    break;
                  }
                }
                if(con_gold){
                  gold_labels(i,j+n_types) = true;
                }else{
                  gold_labels(i,j+n_types) = false;
                }
              }else{
                gold_labels(i,j+n_types) = false;
              }
            }
            if(allfalse){
              for(int k=0;k<n_types;++k){
                gold_labels(i,k) = false;
              }
              if(crac_doc){
                gold_labels(i,n_types-1) = true;//DN
              }else{
                gold_labels(i,0) = true;
              }
            }else{
              for(int k=0;k<n_types;++k){
                gold_labels(i,k) = false;
              }
            }
          }

          gold_clusters[cid][curr_gold_cl_size[cid]] = i;
          curr_gold_cl_size[cid]++;
        }else{
          gold_labels(i,0) = true;
          for(int j=1; j < max_cluster + n_types; ++j){
            gold_labels(i,j) = false;
          }
        }

      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("GoldScoresWithSplitAntecedents").Device(DEVICE_CPU), GoldScoresWithSplitAntecedentsOp);