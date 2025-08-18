#include "test_common_includes.h"

#include "data_structures.h"
#include "line_math.h"
#include "residual_math.h"


// Calculates the ground truth for all buckets, used to compare against the GPU results
void calculate_bucket_answer(BucketGroundTruth &bucket_ground_truth,
                             const std::vector<MDTHit> &mdt_hits,
                             const std::vector<RPCHit> &rpc_hits);

// Calculates the chi2 for all buckets
std::vector<real_t> calculate_chi2(Data &h_data, int num_buckets);