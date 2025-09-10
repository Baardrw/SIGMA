#define EIGEN_DONT_VECTORIZE
#define EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "data_structures.h"
#include "test_common_includes.h"
#include "test_math_utils.h"

// Loads all data from the given MDT and RPC files, organizes it into buckets,
// and returns the bucket IDs, MDT hits, RPC hits, and ground truth for each
// bucket
std::tuple<std::vector<long int>, std::map<long int, std::vector<MDTHit>>,
           std::map<long int, std::vector<RPCHit>>,
           std::vector<BucketGroundTruth>>
load_and_organize_data(const std::string &mdt_filename,
                       const std::string &rpc_filename, int start, int end);

// Sets up the structure of the host data datastructure, basically initializes
// Bucket sizes and allocates memory for the hits
void setup_h_data_buckets(Data &h_data, std::vector<long int> &bucket_ids,
                          std::map<long int, std::vector<MDTHit>> &mdt_bucket,
                          std::map<long int, std::vector<RPCHit>> &rpc_bucket);

// Populates the host data structure with hits from the MDT and RPC buckets
// Also calculates the ground truth for each bucket
void populate_h_data(Data &h_data,
                     std::map<long int, std::vector<MDTHit>> &mdt_bucket,
                     std::map<long int, std::vector<RPCHit>> &rpc_bucket,
                     const std::vector<long int> &bucket_ids,
                     std::vector<BucketGroundTruth> &bucket_ground_truths);

void cleanup_host(Data &h_data);

void cleanup_device(Data &d_data);

void copy_results_to_host(Data &h_data, Data &d_data, int num_buckets);

Data *copy_to_device(const Data &h_data, Data &d_data, int num_measurements,
                     int num_buckets);
