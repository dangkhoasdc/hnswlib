#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <exception>
#include <cstdint>
#include "hnswlib.h"

#include <highfive/H5File.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>


namespace hnswlib {
  struct PQInfo {
    std::vector<float> codebook;
    std::vector<size_t> codebook_dims;

    std::vector<float> discodebook;
    std::vector<size_t> discodebook_dims;

    float query_norm;
    size_t dim;

    PQInfo() = default;

    PQInfo(const PQInfo& other) {
      codebook = other.codebook;
      codebook_dims = other.codebook_dims;

      discodebook = other.discodebook;
      discodebook_dims = other.discodebook_dims;

      query_norm = other.query_norm;
      dim = other.dim;
    }
  };

  static int pq_distance(const void* a, const void* b, const void* c) {
    PQInfo* pInfo = (PQInfo*) c;
    const float* norm_codebook = pInfo->discodebook.data();
    const float* prod_codebook = pInfo->discodebook.data() + 256;
    size_t dim = pInfo->dim;

    const uint8_t * a_byte = reinterpret_cast<const uint8_t*>(a);
    const uint8_t * b_byte = reinterpret_cast<const uint8_t*>(b);

    float norm_a = 0.0f;
    float norm_b = 0.0f;
    float prod = 0.0f;
    for (int i = 0; i < dim; i += 4)
    {
        int idx0a = a_byte[i];
        int idx1a = a_byte[i + 1];
        int idx2a = a_byte[i + 2];
        int idx3a = a_byte[i + 3];

        int idx0b = b_byte[i];
        int idx1b = b_byte[i + 1];
        int idx2b = b_byte[i + 2];
        int idx3b = b_byte[i + 3];

        float norm_a0 = norm_codebook[idx0a];
        float norm_a1 = norm_codebook[idx1a];
        float norm_a2 = norm_codebook[idx2a];
        float norm_a3 = norm_codebook[idx3a];

        float norm_b0 = norm_codebook[idx0b];
        float norm_b1 = norm_codebook[idx1b];
        float norm_b2 = norm_codebook[idx2b];
        float norm_b3 = norm_codebook[idx3b];

        norm_a += norm_a0 + norm_a1 + norm_a2 + norm_a3;
        norm_b += norm_b0 + norm_b1 + norm_b2 + norm_b3;

        int idx0 = (idx0a << 8) | idx0b;
        int idx1 = (idx1a << 8) | idx1b;
        int idx2 = (idx2a << 8) | idx2b;
        int idx3 = (idx3a << 8) | idx3b;

        float prod0 = prod_codebook[idx0];
        float prod1 = prod_codebook[idx1];
        float prod2 = prod_codebook[idx2];
        float prod3 = prod_codebook[idx3];

        prod += prod0 + prod1 + prod2 + prod3;
    }

    // if (norm_a <= 1e-6 || norm_b <= 1e-6)
    // {
    //     if (norm_a <= 1e-6 && norm_b <= 1e-6)
    //     {
    //         return 0;
    //     }
    //     else
    //     {
    //         return 10000;
    //     }
    // }

    float dist = 2.0f - 2.0f * prod / std::sqrt(norm_a) / std::sqrt(norm_b);
    // std::cerr << "dist: " << dist << std::endl;
    int final_dist =  int(dist * 10000);
    // std::cerr << "dist: " << final_dist << std::endl;
    return final_dist;
  }

  PQInfo load_pq_codebook(const std::string& codebook_path) {
    PQInfo info;
    using namespace HighFive;
    File file(codebook_path, File::ReadOnly);

    // load codebook info
    auto codebook = file.getDataSet("data");
    info.codebook_dims = codebook.getDimensions();
    size_t n_codebook_elems = 1;
    for (const auto& dim: info.codebook_dims)
      n_codebook_elems *= dim;

    info.codebook.resize(n_codebook_elems);
    codebook.read<float>(info.codebook.data());

    // load dis codebook data
    auto discodebook = file.getDataSet("dis_data");
    info.discodebook_dims = discodebook.getDimensions();
    size_t n_discodebook_elems = 1;
    for (const auto& dim: info.discodebook_dims)
      n_discodebook_elems *= dim;

    info.discodebook.resize(n_discodebook_elems);
    discodebook.read<float>(info.discodebook.data());

    return info;
  }

  // PQInfo preprocess(const float* data_point, const PQInfo* pq_info, size_t dim) {
  //   PQInfo new_info(*pq_info);
  //   return new_info;
  // }


  class PQSpace: public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    PQInfo orig_pq_info;

  public:
    PQSpace(const std::string& codebook_path, size_t dim) {
      dim_ = dim;
      data_size_ = dim * sizeof(uint8_t);
      orig_pq_info = load_pq_codebook(codebook_path);
      orig_pq_info.dim = dim;
      fstdistfunc_ = pq_distance;
    }
    size_t get_data_size() override {
      return data_size_;
    }

    DISTFUNC<int> get_dist_func() override {
      return fstdistfunc_;
    }

    void *get_dist_func_param() override {
      return &orig_pq_info;
    }

    // void *preprocess(const void* data_point, const void* other_params) {
    //   PQInfo updated_info = preprocess((const float*)data_point, (const PQInfo*) orig_pq_info, dim_);
    //   return &updated_info;
    // }

    ~PQSpace() {}
  };
}
