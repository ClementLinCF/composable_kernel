// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"

#include "block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "grid_gemm_problem.hpp"

// C = A * B
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementFunction,
          typename BElementFunction,
          typename CElementFunction,
          ck_tile::index_t kAAlignment,
          ck_tile::index_t kBAlignment,
          ck_tile::index_t kCAlignment,
          ck_tile::index_t kBlockSize_,
          ck_tile::index_t kMPerBlock_,
          ck_tile::index_t kNPerBlock_,
          ck_tile::index_t kKPerBlock_>
struct Gemm
{
    using GridGemmProblem = ck::tile_program::grid::GridGemmProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CDataType,
                                                                    AElementFunction,
                                                                    BElementFunction,
                                                                    CElementFunction>;

    struct GridGemmPolicy
    {
        static constexpr ck::index_t kBlockSize = kBlockSize_;
        static constexpr ck::index_t kMPerBlock = kMPerBlock_;
        static constexpr ck::index_t kNPerBlock = kNPerBlock_;
        static constexpr ck::index_t kKPerBlock = kKPerBlock_;

        template <typename Problem>
        __host__ __device__ static constexpr auto MakeBlock2TileMap(ck::index_t NumTilesM,
                                                                    ck::index_t NumTilesN)
        {
            using namespace ck;

            const auto unmerge = make_merge_transform(make_tuple(NumTilesN, NumTilesM));

            return [unmerge](index_t block_id) {
                MultiIndex<2> unmerged;
                unmerge.CalculateLowerIndex(unmerged, make_multi_index(block_id));

                return make_multi_index(unmerged.At(Number<1>{}), unmerged.At(Number<0>{}));  

            };
        }

        template <typename Problem>
        __host__ __device__ static constexpr auto GetBlockGemmPipeline()
        {
            using namespace ck;
            using namespace ck::tile_program;
            using namespace ck::tile_program::block;

            using BlockGemmPipelineProblem_ =
                BlockGemmPipelineProblem<ADataType,
                                         BDataType,
                                         AccDataType,
                                         kBlockSize,
                                         TileGemmShape<kMPerBlock, kNPerBlock, kKPerBlock>>;

            return BlockGemmPipelineAGmemBGmemCRegV2<
                BlockGemmPipelineProblem_,
                BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy>{};
        }
    };

    using GridGemm = ck::GridGemmV1<GridGemmProblem, GridGemmPolicy>;

    __device__ void operator()(const ADataType* p_a,
                               const BDataType* p_b,
                               CDataType* p_c,
                               const ck_tile::index_t M,
                               const ck_tile::index_t N,
                               const ck_tile::index_t K,
                               const ck_tile::index_t Lda,
                               const ck_tile::index_t Ldb,
                               const ck_tile::index_t Ldc,
                               const AElementFunction& a_element_func,
                               const BElementFunction& b_element_func,
                               const CElementFunction& c_element_func) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        const auto a_dram = [&] {
            if constexpr(is_same_v<ALayout, ck::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    p_a, ck_tile::make_tuple(M, K), ck_tile::make_tuple(Lda, 1), ck_tile::number<kAAlignment>{}, ck_tile::number<1>{});
            }
            else
            {
                const auto a_k_m_desc = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    p_a, ck_tile::make_tuple(K, M), ck_tile::make_tuple(Lda, 1), ck_tile::number<kAAlignment>{}, ck_tile::number<1>{});

                return ck_tile::transform_tensor_view(
                    a_k_m_desc,
                    ck_tile::make_tuple(ck_tile::make_pass_through_transform(M), ck_tile::make_pass_through_transform(K)),
                    ck_tile::make_tuple(Sequence<1>{}, Sequence<0>{}),
                    ck_tile::make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto b_dram = [&] {
            if constexpr(is_same_v<BLayout, ck::tensor_layout::gemm::ColumnMajor>)
            {
                return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    p_b, ck_tile::make_tuple(N, K), ck_tile::make_tuple(Ldb, 1), ck_tile::number<kBAlignment>{}, ck_tile::number<1>{});
            }
            else
            {
                const auto b_k_n_desc = ck_tile::make_naive_tensor_view<>(
                    p_b, ck_tile::make_tuple(K, N), ck_tile::make_tuple(Ldb, 1), ck_tile::number<kBAlignment>{}, ck_tile::number<1>{});

                return ck_tile::transform_tensor_view(
                    b_k_n_desc,
                    ck_tile::make_tuple(ck_tile::make_pass_through_transform(N), ck_tile::make_pass_through_transform(K)),
                    ck_tile::make_tuple(Sequence<1>{}, Sequence<0>{}),
                    ck_tile::make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto c_dram = [&] {
            if constexpr(is_same_v<CLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    p_c, ck_tile::make_tuple(M, N), ck_tile::make_tuple(Ldc, 1), ck_tile::number<kCAlignment>{}, ck_tile::number<1>{});
            }
            else
            {
                const auto c_n_m_desc = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                    p_c, ck_tile::make_tuple(N, M), ck_tile::make_tuple(Ldc, 1), ck_tile::number<kCAlignment>{}, ck_tile::number<1>{});

                return ck_tile::transform_tensor_view(
                    c_n_m_desc,
                    ck_tile::make_tuple(ck_tile::make_pass_through_transform(M), ck_tile::make_pass_through_transform(N)),
                    ck_tile::make_tuple(Sequence<1>{}, Sequence<0>{}),
                    ck_tile::make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        GridGemm{}(a_dram, b_dram, c_dram, a_element_func, b_element_func, c_element_func);
    }
};
