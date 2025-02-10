// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "block_gemm_asmem_bsmem_creg_problem.hpp"
#include "block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"

namespace ck {
namespace tile_program {
namespace block {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem, typename Policy = ck_tile::BlockGemmASmemBSmemCRegV1DefaultPolicy>
struct BlockGemmASmemBSmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // C += A * B
    template <typename CBlockTensor, typename ABlockWindowTmp, typename BBlockWindowTmp>
    __device__ void operator()(CBlockTensor& c_block_tensor,
                               const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          is_same_v<BDataType, typename BBlockWindowTmp::DataType> &&
                          is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[ck_tile::number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[ck_tile::number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[ck_tile::number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template get<0>())>;

        constexpr index_t MWarp = config.template get<1>();
        constexpr index_t NWarp = config.template get<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = ck_tile::get_warp_id() / NWarp;
        const index_t iNWarp = ck_tile::get_warp_id() % NWarp;

        // construct A-warp-window
        auto a_warp_window_tmp = ck_tile::make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            ck_tile::make_tuple(ck_tile::number<WG::kM>{}, ck_tile::number<WG::kK>{}),
            // a_block_window_tmp.get_window_origin() + MultiIndex<2>{iMWarp * WG::kM, 0},
            {a_block_window_tmp.get_window_origin().at(ck_tile::number<0>{}) + iMWarp * WG::kM, a_block_window_tmp.get_window_origin().at(ck_tile::number<1>{})},
            ck_tile::make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(a_warp_window_tmp), KIterPerWarp>, MIterPerWarp> a_warp_windows{
            {a_warp_window_tmp}};

        for(index_t mIter = 0; mIter < MIterPerWarp; mIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(a_warp_window_tmp), KIterPerWarp>,
                               MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // construct B-warp-window
        auto b_warp_window_tmp = ck_tile::make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            ck_tile::make_tuple(ck_tile::number<WG::kN>{}, ck_tile::number<WG::kK>{}),
            // b_block_window_tmp.GetWindowOrigin() + MultiIndex<2>{iNWarp * WG::kN, 0},
            {b_block_window_tmp.get_window_origin().at(ck_tile::number<0>{}) + iNWarp * WG::kN, b_block_window_tmp.get_window_origin().at(ck_tile::number<1>{})},
            ck_tile::make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(b_warp_window_tmp), KIterPerWarp>,
                               NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths = ck_tile::to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        ck_tile::merge_sequences(ck_tile::sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        ck_tile::merge_sequences(ck_tile::sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    // C = A * B
    template <typename ABlockWindowTmp, typename BBlockWindowTmp>
    __device__ auto operator()(const ABlockWindowTmp& a_block_window_tmp,
                               const BBlockWindowTmp& b_block_window_tmp) const
    {
        static_assert(is_same_v<ADataType, typename ABlockWindowTmp::DataType> &&
                          is_same_v<BDataType, typename BBlockWindowTmp::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindowTmp{}.get_window_lengths()[ck_tile::number<0>{}];
        constexpr index_t NPerBlock = BBlockWindowTmp{}.get_window_lengths()[ck_tile::number<0>{}];
        constexpr index_t KPerBlock = ABlockWindowTmp{}.get_window_lengths()[ck_tile::number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template get(ck_tile::number<0>{}))>;  

        constexpr index_t MWarp = config.template get(ck_tile::number<1>{});
        constexpr index_t NWarp = config.template get(ck_tile::number<2>{});

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        const index_t iMWarp = ck_tile::get_warp_id() / NWarp;
        const index_t iNWarp = ck_tile::get_warp_id() % NWarp;

        // construct A-warp-window
        auto a_warp_window_tmp = ck_tile::make_tile_window(
            a_block_window_tmp.get_bottom_tensor_view(),
            ck_tile::make_tuple(ck_tile::number<WG::kM>{}, ck_tile::number<WG::kK>{}),
            // a_block_window_tmp.get_window_origin() + MultiIndex<2>{iMWarp * WG::kM, 0},
            {a_block_window_tmp.get_window_origin().at(ck_tile::number<0>{}) + iMWarp * WG::kM, a_block_window_tmp.get_window_origin().at(ck_tile::number<1>{})},
            ck_tile::make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(a_warp_window_tmp), KIterPerWarp>, MIterPerWarp> a_warp_windows{
            {a_warp_window_tmp}};

        for(index_t mIter = 0; mIter < MIterPerWarp; mIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(a_warp_window_tmp), KIterPerWarp>,
                               MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // construct B-warp-window
        auto b_warp_window_tmp = ck_tile::make_tile_window(
            b_block_window_tmp.get_bottom_tensor_view(),
            ck_tile::make_tuple(ck_tile::number<WG::kN>{}, ck_tile::number<WG::kK>{}),
            // b_block_window_tmp.GetWindowOrigin() + MultiIndex<2>{iNWarp * WG::kN, 0},
            {b_block_window_tmp.get_window_origin().at(ck_tile::number<0>{}) + iNWarp * WG::kN, b_block_window_tmp.get_window_origin().at(ck_tile::number<1>{})},
            ck_tile::make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

#if 0 // FIXME: using Array will cause register spill
        Array<Array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        StaticallyIndexedArray<StaticallyIndexedArray<decltype(b_warp_window_tmp), KIterPerWarp>,
                               NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        static_assert(is_same_v<CDataType, typename WG::CDataType>, "wrong!");

        // Construct C-Block-Tensor
        constexpr auto c_block_outer_dstr_encoding = ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<MIterPerWarp, MWarp>, ck_tile::sequence<NIterPerWarp, NWarp>>,
            ck_tile::tuple<ck_tile::sequence<1, 2>>,
            ck_tile::tuple<ck_tile::sequence<1, 1>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = ck_tile::detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = ck_tile::make_static_distributed_tensor<CDataType>(c_block_dstr);

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths = to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    // warp GEMM
                    if constexpr(KIterPerWarp == 0)
                    {
                        // c = a * b
                        c_warp_tensor = WG{}(a_warp_tensor, b_warp_tensor);
                    }
                    else
                    {
                        // c += a * b
                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            ck_tile::merge_sequences(ck_tile::sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths));

                        WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    }

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        ck_tile::merge_sequences(ck_tile::sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });

        return c_block_tensor;
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
