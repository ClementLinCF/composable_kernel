// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous pixels(vector size) along seq<M, N>
struct AddShape
{
    static constexpr index_t Block_M = BlockTile::at(number<0>{});
    static constexpr index_t Block_N = BlockTile::at(number<1>{});

    static constexpr index_t Warp_M = WarpTile::at(number<0>{});
    static constexpr index_t Warp_N = WarpTile::at(number<1>{});

    static constexpr index_t Vector_M = Vector::at(number<0>{});
    static constexpr index_t Vector_N = Vector::at(number<1>{});

    static constexpr index_t WarpPerBlock_M = BlockWarps::at(number<0>{});
    static constexpr index_t WarpPerBlock_N = BlockWarps::at(number<1>{});

    static constexpr index_t ThreadPerWarp_M = Warp_M / Vector_M;
    static constexpr index_t ThreadPerWarp_N = Warp_N / Vector_N;

    static constexpr index_t Repeat_M = Block_M / (WarpPerBlock_M * Warp_M);
    static constexpr index_t Repeat_N = Block_N / (WarpPerBlock_N * Warp_N);

    static constexpr index_t BlockSize =
        warpSize * reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});
};

template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename BlockShape_>
struct AddProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
};

// this is our Policy of the kernel
// A policy gives additional information to the kernel about data mapping
// the idea of composing a kernel is that the kernel can be provided with policies written by users or by the library
struct AddDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        // more on this later
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>, // repetition, not used in this example
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>, // levels of distribution
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>, // levels of distribution
                tuple<sequence<1, 2>, sequence<1, 2>>, // 1 means sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M> and 2 means sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>
                tuple<sequence<1, 1>, sequence<2, 2>>, // 0 - Repeat_M, 1 - WarpPerBlock_M, 2 - ThreadPerWarp_M, 3 - Vector_M
                // now read the above two lines vertically from left to right first pair is (1, 1) and second pair is (2, 1) and third pair is (1, 2) and fourth pair is (2, 2)
                // first pair (1, 1) means that the first level of distribution is WarpPerBlock_M
                // second pair (2, 1) the second level of distribution is WarpPerBlock_N
                // third pair (1, 2) the third level of distribution is ThreadPerWarp_M
                // fourth pair (2, 2) the fourth level of distribution is ThreadPerWarp_N
                sequence<1, 1, 2, 2>, // to be deciphered
                sequence<0, 3, 0, 3>>{}); // to be deciphered
    }
};

template <typename Problem_, typename Policy_ = AddDefaultPolicy>
struct Add
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    // body of the kernel
    CK_TILE_DEVICE void operator()(const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        // Tensor on device memory has two parts
        // a wrapper around the raw pointer (TensorView)
        // a wrapper around the shape of the tensor (TensorDescriptor)
        // make_naive_tensor_view is a helper function to create the TensorView with a given shape and raw pointer
        // "naive" indicates that we use default tensor descriptor
        // make_naive_tensor_view<from_address_space>(raw_pointer, shape, strides, lastgarunteedlength, lastgarunteed)
        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global>(
            p_x_a, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});
        
        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global>(
            p_x_b, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M; // calculate the starting point of this block
        
        // this where the ck tile magic happpens
        // we create a tile window for each tensor
        // a tile window is a window that slides over the tensor
        // we can load and store tiles from/to the tensor using the tile window
        // we can also move the tile window to the next tile
        // in a barebone kernel implementation, developers need to manage the tile window manually
        // in a ck tile kernel, the tile window is managed by the ck tile api
        // the idea of composable kernel is that SOME part of the work of a kernel developer can be abstracted away
        // so that the developer can focus on the algorithm itself
        // for example, in this kernel, the developer does not need to write the loop to iterate over the tiles
        // the ck tile api will take care of that
        // he only needs to write the computation that happens in each tile
        // the developer is also not responsible for the synchronization between blocks
        // the ck tile api will take care of that
        auto x_window_a = make_tile_window(x_m_n_a,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());
        
        auto x_window_b = make_tile_window(x_m_n_b,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());
        
        // calculate the number of tiles in the N dimension
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        // for each tile in the N dimension
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa = load_tile(x_window_a); // load a tile from the tensor
            const auto xb = load_tile(x_window_b); // load a tile from the tensor
            auto y_compute = load_tile(y_window); // load a tile from the tensor

            // iterate over the tile
            // the tile is a 2D array
            // we iterate over the tile using two indices
            // the indices are passed to the lambda function
            // the lambda function is the computation that happens in each tile
            // the lambda function is a generic lambda function
            // the indices are passed as a tuple
            // the lambda function is called for each element in the tile
            // the lambda function is called in parallel
            constexpr auto spans = decltype(xa)::get_distributed_spans();
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx          = ck_tile::make_tuple(idx0, idx1);
                    const auto x = ck_tile::type_convert<ComputeDataType>(xa[i_j_idx]);
                    const auto y = ck_tile::type_convert<ComputeDataType>(xb[i_j_idx]);
                    y_compute(i_j_idx) = x + y;
                });
            });

            store_tile(y_window, cast_tile<YDataType>(y_compute)); // store the tile back to the tensor
            move_tile_window(x_window_a, {0, S::Block_N}); // move the tile window to the next tile
            move_tile_window(x_window_b, {0, S::Block_N}); // move the tile window to the next tile
            move_tile_window(y_window, {0, S::Block_N}); // move the tile window to the next tile

            // once again the ck tile api takes care of movement of the tile window (this level of abstraction is the goal of ck tile)
            // the developer only needs to write the computation that happens in each tile
        }
    }
};

} // namespace ck_tile
