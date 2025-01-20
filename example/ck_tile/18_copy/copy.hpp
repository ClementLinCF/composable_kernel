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
struct CopyShape
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
          typename YDataType_,
          typename BlockShape_>
struct CopyProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
};

struct CopyPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }
};

template <typename Problem_, typename Policy_ = CopyPolicy>
struct Copy
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    CK_TILE_DEVICE void operator()(const XDataType* p_x, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});
        
        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            store_tile(y_window, cast_tile<YDataType>(x));
            move_tile_window(x_window, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }
    }
};

} // namespace ck_tile
