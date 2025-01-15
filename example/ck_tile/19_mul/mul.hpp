// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_default_policy.hpp"

namespace ck_tile {

template <typename BlockWarps, // num warps along seq<M, N>
          typename BlockTile,  // block size, seq<M, N>
          typename WarpTile,   // warp size, seq<M, N>
          typename Vector>     // contiguous pixels(vector size) along seq<M, N>
struct Reduce2dShape
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
          typename BlockShape_,
          typename ReduceOp_>
struct MulProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
    using ReduceOp        = ReduceOp_;
};

template <typename Problem_, typename Policy_ = BlockReduce2dDefaultPolicy>
struct Reduce
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

#if 0
    CK_TILE_DEVICE void operator()(const XDataType* p_x, YDataType* p_y, index_t M, index_t N)
    const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M), number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m, make_tuple(number<S::Block_M>{}), {iM});

        const auto f_reduce = [](const auto& v0, const auto& v1) { return v0 + v1; };

        const XDataType reduce_init_value = 0;

        constexpr auto reduce_dims = sequence<1>{};

        auto y_compute = decltype(block_tile_reduce<ComputeDataType>(
            load_tile(x_window), reduce_dims, f_reduce, reduce_init_value)){};

        set_tile(y_compute, reduce_init_value);

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_tile_reduce(y_compute, x, reduce_dims, f_reduce);
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_tile_reduce_sync(y_compute, f_reduce);

        store_tile(y_window, cast_tile<YDataType>(y_compute));
    }
#else
    CK_TILE_DEVICE void operator()(const XDataType* p_x, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto y_m = make_naive_tensor_view_packed<address_space_enum::global>(
            p_y, make_tuple(M), number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window = make_tile_window(x_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        auto y_window = make_tile_window(y_m, make_tuple(number<S::Block_M>{}), {iM});

        __shared__ char smem[Policy::template GetSmemSize<Problem>()];

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        auto reduce_func         = typename Problem::ReduceOp{};
        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        auto block_reduce2d_cross_warp_sync =
            Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        using XTensorType = decltype(load_tile(x_window));
        auto y_compute    = block_reduce2d.template MakeYBlockTile<XTensorType>();
        set_tile(y_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto x = load_tile(x_window);
            block_reduce2d(x, y_compute, reduce_func);
            move_tile_window(x_window, {0, S::Block_N});
        }

        block_reduce2d_sync(y_compute, reduce_func);
        block_reduce2d_cross_warp_sync(y_compute, smem, reduce_func);

        store_tile(y_window, cast_tile<YDataType>(y_compute));
    }
#endif
};

template <typename Problem_, typename Policy_ = BlockReduce2dDefaultPolicy>
struct Mul
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    CK_TILE_DEVICE void operator()(const XDataType* p_x_a, const XDataType* p_x_b, YDataType* p_y, index_t M, index_t N) const
    {
        using S = typename Problem::BlockShape;

        const auto x_m_n_a = make_naive_tensor_view<address_space_enum::global>(
            p_x_a, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});
        
        const auto x_m_n_b = make_naive_tensor_view<address_space_enum::global>(
            p_x_b, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        // const auto y_m = make_naive_tensor_view_packed<address_space_enum::global>(
            // p_y, make_tuple(M), number<1>{});
        
        const auto y_m_n = make_naive_tensor_view<address_space_enum::global>(
            p_y, make_tuple(M, N), make_tuple(N, 1), number<S::Vector_N>{}, number<1>{});

        const auto iM = get_block_id() * S::Block_M;

        auto x_window_a = make_tile_window(x_m_n_a,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());
        
        auto x_window_b = make_tile_window(x_m_n_b,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        // auto y_window = make_tile_window(y_m, make_tuple(number<S::Block_M>{}), {iM});
        auto y_window = make_tile_window(y_m_n,
                                         make_tuple(number<S::Block_M>{}, number<S::Block_N>{}),
                                         {iM, 0},
                                         Policy::template MakeXBlockTileDistribution<Problem>());

        // __shared__ char smem[Policy::template GetSmemSize<Problem>()];

        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(N, S::Block_N));

        auto reduce_func         = typename Problem::ReduceOp{};
        auto block_reduce2d      = Policy::template GetBlockReduce2d<Problem>();
        // auto block_reduce2d_sync = Policy::template GetBlockReduce2dSync<Problem>();
        // auto block_reduce2d_cross_warp_sync =
        //     Policy::template GetBlockReduce2dCrossWarpSync<Problem>();

        // using XTensorType = decltype(load_tile(x_window_a));
        // auto y_compute    = block_reduce2d.template MakeYBlockTile<XTensorType>();
        // set_tile(y_compute, reduce_func.template GetIdentityValue<ComputeDataType>());

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            const auto xa = load_tile(x_window_a);
            const auto xb = load_tile(x_window_b);
            auto y_compute = load_tile(y_window);

            // sweep_tile<XDistributedTensor_>(
            // [&](auto... idx_) {
            //     constexpr auto idx_0 = make_tuple(make_tuple(idx_[number<0>{}]...)[number<0>{}]);
            //     y_tensor(idx_0)      = reduce_func(
            //         y_tensor(idx_0), ck_tile::type_convert<ComputeDataType>(x_tensor[idx_])...);
            // },
            // ReducePacksPerXDim{});



            // block_reduce2d(x, y_compute, reduce_func);
            block_reduce2d(xa, xb, y_compute, reduce_func);
            store_tile(y_window, cast_tile<YDataType>(y_compute));
            // store_tile(y_window, cast_tile<YDataType>(xa));
            move_tile_window(x_window_a, {0, S::Block_N});
            move_tile_window(x_window_b, {0, S::Block_N});
            move_tile_window(y_window, {0, S::Block_N});
        }

        // block_reduce2d_sync(y_compute, reduce_func);
        // block_reduce2d_cross_warp_sync(y_compute, smem, reduce_func);

        // store_tile(y_window, cast_tile<YDataType>(y_compute));
    }
};

} // namespace ck_tile
