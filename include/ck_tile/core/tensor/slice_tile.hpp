// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename StaticDistributedTensor_, index_t... SliceBegins, index_t... SliceEnds>
__host__ __device__ constexpr auto get_slice_tile(const StaticDistributedTensor_& tile,
                                                  sequence<SliceBegins...> slice_begins,
                                                  sequence<SliceEnds...> slice_ends)
{
    using Distribution = decltype(StaticDistributedTensor_::get_tile_distribution());
    using DataType     = typename StaticDistributedTensor_::DataType;

    constexpr auto sliced_dstr_yidx_ylen =
        detail::slice_distribution_from_x(Distribution{}, slice_begins, slice_ends);

    constexpr auto sliced_dstr      = sliced_dstr_yidx_ylen.template get<0>();
    constexpr auto sliced_y_origins = sliced_dstr_yidx_ylen.template get<1>();
    constexpr auto sliced_y_lengths = sliced_dstr_yidx_ylen.template get<2>();

    auto sliced_tensor = make_static_distributed_tensor<DataType>(sliced_dstr);

    sliced_tensor.get_thread_buffer() = tile.get_y_sliced_thread_data(sliced_y_origins, sliced_y_lengths);

    return sliced_tensor;
}

template <typename DstStaticDistributedTensor_,
          typename SrcStaticDistributedTensor_,
          index_t... SliceBegins,
          index_t... SliceEnds>
__host__ __device__ constexpr auto set_slice_tile(DstStaticDistributedTensor_& dst_tile,
                                                  const SrcStaticDistributedTensor_& src_tile,
                                                  sequence<SliceBegins...> slice_begins,
                                                  sequence<SliceEnds...> slice_ends)
{
    using DstDistribution = decltype(DstStaticDistributedTensor_::get_tile_distribution());
    // using SrcDistribution = decltype(SrcStaticDistributedTensor_::GetTileDistribution());

    constexpr auto sliced_dstr_yidx_ylen =
        detail::slice_distribution_from_x(DstDistribution{}, slice_begins, slice_ends);

    // constexpr auto sliced_dstr      = sliced_dstr_yidx_ylen.template At<0>();
    constexpr auto sliced_y_origins = sliced_dstr_yidx_ylen.template get<1>();
    constexpr auto sliced_y_lengths = sliced_dstr_yidx_ylen.template get<2>();

    // static_assert(is_same_v<decltype(sliced_dstr), SrcDistribution>, "wrong!");

    dst_tile.set_y_sliced_thread_data(sliced_y_origins, sliced_y_lengths, src_tile.get_thread_buffer());
}

} // namespace ck_tile
