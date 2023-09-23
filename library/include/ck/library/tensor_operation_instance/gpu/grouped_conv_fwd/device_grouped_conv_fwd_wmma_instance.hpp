// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using BF16 = ck::bhalf_t;
using F16  = ck::half_t;
using F32  = float;
using I8   = int8_t;
using I32  = int32_t;

using Empty_Tuple = ck::Tuple<>;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using namespace ck::tensor_layout::convolution;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;

static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto ConvFwdOddC =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::OddC;

static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <index_t NDSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wmma_f16_instances = std::tuple<
    // clang-format off
        //########################################|    NumDim|       A|       B|       Ds|       E| AData| BData|         Ds|  EData| AccData| CShuffle|            A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|   Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|   DataType|   Type|    Type| DataType|  Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|          |        |        |         |        |      |      |           |       |        |         |    Operation|   Operation|    Operation|               |               |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|          |        |        |         |        |      |      |           |       |        |         |             |            |             |               |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        // generic instance
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,     4,  8,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        // blocksize=256
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,     4,  8,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,    64,   256,     4,  8,    16,   16,       2,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   256,    64,     4,  8,    16,   16,       8,       1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,     8,  8,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        // blocksize=128
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,     4,  8,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,     8,  8,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,   128,     4,  8,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,   128,     8,  8,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    64,     4,  8,    16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    64,     8,  8,    16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    32,   256,     4,  8,    16,   16,       1,       8,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   256,    32,     4,  8,    16,   16,       8,       1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 4>,               8>,      
        // blocksize=64
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    64,     4,  8,    16,   16,       1,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    64,    32,     4,  8,    16,   16,       2,       2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    32,     8,  8,    16,   16,       1,       2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    32,   128,     4,  8,    16,   16,       1,       8,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        // blocksize=32
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    16,    64,     4,  8,    16,   16,       1,       4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    64,    16,     4,  8,    16,   16,       4,       1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    32,    32,     4,  8,    16,   16,       2,       2,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  F16,   F16, DsDatatype,    F16,     F32,      F16,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    16,    16,     4,  8,    16,   16,       1,       1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1,           1,           1,               S<1, 16, 1, 2>,               8>
    // clang-format on
    >;

template <index_t NDSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename DsDatatype,
          typename CDEElementOp,
          ConvolutionForwardSpecialization ConvSpec>
using device_grouped_conv_fwd_wmma_i8_instances = std::tuple<
    // clang-format off
        //########################################|    NumDim|       A|       B|       Ds|       E| AData| BData|         Ds|  EData| AccData| CShuffle|            A|           B|          CDE|    ConvForward|           GEMM| Block|  MPer|  NPer|  KPer| K1|  MPer| NPer| MRepeat| NRepeat|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################|   Spatial|  Layout|  Layout|   Layout|  Layout|  Type|  Type|   DataType|   Type|    Type| DataType|  Elementwise| Elementwise|  Elementwise| Specialization| Specialization|  Size| Block| Block| Block|   |  WMMA| WMMA|        |        |   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|          |        |        |         |        |      |      |           |       |        |         |    Operation|   Operation|    Operation|               |               |      |      |      |      |   |      |     |        |        | Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|          |        |        |         |        |      |      |           |       |        |         |             |            |             |               |               |      |      |      |      |   |      |     |        |        |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        //generic instance
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,     4,  16,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,               1,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,               1,              16,         1,           1,           1,               S<1, 32, 1, 4>,               1>,
        // blocksize=256
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,     4,  16,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,    64,   256,     4,  16,    16,   16,       2,       4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   256,    64,     4,  16,    16,   16,       8,       1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   256,   128,   128,     8,  16,    16,   16,       4,       2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 8>,               8>,
        // blocksize=128
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,     4,  16,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,    64,     8,  16,    16,   16,       2,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,   128,     4,  16,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    64,   128,     8,  16,    16,   16,       2,       4,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    64,     4,  16,    16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   128,    64,     8,  16,    16,   16,       4,       2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,    32,   256,     4,  16,    16,   16,       1,       8,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,   128,   256,    32,     4,  16,    16,   16,       8,       1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,      
        // blocksize=64
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    64,     4,  16,    16,   16,       1,       4,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    64,    32,     4,  16,    16,   16,       2,       2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    32,    32,     8,  16,    16,   16,       1,       2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    64,    32,   128,     4,  16,    16,   16,       1,       8,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        // blocksize=32
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    16,    64,     4,  16,    16,   16,       1,       4,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    64,    16,     4,  16,    16,   16,       4,       1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    32,    32,     4,  16,    16,   16,       2,       2,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>,  
        DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<NDSpatial, ALayout, BLayout, DsLayout, ELayout,  I8,   I8, DsDatatype,    I8,     I32,      I8,  PassThrough, PassThrough, CDEElementOp,       ConvSpec, GemmMNKPadding,    32,    16,    16,     4,  16,    16,   16,       1,       1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              16,              16,         1,     S<2, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              16,              16,         1,           1,           1,               S<1, 16, 1, 2>,               8>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck