// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp"

using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// clang-format off
using DeviceGemmV2_Streamk_Instance = 
    ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle_Streamk_V3<
        ALayout,   BLayout,  CLayout,   
        ADataType,   BDataType,  CDataType,  AccDataType,  CShuffleDataType, 
        PassThrough, PassThrough, PassThrough, GemmDefault, 
        256,
        128, 128, 
        64, 8, 8,
        16,   16,
        4,    4,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        S<8, 32, 1>,  S<1, 0, 2>,  S<1, 0, 2>, 
        2, 8, 8, 0,
        1, 2, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Intrawave,ck::BlockGemmPipelineVersion::v3>;
// clang-format on

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

using ReferenceGemmInstanceGPU = ck::tensor_operation::device::ReferenceGemm<ALayout,
                                                                             BLayout,
                                                                             CLayout,
                                                                             ADataType,
                                                                             BDataType,
                                                                             CDataType,
                                                                             AccDataType,
                                                                             AElementOp,
                                                                             BElementOp,
                                                                             CElementOp>;

#include "run_gemm_example_streamk_v2.inc"

int main(int argc, char* argv[]) { return !run_gemm_universal_streamk_example(argc, argv); }