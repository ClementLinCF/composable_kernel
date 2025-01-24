// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGemm_Streamk_V2<Col,
                                                      Row,
                                                      Row,
                                                      BF16,
                                                      BF16,
                                                      BF16,
                                                      PassThrough,
                                                      PassThrough,
                                                      PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_instances<Intrawave,
                                                                                GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck