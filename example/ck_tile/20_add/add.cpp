#include "ck_tile/host.hpp"
#include "add.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "10240", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType; // data type of input
    using ComputeDataType = float; // data type of computation
    using YDataType       = DataType; // data type of output

    ck_tile::index_t m = arg_parser.get_int("m"); // get m dimension
    ck_tile::index_t n = arg_parser.get_int("n"); // get n dimension
    int do_validation  = arg_parser.get_int("v"); // get validation flag - do we validate the output?
    int warmup         = arg_parser.get_int("warmup"); // get number of warmup iterations - iterations before we start timing
    int repeat         = arg_parser.get_int("repeat"); // get number of iterations to run the kernel for benchmarking

    // What's a HostTensor?
    // HostTensor is a class that wraps a pointer to a buffer on the host (CPU) and provides a convenient way to access the data in the buffer
    // It also provides a way to allocate memory on the host and copy data to and from the device (GPU)
    // In this example, we use HostTensor to create input and output tensors on the host
    // HosTensor has another class within in called TensorDescriptor, which is used to describe the shape of the tensor
    // given the shape {m, n}, TensorDescriptor will calculate the stride of the tensor and the total number of elements in the tensor

    ck_tile::HostTensor<XDataType> x_host_a({m, n}); // create host tensor for input data
    ck_tile::HostTensor<XDataType> x_host_b({m, n}); // create host tensor for input data

    ck_tile::HostTensor<YDataType> y_host_ref({m, n}); // create host tensor for output data
    ck_tile::HostTensor<YDataType> y_host_dev({m, n}); // create host tensor for output data

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_a); // fill input tensor with random values
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_b); // fill input tensor with random values

    // Memory allocation on the device (GPU)
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes()); // a wrapper to call hipMemcpy to copy data from host to device or vice versa
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes()); // a wrapper to call hipMemcpy to copy data from host to device or vice versa
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes()); // a wrapper to call hipMemcpy to copy data from host to device or vice versa

    x_buf_a.ToDevice(x_host_a.data()); // copy data from host to device
    x_buf_b.ToDevice(x_host_b.data()); // copy data from host to device

    // 19xx Gb/s, Original
    // using BlockWarps = ck_tile::sequence<1, 8>;
    // using BlockTile  = ck_tile::sequence<1, 2048>;
    // using WarpTile   = ck_tile::sequence<1, 256>;
    // using Vector     = ck_tile::sequence<1, 4>;

    // 19xx -> 199x, Align 256 thread
    // using BlockWarps = ck_tile::sequence<1, 4>;
    // using BlockTile  = ck_tile::sequence<1, 1024>;
    // using WarpTile   = ck_tile::sequence<1, 256>;
    // using Vector     = ck_tile::sequence<1, 4>;

    // 199x -> 27xx, Utilize vector load/store
    // using BlockWarps = ck_tile::sequence<1, 4>;
    // using BlockTile  = ck_tile::sequence<1, 2048>;
    // using WarpTile   = ck_tile::sequence<1, 512>;
    // using Vector     = ck_tile::sequence<1, 8>;

    // 27xx -> 32xx, 1D block to 2D block
    // Imagine a 2D matrix that covers the entire GPU grid 
    // to map threads to the matrix, we first sub divide the grid into Blocks, then subdivide the blocks into blockTiles
    // each blockTile is further subdivided into warpTiles, and each warpTile.
    // by hardware definition, a warp is 64 threads wide
    // therefore if number of elements in a warp > 64, we need to subdivide the warp into vectors

    using BlockTile  = ck_tile::sequence<2, 1024>; // 1024 threads per block in x direction
    using WarpTile   = ck_tile::sequence<1, 512>; // 512 threads per warp in x direction
    using Vector     = ck_tile::sequence<1, 8>; // 8 threads per vector in x direction
    using BlockWarps = ck_tile::sequence<2, 2>; // 2 warps per block in x and y direction

    // another important bit of information is number of threads per block
    // in this example, we have 32 warps per block, 32 * 32 = 1024 threads per block
    // Note: by hardware definition, a block can have a maximum of 1024 threads
    
    // using BlockWarps = ck_tile::sequence<8, 1>;
    // using BlockTile  = ck_tile::sequence<128, 64>;
    // using WarpTile   = ck_tile::sequence<16, 64>;
    // using Vector     = ck_tile::sequence<4, 4>;


    constexpr ck_tile::index_t kBlockSize  = 256; // number of threads per block (can be extended upto 1024) 
    constexpr ck_tile::index_t kBlockPerCu = 1; // number of blocks per CU

    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape = ck_tile::AddShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Porblem =
        ck_tile::AddProblem<XDataType, ComputeDataType, YDataType, Shape>;

    using Kernel = ck_tile::Add<Porblem>; // in CK we define a kernel as a struct that contains the kernel code, and the parameters to the kernel

    // the kernel struct is then passed to series of host level invocation functions that "do their thing" to get the kernel running on the GPU
    // "do their thing" includes things like setting up the grid and block dimensions, deciding the hip stream, launching the kernel, copying data back to the CPU, benchmarking, etc.
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                       static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m,
                                       n)); 

    std::size_t num_btype = sizeof(XDataType) * m * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_add<XDataType, YDataType>(
           x_host_a, x_host_b, y_host_ref);
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv); // get parameters as command line arguments
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec"); // get precision of input data

    // in a typical composable kernel workflow, we filter throug several parameters on the CPU to determine the best kernel to call
    // in this example, we only have one kernel, so we call it directly based on the precision of the input data
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2; // call appropriate kernel based on precision
    }
}
