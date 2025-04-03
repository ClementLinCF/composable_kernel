#include "ck_tile/host.hpp"
#include "multiply_vector.hpp"
#include <cstring>

auto create_args(int argc, char* argv[])
{   
    // list of arguments that the kernel can accept (easily customizable)
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "m dimension")
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
    using XDataType       = DataType; // input data type
    using ComputeDataType = float; // compute data type
    using YDataType       = DataType; // output data type

    ck_tile::index_t m = arg_parser.get_int("m"); 
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    
    ck_tile::HostTensor<XDataType> x_host_a({m}); // creating a 1D tensor of size m (8192 x 1)
    ck_tile::HostTensor<XDataType> x_host_b({m}); // creating a 1D tensor of size m (8192 x 1)

    ck_tile::HostTensor<YDataType> y_host_ref({m}); // creating a 1D tensor of size m (8192 x 1)
    ck_tile::HostTensor<YDataType> y_host_dev({m}); // creating a 1D tensor of size m (8192 x 1)

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_b);

    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes()); // get the size of the tensor in bytes and allocate memory on device
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(x_host_a.data()); // copy the data from host to device
    x_buf_b.ToDevice(x_host_b.data()); // copy the data from host to device

    // Configuring the block tile, warp tile and thread tile sizes
    // BlockTile: Dimension of one block that covers a part of the entire problem size
    // BlockWarps: How do you want to divide the block into warps
    // WarpTile: Dimension of one warp that covers a part of one block, this is the chunk of work that is assigned to a warp/wavefront
    // Vector: Dimension of one vector that covers a part of one warp, this is the chunk of work that is assigned to an individual thread

    

    constexpr ck_tile::index_t kBlockSize  = 256; 
    constexpr ck_tile::index_t kBlockPerCu = 1; 

    using BlockTile  = ck_tile::sequence<1024>; 
    using WarpTile = ck_tile::sequence<256>; 
    using Vector = ck_tile::sequence<4>; 
    
    using BlockWarps = ck_tile::sequence<4>; // each Block is like in 4 sub divisions

    ck_tile::index_t kGridSize             = (m / BlockTile::at(ck_tile::number<0>{})); // gridDim
    // print BlockTile size, WarpTile size, Vector size and BlockWarps size
    std::cout << "BlockTile: " << BlockTile::at(ck_tile::number<0>{}) << " , " << "WarpTile: " << WarpTile::at(ck_tile::number<0>{}) << " , " 
    << "Vector: " << Vector::at(ck_tile::number<0>{}) << " , " << "BlockWarps: " << BlockWarps::at(ck_tile::number<0>{}) << std::endl; 
    //std::cout << "multiply_vector::block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    //std::cout << "multiply_vector::grid size " << kGridSize << std::endl;

    using Shape = ck_tile::MultiplyShape<BlockWarps, BlockTile, WarpTile, Vector>; // struct that holds the configuration of the block, warp and vector tiles
    
    std::cout << "Shape::Block_M: " << Shape::Block_M << std::endl;
    std::cout << "Shape::Warp_M: " << Shape::Warp_M << std::endl;
    std::cout << "Shape::WarpPerBlock_M: " << Shape::WarpPerBlock_M << std::endl;
    std::cout << "Shape::Vector_M: " << Shape::Vector_M << std::endl;
    std::cout << "Shape::ThreadPerWarp_M: " << Shape::ThreadPerWarp_M << std::endl;
    std::cout << "Shape::Repeat_M: " << Shape::Repeat_M << std::endl;

    // Repeat_M * WarpPerBlock_M * ThreadPerWarp_M * Vector_M = Block_M
    //  4*4*64*4 = 4096
    assert(Shape::Repeat_M  == Shape::Block_M / (Shape::WarpPerBlock_M * Shape::Warp_M));
    printf("assertion Repeat_M == Block_M / (WarpPerBlock_N * Warp_N) passed\n");
    

    using Problem =
        ck_tile::MultiplyProblem<XDataType, ComputeDataType, YDataType, Shape>; // struct that holds the problem size and the data types and kernel configurations

    using Kernel = ck_tile::MultiplyVector<Problem>; // struct that is the kernel implementation

    // In CK, a kernel is not launched directly but through layers of abstractions
    // each layer either adds to the problem description or the kernel description (like assigns a stream)
    // or reads the problem description and invokes one of the many versions the same kernel (like for different data types)
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       kGridSize,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                                       static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()),
                                       static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                                       m));

    std::size_t num_btype = sizeof(XDataType) * m + sizeof(YDataType) * m; // size of two input vectors

    float gb_per_sec = num_btype / 1.E6 / ave_time; 

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_multiply_vector<XDataType, YDataType>(
           x_host_a, x_host_b, y_host_ref);
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
}
