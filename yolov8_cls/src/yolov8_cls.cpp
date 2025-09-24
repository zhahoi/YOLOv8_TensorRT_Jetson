#include "yolov8_cls.h"

YOLOv8_cls::YOLOv8_cls(const std::string& engine_file_path)
{
    // 1. 打开 TensorRT 引擎文件（.engine / .trt）, 以二进制模式读取
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good()); // 确保文件打开成功

    // 2. 定位到文件末尾，获取文件大小
    file.seekg(0, std::ios::end);
    auto size = file.tellg(); // size = 文件字节数
    file.seekg(0, std::ios::beg); // 回到文件开头

    // 3. 在堆上分配一个char数组用于存放文件内容，并读取
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close(); // 关闭文件

    // 4. 初始化TensorRT的自定义Plugin(若使用了任何自定义Layer)
    initLibNvInferPlugins(&this->gLogger, "");

    // 5. 创建运行时对象（IRuntime），后续用于反序列化 Engine
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    // 6. 反序列化 Engine，将二进制流构建为 ICudaEngine 实例
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);

    // 7. 释放已经不再需要的模型流内存
    delete[] trtModelStream;

    // 8. 基于 Engine 创建执行上下文（IExecutionContext），管理推理状态
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);

    // 9. 创建一个 CUDA Stream，用于后续异步拷贝与推理
    cudaStreamCreate(&this->stream);

    // 10. 查询模型中所有的 I/O Tensor 数量（输入 + 输出）
    this->num_bindings = this->engine->getNbIOTensors();

    // 11. 遍历每个 binding，收集其名称、数据类型、尺寸，以及区分输入/输出
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;

        // 11.1 获取第 i 个 binding 的名称（如 "images"、"output0"）
        std::string        name = this->engine->getIOTensorName(i);
        binding.name = name;

        // 11.2 获取该 Tensor 的数据类型（float、int8…），并计算每个元素所占字节大小
        nvinfer1::DataType dtype = this->engine->getTensorDataType(name.c_str());
        binding.dsize = type_to_size(dtype);

        // 11.3 判断该 binding 是否为输入
        bool IsInput = this->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

        // 11.4 获取该 Tensor 在最优配置下的最大 shape（动态 shape 时使用）
        nvinfer1::Dims     dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);

        if (IsInput) {
            // ---- 处理输入 binding ----
            this->num_inputs += 1;

            // 11.5 计算总元素数量
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape

            // 11.6 告诉执行上下文：输入 Tensor 的 shape，供后续动态推理使用
            this->context->setInputShape(name.c_str(), dims);

            std::cout << "input name: " << name << " dims: " << dims.nbDims
                << " input shape:(" << dims.d[0] << "," << dims.d[1] << ","
                << dims.d[2] << "," << dims.d[3] << ")" << std::endl;
        }
        else {
            // ---- 处理输出 binding ----
            // 输出的 shape 可以通过上下文直接查询（动态 shape 时已被固定）
            dims = this->context->getTensorShape(name.c_str());
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;

            std::cout << "ouput name: " << name << " dims: " << dims.nbDims
                << " ouput shape:(" << dims.d[0] << "," << dims.d[1] << ","
                << dims.d[2] << "," << dims.d[3] << ")" << std::endl;
        }
    }
}

YOLOv8_cls::~YOLOv8_cls()
{
    delete this->context;
    delete this->engine;
    delete this->runtime;

    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_cls::make_pipe(bool warmup)
{

    // —— 1. 为所有输入 Binding 分配 Device (GPU) 内存 —— 
    // input_bindings 存储了每个输入张量的元素数量 (binding.size)
    // 和每个元素的字节大小 (binding.dsize)
    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        // 异步地在当前 CUDA 流上分配 GPU 内存：bindings.size * bindings.dsize 字节 
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        // 将分配好的设备指针保存到 device_ptrs 以备后续推理使用
        this->device_ptrs.push_back(d_ptr);
    }

    // —— 2. 为所有输出 Binding 分配 Device 和 Host (Page‑locked) 内存 —— 
    // 输出通常要在 GPU 上计算后再拷贝回 CPU 端做后处理，
    // 所以既需要 Device 内存，也需要 Host 端的 page‑locked (pinned) 内存以加速复制
    for (auto& bindings : this->output_bindings) {
        void* d_ptr, * h_ptr;
        size_t size = bindings.size * bindings.dsize;
        // 在 GPU 上分配同样大小的输出 buffer
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        // 在 Host 上分配 page‑locked 内存，以便后续 cudaMemcpyAsync 高效地从 Device 读出
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    // —— 3. 可选的模型“热身”——当 warmup 为 true 时运行若干次推理
    // 这样可以让 TensorRT JIT 编译、GPU 占用和内存分配提前完成，
    // 减少第一次真实推理的延迟峰值
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            // 3.1 对每个输入执行一次空数据拷贝，模拟真实输入
            for (auto& bindings : this->input_bindings) {
                size_t size = bindings.size * bindings.dsize;
                // 在 CPU 申请一个临时 buffer，并置零
                void* h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                // 异步把“全零”数据拷贝到 GPU 上的输入 buffer
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            // 3.2 调用 infer() 完成一次完整推理
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8_cls::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat nchw;
    auto&   in_binding = this->input_bindings[0];
    auto    width = static_cast<int>(in_binding.dims.d[3]);
    auto    height = static_cast<int>(in_binding.dims.d[2]);

    cv::dnn::blobFromImage(image, nchw, 1 / 255.f, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false, CV_32F);

    nvinfer1::Dims4 dims(1, 3, height, width);
    if (!context->setInputShape(in_binding.name.c_str(), dims)) {
        throw std::runtime_error("Failed to set input shape");
    }
    context->setTensorAddress(in_binding.name.c_str(), device_ptrs[0]);

    size_t byte_size = nchw.total() * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), byte_size, cudaMemcpyHostToDevice, this->stream);

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory copy failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

void YOLOv8_cls::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    cv::dnn::blobFromImage(image, nchw, 1 / 255.f, size, cv::Scalar(0, 0, 0), true, false, CV_32F);

    auto& in_binding = this->input_bindings[0];

    nvinfer1::Dims4 dims(1, 3, size.height, size.width);
    if (!context->setInputShape(in_binding.name.c_str(), dims)) {
        throw std::runtime_error("Failed to set input shape");
    }
    context->setTensorAddress(in_binding.name.c_str(), device_ptrs[0]);
 
    size_t byte_size = nchw.total() * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(this->device_ptrs[0], nchw.ptr<float>(), byte_size, cudaMemcpyHostToDevice, this->stream);

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory copy failed: " + 
                                std::string(cudaGetErrorString(err)));
    }
}

void YOLOv8_cls::preprocessGPU(const cv::Mat& image)
{
    cuda_preprocess(
        image.data, image.cols, image.rows,
        static_cast<float*>(this->device_ptrs[0]), dst_w, dst_h, 
        this->stream
    );
}

void YOLOv8_cls::infer()
{
    // 1. 将所有输入和输出 Tensor 的 GPU 指针绑定到执行上下文
    //    对于每个 binding：
    //      - 如果 i < num_inputs，则是输入 binding；否则是输出 binding（输出索引需要减去 num_inputs）。
    //    调用 setTensorAddress(name, ptr) 告诉 TensorRT 在该 Tensor 上执行推理时，
    //    要从哪个 GPU 内存地址读取/写入数据。缺少这步会导致 enqueueV3 报错“地址未设置”。 :contentReference[oaicite:0]{index=0}
    for (int i = 0; i < this->num_bindings; ++i) {
        const char* tensorName =
            (i < num_inputs ? input_bindings[i].name : output_bindings[i - num_inputs].name).c_str();
        void* devicePtr = device_ptrs[i];
        context->setTensorAddress(tensorName, devicePtr);
    }

    // 2. 发起异步推理任务
    //    使用 enqueueV3 而非旧的 enqueueV2/executeV2，
    //    可以在同一个 CUDA Stream 上实现数据传输和核函数并行化。 :contentReference[oaicite:1]{index=1}
    this->context->enqueueV3(this->stream);

    // 3. 异步将每个输出 Tensor 从 GPU 拷贝回 Host
    //    遍历所有输出 binding（i 从 0 到 num_outputs-1），
    //    他们在 device_ptrs 数组中的索引是 i + num_inputs。
    //    cudaMemcpyAsync 使用同一个 stream，拷贝完成后再做同步。 :contentReference[oaicite:2]{index=2}
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i],  // 目标：CPU 页锁定内存
            this->device_ptrs[i + this->num_inputs],  // 源：GPU 上的输出缓冲
            osize,
            cudaMemcpyDeviceToHost,
            this->stream));
    }

    // 4. 确保当前 Stream 上的所有拷贝和推理都已完成
    //    同步后，host_ptrs 中即存放了此次推理的所有输出数据，可以安全访问。 :contentReference[oaicite:3]{index=3}
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_cls::postprocess(std::vector<Object>& objs)
{
    objs.clear();
    auto num_cls = this->output_bindings[0].dims.d[1];

    float* max_ptr =
        std::max_element(static_cast<float*>(this->host_ptrs[0]), static_cast<float*>(this->host_ptrs[0]) + num_cls);
    Object obj;
    obj.label = std::distance(static_cast<float*>(this->host_ptrs[0]), max_ptr);
    obj.prob  = *max_ptr;
    objs.push_back(obj);
}

void YOLOv8_cls::postprocessGPU(std::vector<Object>& objs)
{
    auto num_cls = this->output_bindings[0].dims.d[1];
    
    // 使用CUDA后处理
    cuda_postprocess(
        objs, 
        static_cast<const float*>(this->device_ptrs[1]), // 直接使用设备指针
        num_cls
    );
}

void YOLOv8_cls::draw_objects(const cv::Mat&                  image,
                              cv::Mat&                        res,
                              const std::vector<Object>&      objs,
                              const std::vector<std::string>& CLASS_NAMES)
{
    res = image.clone();
    char   text[256];
    Object obj = objs[0];
    sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

    int      baseLine   = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
    int      x          = 10;
    int      y          = 10;

    if (y > res.rows)
        y = res.rows;

    cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
    cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
}
