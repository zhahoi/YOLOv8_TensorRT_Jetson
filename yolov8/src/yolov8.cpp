#include "yolov8.h"

YOLOv8::YOLOv8(const std::string& engine_file_path)
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

            std::cout << "output name: " << name << " nbDims: " << dims.nbDims << " shape: (";
            for(int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << (d+1<dims.nbDims?",":"");
            std::cout << ")" << std::endl;
        }
    }
}

YOLOv8::~YOLOv8()
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

void YOLOv8::make_pipe(bool warmup)
{
    // —— 1. 为所有输入 Binding 分配 Device (GPU) 内存 —— 
    // input_bindings 存储了每个输入张量的元素数量 (binding.size)
    // 和每个元素的字节大小 (binding.dsize)
    for (auto& bindings : this->input_bindings) {
        void* d_ptr = nullptr;
        // 异步地在当前 CUDA 流上分配 GPU 内存：bindings.size * bindings.dsize 字节 
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        // 将分配好的设备指针保存到 device_ptrs 以备后续推理使用
        this->device_ptrs.push_back(d_ptr);
    }

    // —— 2. 为所有输出 Binding 分配 Device 和 Host (Page‑locked) 内存 —— 
    // 输出通常要在 GPU 上计算后再拷贝回 CPU 端做后处理，
    // 所以既需要 Device 内存，也需要 Host 端的 page‑locked (pinned) 内存以加速复制
    for (auto& bindings : this->output_bindings) {
        void* d_ptr = nullptr;
        void* h_ptr = nullptr;
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
        // 使用 device_ptrs[0]，不重复 malloc
        int dst_h = static_cast<int>(input_bindings[0].dims.d[2]);
        int dst_w = static_cast<int>(input_bindings[0].dims.d[3]);
        size_t input_bytes = 3 * dst_h * dst_w * sizeof(float);

        std::vector<float> zero(input_bytes / sizeof(float), 0.0f);

        for (int i = 0; i < 10; i++) {
            // 异步拷贝全零数据
            CHECK(cudaMemcpyAsync(device_ptrs[0], zero.data(),
                                  input_bytes,
                                  cudaMemcpyHostToDevice,
                                  this->stream));

            // 推理
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h = size.height;
    const float inp_w = size.width;
    float       height = image.rows;
    float       width = image.cols;

    float r = std::min<float>(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 });

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;
}

void YOLOv8::preprocessGPU(const cv::Mat& image)
{
    cuda_preprocess(image.data, image.cols, image.rows, static_cast<float*>(device_ptrs[0]),
        dst_w, dst_h, stream, pparam);
}

void YOLOv8::postprocessGPU(std::vector<Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();

    // 1. 获取模型输出信息
    int num_channels = this->output_bindings[0].dims.d[1]; // channels
    int num_anchors  = this->output_bindings[0].dims.d[2]; // anchors
    int num_classes  = num_channels - 4;                   // 去掉xywh

    // 2. 获取 GPU 输出指针
    int output_start_idx = static_cast<int>(this->input_bindings.size()); // device_ptrs 中输出起始位置
    float* d_output = static_cast<float*>(this->device_ptrs[output_start_idx]);

    // 3. 调用 cuda_postprocess 返回 CPU 对象列表
    std::vector<Object> tmp_objs = cuda_postprocess(
        d_output,
        num_classes,
        num_anchors,
        this->pparam,
        score_thres,
        iou_thres,
        topk
    );

    // 4. 复制到外部 objs
    objs = std::move(tmp_objs);
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    // 1. 准备一个空的 NCHW 格式容器，用于保存预处理后的图像
    cv::Mat  nchw;

    // 2. 从之前在构造函数中收集的 input_bindings 中取第一个输入 binding
    auto& in_binding = this->input_bindings[0]; 

    // 3. 获取该输入 Tensor 的目标宽度和高度（动态或最大 profile 下）
    //    dims.d[3] 对应 width，dims.d[2] 对应 height（NHWC -> NCHW）
    auto     width64 = in_binding.dims.d[3];  // 640
    auto     height64 = in_binding.dims.d[2];  // 640

    // 4. 安全检查，防止尺寸超出 int 范围
    if (width64 > INT_MAX || height64 > INT_MAX) {
        throw std::runtime_error("Input dimensions too large for cv::Size!");
    }

    // 5. 将尺寸从 64 位转换为 OpenCV 所需的 int
    cv::Size size{ static_cast<int>(width64), static_cast<int>(height64) };

    // 6. 调用 letterbox，将原始 BGR 图像：
    //    - 按比例缩放到目标大小内
    //    - 两端填充灰度（114,114,114）
    //    - 归一化并转换为 NCHW Float Blob
    this->letterbox(image, nchw, size);
    //    执行后：
    //      nchw 是 1×3×H×W 的 CV_32F 矩阵，
    //      nchw.ptr<float>() 指向连续的 NCHW 数据

    // 7. 在执行上下文中设置当前输入的动态 shape
    //    对于动态大小模型，必须在每次推理前指定具体的 H×W
    this->context->setInputShape(in_binding.name.c_str(), nvinfer1::Dims{ 4, {1, 3, height64, width64} });

    // 8. 将预分配好的 GPU 内存指针绑定到该输入 tensor
    //    这样后续 enqueueV3() 时，TensorRT 知道把输入数据从哪里读
    this->context->setTensorAddress(in_binding.name.c_str(), device_ptrs[0]);

    // 9. 异步地将 CPU 上的 blob 数据拷贝到 GPU 输入缓冲中
    //    使用之前创建的 cudaStream，做到数据拷贝与计算重叠
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], // 目标：GPU 输入缓冲区
        nchw.ptr<float>(),    // 源：NCHW blob 内存
        nchw.total() * nchw.elemSize(),  // 拷贝字节数 = H*W*3*4
        cudaMemcpyHostToDevice,  // 方向：主机 -> 设备
        this->stream // 使用本类的专属 CUDA Stream
    ));
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    // 1. 对输入图像执行 letterbox 预处理
    //    - 将原图按给定 size（width, height）缩放并填充，
    //    - 输出 NCHW 格式的 float Blob 存入 nchw
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    //    处理后：nchw 尺寸为 [1, 3, size.height, size.width]

    // 2. 获取第一个输入 binding 的名称
    //    与构造函数中输入绑定时使用的名字必须保持一致
    auto& in_binding = this->input_bindings[0];
    std::string input_name = in_binding.name;

    // 3. 设置动态 Shape
    //    对于使用了动态 profile 的模型，每次推理前都要告诉 TensorRT 本次的 H/W
    //    这里用外部传入的 size 而不是绑定时的最大 dims
    this->context->setInputShape(input_name.c_str(), nvinfer1::Dims{ 4, {1, 3, size.height, size.width} });
    //    N=1, C=3, H=size.height, W=size.width

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], // 目标：GPU 上的输入缓冲
        nchw.ptr<float>(), // 源：CPU 上的 NCHW Blob
        nchw.total() * nchw.elemSize(), // 拷贝字节数 = 1*3*H*W*4
        cudaMemcpyHostToDevice,
        this->stream
    ));
}

void YOLOv8::infer()
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

void YOLOv8::postprocess(
    std::vector<Object>& objs,   // 输出结果容器
    float score_thres,           // 置信度阈值
    float iou_thres,             // NMS IOU阈值
    int topk)                    // 最多保留多少个结果
{
    // 0. 清空上一次结果
    objs.clear();

    // 1. 获取模型输入尺寸
    auto input_h = this->input_bindings[0].dims.d[2];
    auto input_w = this->input_bindings[0].dims.d[3];

    // 2. 找到检测输出 binding
    int num_channels = 0, num_anchors = 0, num_classes = 0;
    bool found = false;
    int bid = -1;
    for (size_t bcnt = 0; bcnt < this->output_bindings.size(); ++bcnt) {
        auto& o = this->output_bindings[bcnt];
        if (o.dims.nbDims == 3) {
            num_channels = o.dims.d[1]; // C
            num_anchors = o.dims.d[2];  // N
            found = true;
            bid = bcnt;
            break;
        }
    }
    assert(found);
    num_classes = num_channels - 4;

    // 3. letterbox 参数
    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    // 4. 封装输出为 Mat
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[bid]));
    output = output.t(); // num_anchors x num_channels

    // 5. 遍历 anchor 提取 box、score、类别
    std::vector<int> labels;
    std::vector<float> scores;
    std::vector<cv::Rect_<float>> bboxes;
    std::vector<cv::Rect> int_bboxes; // NMS 需要整数类型

    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr = output.row(i).ptr<float>();
        auto box_ptr = row_ptr;        // 前4个为 box
        auto score_ptr = row_ptr + 4;  // 后面 num_classes 个为类别

        auto max_score_ptr = std::max_element(score_ptr, score_ptr + num_classes);
        float score = *max_score_ptr;

        if (score > score_thres) {
            // 恢复原图坐标
            float x = *box_ptr++ - dw;
            float y = *box_ptr++ - dh;
            float w = *box_ptr++;
            float h = *box_ptr;

            float x0 = std::clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = std::clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = std::clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = std::clamp((y + 0.5f * h) * ratio, 0.f, height);

            int label = max_score_ptr - score_ptr;
            cv::Rect_<float> bbox(x0, y0, x1 - x0, y1 - y0);
            bboxes.push_back(bbox);

            // NMS 需要整数
            int_bboxes.emplace_back(cv::Rect(
                static_cast<int>(x0),
                static_cast<int>(y0),
                static_cast<int>(x1 - x0),
                static_cast<int>(y1 - y0)
            ));

            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // 6. 非极大抑制
    std::vector<int> indices;
#if defined(BATCHED_NMS)
    cv::dnn::NMSBoxesBatched(int_bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(int_bboxes, scores, score_thres, iou_thres, indices);
#endif

    // 7. 构造最终 Object，保留 topk
    int cnt = 0;
    for (auto idx : indices) {
        if (cnt >= topk) break;

        Object obj;
        obj.rect = bboxes[idx]; // 保留浮点精度
        obj.label = labels[idx];
        obj.prob = scores[idx];
        objs.push_back(obj);
        cnt++;
    }
}


void YOLOv8::draw_objects(
    const cv::Mat& image,                         // 输入原图
    cv::Mat& res,                                 // 输出可视化图
    const std::vector<Object>& objs,              // 检测结果
    const std::vector<std::string>& CLASS_NAMES,  // 类别名称
    const std::vector<std::vector<unsigned int>>& COLORS) // 颜色列表
{
    res = image.clone(); // 拷贝原图

    for (auto& obj : objs) {
        int idx = obj.label;
        cv::Scalar color(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);

        // 1. 绘制边界框
        cv::rectangle(res, obj.rect, color, 2);

        // 2. 绘制类别 + 置信度文本
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = std::max((int)obj.rect.x, 0);
        int y = std::max((int)obj.rect.y, 0);

        // 确保文字不会超出图像边界
        if (x + label_size.width > res.cols)
            x = res.cols - label_size.width;
        if (y - label_size.height < 0)
            y = label_size.height;

        // 3. 绘制文本背景
        cv::rectangle(res, cv::Rect(x, y - label_size.height, label_size.width, label_size.height + baseLine),
                      color, cv::FILLED);

        // 4. 绘制文本
        cv::putText(res, text, cv::Point(x, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
