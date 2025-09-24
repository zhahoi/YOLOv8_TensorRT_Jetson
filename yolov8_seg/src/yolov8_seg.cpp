#include "yolov8_seg.h"


YOLOv8_seg::YOLOv8_seg(const std::string& engine_file_path)
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

YOLOv8_seg::~YOLOv8_seg()
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

void YOLOv8_seg::make_pipe(bool warmup)
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

void YOLOv8_seg::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
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

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image)
{
    // 1. 准备一个空的 NCHW 格式容器，用于保存预处理后的图像
    cv::Mat  nchw;

    // 2. 从之前在构造函数中收集的 input_bindings 中取第一个输入 binding
    auto& in_binding = this->input_bindings[0];

    // 3. 获取该输入 Tensor 的目标宽度和高度（动态或最大 profile 下）
    //    dims.d[3] 对应 width，dims.d[2] 对应 height（NHWC -> NCHW）
    auto     width64 = in_binding.dims.d[3];
    auto     height64 = in_binding.dims.d[2];

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

void YOLOv8_seg::copy_from_Mat(const cv::Mat& image, cv::Size& size)
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

void YOLOv8_seg::preprocessGPU(const cv::Mat& image)
{
    cuda_preprocess(image.data, image.cols, image.rows, static_cast<float*>(device_ptrs[0]),
        dst_w, dst_h, stream, pparam);
}

void YOLOv8_seg::infer()
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

void YOLOv8_seg::postprocessGPU(
    std::vector<Object>& objs,  // 输出结果容器
    float score_thres,  // 置信度阈值
    float iou_thres,    // NMS IOU阈值
    int topk,           // 最多保留多少个结果
    int seg_channels,   // mask 通道数
    int seg_h,          // mask 尺寸
    int seg_w)
{
    // 1. 确定检测输出和原型输出的绑定索引
    int det_bid = -1;
    int proto_bid = -1;
    int num_channels = 0;
    int num_anchors = 0;

    // 遍历所有输出绑定，找到检测输出和原型输出
    for (int i = 0; i < this->output_bindings.size(); ++i) {
        auto& binding = this->output_bindings[i];
        
        // 检测输出是3维的 [1, num_channels, num_anchors]
        if (binding.dims.nbDims == 3) {
            det_bid = i;
            num_channels = binding.dims.d[1];
            num_anchors = binding.dims.d[2];
        }
        // 原型输出是4维的 [1, seg_channels, seg_h, seg_w]
        else if (binding.dims.nbDims == 4) {
            proto_bid = i;
        }
    }

    // 检查是否找到两个输出
    if (det_bid == -1 || proto_bid == -1) {
        std::cerr << "Error: Failed to find detection and prototype outputs" << std::endl;
        return;
    }

    // 2. 获取设备指针
    // 检测输出在 device_ptrs 中的位置：num_inputs + det_bid
    float* d_det_output = static_cast<float*>(this->device_ptrs[this->num_inputs + det_bid]);
    
    // 原型输出在 device_ptrs 中的位置：num_inputs + proto_bid
    float* d_proto_output = static_cast<float*>(this->device_ptrs[this->num_inputs + proto_bid]);

    // 3. 计算类别数
    int num_classes = num_channels - seg_channels - 4;

    // 4. 调用 CUDA 后处理函数
    cuda_seg_postprocess(
        objs,
        d_det_output,
        d_proto_output,
        num_classes,
        num_anchors,
        seg_channels,
        seg_h,
        seg_w,
        this->pparam,
        score_thres,
        iou_thres,
        topk
    );
}

void YOLOv8_seg::postprocess(
    std::vector<Object>& objs,  // 输出结果容器
    float score_thres,  // 置信度阈值
    float iou_thres,    // NMS IOU阈值
    int topk,           // 最多保留多少个结果
    int seg_channels,   // mask 通道数
    int seg_h,          // mask 尺寸
    int seg_w)
{
    // 0. 清空上一次结果
    objs.clear();

    // 1. 从输入 binding 中获取模型输入的尺寸（用于后续坐标映射）
    auto input_h = this->input_bindings[0].dims.d[2];
    auto input_w = this->input_bindings[0].dims.d[3];

    // 2. 找到输出中表示检测和 mask 的 binding 索引
    //    由于输出包括了多个 binding，我们遍历 output_bindings，
    //    找出具有 3 维 (C, H*W 或 anchors 数量) 的那个 binding，
    //    并记下它在 host_ptrs/host_binding 中的序号 bid
    int  num_channels, num_anchors, num_classes;
    bool flag = false;
    int  bid;
    int  bcnt = -1;
    for (auto& o : this->output_bindings) {
        bcnt += 1;
        // ouput name : output0 dims : 3 ouput shape : (1, 116, 8400, 0)
        if (o.dims.nbDims == 3) {
            // 这一binding的dims = {C, anchors}
            num_channels = o.dims.d[1];
            num_anchors = o.dims.d[2];
            flag = true;
            bid = bcnt;
        }
    }
    assert(flag); // 确保找到了检测输出

    // 2.1 计算类别数：总通道 = 4（box） + num_classes + seg_channels
    num_classes = num_channels - seg_channels - 4;

    // 3.恢复letterbox时的参数，用于后续坐标转换
    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    // 4. 将检测输出（host_ptrs[bid]）封装为 Mat，便于后续矩阵操作
    //    原始内存布局：num_channels × num_anchors
    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[bid]));
    // 转为 num_anchors × num_channels
    output = output.t();

    // 5. 将 mask prototype 输出（host_ptrs[其他绑定]）封装为 Mat
    //    prototype 维度：seg_channels × (seg_h * seg_w)
    cv::Mat protos = cv::Mat(seg_channels, seg_h * seg_w, CV_32F, static_cast<float*>(this->host_ptrs[1 - bid]));

    // 6. 遍历每个 anchor 输出，提取 box、score、mask 特征
    std::vector<int>      labels;
    std::vector<float>    scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat>  mask_confs;
    std::vector<int>      indices;

    for (int i = 0; i < num_anchors; i++) {
        // 6.1 获取第 i 行指针
        auto  row_ptr = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;                        // 前 4 个值为 box(x,y,w,h)
        auto  scores_ptr = row_ptr + 4;                    // 接下来 num_classes 个为类别分数
        auto  mask_confs_ptr = row_ptr + 4 + num_classes;  // 后面 seg_channels 个为 mask 特征

        // 6.2 找到 max 类别分数及其指针
        auto  max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score = *max_s_ptr;

        // 6.3 置信度筛选
        if (score > score_thres) {
            // 6.4 恢复原图坐标
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int              label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            // 6.5 保存该 anchor 对应的 mask 特征向量 (1 × seg_channels)
            cv::Mat mask_conf = cv::Mat(1, seg_channels, CV_32F, mask_confs_ptr);

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            mask_confs.push_back(mask_conf);
        }
    }

    // 7. 非极大抑制 (NMS) 去重
#if defined(BATCHED_NMS)
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    // 8. 取 topk 个结果，并构造初始 Object（不含 mask）
    cv::Mat masks;
    int     cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object   obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        objs.push_back(obj);
        cnt += 1;
    }

    // 9. 若无保留结果，直接返回
    if (masks.empty()) {
        // masks is empty
    }
    else {
        // 10. 计算 mask：A × prototype → mask Mat
        //     final_mask_feats: N × seg_channels
        //     protos: seg_channels × (seg_h*seg_w)
        //     matmulRes: N × (seg_h*seg_w)
        cv::Mat matmulRes = (masks * protos).t();
        // 11. 重塑为 N×seg_h×seg_w
        cv::Mat maskMat = matmulRes.reshape(indices.size(), { seg_h, seg_w });

        // 12. 拆分通道
        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);

        // 13. 计算 ROI 区域（去除 letterbox 填充）
        int scale_dw = dw / input_w * seg_w;
        int scale_dh = dh / input_h * seg_h;

        cv::Rect roi(scale_dw, scale_dh, seg_w - 2 * scale_dw, seg_h - 2 * scale_dh);

        // 14. 逐个生成二值 mask 并 resize 回原图大小，再裁剪到 box 区域
        for (int i = 0; i < indices.size(); i++) {
            // 14.1 Sigmoid: sigmoid(x)=1/(1+e^-x)
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);

            // 14.2 裁剪 ROI
            dest = dest(roi);

            // 14.3 插值放大到原图 size
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);

            // 14.4 依据阈值二值化，截取 box 区域
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
        }
    }
}

void YOLOv8_seg::draw_objects(const cv::Mat& image,
    cv::Mat& res,
    const std::vector<Object>& objs,
    const std::vector<std::string>& CLASS_NAMES,
    const std::vector<std::vector<unsigned int>>& COLORS,
    const std::vector<std::vector<unsigned int>>& MASK_COLORS)
{
    res = image.clone();
    cv::Mat mask = image.clone();
    
    for (auto& obj : objs) {
        // 跳过无效的边界框
        if (obj.rect.width <= 0 || obj.rect.height <= 0) {
            continue;
        }
        
        int idx = obj.label;
        cv::Scalar color = cv::Scalar(COLORS[idx][0], COLORS[idx][1], COLORS[idx][2]);
        cv::Scalar mask_color =
            cv::Scalar(MASK_COLORS[idx % 20][0], MASK_COLORS[idx % 20][1], MASK_COLORS[idx % 20][2]);
        
        // 绘制边界框
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[idx].c_str(), obj.prob * 100);
        
        // 绘制掩码（如果有效）
        if (!obj.boxMask.empty() && 
            obj.boxMask.cols == obj.rect.width && 
            obj.boxMask.rows == obj.rect.height &&
            obj.boxMask.type() == CV_8UC1) 
        {
            // 创建与掩码相同尺寸的彩色图像
            cv::Mat color_mask(obj.rect.height, obj.rect.width, CV_8UC3, mask_color);
            
            // 使用掩码将彩色区域应用到图像上
            color_mask.copyTo(mask(obj.rect), obj.boxMask);
        }

        // 绘制标签背景和文本
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = static_cast<int>(obj.rect.x);
        int y = static_cast<int>(obj.rect.y + 1);
        y = std::min(y, res.rows - 1); // 确保不超出图像范围

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);
        cv::putText(res, text, cv::Point(x, y + label_size.height), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
    
    // 合并结果
    cv::addWeighted(res, 0.5, mask, 0.8, 1, res);
}