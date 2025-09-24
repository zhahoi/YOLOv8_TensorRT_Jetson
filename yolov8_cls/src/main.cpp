//
// Created by ubuntu on 4/27/24.
//
#include "opencv2/opencv.hpp"
#include "yolov8_cls.h"
#include <chrono>


int main(int argc, char** argv)
{
    // 需要两个参数：权重文件路径、图片路径
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <tensorrt_engine.trt> <image_path>" << std::endl;
        return -1;
    }

    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string image_path{argv[2]};

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    cv::Size size        = cv::Size{224, 224};

    // 初始化模型
    auto yolov8_cls = new YOLOv8_cls(engine_file_path);
    yolov8_cls->make_pipe(true);

    std::vector<Object> objs;
    auto start = std::chrono::high_resolution_clock::now();

    // yolov8_cls->copy_from_Mat(image, size);
    yolov8_cls->preprocessGPU(image);

    yolov8_cls->infer();
    
    objs.clear();
    // yolov8_cls->postprocess(objs);
    yolov8_cls->postprocessGPU(objs);

    cv::Mat res = image.clone();
    if (!objs.empty()) {
        yolov8_cls->draw_objects(image, res, objs, CLASS_NAMES);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double tc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference time: " << tc << " ms" << std::endl;

    // 保存推理结果
    std::string out_path = "/home/nvidia/Repository/yolov8_cls/build/result.jpg";
    if (cv::imwrite(out_path, res)) {
        std::cout << "Saved result to " << out_path << std::endl;
    } else {
        std::cerr << "Failed to save result image" << std::endl;
    }

    // 可选：窗口显示
    cv::imshow("result", res);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
