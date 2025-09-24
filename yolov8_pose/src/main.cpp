//
// Created by ubuntu on 4/7/23.
//
#include "opencv2/opencv.hpp"
#include "yolov8_pose.h"
#include <chrono>

/*
int main(int argc, char** argv)
{
   // cuda:0
    cudaSetDevice(0);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <tensorrt_engine.trt>" << std::endl;
        return -1;
    }

    const std::string engine_file_path{argv[1]};
    cv::Size size = cv::Size{640, 640};
    int topk = 100;
    float score_thres = 0.25f;
    float iou_thres = 0.65f;
    std::vector<Object> objs;
    cv::Mat image, res;

    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);
  
    bool firstFrameSaved = false;

    // 打开摄像头
    cv::VideoCapture cap(0, cv::CAP_V4L2); // 默认摄像头
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera" << std::endl;
        return -1;
    }

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    while (true) {
        cap >> image;
        if (image.empty()) {
            std::cerr << "Failed to capture frame from camera" << std::endl;
            break;
        }

        if (!firstFrameSaved) {
            if (cv::imwrite("/home/nvidia/zhy/yolov8_seg/build/first_frame.jpg", image)) {
                std::cout << "已保存第一帧到 first_frame.jpg" << std::endl;
            } else {
                std::cerr << "保存第一帧失败" << std::endl;
            }
            firstFrameSaved = true;
        }

        auto start = std::chrono::high_resolution_clock::now();

        // yolov8_pose->copy_from_Mat(image, size);
        yolov8_pose->preprocessGPU(image);

        std::cout << "preprocess..." << std::endl;

        yolov8_pose->infer();
        std::cout << "infer..." << std::endl;

        objs.clear();
        // yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
        yolov8_pose->postprocessGPU(objs, score_thres, iou_thres, topk);
        std::cout << "postprocess..." << std::endl;

        res = image.clone(); // 保证非空
        if (!objs.empty()) {
            yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double tc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Inference time: " << tc << " ms" << std::endl;

        cv::imshow("result", res);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    delete yolov8_pose;
    return 0;
}*/

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

    cv::Size size = cv::Size{640, 640};
    int topk = 100;
    float score_thres = 0.25f;
    float iou_thres = 0.65f;

    auto yolov8_pose = new YOLOv8_pose(engine_file_path);
    yolov8_pose->make_pipe(true);

    std::vector<Object> objs;
    auto start = std::chrono::high_resolution_clock::now();

    // yolov8_pose->copy_from_Mat(image, size);
    yolov8_pose->preprocessGPU(image);

    yolov8_pose->infer();
    
    objs.clear();

    // yolov8_pose->postprocess(objs, score_thres, iou_thres, topk);
    yolov8_pose->postprocessGPU(objs, score_thres, iou_thres, topk);

    cv::Mat res = image.clone();
    if (!objs.empty()) {
        yolov8_pose->draw_objects(image, res, objs, SKELETON, KPS_COLORS, LIMB_COLORS);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double tc = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Inference time: " << tc << " ms" << std::endl;

    // 保存推理结果
    std::string out_path = "/home/nvidia/Repository/yolov8_pose/build/result.jpg";
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