//
// Created by ubuntu on 4/7/23.
//
#ifndef POSE_NORMAL_YOLOv8_pose_HPP
#define POSE_NORMAL_YOLOv8_pose_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include <fstream>

using namespace pose;

const std::vector<std::vector<unsigned int>> KPS_COLORS = {{0, 255, 0}, {0, 255, 0}, {0, 255, 0}, {0, 255, 0},
                                                           {0, 255, 0},{255, 128, 0},{255, 128, 0},{255, 128, 0},
                                                           {255, 128, 0},{255, 128, 0},{255, 128, 0},{51, 153, 255},
                                                           {51, 153, 255},{51, 153, 255}, {51, 153, 255},{51, 153, 255},
                                                           {51, 153, 255}};

const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14},{14, 12},{17, 15},{15, 13},{12, 13},
                                                         {6, 12},{7, 13},{6, 7},{6, 8},{7, 9},{8, 10},
                                                         {9, 11},{2, 3},{1, 2},{1, 3},{2, 4},{3, 5},
                                                         {4, 6},{5, 7}};
                                                         
const std::vector<std::vector<unsigned int>> LIMB_COLORS = {{51, 153, 255},{51, 153, 255},{51, 153, 255},{51, 153, 255},
                                                            {255, 51, 255},{255, 51, 255},{255, 51, 255},{255, 128, 0},
                                                            {255, 128, 0},{255, 128, 0},{255, 128, 0},{255, 128, 0},
                                                            {0, 255, 0},{0, 255, 0}, {0, 255, 0}, {0, 255, 0},
                                                            {0, 255, 0}, {0, 255, 0}, {0, 255, 0}};

class YOLOv8_pose {
public:
    explicit YOLOv8_pose(const std::string& engine_file_path);

    ~YOLOv8_pose();

    void make_pipe(bool warmup = true);

    void copy_from_Mat(const cv::Mat& image);

    void copy_from_Mat(const cv::Mat& image, cv::Size& size);

    void preprocessGPU(const cv::Mat& image);

    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void infer();

    void postprocess(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);

    void postprocessGPU(std::vector<Object>& objs, float score_thres = 0.25f, float iou_thres = 0.65f, int topk = 100);
    
    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs,
                             const std::vector<std::vector<unsigned int>>& SKELETON,
                             const std::vector<std::vector<unsigned int>>& KPS_COLORS,
                             const std::vector<std::vector<unsigned int>>& LIMB_COLORS);

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    int                  dst_w = 640;
    int                  dst_h = 640;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*       engine  = nullptr;
    nvinfer1::IRuntime*          runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif  // POSE_NORMAL_YOLOv8_pose_HPP
