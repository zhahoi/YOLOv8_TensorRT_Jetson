#ifndef OBB_NORMAL_YOLOv8_obb_HPP
#define OBB_NORMAL_YOLOv8_obb_HPP

#include "NvInferPlugin.h"
#include "common.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include <fstream>

using namespace obb;

const std::vector<std::string> CLASS_NAMES = {"plane", "ship", "storage tank", "baseball diamond",
                                              "tennis court", "basketball court", "ground track field",
                                              "harbor", "bridge", "large vehicle", "small vehicle",
                                              "helicopter", "roundabout", "soccer ball field", "swimming pool"};

const std::vector<std::vector<unsigned int>> COLORS = {{0, 114, 189},{217, 83, 25},{237, 177, 32},{126, 47, 142},
                                                       {119, 172, 48},{77, 190, 238},{162, 20, 47},{76, 76, 76},
                                                       {153, 153, 153},{255, 0, 0},{255, 128, 0},{191, 191, 0},
                                                       {0, 255, 0},{0, 0, 255},{170, 0, 255}};


class YOLOv8_obb {
public:
    explicit YOLOv8_obb(const std::string& engine_file_path);
    ~YOLOv8_obb();
    void make_pipe(bool warmup = true);
    void copy_from_Mat(const cv::Mat& image);
    void copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void preprocessGPU(const cv::Mat& image);
    void letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void infer();
    void postprocessGPU(std::vector<Object>& objs,
                     float                score_thres = 0.25f,
                     float                iou_thres   = 0.65f,
                     int                  topk        = 100,
                     int                  num_labels  = 15);
    void postprocess(std::vector<Object>& objs,
                     float                score_thres = 0.25f,
                     float                iou_thres   = 0.65f,
                     int                  topk        = 100,
                     int                  num_labels  = 15);

    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs,
                             const std::vector<std::string>&               CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);

    int                  num_bindings;
    int                  num_inputs  = 0;
    int                  num_outputs = 0;
    int                  dst_w = 1024;
    int                  dst_h = 1024;
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

#endif