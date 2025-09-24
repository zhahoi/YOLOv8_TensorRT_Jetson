//
// Created by ubuntu on 2/8/23.
//
#ifndef SEGMENT_NORMAL_YOLOV8_SEG_HPP
#define SEGMENT_NORMAL_YOLOV8_SEG_HPP
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "common.hpp"
#include "preprocess.h"
#include "postprocess.h"
#include <fstream>
#include <algorithm>

using namespace seg;

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush" };

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0} };

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199} };


class YOLOv8_seg {
public:
    explicit YOLOv8_seg(const std::string& engine_file_path);
    ~YOLOv8_seg();

    void                 make_pipe(bool warmup = true);

    void                 copy_from_Mat(const cv::Mat& image);

    void                 copy_from_Mat(const cv::Mat& image, cv::Size& siWze);

    void                 preprocessGPU(const cv::Mat& image);

    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);

    void                 infer();

    void                 postprocess(std::vector<Object>& objs,
        float                score_thres = 0.25f,
        float                iou_thres = 0.65f,
        int                  topk = 100,
        int                  seg_channels = 32,
        int                  seg_h = 160,
        int                  seg_w = 160);

    void                 postprocessGPU(std::vector<Object>& objs,
        float                score_thres = 0.25f,
        float                iou_thres = 0.65f,
        int                  topk = 100,
        int                  seg_channels = 32,
        int                  seg_h = 160,
        int                  seg_w = 160);
        
    static void          draw_objects(const cv::Mat& image,
        cv::Mat& res,
        const std::vector<Object>& objs,
        const std::vector<std::string>& CLASS_NAMES,
        const std::vector<std::vector<unsigned int>>& COLORS,
        const std::vector<std::vector<unsigned int>>& MASK_COLORS);

private:
    int                  num_bindings;
    int                  num_inputs = 0;
    int                  num_outputs = 0;
    int                  dst_w = 640;
    int                  dst_h = 640;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void*>   host_ptrs;
    std::vector<void*>   device_ptrs;
    PreParam pparam;

    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t                 stream = nullptr;
    Logger                       gLogger{ nvinfer1::ILogger::Severity::kERROR };
};

#endif  // SEGMENT_NORMAL_YOLOV8_SEG_HPP