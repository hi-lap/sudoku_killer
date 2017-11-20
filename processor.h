//
// Created by Fermi on 2017/11/18.
//

#ifndef SUDOKU_KILLER_PROCESSOR_H
#define SUDOKU_KILLER_PROCESSOR_H

#include "sudoku.h"
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

class processor {
public:
    //    processor();
    explicit processor(string& model_file);
    bool tackle(sudoku s);
//    float scale();

private:
    uint32_t identify_digit(Mat& digit_mat);

private:
    Ptr<SVM> _svm;
    HOGDescriptor _hog;
    float _scale;
//    void scale();
//
//    void extract_grid();
//
//    Size find_corners();
//
//    void extract_lines(bool horizontal);
//
//    vector<Rect> find_digits();
//
//    int identify_digit(Mat& digitMat);

    using LineTestFn = function<bool(Rect&, Mat&)>;
    using ExpandRectFn = function<Rect(Rect&, Mat&)>;

private:

    // Expand rects around digit contours by this amount
    const int DIGIT_PADDING = 3;

    // Ignore any contour rect smaller than this on any side
    const int MIN_DIGIT_PIXELS = 20;

    // Scale up puzzle images smaller than this
    const int MIN_PUZZLE_SIZE = 325;

    // Scale down puzzle images larger than this
    const int MAX_PUZZLE_SIZE = 900;

    // Resize digits to this size when exporting to train SVM
    const int EXPORT_DIGIT_SIZE = 28;

    const float MIN_GRID_PCT = 0.3;

    const int CANNY_THRESHOLD = 65;
};


#endif //SUDOKU_KILLER_PROCESSOR_H
