//
// Created by Fermi on 2017/11/20.
//

#ifndef SUDOKU_KILLER_SUDOKU_H
#define SUDOKU_KILLER_SUDOKU_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class sudoku {
public:
    explicit sudoku(string& path);
    explicit sudoku(vector<char>& bytes);

    Mat image();
//    Mat primary_image();
    vector<char> detected_digits();
    vector<char> resolved_digits();
    bool is_parsed();
    bool is_resolved();

//    void set_primary_image(Mat& mat);
    void set_detected_digits(vector<char>& digits);
    void set_resolved_digits(vector<char>& digits);

private:
    Mat _original_image;
//    Mat _primary_image;
    vector<char> _detected_digits;
    vector<char> _resolved_digits;
    bool _is_parsed;
};

#endif //SUDOKU_KILLER_SUDOKU_H
