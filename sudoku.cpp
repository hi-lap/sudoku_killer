//
// Created by Fermi on 2017/11/18.
//

#include "sudoku.h"

sudoku::sudoku(string& path) :
        _is_parsed(false),
//        _original_image(imread(path)) {
        _original_image(imread(path, CV_LOAD_IMAGE_GRAYSCALE)) {
    if (_original_image.empty()) { throw "Read file error"; }
}

sudoku::sudoku(vector<char>& bytes) :
        _is_parsed(false),
        _original_image(bytes) {
}

Mat sudoku::image() {
    return _original_image;
}

//Mat sudoku::primary_image() {
//    return _primary_image;
//}

vector<char> sudoku::detected_digits() {
    return _detected_digits;
}

vector<char> sudoku::resolved_digits() {
    return _resolved_digits;
}

//void sudoku::set_primary_image(Mat& mat) {
//    mat.copyTo(_primary_image);
//}

void sudoku::set_detected_digits(vector<char>& digits) {
    _is_parsed = true;
    _detected_digits = digits;
}

void sudoku::set_resolved_digits(vector<char>& digits) {
    _resolved_digits = digits;
}

bool sudoku::is_parsed() {
    return _is_parsed;
}

bool sudoku::is_resolved() {
//检查横竖方格内的数字之和都为45
    return false;
}