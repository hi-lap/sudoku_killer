//
// Created by Fermi on 2017/11/18.
//

#include <iostream>
#include "processor.h"
using namespace std;
int main() {
    string svm_path("./svm.yml");
    processor p(svm_path);

    string sudoku_path("./samples/800wi.png");
    sudoku s(sudoku_path);
    p.tackle(s);

    sudoku_path = string("./samples/article.jpg");
    s = sudoku(sudoku_path);
    p.tackle(s);

    sudoku_path = string("./samples/cbhsudoku.jpg");
    s = sudoku(sudoku_path);
    p.tackle(s);

    sudoku_path = string("./samples/Newsprint2.jpg");
    s = sudoku(sudoku_path);
    p.tackle(s);

    sudoku_path = string("./samples/NewsprintSudoku.jpg");
    s = sudoku(sudoku_path);
    p.tackle(s);

//    try {
//        sudoku_path = string("./samples/squiggly.jpg");
//        s = sudoku(sudoku_path);
//        p.tackle(s);
//    } catch (const char * s) {
//        cout << s << endl;
//    }


}