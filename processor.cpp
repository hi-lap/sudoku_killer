//
// Created by Fermi on 2017/11/18.
//

#include "processor.h"

processor::processor(string& model_file) :
        _svm(Algorithm::load<SVM>(model_file)), _scale(1.0),
        _hog(HOGDescriptor(Size(28, 28), Size(14, 14), Size(7, 7), Size(14, 14), 9, 1, -1, 0, 0.2, 0, 64, 1)) {
}

bool processor::tackle(sudoku s) {
//    Mat original_image = s.original_image();
    Mat gray = s.image();
//    if (!original_image.data) {
//        cerr << "Problem loading image!!!" << endl;
//    }

//    if (original_image.type() == 2) {
//        original_image.convertTo(original_image, CV_8U, 0.00390625);
//    }


//    Mat cleanedBoard;
    vector<double> gPoints;
    float scale = 1.0;

    // Transform source image to gray if it is not
//    Mat gray;
//    if (original_image.channels() == 3) {
//        cvtColor(original_image, gray, CV_BGR2GRAY);
//    } else {
//        gray = original_image;
//    }

    // make sure image is a reasonable size
//    _scale = scale();
    if (gray.rows > MAX_PUZZLE_SIZE || gray.cols > MAX_PUZZLE_SIZE) {
        scale = max(gray.rows, gray.cols) / float(MAX_PUZZLE_SIZE);
        resize(gray, gray, Size(gray.cols / scale, gray.rows / scale), 0, 0, CV_INTER_AREA);
    } else if (gray.rows < MIN_PUZZLE_SIZE || gray.cols < MIN_PUZZLE_SIZE) {
        scale = min(gray.rows, gray.cols) / float(MAX_PUZZLE_SIZE);
        resize(gray, gray, Size(gray.cols / scale, gray.rows / scale), 0, 0, CV_INTER_CUBIC);
    }


    Mat grid = Mat::zeros(gray.size(), gray.type());

    Mat src_gray;
    blur(gray, src_gray, Size(3, 3));
    Mat denoised;
    fastNlMeansDenoising(src_gray, denoised, 10);
    adaptiveThreshold(~denoised, src_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, -2);

    Mat canny_output;
    Canny(src_gray, canny_output, CANNY_THRESHOLD, CANNY_THRESHOLD * 2, 3);

    //findContours
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
    double largest_area = 0;
    size_t largest_contour_index = 0;
    vector<Point> largestContour;
    Rect bounding_rect;
    for (size_t i = 0; i < contours.size(); i++) {
        double a = contourArea(contours[i], false);  //  Find the area of contour
        if (a > largest_area) {
            largest_area = a;
            largest_contour_index = i;                //Store the index of largest contour
            largestContour = contours[i];
            bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
        }
    }

    float area = gray.cols * gray.rows;
    if (largest_area < area * MIN_GRID_PCT) {
        cout << "largest contour area is only " << (largest_area / area) * 100
             << "% of source; aborting grid extraction" << endl;
        gray.copyTo(grid);
    } else if (largest_contour_index < contours.size()) {// findCorners
        Point2f corners[4];
        Point2f flatCorners[4];
//        Size sz = findCorners(largestContour, corners);

        double dist;
//        float dist;
        double maxDist[4] = {0, 0, 0, 0};
//        float maxDist[4] = {0, 0, 0, 0};

        Moments M = moments(largestContour, true);

//        float cX = float(M.m10 / M.m00);
//        float cY = float(M.m01 / M.m00);
        double cX = M.m10 / M.m00;
        double cY = M.m01 / M.m00;

        for (int i = 0; i < 4; i++) {
            maxDist[i] = 0.0f;
            corners[i] = Point2d(cX, cY);
        }

        Point2d center(cX, cY);

        // find the most distant point in the contour within each quadrant
        for (int i = 0; i < largestContour.size(); i++) {
            Point2f p(largestContour[i].x, largestContour[i].y);
            dist = sqrt(pow(p.x - center.x, 2) + pow(p.y - center.y, 2));
            if (p.x < cX && p.y < cY && maxDist[0] < dist) {
                // top left
                maxDist[0] = dist;
                corners[0] = p;
            } else if (p.x > cX && p.y < cY && maxDist[1] < dist) {
                // top right
                maxDist[1] = dist;
                corners[1] = p;
            } else if (p.x > cX && p.y > cY && maxDist[2] < dist) {
                // bottom right
                maxDist[2] = dist;
                corners[2] = p;
            } else if (p.x < cX && p.y > cY && maxDist[3] < dist) {
                // bottom left
                maxDist[3] = dist;
                corners[3] = p;
            }
        }

        double widthTop = sqrt(pow(corners[0].x - corners[1].x, 2) + pow(corners[0].y - corners[1].y, 2));
        double widthBottom = sqrt(pow(corners[2].x - corners[3].x, 2) + pow(corners[2].y - corners[3].y, 2));

        double heightLeft = sqrt(pow(corners[0].x - corners[3].x, 2) + pow(corners[0].y - corners[3].y, 2));
        double heightRight = sqrt(pow(corners[1].x - corners[2].x, 2) + pow(corners[1].y - corners[2].y, 2));

        Size2d sz(max(widthTop, widthBottom), max(heightLeft, heightRight));

        // draw contour quadrangle
        for (int j = 0; j < 4; j++) {
//            gridPoints.push_back(corners[j].x * scale);
//            gridPoints.push_back(corners[j].y * scale);

            gPoints.push_back(corners[j].x * scale);
            gPoints.push_back(corners[j].y * scale);
        }
        flatCorners[0] = Point2f(0, 0);
        flatCorners[1] = Point2f(sz.width, 0);
        flatCorners[2] = Point2f(sz.width, sz.height);
        flatCorners[3] = Point2f(0, sz.height);
        Mat lambda = getPerspectiveTransform(corners, flatCorners);

        Mat output;
        warpPerspective(denoised, output, lambda, sz);

        GaussianBlur(output, grid, Size(0, 0), 3);
        addWeighted(output, 1.5, grid, -0.5, 0, grid);
    } else {
        throw runtime_error("No grid contour found");
    }

    // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    Mat bw;
    adaptiveThreshold(~grid, bw, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, -2);

    Mat lines = Mat::zeros(bw.size(), bw.type());

    {
        Mat clone = bw.clone();

        LineTestFn lineTest;
        ExpandRectFn expandRect;
        Size size;
        size = Size(bw.cols / 9, 1);
        lineTest = [](Rect& rect, Mat& mat) {
            return rect.height / double(mat.rows) < 0.05 && rect.width / double(mat.cols) > 0.111;
        };
        expandRect = [](Rect& rect, Mat& mat) {
            Rect expanded = rect;
            if (expanded.y > 1) { expanded.y -= 2; }

            if (expanded.y + expanded.height < mat.rows) {
                expanded.height += min(4, mat.rows - expanded.y - expanded.height);
            }
            expanded.x = 0;
            expanded.width = mat.cols;
            return expanded;
        };

        // Create structure element for extracting lines through morphology operations
        Mat structure = getStructuringElement(MORPH_RECT, size);

        // Apply morphology operations
        erode(clone, clone, structure, Point(-1, -1));
        dilate(clone, clone, structure, Point(-1, -1));

        // Find all contours
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(clone, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());

        // Mark contours which pass line test in the destination image
        for (size_t i = 0; i < contours.size(); i++) {
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));
            if (lineTest(boundRect[i], clone)) {
                Rect expanded = expandRect(boundRect[i], clone);
                lines(expanded) |= 255; // set the expanded rect to white
            }
        }
    }

    {
        Mat clone = bw.clone();

        LineTestFn lineTest;
        ExpandRectFn expandRect;
        Size size;
        size = Size(1, bw.rows / 9);
        lineTest = [](Rect& rect, Mat& mat) {
            return rect.width / double(mat.cols) < 0.05 && rect.height / double(mat.rows) > 0.111;
        };
        expandRect = [](Rect& rect, Mat& mat) {
            Rect expanded = rect;
            if (expanded.x > 1) { expanded.x -= 2; }

            if (expanded.x + expanded.width < mat.cols) {
                expanded.width += min(4, mat.cols - expanded.x - expanded.width);
            }
            expanded.y = 0;
            expanded.height = mat.rows;
            return expanded;
        };

        // Create structure element for extracting lines through morphology operations
        Mat structure = getStructuringElement(MORPH_RECT, size);

        // Apply morphology operations
        erode(clone, clone, structure, Point(-1, -1));
        dilate(clone, clone, structure, Point(-1, -1));

        // Find all contours
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(clone, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());

        // Mark contours which pass line test in the destination image
        for (size_t i = 0; i < contours.size(); i++) {
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));
            if (lineTest(boundRect[i], clone)) {
                Rect expanded = expandRect(boundRect[i], clone);
                lines(expanded) |= 255; // set the expanded rect to white
            }
        }
    }




    // subtract grid lines from the black/white image
    // so they don't interfere with digit detection
    Mat clean = bw - lines;
    blur(clean, clean, Size(1, 1));

    vector<Rect> digits;

    {
//        vector<Rect> digits;
        Rect imgRect = Rect(0, 0, clean.cols, clean.rows);

        // Find all contours
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        //findContours( img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
        findContours(clean, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
        vector<vector<Point> > contours_poly(contours.size());
        vector<Rect> boundRect(contours.size());

        for (size_t i = 0; i < contours.size(); i++) {
            approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
            boundRect[i] = boundingRect(Mat(contours_poly[i]));

            if (hierarchy[i][3] < 0 && boundRect[i].height >= MIN_DIGIT_PIXELS) { // find "root" contours
                double aspectRatio = boundRect[i].height / double(boundRect[i].width);
                // check reasonable aspect ratio for digits
                if ((aspectRatio >= 1 && aspectRatio < 3.2)) {
                    int widthToAdd = boundRect[i].height - boundRect[i].width + (2 * DIGIT_PADDING);
                    int pointOffset = int(floor(double(widthToAdd / 2)));
                    boundRect[i] = boundRect[i] - Point(pointOffset, DIGIT_PADDING);
                    boundRect[i] = boundRect[i] + Size(widthToAdd, 2 * DIGIT_PADDING);
                    boundRect[i] &= imgRect;

                    // check white/black pixel ratio to avoid accidental noise getting picked up
                    double wbRatio =
                            countNonZero(clean(boundRect[i])) / double(boundRect[i].width * boundRect[i].height);
                    if (wbRatio > 0.1 && wbRatio < 0.4) {
                        digits.push_back(boundRect[i]); // AND boundRect and imgRect to ensure in boundary
                    }
                }
            }
        }
    }

    map<string, int> digitMap;
    float gridPoints[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    if (digits.size() > 0) {
        // get the bounding box of all digits
        Rect allDigits = digits[0];
        for (size_t i = 0; i < digits.size(); i++) {
            allDigits |= digits[i];
        }
//        if (gPoints.size() == 0) {
        if (gPoints.empty()) {
            gridPoints[0] = allDigits.x * scale;
            gridPoints[1] = allDigits.y * scale;
            gridPoints[2] = (allDigits.x + allDigits.width) * scale;
            gridPoints[3] = allDigits.y * scale;
            gridPoints[4] = allDigits.br().x * scale;
            gridPoints[5] = allDigits.br().y * scale;
            gridPoints[6] = allDigits.x * scale;
            gridPoints[7] = (allDigits.y + allDigits.height) * scale;
        }

        double cellWidth = allDigits.width / 9.0;
        double cellHeight = allDigits.height / 9.0;

        Mat digitBounds = clean.clone();
        Scalar pink = Scalar(255, 105, 180);
        Scalar teal = Scalar(20, 135, 128);
//        if (saveOutput) {
//            cvtColor( digitBounds, digitBounds, COLOR_GRAY2BGR );
//        }

        for (size_t i = 0; i < digits.size(); i++) {
            Point center = (digits[i].br() + digits[i].tl()) * 0.5;
            int row = int(floor((center.y - allDigits.y) / cellHeight));
            char rowChar = "ABCDEFGHI"[row];
            int col = int(floor((center.x - allDigits.x) / cellWidth));

            // save the digit
            Mat digitImg = Mat(clean, digits[i]);
            resize(digitImg, digitImg, Size(EXPORT_DIGIT_SIZE, EXPORT_DIGIT_SIZE), 0, 0, CV_INTER_AREA);
            // despeckle
            fastNlMeansDenoising(digitImg, digitImg, 50.0, 5, clean.cols / 10);

            uint32_t digit = identify_digit(digitImg);

//            cout << digit << endl;
//            if (saveOutput) {
//                rectangle( digitBounds, digits[i], pink, 1, 8, 0 );
//                putText(digitBounds, to_string(digit), center + Point(5, 12), FONT_HERSHEY_PLAIN, 0.8, teal);
//            }

            digitMap[string(1, rowChar) + to_string(col + 1)] = digit;
        }

        rectangle(digitBounds, allDigits, teal, 1, 8, 0);
    }

    string puzzle = "";
    for (int y = 0; y < 9; y++) {
        for (int x = 0; x < 9; x++) {
            string key = string(1, "ABCDEFGHI"[y]) + to_string(x + 1);
            auto search = digitMap.find(key);
            if (search != digitMap.end()) {
                puzzle += to_string(search->second);
            } else {
                puzzle += ".";
            }

        }
    }

    cout << "puzzle parsed as " << puzzle << endl;

//    if (gPoints.size() == 8) {
//        copy(gPoints.begin(), gPoints.end(), gridPoints);
//    }

    return false;
}

uint32_t processor::identify_digit(Mat& digit_mat) {
    vector<float> descriptors;
    vector<Point> positions;

    // Get HOG descriptor
    _hog.compute(digit_mat, descriptors, Size(), Size(), positions);
    //cout << "Computed HOGDescriptor for " << digitMat.cols << "x" << digitMat.rows << " image" << endl;
    // convert HOG descriptor to Mat
    Mat testMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
    for (int x = 0; x < testMat.cols; x++) {
        testMat.at<float>(0, x) = descriptors[x];
    }

    // predict digit
    Mat testResponse;
    imwrite("testdigit.png", digit_mat);
    _svm->predict(testMat, testResponse);
    //cout << "Predicted digit for " << digitMat.cols << "x" << digitMat.rows << " image" << endl;
    // extract prediction
    return uint32_t(testResponse.at<float>(0, 0));
}