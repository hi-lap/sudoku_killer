cmake_minimum_required(VERSION 3.0)
project(sudoku_killer)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp sudoku.cpp trainer.cpp processor.cpp)
add_executable(sudoku_killer ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(sudoku_killer ${OpenCV_LIBS})