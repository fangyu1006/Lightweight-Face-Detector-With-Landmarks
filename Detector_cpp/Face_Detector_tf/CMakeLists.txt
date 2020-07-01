cmake_minimum_required(VERSION 2.8)

project(FaceDetector)

add_definitions(-std=c++11)
add_definitions("-Wall")

set(LIST_LIBS "")

find_package(OpenCV 3.3.1 REQUIRED)
if(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
    list(APPEND LIST_LIBS ${OpenCV_LIBRARIES})
else(OpenCV_FOUND)
    message(WARNING "OpenCV not found")
endif(OpenCV_FOUND)

set(Tensorflow_INCLUDE_DIRS $ENV{TENSORFLOW_DIR}/include)
set(Tensorflow_LIBRARIES $ENV{TENSORFLOW_DIR}/lib/libtensorflow.so)
INCLUDE_DIRECTORIES(${Tensorflow_INCLUDE_DIRS})
list(APPEND LIST_LIBS ${Tensorflow_LIBRARIES})

find_package(Boost 1.61.0 REQUIRED COMPONENTS filesystem system)
if (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
    list(APPEND LIST_LIBS ${Boost_LIBRARIES})
else (Boost_FOUND)
    message(FATAL_ERROR "Need Boost to be included")
endif(Boost_FOUND)
            
set(SRC
        main.cpp
        face_detector.h
        face_detector.cpp
        )

add_executable(FaceDetector ${SRC})
target_link_libraries(FaceDetector ${LIST_LIBS})


set(SRC_FOLDER
        main_folder.cpp
        face_detector.h
        face_detector.cpp
        )
add_executable(FaceDetectorFolder ${SRC_FOLDER})
target_link_libraries(FaceDetectorFolder ${LIST_LIBS})
