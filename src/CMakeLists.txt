find_package(OpenCV REQUIRED)

add_executable(hog_svm
               main.cpp
               dataset_generating.cpp
               data_training.cpp
               object_detection.cpp)

target_link_libraries(hog_svm ${OpenCV_LIBS})
