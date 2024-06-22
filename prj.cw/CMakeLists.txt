# add_library(tsne tsne.cpp)
# target_link_libraries(tsne PRIVATE opencv_core opencv_imgproc opencv_highgui)

include(FetchContent)
FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master)

FetchContent_GetProperties(matplotplusplus)
if(NOT matplotplusplus_POPULATED)
    FetchContent_Populate(matplotplusplus)
    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_subdirectory(tsne)
add_executable(cw main.cpp)
# find_package(Matplot++ CONFIG REQUIRED)
target_link_libraries(cw PRIVATE opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(cw PRIVATE matplot)
# target_include_directories(cw PRIVATE ${CMAKE_SOURCE_DIR}/tsne)