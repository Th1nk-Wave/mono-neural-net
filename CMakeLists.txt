cmake_minimum_required(VERSION 3.5)
project(neural_net C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)

add_library(neural_net
    src/NN.c
    src/RNG.c
    src/fully_connected.c
    src/recurrent.c
)

target_include_directories(neural_net
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(neural_net
    PUBLIC m
)

if (NN_BUILD_TEST)
    add_executable(neural_net_test
        test/main.c
    )

    target_link_libraries(neural_net_test
        PRIVATE neural_net
    )
endif()