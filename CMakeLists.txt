cmake_minimum_required(VERSION 3.5)
project(mono_neural_net
    VERSION 0.2.0
    LANGUAGES C
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build shared library" ON)
option(NN_BUILD_TEST "Build test executable" OFF)

add_library(mono_neural_net
    src/NN.c
    src/RNG.c
    src/fully_connected.c
    src/recurrent.c
)

target_include_directories(mono_neural_net
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(mono_neural_net
    PUBLIC m
)


set_target_properties(mono_neural_net PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    C_VISIBILITY_PRESET hidden
)

target_compile_options(mono_neural_net PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)


if (NN_BUILD_TEST)
    add_executable(mono_neural_net_test
        test/main.c
    )

    target_link_libraries(mono_neural_net_test
        PRIVATE mono_neural_net
    )
endif()



# install

include(GNUInstallDirs)

install(
    TARGETS mono_neural_net
    EXPORT mono_neural_netTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# CMake package

install(
    EXPORT mono_neural_netTargets
    FILE mono_neural_netTargets.cmake
    NAMESPACE mono_neural_net::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mono_neural_net
)

# config file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mono_neural_netConfig.cmake
"include(\"\${CMAKE_CURRENT_LIST_DIR}/mono_neural_netTargets.cmake\")\n"
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/mono_neural_netConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mono_neural_net
)

# pkg-config (who even still uses this?)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/mono_neural_net.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/mono_neural_net.pc
    @ONLY
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/mono_neural_net.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)