cmake_minimum_required(VERSION 3.0)
project(neural_net C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include/)

file(GLOB PROJECT_HEADERS include/*.h)
file(GLOB PROJECT_SOURCES src/*.c)

file(GLOB PROJECT_CONFIGS CMakeLists.txt
                            Readme.md
                            .gitattributes
                            .gitignore
                            .gitmodules)

# source groups
source_group("headers" FILES ${PROJECT_HEADERS})
source_group("sources" FILES ${PROJECT_SOURCES})



add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS}
                                                  ${PROJECT_CONFIGS})


target_link_libraries(${PROJECT_NAME} PUBLIC "m")