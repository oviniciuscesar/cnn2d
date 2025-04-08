cmake_minimum_required(VERSION 3.15)
include(${CMAKE_CURRENT_SOURCE_DIR}/Resources/pd.cmake/pd.cmake)
project(cnn2d-object)

# ╭──────────────────────────────────────╮
# │             Pd External              │
# ╰──────────────────────────────────────╯
# Get all cpp files inside ${CMAKE_SOURCE_DIR}/Sources/ and add them to the project
file(GLOB EXTERNAL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/Sources/*.c")

add_pd_external(cnn2d-object cnn2d ${EXTERNAL_SOURCES})