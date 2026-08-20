add_executable(max_search_tests)
target_sources(max_search_tests
    PRIVATE
        max_search_tests.cu
)

find_package(GTest REQUIRED)

target_link_libraries(max_search_tests
    PRIVATE
        max_search_lib
        gtest_main
        gtest
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(max_search_tests)