add_library(max_search_lib)

target_sources(max_search_lib
    PRIVATE
        max_search.cu

    PUBLIC
        FILE_SET header_max_search_lib
        TYPE HEADERS
        BASE_DIRS
            ../include
        FILES
            ../include/max_search.cuh
)