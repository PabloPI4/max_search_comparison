#include <gtest/gtest.h>
#include "../include/max_search.hpp"

/*****************************
 ***** TEST EMPTY ARRAY ******
 *****  EXPECTED ERROR  ******
 *****************************/
TEST(atomic_max_search, empty_array) {
   thrust::host_vector<int> array(0);

   auto result = max_search::atomic_max(array);

   ASSERT_FALSE(result.has_value());
   EXPECT_EQ(result.error(), max_search::SearchError::EmptyArray);
}

/*************************************
 ***** TEST ONE NUMBER IN ARRAY ******
 *****   EXPECTED THAT NUMBER   ******
 ************************************/
TEST(atomic_max_search, one_element) {
    thrust::host_vector<int> array(1, 2);

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 2);
}



/*************************************
 ***** TEST ARRAY WITH ALL ONES ******
 *****        EXPECTED 1        ******
 ************************************/
TEST(atomic_max_search_integer, all_one) {
    thrust::host_vector<int> array(64, 1);

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
}

/**********************************************
 ***** TEST ARRAY WITH EIGHTEEN 3, ONE 5 ******
 *****             EXPECTED 5            ******
 *********************************************/
TEST(atomic_max_search_integer, size_less_than_1_warp) {
    thrust::host_vector<int> array(19, 3);
    array[11] = 5;

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5);
}

/*******************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS IN ORDER EXCEPT LAST AND ANOTHER ONE ******
 *****                    EXPECTED NUMBER : 16486972                      ******
 ******************************************************************************/
TEST(atomic_max_search_integer, large_size_max_not_in_last_warp) {
    thrust::host_vector<int> array(16486973);
    for (int i = 0; i < 16486972; i++) {
        array[i] = i;
    }
    array[6203] = 16486972;
    array[16486972] = 6203;

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 16486972);
}

/*************************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS IN ORDER AND LAST ACTIVE WARP < 32 THREADS ******
 *****                     EXPECTED LAST NUMBER : 97656                         ******
 ************************************************************************************/
TEST(atomic_max_search_integer, large_size_max_in_last_warp_less_than_32_threads) {
    thrust::host_vector<int> array(97657);
    for (int i = 0; i < 97657; i++) {
        array[i] = i;
    }

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 97656);
}



/*******************************************
 ***** TEST ARRAY WITH ALL ONES FLOAT ******
 *****         EXPECTED 1.0           ******
 ******************************************/
TEST(atomic_max_search_floating_point, all_one) {
    thrust::host_vector<float> array(64, 1.0);

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 1.0);
}

/**************************************************
 ***** TEST ARRAY WITH EIGHTEEN 3.7, ONE 3.9 ******
 *****              EXPECTED 3.9             ******
 *************************************************/
TEST(atomic_max_search_floating_point, size_less_than_1_warp) {
    thrust::host_vector<float> array(19, 3.7);
    array[11] = 3.9;

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 3.9);
}

/**********************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS/10 IN ORDER EXCEPT LAST AND ANOTHER ONE ******
 *****                     EXPECTED NUMBER : 1648697.2                       ******
 *********************************************************************************/
TEST(atomic_max_search_floating_point, large_size_max_not_in_last_warp) {
    thrust::host_vector<float> array(16486973);
    for (int i = 0; i < 8191; i++) {
        array[i] = i/10.0;
    }
    array[6203] = 1648697.2;
    array[16486972] = 620.3;

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 1648697.2);
}

/****************************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS/10 IN ORDER AND LAST ACTIVE WARP < 32 THREADS ******
 *****                      EXPECTED LAST NUMBER : 9765.6                          ******
 ***************************************************************************************/
TEST(atomic_max_search_floating_point, large_size_max_in_last_warp_less_than_32_threads) {
    thrust::host_vector<float> array(97657);
    for (int i = 0; i < 97657; i++) {
        array[i] = i/10.0;
    }

    auto result = max_search::atomic_max(array);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 9765.6);
}





/*****************************
 ***** TEST EMPTY ARRAY ******
 *****  EXPECTED ERROR  ******
 *****************************/
TEST(reduction_max_search, empty_array) {
    thrust::host_vector<int> array(0);

    auto result = max_search::reduction_max(array, false);

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), max_search::SearchError::EmptyArray);
}

/*************************************
 ***** TEST ONE NUMBER IN ARRAY ******
 *****   EXPECTED THAT NUMBER   ******
 ************************************/
TEST(reduction_max_search, one_element) {
    thrust::host_vector<int> array(1, 2);

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 2);
}



/*************************************
 ***** TEST ARRAY WITH ALL ONES ******
 *****        EXPECTED 1        ******
 ************************************/
TEST(reduction_max_search_integer, all_one) {
    thrust::host_vector<int> array(64, 1);

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
}

/**********************************************
 ***** TEST ARRAY WITH EIGHTEEN 3, ONE 5 ******
 *****             EXPECTED 5            ******
 *********************************************/
TEST(reduction_max_search_integer, size_less_than_1_warp) {
    thrust::host_vector<int> array(19, 3);
    array[11] = 5;

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5);
}

/*******************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS IN ORDER EXCEPT LAST AND ANOTHER ONE ******
 *****                    EXPECTED NUMBER : 16486972                      ******
 ******************************************************************************/
TEST(reduction_max_search_integer, large_size_max_not_in_last_warp) {
    thrust::host_vector<int> array(16486973);
    for (int i = 0; i < 16486972; i++) {
        array[i] = i;
    }
    array[6203] = 16486972;
    array[16486972] = 6203;

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 16486972);
}

/*************************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS IN ORDER AND LAST ACTIVE WARP < 32 THREADS ******
 *****                     EXPECTED LAST NUMBER : 97656                         ******
 ************************************************************************************/
TEST(reduction_max_search_integer, large_size_max_in_last_warp_less_than_32_threads) {
    thrust::host_vector<int> array(97657);
    for (int i = 0; i < 97657; i++) {
        array[i] = i;
    }

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 97656);
}



/*******************************************
 ***** TEST ARRAY WITH ALL ONES FLOAT ******
 *****         EXPECTED 1.0           ******
 ******************************************/
TEST(reduction_max_search_floating_point, all_one) {
    thrust::host_vector<float> array(64, 1.0);

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 1.0);
}

/**************************************************
 ***** TEST ARRAY WITH EIGHTEEN 3.7, ONE 3.9 ******
 *****              EXPECTED 3.9             ******
 *************************************************/
TEST(reduction_max_search_floating_point, size_less_than_1_warp) {
    thrust::host_vector<float> array(19, 3.7);
    array[11] = 3.9;

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 3.9);
}

/**********************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS/10 IN ORDER EXCEPT LAST AND ANOTHER ONE ******
 *****                     EXPECTED NUMBER : 1648697.2                       ******
 *********************************************************************************/
TEST(reduction_max_search_floating_point, large_size_max_not_in_last_warp) {
    thrust::host_vector<float> array(16486973);
    for (int i = 0; i < 16486972; i++) {
        array[i] = i/10.0;
    }
    array[6203] = 1648697.2;
    array[16486972] = 620.3;

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 1648697.2);
}

/****************************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS/10 IN ORDER AND LAST ACTIVE WARP < 32 THREADS ******
 *****                      EXPECTED LAST NUMBER : 9765.6                          ******
 ***************************************************************************************/
TEST(reduction_max_search_floating_point, large_size_max_in_last_warp_less_than_32_threads) {
    thrust::host_vector<float> array(97657);
    for (int i = 0; i < 97657; i++) {
        array[i] = i/10.0;
    }

    auto result = max_search::reduction_max(array, false);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 9765.6);
}



/*************************************
 ***** TEST ARRAY WITH ALL ONES ******
 *****        EXPECTED 1        ******
 ************************************/
TEST(reduction_max_search_integer_opt, all_one) {
    thrust::host_vector<int> array(64, 1);

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 1);
}

/**********************************************
 ***** TEST ARRAY WITH EIGHTEEN 3, ONE 5 ******
 *****             EXPECTED 5            ******
 *********************************************/
TEST(reduction_max_search_integer_opt, size_less_than_1_warp) {
    thrust::host_vector<int> array(19, 3);
    array[11] = 5;

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 5);
}

/*******************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS IN ORDER EXCEPT LAST AND ANOTHER ONE ******
 *****                    EXPECTED NUMBER : 16486972                      ******
 ******************************************************************************/
TEST(reduction_max_search_integer_opt, large_size_max_not_in_last_warp) {
    thrust::host_vector<int> array(16486973);
    for (int i = 0; i < 16486972; i++) {
        array[i] = i;
    }
    array[6203] = 16486972;
    array[16486972] = 6203;

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 16486972);
}

/*************************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS IN ORDER AND LAST ACTIVE WARP < 32 THREADS ******
 *****                     EXPECTED LAST NUMBER : 97656                         ******
 ************************************************************************************/
TEST(reduction_max_search_integer_opt, large_size_max_in_last_warp_less_than_32_threads) {
    thrust::host_vector<int> array(97657);
    for (int i = 0; i < 97657; i++) {
        array[i] = i;
    }

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 97656);
}



/*******************************************
 ***** TEST ARRAY WITH ALL ONES FLOAT ******
 *****         EXPECTED 1.0           ******
 ******************************************/
TEST(reduction_max_search_floating_point_opt, all_one) {
    thrust::host_vector<float> array(64, 1.0);

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 1.0);
}

/**************************************************
 ***** TEST ARRAY WITH EIGHTEEN 3.7, ONE 3.9 ******
 *****              EXPECTED 3.9             ******
 *************************************************/
TEST(reduction_max_search_floating_point_opt, size_less_than_1_warp) {
    thrust::host_vector<float> array(19, 3.7);
    array[11] = 3.9;

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 3.9);
}

/**********************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS/10 IN ORDER EXCEPT LAST AND ANOTHER ONE ******
 *****                     EXPECTED NUMBER : 1648697.2                       ******
 *********************************************************************************/
TEST(reduction_max_search_floating_point_opt, large_size_max_not_in_last_warp) {
    thrust::host_vector<float> array(16486973);
    for (int i = 0; i < 16486972; i++) {
        array[i] = i/10.0;
    }
    array[6203] = 1648697.2;
    array[16486972] = 620.3;

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 1648697.2);
}

/****************************************************************************************
 ***** TEST LARGE ARRAY WITH NUMBERS/10 IN ORDER AND LAST ACTIVE WARP < 32 THREADS ******
 *****                      EXPECTED LAST NUMBER : 9765.6                          ******
 ***************************************************************************************/
TEST(reduction_max_search_floating_point_opt, large_size_max_in_last_warp_less_than_32_threads) {
    thrust::host_vector<float> array(97657);
    for (int i = 0; i < 97657; i++) {
        array[i] = i/10.0;
    }

    auto result = max_search::reduction_max(array, true);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result.value(), 9765.6);
}