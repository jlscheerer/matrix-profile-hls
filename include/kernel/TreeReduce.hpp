/**
 * @file    TreeReduce.hpp
 * @author  Jan Luca Scheerer (scheerer@cs.tum.edu)
 * @brief   Maximum of staticly sized array using guaranteed TreeReduction (fully pipelined)
 */

// Code is inspired by / adapted from: https://github.com/definelicht/hlslib/blob/master/include/hlslib/xilinx/TreeReduce.h
// Source: https://github.com/definelicht/hlslib
// Opposed to the original Implementation this only supports the Maximum Operator and Arrays with 2**n elements

#pragma once

namespace TreeReduce {

  namespace {
      template <typename T, int width>
      struct TreeReduceImplementation {
        static T Reduce(T const (&arr)[width]) {
          #pragma HLS INLINE
          
          static constexpr int halfWidth = width / 2;
          T reduced[halfWidth];
          #pragma HLS ARRAY_PARTITION variable=reduced complete

          TreeReduce:
          for (int i = 0; i < halfWidth; ++i) {
            #pragma HLS UNROLL
            reduced[i] = arr[i * 2] > arr[i * 2 + 1] ? arr[i * 2] : arr[i * 2 + 1];
          }

          return TreeReduceImplementation<T, halfWidth>::Reduce(reduced);
        }
      private:
        TreeReduceImplementation() = delete;
        ~TreeReduceImplementation() = delete;
      };

      template <typename T>
      struct TreeReduceImplementation<T, 2> {
        static T Reduce(T const (&arr)[2]) {
          #pragma HLS INLINE
          return arr[0] > arr[1] ? arr[0] : arr[1];
        }
      private:
        TreeReduceImplementation() = delete;
        ~TreeReduceImplementation() = delete;
      };

      template <typename T>
      struct TreeReduceImplementation<T, 1> {
        static T Reduce(T const (&arr)[1]) {
          #pragma HLS INLINE
          return arr[0];
        }
      private:
        TreeReduceImplementation() = delete;
        ~TreeReduceImplementation() = delete;
      };

      template <typename T>
      struct TreeReduceImplementation<T, 0> {
        template <typename RandomAccessType>
        static constexpr T Reduce(RandomAccessType const &) { return 0; }
      private:
        TreeReduceImplementation() = delete;
        ~TreeReduceImplementation() = delete;
      };

  }

  template <typename T, int width>
  T Maximum(T const (&arr)[width]) {
    #pragma HLS INLINE 
    return TreeReduceImplementation<T, width>::Reduce(arr);
  }

}
