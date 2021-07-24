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
        template <typename T, typename Operator, int width>
        struct TreeReduceImplementation {
        static T Reduce(T const (&arr)[width]) {
            #pragma HLS INLINE
            
            static constexpr int halfWidth = width / 2;
            T reduced[halfWidth];
            #pragma HLS ARRAY_PARTITION variable=reduced complete

            TreeReduce:
            for (int i = 0; i < halfWidth; ++i) {
                #pragma HLS UNROLL
                reduced[i] = Operator::Apply(arr[i * 2], arr[i * 2 + 1]);
            }

            return TreeReduceImplementation<T, Operator, halfWidth>::Reduce(reduced);
        }
        private:
            TreeReduceImplementation() = delete;
            ~TreeReduceImplementation() = delete;
        };

        template <typename T, typename Operator>
        struct TreeReduceImplementation<T, Operator, 2> {
            static T Reduce(T const (&arr)[2]) {
                #pragma HLS INLINE
                return Operator::Apply(arr[0], arr[1]);
            }
        private:
            TreeReduceImplementation() = delete;
            ~TreeReduceImplementation() = delete;
        };

        template <typename T, typename Operator>
        struct TreeReduceImplementation<T, Operator, 1> {
            static T Reduce(T const (&arr)[1]) {
                #pragma HLS INLINE
                return arr[0];
            }
        private:
            TreeReduceImplementation() = delete;
            ~TreeReduceImplementation() = delete;
        };

        template <typename T, typename Operator>
        struct TreeReduceImplementation<T, Operator, 0> {
            static constexpr T Reduce(T const (&arr)[0]) { return Operator::identity(); }
        private:
            TreeReduceImplementation() = delete;
            ~TreeReduceImplementation() = delete;
        };

    }

    template <typename T>
    struct Max {
        template <typename T0, typename T1>
        static T Apply(T0 &&a, T1 &&b) {
            #pragma HLS INLINE
            return (a > b) ? a : b;
        }
        static constexpr T identity() { return std::numeric_limits<T>::min(); }
    private:
        Max() = delete;
        ~Max() = delete;
    };

    template <typename T, int width>
    T Maximum(T const (&arr)[width]) {
        #pragma HLS INLINE 
        return TreeReduceImplementation<T, Max<T>, width>::Reduce(arr);
    }

    template <typename T>
    struct Sum {
        template <typename T0, typename T1>
        static T Apply(T0 &&a, T1 &&b) {
            #pragma HLS INLINE
            const T res = a + b;
            HLSLIB_OPERATOR_ADD_RESOURCE_PRAGMA(res);
            return res;
        }
        static constexpr T identity() { return 0; }
    private:
        Sum() = delete;
        ~Sum() = delete;
    };

    template <typename T, int width>
    T Add(T const (&arr)[width]) {
        #pragma HLS INLINE 
        return TreeReduceImplementation<T, Sum<T>, width>::Reduce(arr);
    }

}