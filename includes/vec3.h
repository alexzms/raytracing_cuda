//
// Created by alexzms on 2024/1/2.
//

#ifndef RAYTRACING_CUDA_VEC3_H
#define RAYTRACING_CUDA_VEC3_H

#include "cuda_helpers.h"
#include "algorithm"

namespace rt_cuda {
    template <typename T>
    class vec3 {
    public:
        __host__ __device__ vec3() : _e{0, 0, 0} {}
        __host__ __device__ vec3(T e0, T e1, T e2) : _e{e0, e1, e2} {}

        __host__ __device__ vec3 operator=(const vec3& val) {
            if (this != &val) {                                             // handle self-assignment
                this->_e[0] = val._e[0];
                this->_e[1] = val._e[1];
                this->_e[2] = val._e[2];
            }
            return *this;
        }
        __host__ __device__ vec3 operator-() const { return {-_e[0], -_e[1], -_e[2]}; }
        __host__ __device__ T operator[](int i) const { return _e[i]; }
        __host__ __device__ T& operator[](int i) { return _e[i]; }
        __host__ __device__ vec3& operator+=(const vec3& val) {
            _e[0] += val._e[0];
            _e[1] += val._e[1];
            _e[2] += val._e[2];
            return *this;
        }
        __host__ __device__ vec3& operator-=(const vec3& val) {
            _e[0] -= val._e[0];
            _e[1] -= val._e[1];
            _e[2] -= val._e[2];
            return *this;
        }
        __host__ __device__ vec3& operator*=(const vec3& val) {
            _e[0] *= val._e[0];
            _e[1] *= val._e[1];
            _e[2] *= val._e[2];
            return *this;
        }
        __host__ __device__ vec3& operator*=(const double t) {
            _e[0] *= t;
            _e[1] *= t;
            _e[2] *= t;
            return *this;
        }
        __host__ __device__ vec3& operator/=(const double t) {
            return *(this *= 1/t);
        }



        __host__ __device__ inline T x() const { return _e[0]; }
        __host__ __device__ inline T y() const { return _e[1]; }
        __host__ __device__ inline T z() const { return _e[2]; }
        __host__ __device__ inline T& x() { return _e[2]; }
        __host__ __device__ inline T& y() { return _e[2]; }
        __host__ __device__ inline T& z() { return _e[2]; }

        __host__ __device__ float length_sq() const {
            return _e[0] * _e[0] + _e[1] * _e[1] + _e[2] * _e[2];
        }
        __host__ __device__ float length() const {
            return std::sqrt(this->length_sq());
        }
        __host__ __device__ vec3 normalized() {
            return *this / this->length();
        }
        __host__ __device__ void normalize() {
            *this = *this / this->length();
        }

        template<typename U>
        friend __host__ __device__ inline vec3<U> operator+(const vec3<U>& lhs, const vec3<U>& rhs);
        template<typename U>
        friend __host__ __device__ inline vec3<U> operator-(const vec3<U>& lhs, const vec3<U>& rhs);
        template<typename U>
        friend __host__ __device__ inline vec3<U> operator*(const vec3<U>& lhs, U val);
        template<typename U>
        friend __host__ __device__ inline vec3<U> operator*(U val, const vec3<U>& rhs);
        template<typename U>
        friend __host__ __device__ inline vec3<U> operator/(const vec3<U>& lhs, U val);
        template<typename U>
        friend __host__ __device__ vec3<U> normalize(const vec3<U>& val);
        template<typename U>
        friend __host__ __device__ U dot(const vec3<U>& lhs, const vec3<U>& rhs);
        template<typename U>
        friend __host__ __device__ vec3<U> cross(const vec3<U>& lhs, const vec3<U>& rhs);

    private:
        T _e[3];
    };

    // operators
    template<typename T>
    __host__ __device__ inline vec3<T> operator+(const vec3<T>& lhs, const vec3<T>& rhs) {
        return vec3<T>(lhs._e[0] + rhs._e[0], lhs._e[1] + rhs._e[1], lhs._e[2] + rhs._e[2]);
    }
    template<typename T>
    __host__ __device__ inline vec3<T> operator-(const vec3<T>& lhs, const vec3<T>& rhs) {
        return vec3<T>(lhs._e[0] - rhs._e[0], lhs._e[1] - rhs._e[1], lhs._e[2] - rhs._e[2]);
    }
    template<typename T>
    __host__ __device__ inline vec3<T> operator*(const vec3<T>& lhs, T val) {
        return vec3<T>(lhs._e[0] * val, lhs._e[1] * val, lhs._e[2] * val);
    }
    template<typename T>
    __host__ __device__ inline vec3<T> operator*(T val, const vec3<T>& rhs) {
        return vec3<T>(rhs._e[0] * val, rhs._e[1] * val, rhs._e[2] * val);
    }
    template<typename T>
    __host__ __device__ inline vec3<T> operator/(const vec3<T>& lhs, T val) {
        return vec3<T>(lhs._e[0] / val, lhs._e[1] / val, lhs._e[2] / val);
    }

    template<typename T>
    __host__ __device__ vec3<T> normalize(const vec3<T>& val) { return val / val.length(); }
    template<typename T>
    __host__ __device__ T dot(const vec3<T>& lhs, const vec3<T>& rhs) {
        return lhs._e[0] * rhs._e[0] + lhs._e[1] * rhs._e[1] + lhs._e[2] * rhs._e[2];
    }
    template<typename T>
    __host__ __device__ vec3<T> cross(const vec3<T>& lhs, const vec3<T>& rhs) {
        return vec3<T>(lhs._e[1] * rhs._e[2] - lhs._e[2] * rhs._e[1],
                       lhs._e[2] * rhs._e[0] - lhs._e[0] * rhs._e[2],
                       lhs._e[0] * rhs._e[1] - lhs._e[1] * rhs._e[0]);
    }
}

#endif //RAYTRACING_CUDA_VEC3_H
