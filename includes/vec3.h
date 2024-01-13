//
// Created by alexzms on 2024/1/2.
//

#ifndef RAYTRACING_CUDA_VEC3_H
#define RAYTRACING_CUDA_VEC3_H


#include "cuda_helpers.h"
#include "algorithm"
#include "utilities.h"

namespace rt_cuda {
    template<typename T>
    class vec3 {
    public:
        __device__ vec3() : _e{0, 0, 0} {}

        __device__ vec3(T e0, T e1, T e2) : _e{e0, e1, e2} {}

        __device__ vec3 operator=(const vec3 &val) {
            if (this != &val) {                                             // handle self-assignment
                this->_e[0] = val._e[0];
                this->_e[1] = val._e[1];
                this->_e[2] = val._e[2];
            }
            return *this;
        }

        __device__ vec3 operator-() const { return {-_e[0], -_e[1], -_e[2]}; }

        __device__ T operator[](int i) const { return _e[i]; }

        __device__ T &operator[](int i) { return _e[i]; }

        template<typename U>
        __device__ vec3<T> &operator+=(const vec3<U> &val) {
            _e[0] += val._e[0];
            _e[1] += val._e[1];
            _e[2] += val._e[2];
            return *this;
        }

        __device__ vec3 &operator-=(const vec3 &val) {
            _e[0] -= val._e[0];
            _e[1] -= val._e[1];
            _e[2] -= val._e[2];
            return *this;
        }

        __device__ vec3 &operator*=(const vec3 &val) {
            _e[0] *= val._e[0];
            _e[1] *= val._e[1];
            _e[2] *= val._e[2];
            return *this;
        }

        __device__ vec3 &operator*=(const float t) {
            _e[0] *= t;
            _e[1] *= t;
            _e[2] *= t;
            return *this;
        }

        __device__ vec3 &operator/=(const float t) {
            return (*this) *= 1 / t;
        }

        __device__  T x() const { return _e[0]; }

        __device__  T y() const { return _e[1]; }

        __device__  T z() const { return _e[2]; }

        __device__  T &x() { return _e[2]; }

        __device__  T &y() { return _e[2]; }

        __device__  T &z() { return _e[2]; }

        __device__ float length_sq() const {
            return _e[0] * _e[0] + _e[1] * _e[1] + _e[2] * _e[2];
        }

        __device__ float length() const {
            return std::sqrt(this->length_sq());
        }

        __device__ vec3 normalized() {
            return *this / this->length();
        }

        __device__ void normalize() {
            *this = *this / this->length();
        }

        [[nodiscard]] __device__ bool near_zero() const {
            return (fabs(_e[0]) < CONSTANT_EPSILON)
                   && (fabs(_e[1]) < CONSTANT_EPSILON)
                   && (fabs(_e[2]) < CONSTANT_EPSILON);
        }

        template<typename U, typename V>
        friend __device__  vec3<U> operator+(const vec3<U> &lhs, const vec3<V> &rhs);

        template<typename U, typename V>
        friend __device__  vec3<U> operator-(const vec3<U> &lhs, const vec3<V> &rhs);

        template<typename U, typename V>
        friend __device__  vec3<U> operator*(const vec3<U> &lhs, V val);

        template<typename U, typename V>
        friend __device__  vec3<U> operator*(const vec3<U> &lhs, const vec3<V> &rhs);

        template<typename U, typename V>
        friend __device__  vec3<U> operator*(V val, const vec3<U> &rhs);

        template<typename U, typename V>
        friend __device__  vec3<U> operator/(const vec3<U> &lhs, V val);

        template<typename U>
        friend __device__ vec3<U> normalize(const vec3<U> &val);

        template<typename U>
        friend __device__ U dot(const vec3<U> &lhs, const vec3<U> &rhs);

        template<typename U>
        friend __device__ vec3<U> cross(const vec3<U> &lhs, const vec3<U> &rhs);

    protected:
        T _e[3];
    };

    template<typename U, typename V>
    __device__  vec3<U> operator+(const vec3<U> &lhs, const vec3<V> &rhs) {
        return vec3<U>(lhs._e[0] + rhs._e[0], lhs._e[1] + rhs._e[1], lhs._e[2] + rhs._e[2]);
    }

    template<typename U, typename V>
    __device__  vec3<U> operator-(const vec3<U> &lhs, const vec3<V> &rhs) {
        return vec3<U>(lhs._e[0] - rhs._e[0], lhs._e[1] - rhs._e[1], lhs._e[2] - rhs._e[2]);
    }

    template<typename U, typename V>
    __device__  vec3<U> operator*(const vec3<U> &lhs, V val) {
        return vec3<U>(lhs._e[0] * val, lhs._e[1] * val, lhs._e[2] * val);
    }

    template<typename U, typename V>
    __device__  vec3<U> operator*(V val, const vec3<U> &rhs) {
        return vec3<U>(rhs._e[0] * val, rhs._e[1] * val, rhs._e[2] * val);
    }

    template<typename U, typename V>
    __device__  vec3<U> operator*(const vec3<U> &lhs, const vec3<V> &rhs) {
        return vec3<U>(lhs._e[0] * rhs._e[0], lhs._e[1] * rhs._e[1], lhs._e[2] * rhs._e[2]);
    }

    template<typename U, typename V>
    __device__  vec3<U> operator/(const vec3<U> &lhs, V val) {
        return vec3<U>(lhs._e[0] / val, lhs._e[1] / val, lhs._e[2] / val);
    }

    template<typename T>
    __device__ vec3<T> normalize(const vec3<T> &val) { return val / val.length(); }

    template<typename T>
    __device__ T dot(const vec3<T> &lhs, const vec3<T> &rhs) {
        return lhs._e[0] * rhs._e[0] + lhs._e[1] * rhs._e[1] + lhs._e[2] * rhs._e[2];
    }

    template<typename T>
    __device__ vec3<T> cross(const vec3<T> &lhs, const vec3<T> &rhs) {
        return vec3<T>(lhs._e[1] * rhs._e[2] - lhs._e[2] * rhs._e[1],
                       lhs._e[2] * rhs._e[0] - lhs._e[0] * rhs._e[2],
                       lhs._e[0] * rhs._e[1] - lhs._e[1] * rhs._e[0]);
    }

    __device__  vec3<float> random_vecf_d(curandState *state, float min, float max) {
        return {utilities::random_float_d(state, min, max), utilities::random_float_d(state, min, max),
                utilities::random_float_d(state, min, max)};
    }

// cosine-weighted hemisphere sampling
    __device__  vec3<float> random_cosine_direction_d(curandState *state) {
        auto r1 = utilities::random_float_d(state);
        auto r2 = utilities::random_float_d(state);

        auto phi = 2 * CONSTANT_PI * r1;
        auto x = cos(phi) * sqrt(r2);
        auto y = sin(phi) * sqrt(r2);
        auto z = sqrt(1 - r2);

        return {x, y, z};
    }

    __device__ vec3<float> random_in_unit_disk_d(curandState *state) {
        while (true) {
            auto p = vec3<float>(utilities::random_float_d(state, -1, 1),
                                 utilities::random_float_d(state, -1, 1),
                                 0);
            if (p.length_sq() < 1)
                return p;
        }
    }

    __device__ vec3<float> random_unit_vector_on_sphere_d(curandState *state) {
        auto a = utilities::random_float_d(state, 0, 2 * CONSTANT_PI);
        auto z = utilities::random_float_d(state, -1, 1);
        auto r = sqrtf(1 - z * z);
        return {r * cos(a), r * sin(a), z};
//        return {0, 0,0};
    }

    using vec3f = vec3<float>;
    using point3f = vec3<float>;
    using point3i = vec3<int>;
    using color3f = vec3<float>;
}


#endif //RAYTRACING_CUDA_VEC3_H
