// Include guard to avoid double inclusion of the header file.
// See: https://en.cppreference.com/w/cpp/preprocessor/impl
// Alternative: pragma once
// See: ./staticlib.hpp
#ifndef MYMATH_HPP
#define MYMATH_HPP
namespace mymath {
int add(const int a, const int b);
int add(const int a, const int b, const int c);
int mul(const int a, const int b);
int mul(const int a, const int b, const int c);
} // namespace mymath
#endif // MYMATH_HPP