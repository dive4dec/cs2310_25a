#include "util.hpp"
#include <iostream>

void input_int(int &x) {
    // See: https://en.cppreference.com/w/cpp/io/cout
    std::cout << "Enter an integer: ";
    // See: https://en.cppreference.com/w/cpp/io/cin
    std::cin >> x;
}