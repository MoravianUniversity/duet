#pragma once

#include <array>
#include <utility>

template <class T, size_t... Is, class Function>
constexpr std::array<T, sizeof...(Is)> __make_transformed_array(
    const std::array<T, sizeof...(Is)>& source,
    std::index_sequence<Is...>,
    Function transform
) {
    return {transform(source[Is])...};
}

template <class T, unsigned int N, class Function>
constexpr std::array<T, N> make_transformed_array(
    const std::array<T, N>& source,
    Function transform
) {
    return __make_transformed_array(source, std::make_index_sequence<N>{}, transform);
}

template <class T, size_t... Is, class Function>
constexpr std::array<T, sizeof...(Is)> __make_array(
    std::index_sequence<Is...>,
    Function make
) {
    return {make(Is)...};
}

template <class T, unsigned int N, class Function>
constexpr std::array<T, N> make_array(Function make) {
    return __make_array<T>(std::make_index_sequence<N>{}, make);
}
