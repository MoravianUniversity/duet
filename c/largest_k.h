#pragma once

/**
 * Find the k-th largest element in an array.
 * This runs in O(n log k) time and doesn't modify the input array.
 * Best if log k is very small compared to n.
 */
int largest_k(const int * const values, int n, int k);
