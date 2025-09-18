#include <string.h>
#include <assert.h>
#include "largest_k.h"

/** Swap two integers */
static void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

/** Heapify the node at index i */
static void heapify(int* heap, int n, int i) {
    int smallest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    if (left < n && heap[left] < heap[smallest]) { smallest = left; }
    if (right < n && heap[right] < heap[smallest]) { smallest = right; }
    if (smallest != i) { swap(&heap[i], &heap[smallest]); heapify(heap, n, smallest); }
}

/**
 * Create a fixed-size min heap with the given data.
 * The smallest element is always at the root (index 0).
 */
static void build_heap(int* heap, const int * const values, int n) {
    memcpy(heap, values, n * sizeof(int));
    // Start from the last non-leaf node (parent of the last leaf) and heapify
    // all levels in reverse order
    for (int i = (n - 1) / 2; i >= 0; i--) { heapify(heap, n, i); }
}

/**
 * Find the k-th largest element in an array.
 * This runs in O(n log k) time and doesn't modify the input array.
 * Best if log k is very small compared to n.
 */
int largest_k(const int * const values, int n, int k) {
    assert(k > 0 && k <= n);

    // Create a min heap with the first k elements
    int heap[k];
    build_heap(heap, values, k);

    // Process the remaining elements
    for (int i = k; i < n; i++) {
        if (values[i] > heap[0]) {
            heap[0] = values[i];
            heapify(heap, k, 0);
        }
    }
    return heap[0];
}
