// for memalign

#if __APPLE__
#include <stdlib.h>
static void* memalign(size_t alignment, size_t size) {
    void* p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) { return NULL; }
    return p;
}
#else
#include <malloc.h>
#endif
