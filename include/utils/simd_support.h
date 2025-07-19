#ifndef SIMD_SUPPORT_H
#define SIMD_SUPPORT_H

#define SIMD_AVX_LEVEL_0 0
#define SIMD_AVX_LEVEL_256 256

#if defined(ENABLE_SIMD_AVX2) && defined(__AVX2__)
    #define SIMD_AVX_LEVEL SIMD_AVX_LEVEL_256 
#else
    #define SIMD_AVX_LEVEL SIMD_AVX_LEVEL_0 
#endif

#endif