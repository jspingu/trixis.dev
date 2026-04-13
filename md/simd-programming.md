SIMD Programming with Intrinsics | trixis

<div class="navigation">
    <div>
        <div class="navigation-header">
            <div class="arrow-uturn-left"></div>
            <p class="navigation-header-text">Return</p>
        </div>
        <a href="/">Home</a>
    </div>
</div>

<div class="navigation">
    <div></div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <div class="navigation-header">
            <p class="navigation-header-text">Next</p>
            <div class="chevron-right"></div>
        </div>
        <a href="/simd-motivations" style="text-align: end;">Motivations for Manual Vectorization</a>
    </div>
</div>

# SIMD Programming with Intrinsics

---

## Introduction

Scientific computing and multimedia programs often encounter problems that require operating on many homogeneous values independently of one another. These are known as *embarrassingly parallel problems*, so called because they are embarrassingly easy to split into smaller, independent tasks that can be executed in parallel.

For example, consider a video game that needs to update the display for every frame of processing. This display is a large buffer of pixels, where the output color of each pixel can be determined independently of the output colors of other pixels. Parallelizing this problem is trivial. The display can be split into smaller sub-displays, and the sub-displays can be drawn to in parallel.

Embarrassingly parallel problems open up opportunities for all kinds of optimizations. In particular, I want to focus on *SIMD* optimization—what it is, and how it can be used to accelerate parallel problems.

**SIMD** stands for **S**ame **I**nstruction, **M**ultiple **D**ata. It's a parallelism technique that takes advantage of data parallelism. When programming with SIMD instructions, a single operation can be applied to multiple values simultaneously, which can increase the throughput of a program considerably. 

Contrast this with *multiprocessing*, where multiple CPU cores with different execution contexts operate in parallel. This would be a **MIMD** (**M**ultiple **I**nstruction, **M**ultiple **D**ata) approach to parallelism, in which multiple instructions can be executed on different pieces of data at the same time.

SIMD instructions are directly supported by CPUs as extensions to their instruction set. These SIMD extensions are ubiquitous in modern day processors. If you are using a 64-bit x86 or ARM CPU, it is guaranteed that you have support for SIMD extensions that allow your processor to operate on 128-bit wide registers.

---

## Compiler Intrinsics

If you're a programmer looking to optimize your programs with SIMD, you don't actually have to touch assembly language at all. CPU designers, like Intel and ARM, publish a C interface that expose *intrinsic* functions, which are representations of individual or sequences of machine instructions. Compiler writers, in turn, take the specifications of these functions and integrate them into their compilers. When you use intrinsic functions in your code, the compiler will recognize and map them to the corresponding machine instructions wherever applicable.

Understand that this is still a high-level interface. The compiler still has the final say in what machine code is produced, so you should think of intrinsic functions as suggestions to use specific machine instructions that can be overridden.

Have a look at [Intel's intrinsics documentation here](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index#) and [ARM's intrinsics documentation here](https://developer.arm.com/architectures/instruction-sets/intrinsics/)

From this point on, I'm going to focus solely on SIMD intrinsics for the x86 architecture, but many of the concepts presented will also be applicable to ARM SIMD.

To get started, open up a new C source file and include the `<immintrin.h>` header. This will pull in most of the intrinsic functions for x86, and it's probably the only header you'll ever need.

```c
/* Pulls in most x86 intrinsic functions */
#include <immintrin.h>
```

SIMD intrinsic functions work with special, wide data types. Conventionally, these are called *vectors*, but I'm going to call them *strides* instead, because the terminology can get confusing when we start using these types to perform computations on *geometric* vectors (the kind used to represent point coordinates and directions). Several strided types are available depending on which SIMD extensions are present on the system.

The *Streaming SIMD Extensions* (SSE) instruction set, present on all 64-bit x86 processors, allow us to work with 128-bit wide strides. Beyond the SSE extensions, there are also the *Advanced Vector Extensions* (AVX), and more recently, the AVX-512 extensions, which operate on 256-bit and 512-bit strides, respectively. Support for these extensions is more limited, so you'll have to carefully consider your target platform before writing code that use them.

Floating point numbers and integers of varying bit-widths can be stored in a stride. For 128-bit strides containing single-precision (32-bit) floating point numbers, the `__m128` type is used. Being 128-bits wide, this type can hold four floating point numbers at a time, and by using intrinsic functions, we can perform parallel operations on them all at once.

Here is an example of a function that uses the `__m128` type.

```c
__m128 double_sse(__m128 in) {
    return _mm_add_ps(in, in);
}
```

This is a simple function that takes in a stride of floating point numbers, doubles each of them, and returns the result as a stride. A single intrinsic function, `_mm_add_ps`, is used to do this. The naming of these intrinsic functions can seem cryptic, but they all follow a general sequence.

1. Prefix: Depends on the extension being used
    - SSE: `_mm`
    - AVX: `_mm256`
    - AVX512: `_mm512`
2. Operation: The operation to perform, such as `add` or `mul`
3. Type: The type of data operated on, such as `ps`, which stands for "packed singles", i.e. a stride of 32-bit floating point numbers

So looking back at our function, a stride of floating point numbers will be passed in, added to itself, and then returned. This addition is performed *element-wise* in parallel, which is why it serves as a doubling function. Here's a visualization.

```txt
+---------+---------+---------+---------+
|    1    |    2    |    3    |    4    |
+---------+---------+---------+---------+

                    +

+---------+---------+---------+---------+
|    1    |    2    |    3    |    4    |
+---------+---------+---------+---------+

                    =

+---------+---------+---------+---------+
|    2    |    4    |    6    |    8    |
+---------+---------+---------+---------+
```

Another fact I hope to make clear with this example is that `__m128` and other strided types are *first-class objects*, meaning that you can treat them like any other standard type. You may pass them to and return them from functions, assign values to them, and take their address. Don't mistake them to be low-level constructs that represent the actual CPU registers used behind the scenes. The compiler will handle all register allocation and stack spilling wherever necessary.

---

## Working with Memory

SIMD code often needs to read from and write to large blocks of memory. There are a couple guidelines you should be aware of when it comes to moving data between memory and strides. The standard intrinsic functions for loading and storing memory [require the memory argument to be aligned at specific byte-boundaries](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index#ig_expand=4013,5782,3107,98,89,4769,7051,6994,6047,484,4923,4928,6994,7051,6047,4601,482,6047,4013,4013&text=load_ps), corresponding to the byte-length of the relevant SIMD type. Not doing so will raise a general-protection exception, causing your program to fail.

Of course, the memory locations you work with may not always be directly managed by yourself, so you can't always make guarantees about whether an address is suitably aligned. In such cases, be sure to use the `loadu` and `storeu` function variants, which will emit unaligned move instructions that do not have this limitation.

```c
/* 16-byte aligned memory address */
float *align16 = aligned_alloc(16, sizeof(__m256));

/* 32-byte aligned memory address */
float *align32 = aligned_alloc(32, sizeof(__m256));

/* Requires 16-byte alignment */
__m128 sd16 = _mm_load_ps(align16);       // Ok.
       sd16 = _mm_load_ps(align16 + 1);   // Segmentation fault.
       sd16 = _mm_loadu_ps(align16 + 1);  // Ok.

/* Requires 32-byte alignment */
__m256 sd32 = _mm256_load_ps(align32);    // Ok.
       sd32 = _mm256_load_ps(align16);    // Possible execption, if `align16` is not 32-byte aligned.
       sd32 = _mm256_loadu_ps(align16);   // Ok.
```

Bear in mind that **these restrictions also apply to assignment expressions**. Assigning to an lvalue of strided type will generate an aligned move instruction.

```c
/* 16-byte aligned memory address */
__m256 *align16 = aligned_alloc(16, sizeof(__m256));

/* 32-byte aligned memory address */
__m256 *align32 = aligned_alloc(32, sizeof(__m256));

/* Requires 32-byte alignment */
__m256 sd32 = *align32;                           // Ok.
       sd32 = *align16;                           // Possible execption, if `align16` is not 32-byte aligned.
       sd32 = _mm256_loadu_ps((float *)align16);  // Ok.
```

From what I understand, there is actually no difference between aligned and unaligned load/store instructions with regards to performance, at least on modern x86 processors. That being said, the alignment of the memory address **does** matter, which is to say that, when using an unaligned load/store instruction, passing in an unaligned address may result in a performance penalty, while passing in an aligned address will not.

The reason for this is cache. When computers fetch data from memory into cache, they typically read 64-byte chunks from 64-byte aligned addresses at a time. This is called a *cache line*. The issue with memory accesses at unaligned addresses is that they may cross cache line boundaries. Imagine a scenario where 16 bytes of memory need to be loaded from an address which is not aligned to 16 bytes. It is possible for these 16 bytes to start at the 56th byte of a cache line, meaning that it will overflow into the next cache line. Reading this memory will require two cache line transfers, taking more time and placing more strain on the cache. For this reason, you should prefer to allocate memory that is suitably aligned to work with the SIMD types you are using, if possible.

---

## Indexing Strides

Now let's have a look at how we can manipulate the individual elements of a stride. The `_mm_set_ps` intrinsic will allow us to set the individual floating point elements in a 128-bit stride. Below is an example of its usage.

```c
float elems[4];
__m128 sd = _mm_set_ps(1, 2, 3, 4);
_mm_storeu_ps(elems, sd);

for (int i = 0; i < 4; ++i)
    printf("%g ", elems[i]);
```

It may come as a surprise to you that the output of this program is actually "4 3 2 1 ". To see why, [look at Intel's documentation for this intrinsic](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index#text=set_ps&ig_expand=5763,5763,5763). The elements are actually specified in reverse, with the fourth element, `e3`, being passed as the first argument, and the first element, `e0`, being passed as the last argument. More interestingly, the pseudocode that Intel provides for this operation is as follows.

```txt
dst[31:0] := e0
dst[63:32] := e1
dst[95:64] := e2
dst[127:96] := e3
```

Indexes increase from **right to left**. Keep the following mental model in mind when working with strided types.

```txt
Bit 127                                 Bit 0
|                                           |
+----------+----------+----------+----------+
|    e3    |    e2    |    e1    |    e0    |
+----------+----------+----------+----------+
```

It may seem jarring at first, but this is actually a very helpful way to visualize memory. x86 is a *little-endian* architecture, which means that the bytes that make up a number are ordered from **least significant to most significant**, with byte significance increasing with higher memory addresses. We tend to think of numbers the opposite way around, writing digits in order of **most significant to least significant**.

This conflict can cause a lot of confusion, especially when it comes to bit-shift operations. A left bit-shift operation, for instance, moves bits in the direction of least significant to most significant. On a little-endian system, if we visualize ascending memory addresses from left to right, **left** shifting will actually appear to move bits to the **right**, and vice versa. For this reason, it is more helpful to visualize memory addresses growing from right to left.

Be sure to keep this in mind, because all of the SIMD intrinsic functions that involve indexing require this mental model. Take the `_mm_shuffle_epi32` intrinsic, for example, which takes in a stride of 32-bit integers and produces a new stride based on a control byte that indexes the input stride.

```c
int elems[4];
__m128i input = _mm_set_epi32(4, 3, 2, 1);
__m128i new = _mm_shuffle_epi32(input, 0b00'10'01'11);

_mm_storeu_si128((__m128i *)elems, new);

for (int i = 0; i < 4; ++i)
    printf("%i ", elems[i]);

/* Output: "4 2 3 1 " */
```

Each pair of bits in the constant `0b00'10'01'11` correspond to an element in the output stride. Each of those 2-bit values index the input stride, selecting one of its four elements to store in the corresponding element of the output stride.

Remember, elements are indexed right to left, so if the rightmost pair of bits have a value of 3 (`0b11`), it means that the element at index 0 of the output stride will have the value of the element at index 3 of the input stride, which in this case is 4.

---

## Vectorizing Loops

Now that the fine details are out of the way, let's see how we can optimize programs using SIMD intrinsics. SIMD works best when the workload consists of independent operations on large quantities of data. Non-SIMD code, or *scalar* code, will perform these computations by simply iterating over the data in a loop, performing the necessary operations on each individual data element before moving on to the next.

The process of converting scalar code into SIMD-accelerated code is called *vectorization*. Vectorized code iterates over **strides** of data elements, rather than **individual** data elements, performing operations on all of the elements within a stride at the same time.

The relevant quantity here is the number of data elements that can fit within a single stride. I like to call this quantity the *stride length*, since in a vectorized loop, this corresponds to the number of elements passed over in each iteration. For an `N`-bit wide stride and `M`-bit wide data elements, the stride length is given by `N/M`.

Consider the following scalar function, which squares each floating point element in an array.

```c
void square_scalar(float *dst, float *src, size_t length) {
    for (size_t i = 0; i < length; ++i)
        dst[i] = src[i] * src[i];
}
```

Using SSE intrinsics, we can vectorize this function. Instead of iterating through individual floating point elements, we will iterate through and square four of them at a time.

```c
void square_sse(float *dst, float *src, size_t length) {
    for (size_t i = 0; i < length; i += 4) {
        __m128 stride = _mm_load_ps(src + i);
        __m128 square = _mm_mul_ps(stride, stride);
        _mm_store_ps(dst + i, square);
    }
}
```

But now there's a problem. If `length` is not a multiple of the stride length, the remaining elements will be skipped. If you don't have control over how the buffers of memory are allocated, there's only one way to deal with this. A scalar clean-up loop will have to be placed at the end of the function to take care of the remaining elements.

```c
void square_sse(float *dst, float *src, size_t length) {
    for (size_t i = 0; i < length; i += 4) {
        __m128 stride = _mm_load_ps(src + i);
        __m128 square = _mm_mul_ps(stride, stride);
        _mm_store_ps(dst + i, square);
    }

    for (size_t i = length / 4 * 4; i < length; ++i)
        dst[i] = src[i] * src[i];
}
```

If you **do** have control over how the memory is allocated, however, you can choose to always allocate the minimum number of strides required to hold all of the elements in your array. Then, in the vectorized loop, iterate over that number of strides, which will cover the entire array without overflowing.

```c
float *alloc_strides(size_t nfloats) {
    size_t nstrides = (nfloats + 3) / 4;
    return aligned_alloc(alignof(__m128), sizeof(__m128) * nstrides);
}

/* `dst` and `src` are assumed to have been allocated with `alloc_strides` */
void square_sse(float *dst, float *src, size_t length) {
    size_t nstrides = (length + 3) / 4;

    for (size_t i = 0; i < nstrides; ++i) {
        __m128 stride = _mm_load_ps(src + i * 4);
        __m128 square = _mm_mul_ps(stride, stride);
        _mm_store_ps(dst + i * 4, square);
    }
}
```

Notice how there is no longer any code that operates on individual floating point elements anymore. We can make the syntax much more readable by using pointers to `__m128` instead of pointers to `float`.

```c
__m128 *alloc_strides(size_t nfloats) {
    size_t nstrides = (nfloats + 3) / 4;
    return aligned_alloc(alignof(__m128), sizeof(__m128) * nstrides);
}

/* `dst` and `src` are assumed to have been allocated with `alloc_strides` */
void square_sse(__m128 *dst, __m128 *src, size_t length) {
    size_t nstrides = (length + 3) / 4;

    for (size_t i = 0; i < nstrides; ++i)
        dst[i] = _mm_mul_ps(src[i], src[i]);
}
```

Great. Now the code for the vectorized function doesn't look too far off from the code for its scalar counterpart.

---

## Performance Benchmark

I will close with a benchmark that demonstrates the performance gain from utilizing SIMD. In the code below, I've implemented three different variations of the array squaring function—one scalar, one accelerated with SSE, and one accelerated with AVX. For the sake of simplicity, the buffer length is a constant, 256, which is divisible by both stride lengths and makes the buffer small enough to fit inside L1 cache. After a warm-up period, the benchmark program will time how long it takes to complete 100 million iterations for each array squaring function.


```c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#define ITERATIONS  100000000
#define LENGTH      256
#define NBENCH      3

/* Avoid aliasing issues by using `unsigned char *` */
unsigned char *dst, *src;

void square_scalar(void) {
    float *fdst = (float *)dst;
    float *fsrc = (float *)src;

    for (size_t i = 0; i < LENGTH; ++i)
        fdst[i] = fsrc[i] * fsrc[i];
}

void square_sse(void) {
    __m128 *vdst = (__m128 *)dst;
    __m128 *vsrc = (__m128 *)src;
    size_t nstrides = LENGTH * sizeof(float) / sizeof(__m128);

    for (size_t i = 0; i < nstrides; ++i)
        vdst[i] = _mm_mul_ps(vsrc[i], vsrc[i]);
}

void square_avx(void) {
    __m256 *vdst = (__m256 *)dst;
    __m256 *vsrc = (__m256 *)src;
    size_t nstrides = LENGTH * sizeof(float) / sizeof(__m256);

    for (size_t i = 0; i < nstrides; ++i)
        vdst[i] = _mm256_mul_ps(vsrc[i], vsrc[i]);
}

int main(void) {
    char *names[NBENCH] = { "Scalar", "SSE", "AVX" };
    void (*benchmarks[NBENCH])(void) = { square_scalar, square_sse, square_avx };

    dst = aligned_alloc(alignof(__m256), sizeof(float) * LENGTH);
    src = aligned_alloc(alignof(__m256), sizeof(float) * LENGTH);
    float *fdst = (float *)dst;
    float *fsrc = (float *)src;

    for (size_t i = 0; i < LENGTH; ++i) {
        fdst[i] = 0;
        fsrc[i] = rand();
    }

    puts("Warming up...");

    for (int i = 0; i < NBENCH; ++i)
        for (size_t j = 0; j < ITERATIONS; ++j)
            benchmarks[i]();

    puts("Begin benchmark");

    for (int i = 0; i < NBENCH; ++i) {
        clock_t start = clock();

        for (size_t j = 0; j < ITERATIONS; ++j)
            benchmarks[i]();

        double time = (double)(clock() - start) / CLOCKS_PER_SEC;
        printf("%s: %gs\n", names[i], time);
    }

    return 0;
}
```

gcc's autovectorizer kicks in at the `-O3` optimization level, so in order to compare performance between scalar and vectorized code, I compiled the program with the `-O2` flag instead. Below are the results after compiling with `gcc -O2 -march=native` and running the benchmark on my machine.

```txt
Scalar: 8.06339s
SSE: 1.92887s
AVX: 1.01583s
```

These results are as expected. The SSE variant ran about 4x faster than the scalar variant, and the AVX variant ran about 8x faster than the scalar variant. In theory, you can expect your program's efficiency to increase proportionally with the stride length of the underlying SIMD type you are working with.

---

*Last edited on April 13, 2026*

<div class="navigation">
    <div></div>
    <div>
        <div class="navigation-header" style="justify-content: flex-end;">
            <p class="navigation-header-text">Next</p>
            <div class="chevron-right"></div>
        </div>
        <a href="/simd-motivations">Motivations for Manual Vectorization</a>
    </div>
</div>

<div class="navigation">
    <div>
        <div class="navigation-header">
            <div class="arrow-uturn-left"></div>
            <p class="navigation-header-text">Return</p>
        </div>
        <a href="/">Home</a>
    </div>
</div>
