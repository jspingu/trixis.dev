Scalable Vectorization | trixis

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
    <div>
        <div class="navigation-header">
            <div class="chevron-left"></div>
            <p class="navigation-header-text">Previous</p>
        </div>
        <a href="/simd-motivations">Motivations for Manual Vectorization</a>
    </div>
</div>

# Scalable Vectorization

---

## Units of Computation

I want to expand upon my points on structured data from the previous article, because it marks a subtle, but important shift in perspective on how SIMD types should be treated. This shift in perspective is key to understanding how vectorized programs can be written portably.

Consider the following two approaches to adding 4-dimensional vectors together using SIMD.

The first approach interleaves vector components within a stride. Assuming the stride length is four, one vector can fit snugly within a single stride. Adding vectors then, becomes as simple as adding two strides together, using packed SIMD arithmetic. With this approach, one strided addition yields one vector result.

```txt
+---------+---------+---------+---------+
|    x    |    y    |    z    |    w    |
+---------+---------+---------+---------+

                    +

+---------+---------+---------+---------+
|    x    |    y    |    z    |    w    |
+---------+---------+---------+---------+
```

The second approach is to hold multiple, *homogeneous* vector components within a single stride. By doing this, we lose the ability to add a single pair of vectors, instead requiring the addition of four pairs at a time. Here, four strided additions yield four vector results.

```txt
+---------+---------+---------+---------+  +---------+---------+---------+---------+  +---------+  +---------+
|    x    |    x    |    x    |    x    |  |    y    |    x    |    y    |    y    |  |   ...   |  |   ...   |
+---------+---------+---------+---------+  +---------+---------+---------+---------+  +---------+  +---------+
                                                                                                              
                    +                                          +                           +            +     
                                                                                                              
+---------+---------+---------+---------+  +---------+---------+---------+---------+  +---------+  +---------+
|    x    |    x    |    x    |    x    |  |    y    |    y    |    y    |    y    |  |   ...   |  |   ...   |
+---------+---------+---------+---------+  +---------+---------+---------+---------+  +---------+  +---------+
```

Note that both approaches offer the same throughput, with an average of one vector result produced for each packed SIMD addition.

When starting out with SIMD programming, it's easy to find yourself in favor of the first approach, because it appears to be more simple and flexible. However, as elaborated upon in the previous article, this approach falls apart as soon as we introduce the need for operations between **different** vector components.

The primary advantage of the second approach is that by only holding homogeneous elements within strides, each stride can be treated as an *indivisible unit of computation* in the same way that, for example, `float` data types are. There is never any structure within a stride that hides the complexity of the underlying computation being performed. Thinking back to the example of adding 4D vectors, this is the difference between having vector addition be composed of four different addition operations—arguably the more natural approach that mimics scalar code—rather than adding an entire pair of vectors at once.

Within a loop that performs independent computations on a collection of data values, using this approach makes vectorization trivial, because each scalar instruction can be directly mapped to its SIMD counterpart. In this way, operations on the elements of a stride become analogous to consecutive scalar operations across multiple iterations of a loop.

I'll illustrate this using the cross product operation, which is a fairly complicated operation between 3D vectors that involves multiplications and subtractions between different vector components. The following is a scalar implementation of a function that computes the cross product between vectors in two arrays.

```c
typedef struct vec3 {
    float x, y, z;
} vec3;

void cross_scalar(vec3 *dst, vec3 *lhs, vec3 *rhs, size_t length) {
    for (size_t i = 0; i < length; ++i)
        dst[i] = (vec3) {
            .x = lhs[i].y * rhs[i].z - lhs[i].z * rhs[i].y,
            .y = lhs[i].z * rhs[i].x - lhs[i].x * rhs[i].z,
            .z = lhs[i].x * rhs[i].y - lhs[i].y * rhs[i].x
        };
}
```

Using the AoSoA data layout, we can vectorize this using SSE intrinsics easily. All we need to do is substitute the `vec3` structure for a strided 3D vector structure, adjust the number of loop iterations according to the stride length, and substitute each arithmetic operation with its SSE counterpart.

```c
typedef struct sd_vec3 {
    __m128 x, y, z;
} sd_vec3;

void cross_sse(sd_vec3 *dst, sd_vec3 *lhs, sd_vec3 *rhs, size_t length) {
    size_t sd_len = (length + 3) / 4;

    /*
     * Each packed arithmetic instruction here corresponds to the same scalar
     * instruction across four consecutive loop iterations in `cross_scalar`
     */
    for (size_t i = 0; i < sd_len; ++i)
        dst[i] = (sd_vec3) {
            .x = _mm_sub_ps(_mm_mul_ps(lhs[i].y, rhs[i].z), _mm_mul_ps(lhs[i].z, rhs[i].y)),
            .y = _mm_sub_ps(_mm_mul_ps(lhs[i].z, rhs[i].x), _mm_mul_ps(lhs[i].x, rhs[i].z)),
            .z = _mm_sub_ps(_mm_mul_ps(lhs[i].x, rhs[i].y), _mm_mul_ps(lhs[i].y, rhs[i].x)),
        };
}
```

Notice how consistent this process is, regardless of the underlying SIMD extension. If we wanted to vectorize using AVX, it's as simple as adjusting for a different stride length and substituting in the relevant AVX types and functions.

```c
typedef struct sd_vec3 {
    __m256 x, y, z;
} sd_vec3;

void cross_avx(sd_vec3 *dst, sd_vec3 *lhs, sd_vec3 *rhs, size_t length) {
    size_t sd_len = (length + 7) / 8;

    for (size_t i = 0; i < sd_len; ++i)
        dst[i] = (sd_vec3) {
            .x = _mm256_sub_ps(_mm256_mul_ps(lhs[i].y, rhs[i].z), _mm256_mul_ps(lhs[i].z, rhs[i].y)),
            .y = _mm256_sub_ps(_mm256_mul_ps(lhs[i].z, rhs[i].x), _mm256_mul_ps(lhs[i].x, rhs[i].z)),
            .z = _mm256_sub_ps(_mm256_mul_ps(lhs[i].x, rhs[i].y), _mm256_mul_ps(lhs[i].y, rhs[i].x)),
        };
}
```

By enforcing the homogeneity of stride elements, we can see an interesting pattern emerge. Both vectorized functions, `cross_sse` and `cross_avx`, share the exact same structure despite the fact that they use different SIMD extensions. In fact, `cross_scalar` can also be seen as a variation of this pattern, where the stride length is one.

This hints at a possible method for writing vectorized code that is *scalable* to multiple SIMD extensions with different stride lengths.

---

## Vector-Length Agnostic Programming

Writing vectorized code in a way that does not depend on any particular stride length is known as *Vector-Length Agnostic* (VLA) programming. To write VLA code, we must rely on a set of abstractions that represent variable-width SIMD types and functions.

First, fixed-width types need to go. Data types like `float`, `__m128`, and `__m256` are specific, fixed-width implementations of strides that hold floating point numbers. To generalize them, let's declare an abstract `sd_float` type, which represents a variable-width stride of floating point numbers.

```c
typedef /* Implementation-defined */ sd_float;
```

Both the stride length and alignment can be computed based on this abstract type. The stride length depends on the bit-width of the underlying SIMD type, as well as the size of the elements being stored. For the sake of simplicity, I'll assume that only 32-bit elements will be stored.

```c
constexpr size_t SD_LENGTH = sizeof(sd_float) / sizeof(unsigned char [4]);
constexpr size_t SD_ALIGN = alignof(sd_float);
```

Moving on, operations like `+`, `_mm_add_ps`, and `_mm256_add_ps` also depend on specific stride lengths. These must be abstracted away into functions that operate on and return variable-width strides.

```c
sd_float sd_float_add(sd_float lhs, sd_float rhs) {
    /* Implementation-defined */
}
```

The above abstractions form the building blocks for writing VLA programs. Their power comes from the fact that their implementations can be swapped out at any time to use different SIMD extensions, and by doing so, all dependent VLA code will automatically switch to using those extensions. What we need now, of course, is a piece of software that automates the process of filling in those implementations depending on what SIMD extensions are present on the target system. The C preprocessor is suitable for this task.

Most C compilers will predefine a handful of macros based on enabled x86 CPU extensions. By using conditional compilation directives, we can compile the appropriate SIMD implementations of our VLA abstractions according to which of these macros are defined. I will demonstrate this with the following predefined macros.

```txt
__SSE2__
__AVX2__
__AVX512F__
```

For each SIMD extension corresponding to these macros, we will write an implementation of each VLA type and function using that extension's intrinsic types and functions, guarding its inclusion through conditional compilation. In the case that *none* of the macros are defined, a scalar fallback implementation will be provided.

All of the above definitions can be placed in a header file, which is to be included by source files using VLA abstractions. Since function definitions are directly included in the header, they will be marked as `static`—for internal linkage—and `inline`—for efficient code generation.

So far, the VLA SIMD wrapper looks like the following.

```c
#ifdef __AVX512F__
    typedef __m512 sd_float;
#elifdef __AVX2__
    typedef __m256 sd_float;
#elifdef __SSE2__
    typedef __m128 sd_float;
#else
    typedef float sd_float;
#endif

constexpr size_t SD_LENGTH = sizeof(sd_float) / sizeof(unsigned char [4]);
constexpr size_t SD_ALIGN = alignof(sd_float);

static inline sd_float sd_float_add(sd_float lhs, sd_float rhs) {
#ifdef __AVX512F__
    return _mm512_add_ps(lhs, rhs);
#elifdef __AVX2__
    return _mm256_add_ps(lhs, rhs);
#elifdef __SSE2__
    return _mm_add_ps(lhs, rhs);
#else
    return lhs + rhs;
#endif
}
```

We can easily extend our VLA programming interface by building on top of these abstractions. One useful function to define is a *bounding size* function, which is a generalization of the computation that determines the number of strides needed to hold a given number of elements. This can be defined as follows, slightly modified from previous examples to handle the overflow case.

```c
static inline size_t sd_bounding_size(size_t n) {
    return n ? (n - 1) / SD_LENGTH + 1 : 0;
}
```

Let's now see an example of a VLA program using this interface. Below is a function that doubles each floating point number in an array, implemented using VLA abstractions.

```c
void double_vla(sd_float *dst, sd_float *src, size_t length) {
    size_t sd_len = sd_bounding_size(length);

    for (size_t i = 0; i < sd_len; ++i)
        dst[i] = sd_float_add(src[i], src[i]);
}
```

Now, by compiling this code with different SIMD extensions enabled, we can automatically generate variants of the `double_vla` function that use different SIMD instructions, without having to maintain multiple implementations of `double_vla` itself.

We can visualize how this works by passing in the `-E` flag to the compiler, which instructs it to output the source code after preprocessing. For example, when compiling with the `-mavx2` flag, the `__AVX2__` macro will be defined, so the code for the VLA abstractions will look like the following.

```c
typedef __m256 sd_float;

constexpr size_t SD_LENGTH = sizeof(sd_float) / sizeof(unsigned char [4]);
constexpr size_t SD_ALIGN = alignof(sd_float);

static inline size_t sd_bounding_size(size_t n) {
    return n ? (n - 1) / SD_LENGTH + 1 : 0;
}

static inline sd_float sd_float_add(sd_float lhs, sd_float rhs) {
    return _mm256_add_ps(lhs, rhs);
}
```

And so, with AVX2 extensions enabled, these are the VLA implementations that will be used. Optimizing compilers will directly inline this code into `double_vla`, so it'd be as if it was written with AVX intrinsics in the first place.

This brief overview should give you a good idea of the flexibility and power of VLA programming. To close this section, let's revisit the array cross product function, this time implemented using VLA abstractions to demonstrate its capability in more complex scenarios. The incredible thing to note here is that our previous implementations of this function (`cross_scalar`, `cross_sse`, and `cross_avx`) have all been distilled into this general form.

```c
typedef struct sd_vec3 {
    sd_float x, y, z;
} sd_vec3;

void cross_vla(sd_vec3 *dst, sd_vec3 *lhs, sd_vec3 *rhs, size_t length) {
    size_t sd_len = sd_bounding_size(length);

    for (size_t i = 0; i < sd_len; ++i)
        dst[i] = (sd_vec3) {
            .x = sd_float_sub(sd_float_mul(lhs[i].y, rhs[i].z), sd_float_mul(lhs[i].z, rhs[i].y)),
            .y = sd_float_sub(sd_float_mul(lhs[i].z, rhs[i].x), sd_float_mul(lhs[i].x, rhs[i].z)),
            .z = sd_float_sub(sd_float_mul(lhs[i].x, rhs[i].y), sd_float_mul(lhs[i].y, rhs[i].x)),
        };
}
```

---

## Dynamic Dispatch

Even though we now have this powerful VLA programming interface, the fact still remains that our compiled code only supports one SIMD extension at a time. This means that when distributing binaries, we'd have to build multiple versions of the same binary for different SIMD extensions, and then have the end user select the appropriate one to install for their system. This method of distribution is obviously ridiculous. We can't expect users to know which SIMD extensions are supported by their CPU, let alone understand what SIMD even is.

The ideal solution would be to *build once, run everywhere*. With this approach, a single binary is compiled with support for multiple SIMD extensions, and at runtime, the program probes the CPU for supported SIMD extensions and dynamically selects the appropriate implementations of functions to be executed. This is known as *dynamic dispatch*.

In order to have such a system, we need a way to automate building multiple SIMD variants of the same function. This is a task that the C preprocessor won't be able to handle on its own, because it requires an automation system that can compile the same source file multiple times with different compiler switches to enable different SIMD extensions. For this, a dedicated build system is needed. I will show how this can be done with a Makefile.

Let's assume our project directory has the following structure.

```txt
project
│ Makefile
│ main.c
│ stride.h
└ vla.c
```

This project has two source files: `main.c`, which contains the main function that calls vectorized code, and `vla.c`, which contains VLA code. Our VLA SIMD wrapper is located in the `stride.h` header file, which is included by `vla.c`.

In order to build this project, the Makefile will have to compile `vla.c` multiple times with different SIMD extensions enabled, while compiling `main.c` as usual. The resulting object files then need to be linked together to produce a single binary.

Begin writing the Makefile by defining the standard variables for the C compiler, compiler flags, output binary, source files, and object files.

```make
CC ?= gcc
CFLAGS += -O3 -std=c23 -Wall -Wextra -Wpedantic

BIN = out
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
```

The files in `OBJS` will serve as the *baseline* object files, which are compiled with the base feature set determined at build time; it's not necessarily the case that they will be compiled without any SIMD extensions at all.

Next, define a vectorized source list, which contains the source files written using VLA abstractions. These files are eligible to be compiled with multiple SIMD extensions, so we will define corresponding object file lists for each supported SIMD extension.

```make
SRCS_VECTORIZE = vla.c
OBJS_VECTORIZE_AVX512F = $(SRCS_VECTORIZE:.c=_avx512f.o)
OBJS_VECTORIZE_AVX2 = $(SRCS_VECTORIZE:.c=_avx2.o)
OBJS_VECTORIZE_SSE2 = $(SRCS_VECTORIZE:.c=_sse2.o)
```

Now, we need to determine which SIMD variants should be built *on top* of the baseline object files. This will help prevent us from building unnecessary SIMD variants that are already supported at the baseline level. To do this, we first need to obtain a list of baseline SIMD extensions, which can be done by querying the predefined macros.

On gcc, extracting the predefined macros can be done with the following command.

```bash
gcc -dM -E - < /dev/null
```

This can be called and saved into a Makefile variable through the `$(shell ...)` function. Then, specific extensions can be queried from this variable using the `$(findstring ...)` function. Note that I have inserted the `CFLAGS` variable into the command, so that the predefined macros will reflect the SIMD extensions enabled by the compiler flags.

```make
PREDEFINED_MACROS := $(shell $(CC) $(CFLAGS) -dM -E - < /dev/null)

BASELINE_EXTENSIONS = $(findstring AVX512F,$(PREDEFINED_MACROS)) \
                      $(findstring AVX2,$(PREDEFINED_MACROS))    \
                      $(findstring SSE2,$(PREDEFINED_MACROS))
```

The `$(findstring ...)` function will return the string being searched for if it is found, and an empty string otherwise. Therefore, `BASELINE_EXTENSIONS` will contain the list of baseline SIMD extensions. With this, we can filter out the baseline extensions from the list of supported extensions to obtain the list of SIMD variants to be built.

```make
SIMD_VARIANTS = $(filter-out $(BASELINE_EXTENSIONS),AVX512F AVX2 SSE2)
```

A final vectorized object list can then be obtained by iterating over the `SIMD_VARIANTS` list and aggregating the corresponding object file lists using computed variable names.

```make
OBJS_VECTORIZE = $(foreach v,$(SIMD_VARIANTS),$(OBJS_VECTORIZE_$v))
```

We can now begin writing Makefile rules, beginning with the rule for the output binary. The output binary will have both the vectorized and baseline object files as prerequisites, linking them together in the recipe to produce the executable.

```make
$(BIN): $(OBJS_VECTORIZE) $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@
```

As for the object files, a static pattern rule will be defined for each of the baseline and vectorized object file lists. The rule for building the baseline objects is simple enough; just compile without any special flags. For each list of vectorized objects, however, we will have to insert the appropriate flags in the recipe to enable the corresponding SIMD extension.

```make
$(OBJS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJS_VECTORIZE_AVX512F): %_avx512f.o: %.c
	$(CC) $(CFLAGS) -mavx512f -c $< -o $@

$(OBJS_VECTORIZE_AVX2): %_avx2.o: %.c
	$(CC) $(CFLAGS) -mavx2 -mfma -c $< -o $@

$(OBJS_VECTORIZE_SSE2): %_sse2.o: %.c
	$(CC) $(CFLAGS) -msse2 -c $< -o $@
```

With this, the build system is complete—but we aren't quite ready to build just yet. One glaring issue you may have noticed is that our program will not link. To see why, consider the following contents of `vla.c`.

```c
#include "stride.h"

sd_float *even_nums(size_t len) {
    size_t sd_len = sd_bounding_size(len);
    sd_float *out = aligned_alloc(SD_ALIGN, sizeof(sd_float) * sd_len);

    for (size_t i = 0; i < sd_len; ++i) {
        /* `sd_float_set` sets all stride elements to some number */
        sd_float two = sd_float_set(2);
        sd_float offset = sd_float_set(i * SD_LENGTH);

        /* `sd_float_range` sets each stride element to its index */
        sd_float nums = sd_float_range();
                 nums = sd_float_add(nums, offset);
                 nums = sd_float_mul(nums, two);

        out[i] = nums;
    }

    return out;
}
```

The build system will compile this source file up to four times—each with a different SIMD extension enabled. Since `even_nums` is written using VLA abstractions, the code generated for the body of the function will be different for each of the compiled object files. However, the function's symbol name remains the same in each compilation, so we will end up with multiple definitions of `even_nums`, leading to a linker error.

This can be avoided by mangling the symbol names of vectorized functions to include the name of the currently enabled SIMD extension. To automate this, we can extend the VLA SIMD wrapper to include a function-like macro that takes in a function name and appends the name of the currently enabled SIMD extension to it.

```c
#ifdef __AVX512F__
    #define SD_VARIANT(fnname)  fnname##_avx512f 
#elifdef __AVX2__
    #define SD_VARIANT(fnname)  fnname##_avx2 
#elifdef __SSE2__
    #define SD_VARIANT(fnname)  fnname##_sse2 
#else
    #define SD_VARIANT(fnname)  fnname##_scalar 
#endif
```

To use this `SD_VARIANT` macro, simply wrap the name of each vectorized function with it, like so.

```c
#include "stride.h"

sd_float *SD_VARIANT(even_nums)(size_t len) {
    /* ... */
}
```

With this, each vectorized variant of `even_nums` will be defined under a unique symbol name. Corresponding to each of these definitions should be a declaration that makes the function available to be called from other source files. Once again, a function-like macro can be used to automate this process.

```c
#define SD_DECLARE(rettype, fnname, __VA_ARGS__)    \
    typeof(rettype) fnname##_avx512f(__VA_ARGS__);  \
    typeof(rettype) fnname##_avx2(__VA_ARGS__);     \
    typeof(rettype) fnname##_sse2(__VA_ARGS__);     \
    typeof(rettype) fnname##_scalar(__VA_ARGS__);
```

Now, if we wish to call `even_nums` from `main.c`, we should first declare each of its vectorized variants using the `SD_DECLARE` macro.

```c
#include "stride.h"

SD_DECLARE(sd_float *, even_nums, size_t len)

int main(void) {
    sd_float *nums = TODO;  /* Call `even_nums` somehow... */

    for (size_t i = 0; i < 256; ++i)
        printf("%g\n", ((float *)nums)[i]);

    return 0;
}
```

The next problem to tackle is how to actually call the correct vectorized function variation. Here, it's important to take into account whether or not we have already entered a *vectorized context*, as this has implications on how calls to vectorized functions should be handled.

Consider the following scenario, where we wish to call `increment` from `increment_double`, both of which are vectorized functions.
```c
#include "stride.h"

void SD_VARIANT(increment)(sd_float *buf, size_t len) {
    size_t sd_len = sd_bounding_size(len);
    sd_float one = sd_float_set(1);

    for (size_t i = 0; i < sd_len; ++i)
        buf[i] = sd_float_add(buf[i], one);
}

void SD_VARIANT(increment_double)(sd_float *buf, size_t len) {
    size_t sd_len = sd_bounding_size(len);

    /* Branch to vectorized code path not required, we are already in a vectorized context */
    SD_VARIANT(increment)(buf, len);

    for (size_t i = 0; i < sd_len; ++i)
        buf[i] = sd_float_add(buf[i], buf[i]);
```

The fact that we are in the body of `increment_double` means that we are currently executing code for one of its vectorized variants. Because of this, we can simply reuse the `SD_VARIANT` macro to obtain the correct variant of `increment` to call, since this will resolve to the SIMD variant of `increment` corresponding to the SIMD variant of `increment_double` currently being compiled.

This same strategy **will not work** in an *unvectorized context*, however. This is because unvectorized functions are only compiled once, with the baseline feature set, meaning that `SD_VARIANT` will always resolve to the baseline variant of whatever function we give it.

```c
#include "stride.h"

SD_DECLARE(sd_float *, even_nums, size_t len)

int main(void) {
    /*
     * Static dispatch to the baseline variant of `even_nums` from an unvectorized context
     * Variants compiled with higher level SIMD extensions will never be called...
     */
    sd_float *nums = SD_VARIANT(even_nums)(256);

    /* ... */
}
```

What we want instead is to have some kind of mechanism that produces *diverging code paths* when we call vectorized functions from an unvectorized context. At the call site, we need to query the runtime CPU's supported SIMD extensions, and then branch to the appropriate vectorized function variation accordingly. This is a runtime dynamic dispatch system, and it essentially serves as a gateway between the unvectorized and vectorized worlds.

Determining which SIMD extensions are available at runtime can be quite tricky to do portably. Libraries can help streamline this process ([SDL's cpuinfo API](https://wiki.libsdl.org/SDL3/CategoryCPUInfo) is great for this), but for simplicity, I'll just rely on [gcc's `__builtin_cpu_supports` function](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions#index-_005f_005fbuiltin_005fcpu_005fsupports-1). Below is an example of how to perform runtime checks for supported SIMD extensions using it.

```c
printf("AVX512F support: %s\n", __builtin_cpu_supports("avx512f") ? "Yes" : "No");
printf("AVX2 support: %s\n", __builtin_cpu_supports("avx2") ? "Yes" : "No");
printf("SSE2 support: %s\n", __builtin_cpu_supports("sse2") ? "Yes" : "No");
```

With this, we can define yet another function-like macro that takes in a function name and expands to an expression that evaluates to the correct vectorized variant of that function, using chained conditional operators. I will call this the `SD_SELECT` macro, and it can be defined as follows.

```c
#ifdef __AVX512F__
    #define SD_SELECT(fnname)  ( fnname##_avx512f )
#elifdef __AVX2__
    #define SD_SELECT(fnname)  ( __builtin_cpu_supports("avx512f") ? fnname##_avx512f :  \
                                                                     fnname##_avx2    )
#elifdef __SSE2__
    #define SD_SELECT(fnname)  ( __builtin_cpu_supports("avx512f") ? fnname##_avx512f :  \
                                 __builtin_cpu_supports("avx2")    ? fnname##_avx2    :  \
                                                                     fnname##_sse2    )
#else
    #define SD_SELECT(fnname)  ( __builtin_cpu_supports("avx512f") ? fnname##_avx512f :  \
                                 __builtin_cpu_supports("avx2")    ? fnname##_avx2    :  \
                                 __builtin_cpu_supports("sse2")    ? fnname##_sse2    :  \
                                                                     fnname##_scalar  )
#endif
```

Let us now complete the implementation of `main` by using the `SD_SELECT` macro to call the appropriate vectorized variant of `even_nums` at runtime.

```c
#include "stride.h"

SD_DECLARE(sd_float *, even_nums, size_t len)

int main(void) {
    sd_float *nums = SD_SELECT(even_nums)(256);

    for (size_t i = 0; i < 256; ++i)
        printf("%g\n", ((float *)nums)[i]);

    return 0;
}
```

We can now finally build the project. On my system, running `make` will produce the following build artefacts.

```txt
out
main.o
vla.o
vla_avx2.o
vla_avx512f.o
```

The compiler is configured to build for baseline 64-bit x86, so SSE2 extensions are enabled by default. On top of the baseline object files, AVX2 and AVX512F accelerated objects have also been built for `vla.c`. These are all linked together to produce `out`, a portable binary that will run on any 64-bit x86 CPU, while still being able to harness the power of higher-level SIMD extensions on supported devices.

By combining VLA programming, a dedicated build system, and dynamic dispatch, we have created a powerful vectorization system that is both portable and efficient, maintaining the benefits of intrinsics programming while not sacrificing the ease with which users can install and run our software.

---

## Beyond Fixed-Width SIMD

What we have just built may be described as a software-based scalable vectorization system. It works by transforming source code into various different forms that each leverage the SIMD processing capabilities of different instruction sets. As we have explored in this article, the key insight in this process is that vector-length agnosticism allows vectorized computations to be expressed independently of any particular SIMD extension.

Take a step back and ponder on this idea. We know it is possible to write generic vectorized code. If this is the case, why do we have to work with fixed-width SIMD instruction sets that require us to build a software layer on top of them to achieve scalability? Couldn't the instruction sets *themselves* be vector-length agnostic?

Processors that implement VLA instruction sets do indeed exist, and they are known as *vector processors*. In fact, such processors are by no means novel; they've been around for decades, with the first notable instance of one appearing in the famous [Cray-1](https://en.wikipedia.org/wiki/Cray-1#) supercomputer, released in 1976.

With the continued advancements in semiconductor technology, microprocessors became cheaper and more cost-effective. As such, the high-performance computing market shifted its focus towards massively parallel MIMD architectures, leading to the gradual decline in popularity of vector processors.

SIMD entered the consumer desktop space at a time when multimedia and gaming applications were becoming increasingly common. To satisfy the demand for real-time performance in these applications, CPU designers introduced fixed-width, or "packed", SIMD instruction sets, which were a simpler solution to achieve data-parallelism than vector processing. Packed SIMD remains the dominant form of SIMD in consumer CPUs, but the limitations of this design have become more apparent over time. The x86 platform, in particular, has been criticized for the ballooning size of its instruction set, largely caused by the continued addition of SIMD extensions that double the vector length of its previous iteration.

Vector instruction sets, being the more elegant and scalable design, have recently been undergoing a resurgence in popularity. Of note are the [Scalable Vector Extension](https://developer.arm.com/Architectures/Scalable%20Vector%20Extensions) (SVE) on the ARM architecture, and the [RISC-V Vector Extension](https://docs.riscv.org/reference/isa/extensions/vector/_attachments/riscv-v-spec.pdf) (RVV) on the RISC-V architecture. These instruction sets, being vector-length agnostic, do not encode the register width in their instructions, instead leaving it as an implementation detail decided upon by the CPU vendor. This allows vector code that is written using these extensions to be scalable, without the need for any kind of software layer.

As of now, vector processors are not readily available on consumer devices. Regardless, the ideas behind them are still very much relevant to writing portable, high-performance software. I hope this article has given you valuable insights into scalable vectorization and VLA programming techniques, which you may consider integrating into your own projects to achieve better performance and portability.

---

*Last edited on April 15, 2026*

<div class="navigation">
    <div>
        <div class="navigation-header">
            <div class="chevron-left"></div>
            <p class="navigation-header-text">Previous</p>
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
