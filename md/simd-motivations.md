Motivations for Manual Vectorization | trixis

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
        <a href="/simd-programming">SIMD Programming with Intrinsics</a>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <div class="navigation-header">
            <p class="navigation-header-text">Next</p>
            <div class="chevron-right"></div>
        </div>
        <a href="/simd-scalable" style="text-align: end;">Scalable Vectorization</a>
    </div>
</div>

# Motivations for Manual Vectorization

---

## Automatic Vectorization

At the end of the previous article, I mentioned gcc's ability to automatically vectorize scalar code—a feature of most widely accepted C compilers. The benchmark program I ran was compiled at a lower optimization level where auto-vectorization is disabled, which allowed me to showcase performance gains from SIMD.

Let's recompile the benchmark program at the highest optimization level to see how well auto-vectorized scalar code produced by the compiler performs. Compiling with `gcc -O3 -march=native`, these are the results of the same benchmark on my system.

```txt
Scalar: 1.03859s
SSE: 1.9258s
AVX: 1.0152s
```

The auto-vectorized scalar code performs about as well as the manually vectorized code using AVX intrinsics.

These results beg the question—why would anyone ever choose to manually vectorize their code using SIMD intrinsics, even in the context of writing performance-critical software? Manually vectorized code is not portable across CPU architectures, and even *within* the same architecture. Additionally, manually vectorized code is more difficult to read and maintain, using function calls for simple arithmetic expressions and introducing concerns regarding memory alignment. If the same performance gains can be achieved through auto-vectorization without these sacrifices, then why even bother?

I will discuss the limitations of automatic vectorization and how they can be overcome through manual vectorization with regards to three areas.

**Code complexity**: compilers produce longer and more complex vectorized code due to relaxed assumptions

**Delegation**: compilers will not vectorize delegated computations not known at compile-time

**Structured data**: compilers will not arrange data structures to benefit from vectorization for you

---

## Code Complexity

Let's revisit the two array squaring functions from the previous article.

```c
void square_scalar(float *dst, float *src, size_t length) {
    for (size_t i = 0; i < length; ++i)
        dst[i] = src[i] * src[i];
}

void square_sse(__m128 *dst, __m128 *src, size_t length) {
    size_t nstrides = (length + 3) / 4;

    for (size_t i = 0; i < nstrides; ++i)
        dst[i] = _mm_mul_ps(src[i], src[i]);
}
```

[Using gcc with the highest optimization level](https://godbolt.org/z/r3abzM7z6), `square_scalar` compiles to 49 machine code instructions, while `square_sse` compiles to only 12 instructions. That's **four times** the number of instructions to implement the seemingly simpler, scalar function.

Recall that the `square_sse` function makes certain assumptions about the data which allow it to be expressed almost as concisely as the scalar function. Namely, it assumes that the memory for the `dst` and `src` arrays have been allocated such that they can contain a number of floating point elements that is a multiple of the stride length. This makes a scalar clean-up loop unnecessary, because even if the actual number of floating point elements stored in the array is not divisible by the stride length, the array is padded so that a whole number of strides can fit inside it. Instead of overflowing the buffer, the final stride will simply operate on allocated, but partially uninitialized memory.

The compiler's auto-vectorizer can make no such assumptions. Therefore, additional code and branching logic must be inserted to deal with these situations.

That's not the end of the story. *Data dependencies* are another complication that must be considered. Imagine a scenario where the `dst` and `src` arrays overlap, such that the base address of `dst` is two elements ahead of the base address of `src`.

```txt
|----------------- src -----------------|

+---------+---------+---------+---------+---------+---------+
|         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+

                    |----------------- dst -----------------|
```

What the `square_scalar` function will do when given these arguments is equivalent to the following code.

```c
for (size_t i = 0; i < length; ++i)
    dst[i] = dst[i - 2] * dst[i - 2];
```

Each element written to `dst` has a dependence on the element two places before it. Because the distance between dependent elements is **less** than the stride length—four, with SSE—vectorization in a way that preserves the original function's behaviour is **not possible** because of a *data hazard*.

Consider what happens when we try to vectorize this using SSE anyway. The higher two elements of each stride must be set to the squares of the lower two elements, which themselves need to be set according to the two elements that come before them. Since we are working on four elements at a time, it's not possible to set the lower two elements before the upper two elements, so the behaviour of the vectorized function will diverge from the behaviour of the scalar function.

```txt
Before:

|----------------- src -----------------|

+---------+---------+---------+---------+---------+---------+
|    A    |    B    |    C    |    D    |    E    |    F    |
+---------+---------+---------+---------+---------+---------+

                    |----------------- dst -----------------|

After (scalar):

|----------------- src -----------------|

+---------+---------+---------+---------+---------+---------+
|    A    |    B    |   A^2   |   B^2   |   A^4   |   B^4   |
+---------+---------+---------+---------+---------+---------+

                    |----------------- dst -----------------|

After (vectorized, stride length = 4):

|----------------- src -----------------|

+---------+---------+---------+---------+---------+---------+
|    A    |    B    |   A^2   |   B^2   |   C^2   |   D^2   |
+---------+---------+---------+---------+---------+---------+

                    |----------------- dst -----------------|
```

Because of this, the compiler must emit instructions that check the offset between the `dst` and `src` arrays, conditionally branching to a scalar loop if it detects a data hazard.

To make matters worse, think about what happens when more SIMD extensions are made available. With AVX and AVX-512, the compilier can additionally work with stride lengths of 8 and 16. It's possible for data hazards to exist with one stride length, but not for lower ones. In situations like these, should multiple vectorized loop variations be created for every possible stride length supported by the system?

The availability of multiple stride lengths also leads to the possibility for the clean-up step to be vectorized. This can be accomplished by using strides of lower bit-widths than the ones used in the main loop. The handling of all these cases will require even more branching paths and code complexity.

Compilers don't quite go to the extent of creating a bunch of vectorized loop variations, but nevertheless the amount of code they produce when more SIMD features are available can be staggering. With AVX-512 extensions enabled, [gcc will compile `square_scalar` to 90 machine code instructions](https://godbolt.org/z/ea83nao79).

Be aware of this if you are relying on the auto-vectorizer to optimize your code. Even the simplest functions can be blown up to massive code sizes because the compiler has to operate under relaxed assumptions. By understanding vectorization and manually tuning your code for it, you can explicitly state your assumptions to circumvent these limitations.

One more thing I should mention is the `restrict` keyword. [`restrict` is a type qualifier](https://en.cppreference.com/w/c/language/restrict) that is used to assert that when an object is modified through *some* pointer, it can only be accessed through *that* pointer. This can be used to eliminate code dealing with data dependencies generated by the compiler. In the case of the `square_scalar` function, quite a few instructions can be cut down by replacing the declaration `float *dst` with `float *restrict dst`. This is a nice, portable optimization that doesn't require manual vectorization.

---

## Delegation

Previously, with the array squaring function, we had a compile-time *known* computation that would be applied to each element of an array. Now consider a higher-order mapping function that takes in, as an argument, a compile-time *unknown* function that serves as the computation to be applied to each element of an array. In computer graphics, the rendering pipeline and programmable [fragment shaders](https://en.wikipedia.org/wiki/Shader#Fragment_shaders) serve as an excellent example of this. Such a mapping function, implemented with scalar code, might look like the following.

```c
void map_scalar(float *dst, float *src, float (*fn)(float), size_t length) {
    for (size_t i = 0; i < length; ++i)
        dst[i] = fn(src[i]);
}
```

Now, regardless of how you compile this code, that delegated computation simply cannot be vectorized. Here is the disassembly of this function's loop.

```x86asm
.L3:
        movss   xmm0, DWORD PTR [r14+rbx*4]
        call    r13
        movss   DWORD PTR [r12+rbx*4], xmm0
        add     rbx, 1
        cmp     rbp, rbx
        jne     .L3
```

Here, `movss` represents a *move scalar single* instruction. This indicates that floating point numbers are being passed to the callback function *one at a time* rather than *strides at a time*. The reason for this is simple. The function pointed to by `fn` is specified to take in and return a single floating point number, and this interface must be adhered to.

In order to vectorize this function, we must change the interface of the callback function so that it uses strided types directly.

```c
void map_sse(__m128 *dst, __m128 *src, __m128 (*fn)(__m128), size_t length) {
    size_t nstrides = (length + 3) / 4;

    for (size_t i = 0; i < nstrides; ++i)
        dst[i] = fn(src[i]);
}
```

By doing this, the functions passed into `map_sse` must be **explicitly** vectorized. Notice that we have also lost the ability to operate on individual floating point elements at a time, meaning that padding the `dst` and `src` arrays to a whole number of strides becomes especially important here.

The nice thing about passing SIMD types between functions is that this usage is often directly supported through *calling conventions*, which determine how parameters and return values are passed between functions, including which CPU registers are used. For example, the calling convention for the System V ABI, used by Unix-like operating systems, will pass/return `__m128` types through the wide, 128-bit `xmm` registers instead of using the stack, which is slower. When compiled for this calling convention, here is what the disassembly of the `map_sse` function's loop looks like.

```x86asm
.L3:
        mov     rbp, rbx
        add     rbx, 1
        sal     rbp, 4
        movaps  xmm0, XMMWORD PTR [r15+rbp]
        call    r14
        movaps  XMMWORD PTR [r13+0+rbp], xmm0
        cmp     r12, rbx
        jne     .L3
```

We can now see that the `movaps` (*move aligned packed singles*) instruction is being used to pass a stride of floating point data to the vectorized callback function in each loop iteration.

In summary, when the workload consists of delegating computations to some callback that isn't known ahead of time, auto-vectorization won't be able to help. If you *need* this to be fast, you'll have to manually vectorize to get the benefits of SIMD.

---

## Structured Data

Typically, the data values that you work with in a program don't just consist of a single primitive type, like a single integer or a single floating point number. Data is often structured, being composed of *multiple* primitive types. Take a 3-dimensional vector for example, which is a unit of data composed of three numerical values—an x, y, and z component. Here is how you might define a structure that represents a 3D vector.

```c
typedef struct vec3 {
    float x, y, z;
} vec3;
```

Let's now imagine a parallelizable workload involving the `vec3` structure. Say you have an array of 3D vectors, and you want to go through each vector, take its dot product with some fixed vector, and store the result in another array. The scalar implementation of a function that does this would look like the following.

```c
void dot_scalar(float *dst, vec3 *src, vec3 rhs, size_t length) {
    for (size_t i = 0; i < length; ++i)
        dst[i] = src[i].x * rhs.x
               + src[i].y * rhs.y
               + src[i].z * rhs.z;
}
```

The disassembly of this function's loop, when compiled *without* auto-vectorization, is quite simple.

```x86asm
.L3:
        movss   xmm0, DWORD PTR [rsi]    ; src[i].x
        movss   xmm2, DWORD PTR [rsi+4]  ; src[i].y
        add     rdi, 4
        add     rsi, 12
        mulss   xmm2, xmm3
        mulss   xmm0, xmm4
        addss   xmm0, xmm2
        movss   xmm2, DWORD PTR [rsi-4]  ; src[i].z
        mulss   xmm2, xmm1
        addss   xmm0, xmm2
        movss   DWORD PTR [rdi-4], xmm0  ; dst[i] = result
        cmp     rax, rdi
        jne     .L3
```

Each loop consists of three move instructions which load the x, y, and z components of the current vector, five arithmetic instructions to compute the dot product with `rhs`, and one more move instruction to store the result into `dst`.

Given this, it would be reasonable to expect the auto-vectorized version of `dot_scalar` to essentially have the same loop, just with the scalar instructions replaced with their packed SIMD counterparts. In reality, however, this is not the case.

Here is the disassembly of the primary loop when `dot_scalar` is compiled with auto-vectorization enabled.

```x86asm
.L6:
        movups  xmm3, XMMWORD PTR [rax+16]
        movups  xmm2, XMMWORD PTR [rax]
        add     rdx, 16
        add     rax, 48
        movups  xmm4, XMMWORD PTR [rax-16]
        movaps  xmm0, xmm2
        movaps  xmm5, xmm3
        movaps  xmm9, xmm3
        shufps  xmm5, xmm4, 175
        shufps  xmm0, xmm3, 5
        shufps  xmm0, xmm5, 136
        movaps  xmm5, xmm2
        mulps   xmm0, xmm7
        shufps  xmm9, xmm4, 90
        shufps  xmm5, xmm9, 140
        shufps  xmm2, xmm3, 90
        mulps   xmm5, xmm8
        shufps  xmm2, xmm4, 200
        mulps   xmm2, xmm6
        addps   xmm0, xmm5
        addps   xmm0, xmm2
        movups  XMMWORD PTR [rdx-16], xmm0
        cmp     rdx, r8
        jne     .L6
```

The four move instructions and five arithmetic instructions appear again, this time operating on wide SIMD registers, but besides that, a bunch of extra `movaps` and `shufps` instructions have been inserted. Why is this?

Let's consider the steps required to vectorize `dot_scalar`. The goal is to maximize efficiency by fully utilizing the increased throughput from SIMD. When this function is vectorized, each addition and multiplication operation involved in computing the dot product should always be performed on four values—assuming SSE—at a time, up until the last addition operation, which will produce the final result containing four dot products.

Producing four dot products with a fixed vector will require accessing four vectors from `src` at a time. In total, these four vectors contain 12 components, which can fit within three strides. Therefore, the vectorized loop we create should iterate over the `src` array in steps of three strides at a time—a unit I will refer to as a *strided 3D vector*.

```txt
      |------ 1 stride -------|------ 1 stride -------|------ 1 stride -------|

      |------ v0  ------|------ v1  ------|------ v2  ------|------ v3  ------|

      +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
src:  |  x  |  y  |  z  |  x  |  y  |  z  |  x  |  y  |  z  |  x  |  y  |  z  | ... |
      +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

                                       do work...

                                           |
                                           |
                                           v

      |------------------------------ 1 stride -------------------------------|

      +-----------------+-----------------+-----------------+-----------------+-----+
dst:  |   dot(v0,rhs)   |   dot(v1,rhs)   |   dot(v2,rhs)   |   dot(v3,rhs)   | ... |
      +-----------------+-----------------+-----------------+-----------------+-----+
```

The issue here is that the components of the strided 3D vectors are *interleaved*, meaning that they come one after another. This is a problem because the dot product requires the vector components to be added together after they are multiplied, but we can't do that if the vector components are next to each other in the *same* stride.

A *deinterleaving* step is required, which will extract the like-components of each vector and store them into their own respective strides. Here's a visualization of how the dot product computation needs to be done for each strided 3D vector.

```txt
                    |------ 1 stride -------|------ 1 stride -------|------ 1 stride -------|

                    |------ v0  ------|------ v1  ------|------ v2  ------|------ v3  ------|

                    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
Strided 3D vector:  |  x  |  y  |  z  |  x  |  y  |  z  |  x  |  y  |  z  |  x  |  y  |  z  |
                    +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

                                                  deinterleave

                                                        |
                             +--------------------------+--------------------------+
                             |                          |                          |
                             v                          v                          v

                 +-----+-----+-----+-----+  +-----+-----+-----+-----+  +-----+-----+-----+-----+
                 |  x  |  x  |  x  |  x  |  |  y  |  y  |  y  |  y  |  |  z  |  z  |  z  |  z  |
                 +-----+-----+-----+-----+  +-----+-----+-----+-----+  +-----+-----+-----+-----+

                             x                          x                          x

                 +-----+-----+-----+-----+  +-----+-----+-----+-----+  +-----+-----+-----+-----+
                 |rhs_x|rhs_x|rhs_x|rhs_x|  |rhs_y|rhs_y|rhs_y|rhs_y|  |rhs_z|rhs_z|rhs_z|rhs_z|
                 +-----+-----+-----+-----+  +-----+-----+-----+-----+  +-----+-----+-----+-----+

                             =                          =                          =

                 +-----+-----+-----+-----+  +-----+-----+-----+-----+  +-----+-----+-----+-----+
                 |lxr_x|lxr_x|lxr_x|lxr_x|  |lxr_y|lxr_y|lxr_y|lxr_y|  |lxr_z|lxr_z|lxr_z|lxr_z|
                 +-----+-----+-----+-----+  +-----+-----+-----+-----+  +-----+-----+-----+-----+

                             |                          |                          |
                             +--------------------------+--------------------------+
                                                        |
                                                       sum
                                                        |
                                                        v

                    +-----------------+-----------------+-----------------+-----------------+
                    |   dot(v0,rhs)   |   dot(v1,rhs)   |   dot(v2,rhs)   |   dot(v3,rhs)   |
                    +-----------------+-----------------+-----------------+-----------------+
```

The need for deinterleaving is precisely why the auto-vectorized loop contains many `shufps` instructions. This is an instruction that shuffles the elements of strides through indexing, and it is used to transform the vector data into a usable state. All of this shuffling creates a large overhead, and its necessity stems from the usage of inefficient data structures with interleaved members.

The obvious question to ask is why structure members are even interleaved in the first place. In the same way that we can simplify vectorization by assuming that arrays are padded to be able to fit a whole number of strides, couldn't we also assume that data is laid out in a way that doesn't require deinterleaving to work with?

Think back to definition of the `vec3` structure from the start of this section.

```c
typedef struct vec3 {
    float x, y, z;
} vec3;
```

This is a very natural way to arrange the components of a 3D vector, where each vector occupies a contiguous block of memory that contains all of its components. To store an array of vectors, we simply pack these structures next to one another. This is known as an *Array of Structures* (AoS) layout.

The AoS layout leads to inefficiencies in vectorization. To resolve this, we need to adopt a new way of arranging structure members that is compatible with data parallelism. In the case of vectorizing dot products of 3D vectors, data parallelism comes in the form of parallel operations on like-components. Therefore, an array of 3D vectors should be arranged such that the like-components of consecutive vectors can be loaded into strides all at once, without any shuffling.

One way to do this is by inverting the composition of the AoS layout to obtain a *Structure of Arrays* (SoA) layout.

```c
typedef struct vec3_arr {
    float *xs, *ys, *zs;
} vec3_arr;
```

With this layout, rather than having a single array of interleaved vector components, three separate, parallel arrays containing like-components are held in a structure. Each 3D vector is scattered across different memory regions, and they are extracted by accessing the three arrays with the same index.

Here is `dot_scalar` rewritten to use the SoA layout.

```c
void dot_scalar_soa(float *dst, vec3_arr src, vec3 rhs, size_t length) {
    for (size_t i = 0; i < length; ++i)
        dst[i] = src.xs[i] * rhs.x
               + src.ys[i] * rhs.y
               + src.zs[i] * rhs.z;
}
```

And here is the disassembly of its primary loop, when compiled with auto-vectorization enabled. We can see that all of the shuffling instructions have been eliminated, since vector components are no longer interleaved.

```x86asm
.L39:
        movups  xmm0, XMMWORD PTR [rcx+rax]
        movups  xmm2, XMMWORD PTR [rsi+rax]
        mulps   xmm0, xmm5
        mulps   xmm2, xmm4
        addps   xmm0, xmm2
        movups  xmm2, XMMWORD PTR [r8+rax]
        mulps   xmm2, xmm3
        addps   xmm0, xmm2
        movups  XMMWORD PTR [rdi+rax], xmm0
        add     rax, 16
        cmp     rax, r9
        jne     .L39
```

As visible, the SoA layout leads to more efficient vectorization. However, it still leaves some things to be desired. With regards to performance, the SoA layout lacks *spatial locality* between structure members, which is to say that the members of a structure do not occupy nearby memory locations. This is less cache friendly, because the access pattern of these structures requires more cache lines to be loaded, reducing memory efficiency.

Additionally, by separating structure members into parallel arrays, the SoA layout leads to reduced expressiveness. In particular, it's no longer possible to address a single structure as a single unit of data. Instead, structures always exist inside arrays, and their individual members need to be accessed through awkward indexing operations. This awkwardness becomes especially clear when you try to write something like a mapping function, where structures need to be passed between functions.

```c
void map_vec3_arr(vec3_arr dst, vec3_arr src, vec3 (*fn)(vec3), size_t length) {
    for (size_t i = 0; i < length; ++i) {
        vec3 out = fn((vec3) {
            .x = src.xs[i],
            .y = src.ys[i],
            .z = src.zs[i],
        });

        dst.xs[i] = out.x;
        dst.ys[i] = out.y;
        dst.zs[i] = out.z;
    }

    /* As we've seen in the Delegation section, none of this will be vectorized anyway! */
}
```

There is another data layout that addresses both of these concerns, known as the *Array of Structures of Arrays* (AoSoA) layout. This layout is essentially the AoS layout, but instead of an array of interleaved structure members, it's an array of interleaved *strides* of structure members. Each structure becomes composed of strides, being able to hold a stride length number of individual, scalar structures.

```c
typedef struct sd_vec3 {
    __m128 x, y, z;
} sd_vec3;
```

Now that strided types are being used directly, manual vectorization is required. Here is the implementation of the array dot product function, using the AoSoA layout.

```c
/* `src` is assumed to be padded to a whole number of strided 3D vectors */
void dot_sse_aosoa(__m128 *dst, sd_vec3 *src, sd_vec3 rhs, size_t length) {
    size_t sd_len = (length + 3) / 4;

    for (size_t i = 0; i < sd_len; ++i)
        dst[i] = _mm_add_ps(_mm_mul_ps(src[i].x, rhs.x),
                 _mm_add_ps(_mm_mul_ps(src[i].y, rhs.y),
                            _mm_mul_ps(src[i].z, rhs.z)));
}
```

Just as with the AoS layout, the `src` array is iterated through in steps of strided 3D vectors. The key difference is that no deinterleaving is required, because vector components are not interleaved within strides. Here is the disassembly of this function's loop, once again showing no shuffling instructions.

```x86asm
.L3:
        movaps  xmm0, XMMWORD PTR [rsi+32]
        movaps  xmm1, XMMWORD PTR [rsi+16]
        add     rdi, 16
        add     rsi, 48
        mulps   xmm1, xmm3
        mulps   xmm0, xmm4
        addps   xmm0, xmm1
        movaps  xmm1, XMMWORD PTR [rsi-48]
        mulps   xmm1, xmm2
        addps   xmm0, xmm1
        movaps  XMMWORD PTR [rdi-16], xmm0
        cmp     rax, rdi
        jne     .L3
```

The AoSoA layout also works great with delegated computations. Here's an implementation of the 3D vector mapping function using this layout, which the SoA approach previously struggled with. On top of being more concise, it is also fully vectorized.

```c
void map_sd_vec3(sd_vec3 *dst, sd_vec3 *src, sd_vec3 (*fn)(sd_vec3), size_t length) {
    size_t sd_len = (length + 3) / 4;

    for (size_t i = 0; i < sd_len; ++i)
        dst[i] = fn(src[i]);
}
```

To conclude this section, it is vital to understand that effective vectorization is not only a matter of utilizing SIMD instructions. When working with structured data, the way in which that data is arranged significantly impacts parallel performance. By remaining oblivious to vectorization and data parallelism, we introduce inefficient structures that auto-vectorizers can only do so much with. Through manual vectorization with the AoSoA data layout, highly efficient SIMD code can be produced without sacrificing expressiveness.

---

## Conclusion

Hopefully this article has shed light on the importance and relevance of manual SIMD optimization. Modern auto-vectorizers are highly advanced and incredibly clever systems, but it's important to understand their limits. Scalar programs, by their nature, contain many inefficiencies that make vectorization difficult. It is only through an awareness of these inefficiencies that steps can be made towards creating well optimized programs. Depending on your requirements, these steps could range from rearranging data structures to intrinsics programming.

One of the biggest deterrents to using SIMD intrinsics is, understandably, their lack of portability. In the next article, I want to share a technique for *scalable vectorization*, which allows for the creation of vectorized programs that can be ported across various systems with different SIMD extensions.

---

*Last edited on April 13, 2026*

<div class="navigation">
    <div>
        <div class="navigation-header">
            <div class="chevron-left"></div>
            <p class="navigation-header-text">Previous</p>
        </div>
        <a href="/simd-programming">SIMD Programming with Intrinsics</a>
    </div>
    <div>
        <div class="navigation-header" style="justify-content: flex-end;">
            <p class="navigation-header-text">Next</p>
            <div class="chevron-right"></div>
        </div>
        <a href="simd-scalable" style="text-align: end;">Scalable Vectorization</a>
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
