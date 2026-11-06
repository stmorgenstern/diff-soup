// The complex scalar of the cuDSS contract's CUDA kernels: two T, the real part then the imaginary
// part, laid out as std::complex<T> and as cuDSS's CUDA_C_64F and CUDA_C_32F values, with the few
// operations the kernels need. Each part follows IEEE arithmetic, so a non-finite part stays
// non-finite through a sum or a product.

#ifndef DIFF_SOUP_CUDSS_COMPLEX_SCALAR_CUH_
#define DIFF_SOUP_CUDSS_COMPLEX_SCALAR_CUH_

namespace diff_soup_complex {

template <typename T>
struct alignas(2 * sizeof(T)) Complex {
    T re;
    T im;

    Complex() = default;
    __host__ __device__ explicit Complex(int value)
        : re(static_cast<T>(value)), im(static_cast<T>(0)) {}
    __host__ __device__ Complex(T real, T imag) : re(real), im(imag) {}
    template <typename U>
    __host__ __device__ explicit Complex(const Complex<U>& other)
        : re(static_cast<T>(other.re)), im(static_cast<T>(other.im)) {}

    __host__ __device__ Complex& operator+=(const Complex& other) {
        re += other.re;
        im += other.im;
        return *this;
    }
};

template <typename T>
__host__ __device__ inline Complex<T> operator*(const Complex<T>& a, const Complex<T>& b) {
    return Complex<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

template <typename T>
__host__ __device__ inline Complex<T> operator*(const Complex<T>& a, T b) {
    return Complex<T>(a.re * b, a.im * b);
}

template <typename T>
__host__ __device__ inline bool operator==(const Complex<T>& a, const Complex<T>& b) {
    return a.re == b.re && a.im == b.im;
}

// Whether a scalar type of the kernels is complex.
template <typename T>
struct IsComplex {
    static constexpr bool value = false;
};
template <typename T>
struct IsComplex<Complex<T>> {
    static constexpr bool value = true;
};

}  // namespace diff_soup_complex

#endif  // DIFF_SOUP_CUDSS_COMPLEX_SCALAR_CUH_
