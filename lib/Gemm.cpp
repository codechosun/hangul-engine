// lib/Gemm.cpp

#include "Gemm.hpp"

#include <cstring>
#include <chrono>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace
{

// 블록 크기. C5 본문에서 재보고 정한 값이다.
//
// **N(열)은 안 쪼갠다.** 안쪽 루프를 길게 유지해야 컴파일러가 자동
// 벡터화를 제대로 하고 루프 준비 비용도 묻힌다. 쪼개는 것은 M 과 K 뿐이다.
// 처음에는 셋 다 쪼갰는데 오히려 느렸다. C5 본문 참고.
const size_t GBlockM = 64;
const size_t GBlockK = 128;

} // namespace

bool CpuHasAvx2()
{
#if defined(_MSC_VER)
    int Info[4] = {};

    __cpuid(Info, 0);
    if (Info[0] < 7)
    {
        return false;
    }

    __cpuid(Info, 1);
    bool bHasFma = ((Info[2] >> 12) & 1) != 0;
    bool bHasOsXsave = ((Info[2] >> 27) & 1) != 0;
    bool bHasAvx = ((Info[2] >> 28) & 1) != 0;

    if (!bHasFma || !bHasAvx || !bHasOsXsave)
    {
        return false;
    }

    // 운영체제가 YMM 레지스터를 저장해주는지도 확인해야 한다.
    // CPU 가 지원해도 OS 가 모르면 문맥 전환에서 값이 날아간다.
    unsigned long long Mask = _xgetbv(0);
    if ((Mask & 0x6) != 0x6)
    {
        return false;
    }

    __cpuidex(Info, 7, 0);
    return ((Info[1] >> 5) & 1) != 0;
#else
    return false;
#endif
}

int CpuThreadCount()
{
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void GemmNaive(const Real* A, const Real* B, Real* C,
               size_t M, size_t N, size_t K)
{
    // 교과서에 나오는 그대로. 각 C[i][j] 를 하나씩 완성한다.
    //
    // 문제는 B 를 세로로 훑는다는 것이다. B[k * N + j] 에서 k 가 늘 때마다
    // N 칸씩 건너뛴다. 캐시 라인 하나를 읽어 4바이트만 쓰고 버린다.
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            Real Sum = Real(0);
            for (size_t k = 0; k < K; k++)
            {
                Sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = Sum;
        }
    }
}

void GemmReorder(const Real* A, const Real* B, Real* C,
                 size_t M, size_t N, size_t K)
{
    // 같은 계산, 다른 순서.
    //
    // 안쪽 루프가 j 를 돌면 B[k*N + j] 와 C[i*N + j] 둘 다 **순차 접근**이다.
    // A[i*K + k] 는 안쪽 루프 안에서 안 변하므로 레지스터에 올라간다.
    std::memset(C, 0, M * N * sizeof(Real));

    for (size_t i = 0; i < M; i++)
    {
        for (size_t k = 0; k < K; k++)
        {
            Real Scale = A[i * K + k];
            const Real* BRow = B + k * N;
            Real* CRow = C + i * N;

            for (size_t j = 0; j < N; j++)
            {
                CRow[j] += Scale * BRow[j];
            }
        }
    }
}

void GemmBlockedTuned(const Real* A, const Real* B, Real* C,
                      size_t M, size_t N, size_t K,
                      size_t BlockM, size_t BlockK, size_t BlockN)
{
    // 블록으로 쪼갠다. 한 블록이 캐시에 들어가면 그 안에서는 메모리를
    // 다시 안 읽는다.
    std::memset(C, 0, M * N * sizeof(Real));

    if (BlockN == 0)
    {
        BlockN = N;   // N 은 안 쪼갠다
    }

    for (size_t i0 = 0; i0 < M; i0 += BlockM)
    {
        size_t iStop = (i0 + BlockM < M) ? (i0 + BlockM) : M;

        for (size_t k0 = 0; k0 < K; k0 += BlockK)
        {
            size_t kStop = (k0 + BlockK < K) ? (k0 + BlockK) : K;

            for (size_t j0 = 0; j0 < N; j0 += BlockN)
            {
                size_t jStop = (j0 + BlockN < N) ? (j0 + BlockN) : N;

                for (size_t i = i0; i < iStop; i++)
                {
                    Real* CRow = C + i * N;

                    for (size_t k = k0; k < kStop; k++)
                    {
                        Real Scale = A[i * K + k];
                        const Real* BRow = B + k * N;

                        for (size_t j = j0; j < jStop; j++)
                        {
                            CRow[j] += Scale * BRow[j];
                        }
                    }
                }
            }
        }
    }
}

void GemmBlocked(const Real* A, const Real* B, Real* C,
                 size_t M, size_t N, size_t K)
{
    GemmBlockedTuned(A, B, C, M, N, K, GBlockM, GBlockK, 0);
}

#if defined(_MSC_VER)

namespace
{

// 한 줄(C[i][j0..j1])을 AVX2 로 갱신한다.
inline void AddScaledRow(Real* CRow, const Real* BRow, Real Scale,
                         size_t From, size_t To)
{
    if (sizeof(Real) == 4)
    {
        __m256 ScaleVector = _mm256_set1_ps((float)Scale);

        size_t j = From;
        for (; j + 8 <= To; j += 8)
        {
            __m256 CValue = _mm256_loadu_ps((const float*)(CRow + j));
            __m256 BValue = _mm256_loadu_ps((const float*)(BRow + j));

            // 곱하고 더하기를 **한 명령**으로. 반올림도 한 번만 일어난다.
            CValue = _mm256_fmadd_ps(BValue, ScaleVector, CValue);

            _mm256_storeu_ps((float*)(CRow + j), CValue);
        }

        // 8 로 안 나눠떨어지는 꼬리는 그냥 돈다.
        for (; j < To; j++)
        {
            CRow[j] += Scale * BRow[j];
        }
    }
    else
    {
        __m256d ScaleVector = _mm256_set1_pd((double)Scale);

        size_t j = From;
        for (; j + 4 <= To; j += 4)
        {
            __m256d CValue = _mm256_loadu_pd((const double*)(CRow + j));
            __m256d BValue = _mm256_loadu_pd((const double*)(BRow + j));

            CValue = _mm256_fmadd_pd(BValue, ScaleVector, CValue);

            _mm256_storeu_pd((double*)(CRow + j), CValue);
        }

        for (; j < To; j++)
        {
            CRow[j] += Scale * BRow[j];
        }
    }
}

void GemmAvx2Core(const Real* A, const Real* B, Real* C,
                  size_t M, size_t N, size_t K,
                  size_t RowFrom, size_t RowTo)
{
    for (size_t i0 = RowFrom; i0 < RowTo; i0 += GBlockM)
    {
        size_t iStop = (i0 + GBlockM < RowTo) ? (i0 + GBlockM) : RowTo;

        for (size_t k0 = 0; k0 < K; k0 += GBlockK)
        {
            size_t kStop = (k0 + GBlockK < K) ? (k0 + GBlockK) : K;

            for (size_t i = i0; i < iStop; i++)
            {
                Real* CRow = C + i * N;

                for (size_t k = k0; k < kStop; k++)
                {
                    AddScaledRow(CRow, B + k * N, A[i * K + k], 0, N);
                }
            }
        }
    }
}

} // namespace

#endif

void GemmAvx2(const Real* A, const Real* B, Real* C,
              size_t M, size_t N, size_t K)
{
#if defined(_MSC_VER)
    if (!CpuHasAvx2())
    {
        GemmBlocked(A, B, C, M, N, K);
        return;
    }

    std::memset(C, 0, M * N * sizeof(Real));
    GemmAvx2Core(A, B, C, M, N, K, 0, M);
#else
    GemmBlocked(A, B, C, M, N, K);
#endif
}

void GemmParallel(const Real* A, const Real* B, Real* C,
                  size_t M, size_t N, size_t K)
{
#if defined(_MSC_VER) && defined(_OPENMP)
    if (!CpuHasAvx2())
    {
        GemmBlocked(A, B, C, M, N, K);
        return;
    }

    std::memset(C, 0, M * N * sizeof(Real));

    // 행을 나눠 맡는다.
    //
    // **겹치지 않게 나누는 것**이 요점이다. 스레드마다 C 의 서로 다른 줄만
    // 건드리므로 같은 자리에 두 스레드가 쓰는 일이 없다. 잠금도 필요 없다.
    const int Threads = omp_get_max_threads();

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < Threads; t++)
    {
        size_t From = (size_t)((long long)M * t / Threads);
        size_t To = (size_t)((long long)M * (t + 1) / Threads);

        // 블록 경계에 맞춰 잘라야 블록 논리가 깨지지 않는다.
        if (From < To)
        {
            GemmAvx2Core(A, B, C, M, N, K, From, To);
        }
    }
#else
    GemmAvx2(A, B, C, M, N, K);
#endif
}

double MeasurePeakGflops(int ThreadCount)
{
#if defined(_MSC_VER)
    if (!CpuHasAvx2())
    {
        return 0.0;
    }

    const long long Rounds = 2000000;
    const int Chains = 8;   // 의존성을 끊어 파이프라인을 채운다

    auto Begin = std::chrono::steady_clock::now();

    double Guard = 0.0;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) reduction(+:Guard) num_threads(ThreadCount)
#endif
    for (int t = 0; t < ThreadCount; t++)
    {
        __m256 Accumulator[Chains];
        __m256 One = _mm256_set1_ps(1.0000001f);

        for (int c = 0; c < Chains; c++)
        {
            Accumulator[c] = _mm256_set1_ps((float)(c + 1));
        }

        for (long long r = 0; r < Rounds; r++)
        {
            for (int c = 0; c < Chains; c++)
            {
                Accumulator[c] = _mm256_fmadd_ps(Accumulator[c], One, One);
            }
        }

        float Out[8];
        for (int c = 0; c < Chains; c++)
        {
            _mm256_storeu_ps(Out, Accumulator[c]);
            Guard += (double)Out[0];
        }
    }

    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;

    // FMA 하나가 곱셈 + 덧셈 = 2 연산. 한 명령이 8개(float)를 한다.
    double Operations = (double)Rounds * Chains * 8.0 * 2.0 * ThreadCount;

    if (Guard == 12345.6789)   // 최적화가 계산을 지우지 못하게 붙잡아둔다
    {
        return 0.0;
    }

    return Operations / Elapsed.count() / 1e9;
#else
    (void)ThreadCount;
    return 0.0;
#endif
}
