// ch-C05-simd/Main.cpp
//
// C5. CPU 최적화
//
// 저장소 루트에서 실행할 것. /openmp 를 켜고 빌드해야 한다.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Gemm.hpp"
#include "Matrix.hpp"
#include "Nn.hpp"
#include "Random.h"
#include "Vector.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define BOOK_SEED 20260914ull

// C3 의 출력층 크기. 배치 64 개를 한꺼번에 처리하면 이 모양이 된다.
#define NPLM_M 512
#define NPLM_N 64
#define NPLM_K 64

// 제대로 된 크기의 행렬 곱셈.
#define BIG_M 1024
#define BIG_N 1024
#define BIG_K 1024

namespace
{

using FGemm = void (*)(const Real*, const Real*, Real*, size_t, size_t, size_t);

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

struct FResult
{
    double Seconds = 0.0;
    double Gflops = 0.0;
    double WorstGap = 0.0;
};

FResult RunGemm(FGemm Function, const std::vector<Real>& A,
                const std::vector<Real>& B, std::vector<Real>& C,
                const std::vector<Real>& Reference,
                size_t M, size_t N, size_t K, int Repeat)
{
    FResult Result;

    // 한 번 미리 돌려 캐시를 덥힌다. 첫 회는 늘 느리다.
    Function(A.data(), B.data(), C.data(), M, N, K);

    auto Begin = std::chrono::steady_clock::now();
    for (int r = 0; r < Repeat; r++)
    {
        Function(A.data(), B.data(), C.data(), M, N, K);
    }
    Result.Seconds = Seconds(Begin) / Repeat;

    double Operations = 2.0 * (double)M * (double)N * (double)K;
    Result.Gflops = Operations / Result.Seconds / 1e9;

    if (!Reference.empty())
    {
        for (size_t i = 0; i < C.size(); i++)
        {
            double Gap = std::fabs((double)C[i] - (double)Reference[i]);
            if (Gap > Result.WorstGap)
            {
                Result.WorstGap = Gap;
            }
        }
    }

    return Result;
}

void PrintResult(const char* Name, const FResult& R, double BaseSeconds)
{
    char Buffer[64];

    printf("  ");
    PrintPadded(Name, 22);

    snprintf(Buffer, sizeof(Buffer), "%.4f 초", R.Seconds);
    PrintPaddedRight(Buffer, 14);

    snprintf(Buffer, sizeof(Buffer), "%.1f", R.Gflops);
    PrintPaddedRight(Buffer, 12);

    snprintf(Buffer, sizeof(Buffer), "%.1f 배", BaseSeconds / R.Seconds);
    PrintPaddedRight(Buffer, 12);

    snprintf(Buffer, sizeof(Buffer), "%.1e", R.WorstGap);
    PrintPaddedRight(Buffer, 12);

    printf("\n");
}

void FillRandom(std::vector<Real>& Values, FRandom& Rng)
{
    for (size_t i = 0; i < Values.size(); i++)
    {
        Values[i] = (Real)RandomRange(&Rng, 1.0);
    }
}

} // namespace

int main(void)
{
    printf("C5. CPU 최적화\n\n");

    printf("  Real     = %s\n", (sizeof(Real) == 8) ? "double" : "float");
    printf("  AVX2/FMA = %s\n", CpuHasAvx2() ? "쓸 수 있다" : "없다");
    printf("  스레드   = %d\n\n", CpuThreadCount());

    CHECK(CpuThreadCount() >= 1);

    // ---- C5-1. 무엇이 느린가 ----
    printf("[C5-1] 먼저 어디가 느린지부터 본다\n\n");

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        FLinear HiddenLayer(48, 64);
        FLinear OutputLayer(64, 512);
        HiddenLayer.InitScaled(Rng);
        OutputLayer.InitScaled(Rng);

        FVector X(48);
        for (size_t i = 0; i < X.Size(); i++)
        {
            X[i] = (Real)RandomRange(&Rng, 1.0);
        }

        const int Rounds = 20000;

        auto Begin = std::chrono::steady_clock::now();
        for (int r = 0; r < Rounds; r++)
        {
            FVector H = HiddenLayer.Forward(X);
            if (H[0] == Real(12345)) printf("never\n");
        }
        double HiddenSeconds = Seconds(Begin);

        FVector H = Tanh(HiddenLayer.Forward(X));

        Begin = std::chrono::steady_clock::now();
        for (int r = 0; r < Rounds; r++)
        {
            FVector S = OutputLayer.Forward(H);
            if (S[0] == Real(12345)) printf("never\n");
        }
        double OutputSeconds = Seconds(Begin);

        Begin = std::chrono::steady_clock::now();
        FVector S = OutputLayer.Forward(H);
        for (int r = 0; r < Rounds; r++)
        {
            FVector P = Softmax(S);
            if (P[0] == Real(12345)) printf("never\n");
        }
        double SoftmaxSeconds = Seconds(Begin);

        double Total = HiddenSeconds + OutputSeconds + SoftmaxSeconds;

        printf("  ");
        PrintPadded("단계", 18);
        PrintPaddedRight("곱셈 횟수", 14);
        PrintPaddedRight("시간", 12);
        PrintPaddedRight("비중", 10);
        printf("\n");

        struct { const char* Name; double Time; long long Mults; } Parts[3] = {
            { "은닉층 (48->64)", HiddenSeconds, 48LL * 64 },
            { "출력층 (64->512)", OutputSeconds, 64LL * 512 },
            { "softmax (512)", SoftmaxSeconds, 0 },
        };

        for (int i = 0; i < 3; i++)
        {
            char Buffer[32];
            printf("  ");
            PrintPadded(Parts[i].Name, 18);
            snprintf(Buffer, sizeof(Buffer), "%lld", Parts[i].Mults);
            PrintPaddedRight(Buffer, 14);
            snprintf(Buffer, sizeof(Buffer), "%.3f 초", Parts[i].Time);
            PrintPaddedRight(Buffer, 12);
            snprintf(Buffer, sizeof(Buffer), "%.1f%%",
                     Parts[i].Time * 100.0 / Total);
            PrintPaddedRight(Buffer, 10);
            printf("\n");
        }

        printf("\n  **출력층이 대부분이다.** 여기를 안 고치면 나머지를 고쳐봐야\n");
        printf("  전체가 안 빨라진다.\n\n");

        CHECK(OutputSeconds > HiddenSeconds);
    }

    // ---- 데이터 준비 ----
    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    // ---- C5-2. NPLM 출력층 크기로 ----
    printf("[C5-2] NPLM 출력층을 배치 %d개로 묶으면 (%d x %d x %d)\n\n",
           NPLM_N, NPLM_M, NPLM_N, NPLM_K);

    {
        std::vector<Real> A((size_t)NPLM_M * NPLM_K);
        std::vector<Real> B((size_t)NPLM_K * NPLM_N);
        std::vector<Real> C((size_t)NPLM_M * NPLM_N);
        std::vector<Real> Reference((size_t)NPLM_M * NPLM_N);

        FillRandom(A, Rng);
        FillRandom(B, Rng);

        GemmNaive(A.data(), B.data(), Reference.data(), NPLM_M, NPLM_N, NPLM_K);

        printf("  ");
        PrintPadded("방법", 22);
        PrintPaddedRight("시간", 14);
        PrintPaddedRight("GFLOP/s", 12);
        PrintPaddedRight("배속", 12);
        PrintPaddedRight("최대 차이", 12);
        printf("\n");

        FResult Base = RunGemm(GemmNaive, A, B, C, Reference,
                               NPLM_M, NPLM_N, NPLM_K, 200);
        PrintResult("1. 교과서 (i,j,k)", Base, Base.Seconds);

        FResult Reordered = RunGemm(GemmReorder, A, B, C, Reference,
                                    NPLM_M, NPLM_N, NPLM_K, 200);
        PrintResult("2. 순서 바꾸기", Reordered, Base.Seconds);

        FResult Blocked = RunGemm(GemmBlocked, A, B, C, Reference,
                                  NPLM_M, NPLM_N, NPLM_K, 200);
        PrintResult("3. 블록", Blocked, Base.Seconds);

        FResult Simd = RunGemm(GemmAvx2, A, B, C, Reference,
                               NPLM_M, NPLM_N, NPLM_K, 200);
        PrintResult("4. AVX2 + FMA", Simd, Base.Seconds);

        FResult Parallel = RunGemm(GemmParallel, A, B, C, Reference,
                                   NPLM_M, NPLM_N, NPLM_K, 200);
        PrintResult("5. 스레드", Parallel, Base.Seconds);

        printf("\n");

        // 전부 같은 답을 내야 한다. 완전히 같지는 않다 — 더하는 순서가 다르다.
        const double Tolerance = (sizeof(Real) == 8) ? 1e-9 : 1e-3;
        CHECK(Reordered.WorstGap < Tolerance);
        CHECK(Blocked.WorstGap < Tolerance);
        CHECK(Simd.WorstGap < Tolerance);
        CHECK(Parallel.WorstGap < Tolerance);

        CHECK(Simd.Seconds < Base.Seconds);

        printf("  이 크기에서도 스레드가 도움이 되긴 한다. 다만 %d x %d x %d\n",
               BIG_M, BIG_N, BIG_K);
        printf("  에서 얻는 배속에는 한참 못 미친다. 일감이 작으면 나누고\n");
        printf("  모으는 값이 계산에 비해 커지기 때문이다.\n\n");
    }

    // ---- C5-3. 제대로 된 크기로 ----
    printf("[C5-3] %d x %d x %d 로 키우면\n\n", BIG_M, BIG_N, BIG_K);

    std::vector<Real> BigA((size_t)BIG_M * BIG_K);
    std::vector<Real> BigB((size_t)BIG_K * BIG_N);
    std::vector<Real> BigC((size_t)BIG_M * BIG_N);
    std::vector<Real> BigReference((size_t)BIG_M * BIG_N);

    FillRandom(BigA, Rng);
    FillRandom(BigB, Rng);

    GemmNaive(BigA.data(), BigB.data(), BigReference.data(), BIG_M, BIG_N, BIG_K);

    {
        printf("  ");
        PrintPadded("방법", 22);
        PrintPaddedRight("시간", 14);
        PrintPaddedRight("GFLOP/s", 12);
        PrintPaddedRight("배속", 12);
        PrintPaddedRight("최대 차이", 12);
        printf("\n");

        FResult Base = RunGemm(GemmNaive, BigA, BigB, BigC, BigReference,
                               BIG_M, BIG_N, BIG_K, 3);
        PrintResult("1. 교과서 (i,j,k)", Base, Base.Seconds);

        FResult Reordered = RunGemm(GemmReorder, BigA, BigB, BigC, BigReference,
                                    BIG_M, BIG_N, BIG_K, 5);
        PrintResult("2. 순서 바꾸기", Reordered, Base.Seconds);

        FResult Blocked = RunGemm(GemmBlocked, BigA, BigB, BigC, BigReference,
                                  BIG_M, BIG_N, BIG_K, 5);
        PrintResult("3. 블록", Blocked, Base.Seconds);

        FResult Simd = RunGemm(GemmAvx2, BigA, BigB, BigC, BigReference,
                               BIG_M, BIG_N, BIG_K, 10);
        PrintResult("4. AVX2 + FMA", Simd, Base.Seconds);

        FResult Parallel = RunGemm(GemmParallel, BigA, BigB, BigC, BigReference,
                                   BIG_M, BIG_N, BIG_K, 10);
        PrintResult("5. 스레드", Parallel, Base.Seconds);

        printf("\n");

        const double Tolerance = (sizeof(Real) == 8) ? 1e-9 : 1e-2;
        CHECK(Reordered.WorstGap < Tolerance);
        CHECK(Blocked.WorstGap < Tolerance);
        CHECK(Simd.WorstGap < Tolerance);
        CHECK(Parallel.WorstGap < Tolerance);

        CHECK(Parallel.Seconds < Base.Seconds);
        CHECK(Simd.Seconds < Reordered.Seconds);

        // ---- C5-4. 기계 최대치와 비교 ----
        printf("[C5-4] 이 기계가 낼 수 있는 최대치와 비교\n\n");

        double PeakOne = MeasurePeakGflops(1);
        double PeakAll = MeasurePeakGflops(CpuThreadCount());

        printf("  ");
        PrintPadded("", 22);
        PrintPaddedRight("GFLOP/s", 14);
        PrintPaddedRight("최대치 대비", 14);
        printf("\n");

        char Buffer[64];

        printf("  ");
        PrintPadded("한 스레드 최대치", 22);
        snprintf(Buffer, sizeof(Buffer), "%.0f", PeakOne);
        PrintPaddedRight(Buffer, 14);
        PrintPaddedRight("100%", 14);
        printf("\n");

        printf("  ");
        PrintPadded("우리 AVX2 (1스레드)", 22);
        snprintf(Buffer, sizeof(Buffer), "%.1f", Simd.Gflops);
        PrintPaddedRight(Buffer, 14);
        snprintf(Buffer, sizeof(Buffer), "%.1f%%", Simd.Gflops * 100.0 / PeakOne);
        PrintPaddedRight(Buffer, 14);
        printf("\n");

        printf("  ");
        PrintPadded("전체 스레드 최대치", 22);
        snprintf(Buffer, sizeof(Buffer), "%.0f", PeakAll);
        PrintPaddedRight(Buffer, 14);
        PrintPaddedRight("100%", 14);
        printf("\n");

        printf("  ");
        PrintPadded("우리 병렬판", 22);
        snprintf(Buffer, sizeof(Buffer), "%.1f", Parallel.Gflops);
        PrintPaddedRight(Buffer, 14);
        snprintf(Buffer, sizeof(Buffer), "%.1f%%",
                 Parallel.Gflops * 100.0 / PeakAll);
        PrintPaddedRight(Buffer, 14);
        printf("\n\n");

        CHECK(PeakOne > 0.0);
        CHECK(Simd.Gflops < PeakOne * 1.2);   // 최대치를 넘을 수는 없다
    }

    // ---- C5-5. 효과가 없던 것 ----
    printf("[C5-5] 해봤는데 오히려 느려진 것\n\n");

    {
        printf("  M, K 만 쪼개는 대신 **N 까지 쪼개면** 어떻게 되는가\n");
        printf("  (%d x %d x %d, 한 스레드, lib/Gemm 의 같은 코드)\n\n",
               BIG_M, BIG_N, BIG_K);

        printf("  ");
        PrintPaddedRight("블록", 10);
        PrintPaddedRight("M,K 만", 14);
        PrintPaddedRight("M,N,K 전부", 16);
        printf("\n");

        const size_t Sizes[4] = { 32, 64, 128, 256 };

        double BestTwo = 0.0;
        double BestThree = 0.0;
        size_t BestTwoSize = 0;

        for (int s = 0; s < 4; s++)
        {
            size_t Block = Sizes[s];
            const int Repeat = 3;

            auto Begin = std::chrono::steady_clock::now();
            for (int r = 0; r < Repeat; r++)
            {
                GemmBlockedTuned(BigA.data(), BigB.data(), BigC.data(),
                                 BIG_M, BIG_N, BIG_K, Block, Block, 0);
            }
            double TwoGflops = 2.0 * BIG_M * BIG_N * BIG_K
                             / (Seconds(Begin) / Repeat) / 1e9;

            Begin = std::chrono::steady_clock::now();
            for (int r = 0; r < Repeat; r++)
            {
                GemmBlockedTuned(BigA.data(), BigB.data(), BigC.data(),
                                 BIG_M, BIG_N, BIG_K, Block, Block, Block);
            }
            double ThreeGflops = 2.0 * BIG_M * BIG_N * BIG_K
                               / (Seconds(Begin) / Repeat) / 1e9;

            if (TwoGflops > BestTwo)
            {
                BestTwo = TwoGflops;
                BestTwoSize = Block;
            }
            if (ThreeGflops > BestThree)
            {
                BestThree = ThreeGflops;
            }

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%zu", Block);
            PrintPaddedRight(Buffer, 10);
            snprintf(Buffer, sizeof(Buffer), "%.1f", TwoGflops);
            PrintPaddedRight(Buffer, 14);
            snprintf(Buffer, sizeof(Buffer), "%.1f", ThreeGflops);
            PrintPaddedRight(Buffer, 16);
            printf("\n");
        }

        printf("\n  가장 좋았던 블록 크기 = %zu (%.1f GFLOP/s)\n", BestTwoSize,
               BestTwo);
        printf("  N 까지 쪼개면 최고가 %.1f 로 떨어진다\n\n", BestThree);

        CHECK(BestTwo > BestThree);

        printf("  교과서에는 '캐시 블로킹은 세 축을 다 쪼갠다' 고 나온다.\n");
        printf("  실제로 재보니 이 코드에서는 **손해였다.** 안쪽 루프가\n");
        printf("  짧아지면서 컴파일러의 자동 벡터화가 힘을 잃기 때문이다.\n");
        printf("  세 축을 다 쪼개고도 이기려면 패킹과 레지스터 타일링이\n");
        printf("  필요한데, 그건 이 장의 범위를 넘는다.\n\n");
    }

    return ReportResult();
}
