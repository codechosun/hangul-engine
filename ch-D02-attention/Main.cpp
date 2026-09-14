// ch-D02-attention/Main.cpp
//
// D2. 어텐션 순전파
//
// 저장소 루트에서 실행할 것.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Attention.hpp"
#include "Random.h"
#include "Tensor.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define BOOK_SEED 20260914ull

#define BATCH  2
#define HEADS  4
#define LENGTH 6
#define DIM    8

namespace
{

void FillRandom(FTensor& T, FRandom& Rng, double Range)
{
    for (size_t i = 0; i < T.Count(); i++)
    {
        T.At(i) = (Real)RandomRange(&Rng, Range);
    }
}

// 분포가 얼마나 퍼져 있는가. 1 에 가까우면 균등, 0 에 가까우면 한 곳에 몰림.
double NormalizedEntropy(const Real* Row, size_t Count)
{
    double Entropy = 0.0;
    for (size_t i = 0; i < Count; i++)
    {
        double P = (double)Row[i];
        if (P > 1e-12)
        {
            Entropy -= P * std::log(P);
        }
    }
    return Entropy / std::log((double)Count);
}

} // namespace

int main(void)
{
    printf("D2. 어텐션 순전파\n\n");

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    // ---- D2-1. 손으로 따라갈 수 있는 크기 ----
    printf("[D2-1] 위치 셋, 차원 둘짜리 어텐션\n\n");

    {
        // (배치 1, 헤드 1, 위치 3, 차원 2)
        FTensor Q({ 1, 1, 3, 2 });
        FTensor K({ 1, 1, 3, 2 });
        FTensor V({ 1, 1, 3, 2 });

        // Q 의 각 줄이 K 의 어느 줄과 닮았는지 눈에 보이게 만든다.
        Q(0, 0, 0, 0) = Real(1); Q(0, 0, 0, 1) = Real(0);
        Q(0, 0, 1, 0) = Real(0); Q(0, 0, 1, 1) = Real(1);
        Q(0, 0, 2, 0) = Real(1); Q(0, 0, 2, 1) = Real(1);

        K(0, 0, 0, 0) = Real(1); K(0, 0, 0, 1) = Real(0);
        K(0, 0, 1, 0) = Real(0); K(0, 0, 1, 1) = Real(1);
        K(0, 0, 2, 0) = Real(1); K(0, 0, 2, 1) = Real(1);

        V(0, 0, 0, 0) = Real(10); V(0, 0, 0, 1) = Real(0);
        V(0, 0, 1, 0) = Real(0);  V(0, 0, 1, 1) = Real(10);
        V(0, 0, 2, 0) = Real(5);  V(0, 0, 2, 1) = Real(5);

        FAttention Result = Attend(Q, K, V, false);

        printf("  Q = [1 0; 0 1; 1 1], K 도 같다, V = [10 0; 0 10; 5 5]\n\n");

        printf("  어텐션 가중치 (각 줄의 합이 1)\n");
        for (size_t q = 0; q < 3; q++)
        {
            printf("    ");
            double Sum = 0.0;
            for (size_t k = 0; k < 3; k++)
            {
                printf("%8.5f", (double)Result.Weights(0, 0, q, k));
                Sum += (double)Result.Weights(0, 0, q, k);
            }
            printf("   합 %.9f\n", Sum);

            CHECK_NEAR(Sum, 1.0, 1e-6);
        }

        printf("\n  출력\n");
        for (size_t q = 0; q < 3; q++)
        {
            printf("    %8.4f %8.4f\n", (double)Result.Output(0, 0, q, 0),
                   (double)Result.Output(0, 0, q, 1));
        }
        printf("\n");

        // 손계산: Q 첫 줄 (1,0) 과 K 의 내적은 (1, 0, 1)/sqrt(2) = (0.7071, 0, 0.7071)
        // softmax(0.7071, 0, 0.7071) = (0.4223, 0.1554, 0.4223)
        printf("  첫 줄 손계산\n");
        printf("    점수 = (1, 0, 1) / sqrt(2) = (%.4f, %.4f, %.4f)\n",
               1.0 / std::sqrt(2.0), 0.0, 1.0 / std::sqrt(2.0));

        double E1 = std::exp(1.0 / std::sqrt(2.0));
        double E2 = std::exp(0.0);
        double Total = E1 + E2 + E1;

        printf("    softmax = (%.6f, %.6f, %.6f)\n", E1 / Total, E2 / Total,
               E1 / Total);
        printf("    C++ 결과 = (%.6f, %.6f, %.6f)\n\n",
               (double)Result.Weights(0, 0, 0, 0),
               (double)Result.Weights(0, 0, 0, 1),
               (double)Result.Weights(0, 0, 0, 2));

        CHECK_NEAR(Result.Weights(0, 0, 0, 0), E1 / Total, 1e-6);
        CHECK_NEAR(Result.Weights(0, 0, 0, 1), E2 / Total, 1e-6);
        CHECK_NEAR(Result.Weights(0, 0, 0, 2), E1 / Total, 1e-6);

        // 대칭성: Q 첫 줄과 둘째 줄은 서로 뒤집힌 모양이다.
        CHECK_NEAR(Result.Weights(0, 0, 0, 0), Result.Weights(0, 0, 1, 1), 1e-6);
    }

    // ---- D2-2. 모든 줄의 합이 1인가 ----
    printf("[D2-2] 제대로 된 크기에서 줄의 합\n\n");

    {
        FTensor Q({ BATCH, HEADS, LENGTH, DIM });
        FTensor K({ BATCH, HEADS, LENGTH, DIM });
        FTensor V({ BATCH, HEADS, LENGTH, DIM });

        FillRandom(Q, Rng, 1.0);
        FillRandom(K, Rng, 1.0);
        FillRandom(V, Rng, 1.0);

        FAttention Result = Attend(Q, K, V, false);

        printf("  모양 (%zu, %zu, %zu, %zu)\n", Q.Size(0), Q.Size(1), Q.Size(2),
               Q.Size(3));
        printf("  가중치 모양 (%zu, %zu, %zu, %zu)\n",
               Result.Weights.Size(0), Result.Weights.Size(1),
               Result.Weights.Size(2), Result.Weights.Size(3));
        printf("  출력 모양 (%zu, %zu, %zu, %zu)\n\n",
               Result.Output.Size(0), Result.Output.Size(1),
               Result.Output.Size(2), Result.Output.Size(3));

        CHECK(Result.Weights.Size(2) == LENGTH);
        CHECK(Result.Weights.Size(3) == LENGTH);
        CHECK(Result.Output.Size(3) == DIM);

        double WorstGap = 0.0;
        int Rows = 0;

        for (size_t b = 0; b < BATCH; b++)
        for (size_t h = 0; h < HEADS; h++)
        for (size_t q = 0; q < LENGTH; q++)
        {
            double Sum = 0.0;
            for (size_t k = 0; k < LENGTH; k++)
            {
                Sum += (double)Result.Weights(b, h, q, k);
            }

            double Gap = std::fabs(Sum - 1.0);
            if (Gap > WorstGap) WorstGap = Gap;
            Rows++;
        }

        printf("  줄 %d개 전부에서 합이 1 인가. 최대 차이 = %.3e\n\n",
               Rows, WorstGap);

        CHECK(WorstGap < 1e-5);
        CHECK(Rows == BATCH * HEADS * LENGTH);
    }

    // ---- D2-3. 인과 마스크 ----
    printf("[D2-3] 뒤를 못 보게 막는다\n\n");

    {
        FTensor Q({ 1, 1, LENGTH, DIM });
        FTensor K({ 1, 1, LENGTH, DIM });
        FTensor V({ 1, 1, LENGTH, DIM });

        FillRandom(Q, Rng, 1.0);
        FillRandom(K, Rng, 1.0);
        FillRandom(V, Rng, 1.0);

        FAttention Free = Attend(Q, K, V, false);
        FAttention Masked = Attend(Q, K, V, true);

        printf("  마스크 없이\n");
        for (size_t q = 0; q < LENGTH; q++)
        {
            printf("    ");
            for (size_t k = 0; k < LENGTH; k++)
            {
                printf("%8.4f", (double)Free.Weights(0, 0, q, k));
            }
            printf("\n");
        }

        printf("\n  마스크를 걸면\n");
        for (size_t q = 0; q < LENGTH; q++)
        {
            printf("    ");
            for (size_t k = 0; k < LENGTH; k++)
            {
                printf("%8.4f", (double)Masked.Weights(0, 0, q, k));
            }
            printf("\n");
        }
        printf("\n");

        // 우상단이 정확히 0 이어야 한다.
        double WorstUpper = 0.0;
        for (size_t q = 0; q < LENGTH; q++)
        {
            for (size_t k = q + 1; k < LENGTH; k++)
            {
                double Value = (double)Masked.Weights(0, 0, q, k);
                if (Value > WorstUpper) WorstUpper = Value;
            }
        }

        printf("  우상단의 최대값 = %.3e (0 이어야 한다)\n", WorstUpper);
        CHECK(WorstUpper < 1e-12);

        // 그리고 각 줄의 합은 여전히 1 이어야 한다.
        double WorstGap = 0.0;
        for (size_t q = 0; q < LENGTH; q++)
        {
            double Sum = 0.0;
            for (size_t k = 0; k < LENGTH; k++)
            {
                Sum += (double)Masked.Weights(0, 0, q, k);
            }
            double Gap = std::fabs(Sum - 1.0);
            if (Gap > WorstGap) WorstGap = Gap;
        }

        printf("  줄의 합은 여전히 1 인가. 최대 차이 = %.3e\n", WorstGap);
        CHECK(WorstGap < 1e-5);

        // 첫 줄은 자기 자신만 보므로 반드시 1 이다.
        printf("  첫 줄은 자기 자신만 본다 -> %.9f\n\n",
               (double)Masked.Weights(0, 0, 0, 0));
        CHECK_NEAR(Masked.Weights(0, 0, 0, 0), 1.0, 1e-9);

        // 마스크를 걸면 출력이 달라져야 한다. 안 달라지면 마스크가 안 걸린 것이다.
        double OutputGap = 0.0;
        for (size_t i = 0; i < Free.Output.Count(); i++)
        {
            double Gap = std::fabs((double)Free.Output.At(i)
                                 - (double)Masked.Output.At(i));
            if (Gap > OutputGap) OutputGap = Gap;
        }

        printf("  마스크가 출력을 바꾸는가. 최대 차이 = %.4f\n\n", OutputGap);
        CHECK(OutputGap > 1e-3);
    }

    // ---- D2-4. 왜 sqrt(d) 로 나누는가 ----
    printf("[D2-4] 차원이 커지면 점수가 커진다\n\n");
    printf("  ");
    PrintPaddedRight("차원", 8);
    PrintPaddedRight("내적의 표준편차", 18);
    PrintPaddedRight("나누기 전 엔트로피", 22);
    PrintPaddedRight("나눈 뒤", 12);
    printf("\n");

    {
        const size_t Dims[5] = { 4, 16, 64, 256, 1024 };

        double FirstRaw = 0.0;
        double LastRaw = 0.0;

        for (int s = 0; s < 5; s++)
        {
            const size_t D = Dims[s];
            const size_t T = 16;

            FTensor Q({ 1, 1, T, D });
            FTensor K({ 1, 1, T, D });
            FillRandom(Q, Rng, 1.0);
            FillRandom(K, Rng, 1.0);

            // 내적을 직접 구해 퍼진 정도를 본다.
            std::vector<double> Dots;
            for (size_t q = 0; q < T; q++)
            {
                for (size_t k = 0; k < T; k++)
                {
                    double Sum = 0.0;
                    for (size_t d = 0; d < D; d++)
                    {
                        Sum += (double)Q(0, 0, q, d) * (double)K(0, 0, k, d);
                    }
                    Dots.push_back(Sum);
                }
            }

            double Mean = 0.0;
            for (double Value : Dots) Mean += Value;
            Mean /= (double)Dots.size();

            double Variance = 0.0;
            for (double Value : Dots) Variance += (Value - Mean) * (Value - Mean);
            Variance /= (double)Dots.size();

            double Deviation = std::sqrt(Variance);

            // 나누기 전 / 후의 softmax 엔트로피
            FTensor Raw({ 1, 1, T, T });
            FTensor Scaled({ 1, 1, T, T });

            for (size_t i = 0; i < Dots.size(); i++)
            {
                Raw.At(i) = (Real)Dots[i];
                Scaled.At(i) = (Real)(Dots[i] / std::sqrt((double)D));
            }

            SoftmaxLastAxis(Raw);
            SoftmaxLastAxis(Scaled);

            double RawEntropy = 0.0;
            double ScaledEntropy = 0.0;
            for (size_t q = 0; q < T; q++)
            {
                RawEntropy += NormalizedEntropy(Raw.Data() + q * T, T);
                ScaledEntropy += NormalizedEntropy(Scaled.Data() + q * T, T);
            }
            RawEntropy /= (double)T;
            ScaledEntropy /= (double)T;

            if (s == 0) FirstRaw = RawEntropy;
            LastRaw = RawEntropy;

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%zu", D);
            PrintPaddedRight(Buffer, 8);
            snprintf(Buffer, sizeof(Buffer), "%.3f", Deviation);
            PrintPaddedRight(Buffer, 18);
            snprintf(Buffer, sizeof(Buffer), "%.4f", RawEntropy);
            PrintPaddedRight(Buffer, 22);
            snprintf(Buffer, sizeof(Buffer), "%.4f", ScaledEntropy);
            PrintPaddedRight(Buffer, 12);
            printf("\n");
        }

        printf("\n  표준편차가 sqrt(차원) 에 비례해 커진다.\n");
        printf("  나누지 않으면 엔트로피가 %.4f 에서 %.4f 로 무너진다.\n",
               FirstRaw, LastRaw);
        printf("  한 곳만 1 이고 나머지가 0 이면 **그래디언트가 흐르지 않는다.**\n\n");

        CHECK(LastRaw < FirstRaw);
    }

    // ---- D2-5. 헤드 나누기와 합치기 ----
    printf("[D2-5] 헤드를 나눴다 합치면 제자리인가\n\n");

    {
        const size_t Model = HEADS * DIM;

        FTensor X({ BATCH, LENGTH, Model });
        for (size_t i = 0; i < X.Count(); i++)
        {
            X.At(i) = (Real)i;
        }

        FTensor Split = SplitHeads(X, HEADS);
        FTensor Merged = MergeHeads(Split);

        printf("  (%zu, %zu, %zu) -> (%zu, %zu, %zu, %zu) -> (%zu, %zu, %zu)\n\n",
               X.Size(0), X.Size(1), X.Size(2),
               Split.Size(0), Split.Size(1), Split.Size(2), Split.Size(3),
               Merged.Size(0), Merged.Size(1), Merged.Size(2));

        CHECK(Split.Rank() == 4);
        CHECK(Split.Size(1) == HEADS);
        CHECK(Split.Size(2) == LENGTH);
        CHECK(Split.Size(3) == DIM);

        int Same = 1;
        for (size_t i = 0; i < X.Count(); i++)
        {
            if (Merged.At(i) != X.At(i))
            {
                Same = 0;
            }
        }

        printf("  왕복해서 제자리인가 = %s\n", Same ? "예" : "아니오");
        CHECK(Same == 1);

        // 나뉜 뒤의 자리도 확인한다.
        int Placed = 1;
        for (size_t b = 0; b < BATCH; b++)
        for (size_t t = 0; t < LENGTH; t++)
        for (size_t h = 0; h < HEADS; h++)
        for (size_t d = 0; d < DIM; d++)
        {
            if (Split(b, h, t, d) != X(b, t, h * DIM + d))
            {
                Placed = 0;
            }
        }

        printf("  Split(b,h,t,d) == X(b,t,h*DIM+d) 인가 = %s\n\n",
               Placed ? "예" : "아니오");
        CHECK(Placed == 1);
    }

    // ---- D2-6. 헤드가 여럿이면 무엇이 달라지는가 ----
    printf("[D2-6] 헤드 하나와 여럿\n\n");

    {
        const size_t Model = 32;
        FTensor X({ 1, LENGTH, Model });
        FillRandom(X, Rng, 1.0);

        for (size_t HeadCount : { (size_t)1, (size_t)2, (size_t)4, (size_t)8 })
        {
            FTensor Split = SplitHeads(X, HeadCount);
            FAttention Result = Attend(Split, Split, Split, true);
            FTensor Merged = MergeHeads(Result.Output);

            printf("  헤드 %zu개 -> 헤드당 차원 %zu, 출력 모양 (%zu, %zu, %zu)\n",
                   HeadCount, Model / HeadCount, Merged.Size(0), Merged.Size(1),
                   Merged.Size(2));

            CHECK(Merged.Size(2) == Model);
        }

        printf("\n  **어느 경우에도 출력 크기가 같다.** 헤드 수는 안에서\n");
        printf("  일을 몇 갈래로 나눌지만 정한다. 계산량도 거의 같다.\n\n");
    }

    return ReportResult();
}
