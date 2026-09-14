// ch-D03-recipe/Main.cpp
//
// D3. 현대적 레시피
//
// 저장소 루트에서 실행할 것.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Attention.hpp"
#include "Block.hpp"
#include "Random.h"
#include "Tensor.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define BOOK_SEED 20260914ull

#define BATCH  2
#define LENGTH 8
#define MODEL  64
#define HEADS  4
#define HIDDEN 256

namespace
{

void FillRandom(FTensor& T, FRandom& Rng, double Range)
{
    for (size_t i = 0; i < T.Count(); i++)
    {
        T.At(i) = (Real)RandomRange(&Rng, Range);
    }
}

// 마지막 축을 따라 잰 RMS 의 평균.
double MeanRms(const FTensor& T)
{
    const size_t Last = T.Size(T.Rank() - 1);
    const size_t Rows = T.Count() / Last;

    double Total = 0.0;
    for (size_t r = 0; r < Rows; r++)
    {
        const Real* Row = T.Data() + r * Last;

        double Square = 0.0;
        for (size_t i = 0; i < Last; i++)
        {
            Square += (double)Row[i] * (double)Row[i];
        }

        Total += std::sqrt(Square / (double)Last);
    }

    return Total / (double)Rows;
}

// 블록을 Depth 개 쌓아 통과시킨 뒤 크기를 잰다.
double StackAndMeasure(const FBlockConfig& Config, int Depth, uint64_t Seed)
{
    FRandom Rng;
    RandomSeed(&Rng, Seed);

    FTensor X({ BATCH, LENGTH, MODEL });
    FillRandom(X, Rng, 1.0);

    std::vector<FBlock> Blocks;
    for (int i = 0; i < Depth; i++)
    {
        FBlock Block(Config);
        Block.Init(Rng);
        Blocks.push_back(std::move(Block));
    }

    FTensor Current = X;
    for (int i = 0; i < Depth; i++)
    {
        Current = Blocks[(size_t)i].Forward(Current);
    }

    return MeanRms(Current);
}

} // namespace

int main(void)
{
    printf("D3. 현대적 레시피\n\n");

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    // ---- D3-1. RMSNorm ----
    printf("[D3-1] RMSNorm — 크기를 1 로 되돌린다\n\n");

    {
        FTensor X({ 1, 1, 4 });
        X(0, 0, 0) = Real(3); X(0, 0, 1) = Real(4);
        X(0, 0, 2) = Real(0); X(0, 0, 3) = Real(0);

        FTensor Gain({ 4 });
        Gain.Fill(Real(1));

        FTensor Normed = RmsNorm(X, Gain, (Real)0);

        // RMS = sqrt((9 + 16 + 0 + 0) / 4) = sqrt(6.25) = 2.5
        printf("  x = (3, 4, 0, 0)\n");
        printf("  RMS = sqrt((9+16)/4) = %.4f\n", std::sqrt(25.0 / 4.0));
        printf("  결과 = (%.4f, %.4f, %.4f, %.4f)\n",
               (double)Normed(0, 0, 0), (double)Normed(0, 0, 1),
               (double)Normed(0, 0, 2), (double)Normed(0, 0, 3));
        printf("  손계산 (1.2, 1.6, 0, 0)\n\n");

        CHECK_NEAR(Normed(0, 0, 0), 1.2, 1e-5);
        CHECK_NEAR(Normed(0, 0, 1), 1.6, 1e-5);
        CHECK_NEAR(Normed(0, 0, 2), 0.0, 1e-6);

        // 정규화한 것의 RMS 는 1 이어야 한다.
        printf("  정규화한 것의 RMS = %.9f\n", MeanRms(Normed));
        CHECK_NEAR(MeanRms(Normed), 1.0, 1e-5);

        // 입력을 100배로 키워도 결과가 같아야 한다.
        FTensor Bigger = X * Real(100);
        FTensor BiggerNormed = RmsNorm(Bigger, Gain, (Real)0);

        printf("  입력을 100배로 키우면 = (%.4f, %.4f, ...)\n",
               (double)BiggerNormed(0, 0, 0), (double)BiggerNormed(0, 0, 1));
        printf("  **크기에 무관하다.** 이게 정규화가 하는 일이다.\n\n");

        CHECK_NEAR(BiggerNormed(0, 0, 0), 1.2, 1e-4);

        // 이득(Gain)이 곱해지는지도 확인한다.
        FTensor Doubled({ 4 });
        Doubled.Fill(Real(2));
        FTensor Scaled = RmsNorm(X, Doubled, (Real)0);
        CHECK_NEAR(Scaled(0, 0, 0), 2.4, 1e-5);

        // 전부 0 인 줄에서도 안 죽어야 한다. Epsilon 이 하는 일.
        FTensor Zero({ 1, 1, 4 });
        FTensor ZeroNormed = RmsNorm(Zero, Gain, (Real)1e-6);
        int bFinite = 1;
        for (size_t i = 0; i < ZeroNormed.Count(); i++)
        {
            if (!std::isfinite((double)ZeroNormed.At(i)))
            {
                bFinite = 0;
            }
        }
        printf("  전부 0 인 줄도 무사한가 = %s (Epsilon 덕분)\n\n",
               bFinite ? "예" : "아니오");
        CHECK(bFinite == 1);
    }

    // ---- D3-2. SiLU ----
    printf("[D3-2] SiLU — ReLU 의 부드러운 사촌\n\n");
    printf("  ");
    PrintPaddedRight("x", 10);
    PrintPaddedRight("relu", 12);
    PrintPaddedRight("silu", 12);
    printf("\n");

    {
        const Real Xs[7] = { Real(-4), Real(-2), Real(-1), Real(0),
                             Real(1), Real(2), Real(4) };

        for (int i = 0; i < 7; i++)
        {
            double Relu = ((double)Xs[i] > 0.0) ? (double)Xs[i] : 0.0;
            printf("  %9.2f %11.6f %11.6f\n", (double)Xs[i], Relu,
                   (double)Silu(Xs[i]));
        }

        printf("\n  음수 쪽이 완전히 0 이 아니다. 그래디언트가 조금 흐른다.\n\n");

        // 손계산: silu(0) = 0, silu(1) = 1/(1+e^-1) = 0.731059
        CHECK_NEAR(Silu(Real(0)), 0.0, 1e-9);
        CHECK_NEAR(Silu(Real(1)), 1.0 / (1.0 + std::exp(-1.0)), 1e-6);

        // 아주 큰 값에서도 안 넘쳐야 한다.
        CHECK(std::isfinite((double)Silu(Real(1000))));
        CHECK(std::isfinite((double)Silu(Real(-1000))));
        CHECK_NEAR(Silu(Real(-1000)), 0.0, 1e-6);
    }

    // ---- D3-3. 잔차 연결이 없으면 ----
    printf("[D3-3] 층을 쌓으면 신호가 어떻게 되는가\n\n");
    printf("  (입력의 RMS 는 약 %.3f)\n\n", 1.0 / std::sqrt(3.0));

    printf("  ");
    PrintPaddedRight("깊이", 8);
    PrintPaddedRight("맨몸", 16);
    PrintPaddedRight("잔차만", 16);
    PrintPaddedRight("Post-Norm", 16);
    PrintPaddedRight("Pre-Norm", 16);
    printf("\n");

    {
        const int Depths[5] = { 1, 2, 4, 8, 16 };

        double BareLast = 0.0;
        double PreLast = 0.0;

        for (int d = 0; d < 5; d++)
        {
            FBlockConfig Bare;
            Bare.Model = MODEL; Bare.Heads = HEADS; Bare.Hidden = HIDDEN;
            Bare.bResidual = false; Bare.bNorm = false;

            FBlockConfig ResidualOnly = Bare;
            ResidualOnly.bResidual = true;

            FBlockConfig PostNorm = ResidualOnly;
            PostNorm.bNorm = true; PostNorm.bPreNorm = false;

            FBlockConfig PreNorm = ResidualOnly;
            PreNorm.bNorm = true; PreNorm.bPreNorm = true;

            double A = StackAndMeasure(Bare, Depths[d], BOOK_SEED);
            double B = StackAndMeasure(ResidualOnly, Depths[d], BOOK_SEED);
            double C = StackAndMeasure(PostNorm, Depths[d], BOOK_SEED);
            double E = StackAndMeasure(PreNorm, Depths[d], BOOK_SEED);

            if (Depths[d] == 16)
            {
                BareLast = A;
                PreLast = E;
            }

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%d", Depths[d]);
            PrintPaddedRight(Buffer, 8);
            snprintf(Buffer, sizeof(Buffer), "%.3e", A);
            PrintPaddedRight(Buffer, 16);
            snprintf(Buffer, sizeof(Buffer), "%.3e", B);
            PrintPaddedRight(Buffer, 16);
            snprintf(Buffer, sizeof(Buffer), "%.3e", C);
            PrintPaddedRight(Buffer, 16);
            snprintf(Buffer, sizeof(Buffer), "%.3e", E);
            PrintPaddedRight(Buffer, 16);
            printf("\n");
        }

        printf("\n  **맨몸은 신호가 사라진다.** 16층에서 %.3e 다.\n", BareLast);
        printf("  잔차와 정규화를 걸면 %.3e 로 유지된다.\n\n", PreLast);

        CHECK(BareLast < PreLast);
        CHECK(BareLast < 1e-2);
    }

    // ---- D3-4. Pre-Norm 과 Post-Norm ----
    printf("[D3-4] 정규화를 앞에 두느냐 뒤에 두느냐\n\n");

    {
        printf("  Post-Norm   x -> Norm(x + Attn(x))\n");
        printf("  Pre-Norm    x -> x + Attn(Norm(x))\n\n");

        printf("  차이는 **잔차 경로에 정규화가 끼어 있느냐**다.\n");
        printf("  Pre-Norm 은 입력이 출력까지 아무 변형 없이 이어진다.\n");
        printf("  그 길을 따라 그래디언트도 손실 없이 내려온다.\n\n");

        // 잔차 경로가 살아 있는지 확인한다.
        // 가중치를 전부 0 으로 만들면 Pre-Norm 은 입력을 그대로 내보내야 한다.
        FBlockConfig Config;
        Config.Model = MODEL; Config.Heads = HEADS; Config.Hidden = HIDDEN;
        Config.bNorm = true; Config.bPreNorm = true; Config.bResidual = true;

        FBlock Block(Config);
        Block.Query.Weight.Fill(Real(0));
        Block.Key.Weight.Fill(Real(0));
        Block.Value.Weight.Fill(Real(0));
        Block.Project.Weight.Fill(Real(0));
        Block.Up.Weight.Fill(Real(0));
        Block.Down.Weight.Fill(Real(0));

        FTensor X({ BATCH, LENGTH, MODEL });
        FillRandom(X, Rng, 1.0);

        FTensor Y = Block.Forward(X);

        double WorstGap = 0.0;
        for (size_t i = 0; i < X.Count(); i++)
        {
            double Gap = std::fabs((double)Y.At(i) - (double)X.At(i));
            if (Gap > WorstGap) WorstGap = Gap;
        }

        printf("  가중치를 전부 0 으로 두면 Pre-Norm 블록은 입력을\n");
        printf("  그대로 내보내야 한다. 최대 차이 = %.3e\n\n", WorstGap);

        CHECK(WorstGap < 1e-6);

        printf("  **처음에 아무것도 안 하는 것이 좋은 출발점**이다.\n");
        printf("  깊은 모델일수록 '일단 그대로 흘려보내는' 상태에서\n");
        printf("  시작해야 훈련이 무너지지 않는다.\n\n");
    }

    // ---- D3-5. QK-Norm ----
    printf("[D3-5] QK-Norm — sqrt(d) 로도 모자랄 때\n\n");

    {
        printf("  ");
        PrintPaddedRight("Q,K 가중치 크기", 20);
        PrintPaddedRight("QK-Norm 없이", 18);
        PrintPaddedRight("QK-Norm 켜고", 18);
        printf("\n");

        const double Scales[4] = { 1.0, 4.0, 16.0, 64.0 };

        double WithoutFirst = 0.0;
        double WithoutLast = 0.0;
        double WithFirst = 0.0;
        double WithLast = 0.0;

        for (int s = 0; s < 4; s++)
        {
            for (int Mode = 0; Mode < 2; Mode++)
            {
                FBlockConfig Config;
                Config.Model = MODEL; Config.Heads = HEADS;
                Config.Hidden = HIDDEN;
                Config.bQkNorm = (Mode == 1);

                FRandom Local;
                RandomSeed(&Local, BOOK_SEED);

                FBlock Block(Config);
                Block.Init(Local);

                // Q, K 가중치만 키운다. 훈련 중에 실제로 일어나는 일이다.
                for (size_t i = 0; i < Block.Query.Weight.Count(); i++)
                {
                    Block.Query.Weight.At(i) *= (Real)Scales[s];
                    Block.Key.Weight.At(i) *= (Real)Scales[s];
                }

                FTensor X({ 1, LENGTH, MODEL });
                FRandom Fill;
                RandomSeed(&Fill, BOOK_SEED + 7);
                FillRandom(X, Fill, 1.0);

                // 어텐션 가중치의 엔트로피를 본다.
                FTensor Normed = RmsNorm(X, Block.AttentionGain, (Real)1e-6);
                FTensor Q = SplitHeads(Block.Query.Forward(Normed), HEADS);
                FTensor K = SplitHeads(Block.Key.Forward(Normed), HEADS);
                FTensor V = SplitHeads(Block.Value.Forward(Normed), HEADS);

                if (Config.bQkNorm)
                {
                    Q = RmsNorm(Q, Block.QueryGain, (Real)1e-6);
                    K = RmsNorm(K, Block.KeyGain, (Real)1e-6);
                }

                FAttention Attended = Attend(Q, K, V, true);

                // 마지막 위치의 분포가 얼마나 퍼져 있는가.
                double Entropy = 0.0;
                for (size_t h = 0; h < HEADS; h++)
                {
                    double E = 0.0;
                    for (size_t k = 0; k < LENGTH; k++)
                    {
                        double P = (double)Attended.Weights(0, h, LENGTH - 1, k);
                        if (P > 1e-12) E -= P * std::log(P);
                    }
                    Entropy += E / std::log((double)LENGTH);
                }
                Entropy /= (double)HEADS;

                if (Mode == 0)
                {
                    if (s == 0) WithoutFirst = Entropy;
                    WithoutLast = Entropy;
                }
                else
                {
                    if (s == 0) WithFirst = Entropy;
                    WithLast = Entropy;

                    char Buffer[32];
                    printf("  ");
                    snprintf(Buffer, sizeof(Buffer), "x%.0f", Scales[s]);
                    PrintPaddedRight(Buffer, 20);
                    snprintf(Buffer, sizeof(Buffer), "%.4f", WithoutLast);
                    PrintPaddedRight(Buffer, 18);
                    snprintf(Buffer, sizeof(Buffer), "%.4f", Entropy);
                    PrintPaddedRight(Buffer, 18);
                    printf("\n");
                }
            }
        }

        printf("\n  가중치가 커지면 QK-Norm 없이는 엔트로피가\n");
        printf("  %.4f 에서 %.4f 로 무너진다. QK-Norm 을 켜면 %.4f -> %.4f.\n\n",
               WithoutFirst, WithoutLast, WithFirst, WithLast);

        CHECK(WithoutLast < WithoutFirst);
        CHECK(WithLast > WithoutLast);
    }

    // ---- D3-6. 블록 하나 ----
    printf("[D3-6] 블록 하나를 조립한다\n\n");

    {
        FBlockConfig Config;
        Config.Model = MODEL; Config.Heads = HEADS; Config.Hidden = HIDDEN;

        FBlock Block(Config);
        Block.Init(Rng);

        FTensor X({ BATCH, LENGTH, MODEL });
        FillRandom(X, Rng, 1.0);

        FTensor Y = Block.Forward(X);

        size_t Parameters = Block.Query.Weight.Count()
                          + Block.Key.Weight.Count()
                          + Block.Value.Weight.Count()
                          + Block.Project.Weight.Count()
                          + Block.Up.Weight.Count()
                          + Block.Down.Weight.Count()
                          + Block.AttentionGain.Count()
                          + Block.FeedGain.Count();

        printf("  모델차원 %zu, 헤드 %zu, 앞먹임 %zu\n", Config.Model,
               Config.Heads, Config.Hidden);
        printf("  가중치 %zu개\n", Parameters);
        printf("  입력 (%zu, %zu, %zu) -> 출력 (%zu, %zu, %zu)\n\n",
               X.Size(0), X.Size(1), X.Size(2),
               Y.Size(0), Y.Size(1), Y.Size(2));

        CHECK(Y.Rank() == 3);
        CHECK(Y.Size(0) == BATCH);
        CHECK(Y.Size(1) == LENGTH);
        CHECK(Y.Size(2) == MODEL);

        // 모든 값이 유한해야 한다.
        int bFinite = 1;
        for (size_t i = 0; i < Y.Count(); i++)
        {
            if (!std::isfinite((double)Y.At(i)))
            {
                bFinite = 0;
            }
        }
        CHECK(bFinite == 1);

        // **인과성 검사.** 뒤쪽 입력을 바꿔도 앞쪽 출력은 안 바뀌어야 한다.
        FTensor Changed = X;
        for (size_t d = 0; d < MODEL; d++)
        {
            Changed(0, LENGTH - 1, d) = (Real)RandomRange(&Rng, 5.0);
        }

        FTensor Y2 = Block.Forward(Changed);

        double EarlyGap = 0.0;
        double LateGap = 0.0;

        for (size_t t = 0; t < LENGTH; t++)
        for (size_t d = 0; d < MODEL; d++)
        {
            double Gap = std::fabs((double)Y(0, t, d) - (double)Y2(0, t, d));
            if (t < LENGTH - 1)
            {
                if (Gap > EarlyGap) EarlyGap = Gap;
            }
            else
            {
                if (Gap > LateGap) LateGap = Gap;
            }
        }

        printf("  마지막 위치의 입력만 바꿨을 때\n");
        printf("    앞쪽 출력의 변화 = %.3e (0 이어야 한다)\n", EarlyGap);
        printf("    마지막 출력의 변화 = %.3e (커야 한다)\n\n", LateGap);

        CHECK(EarlyGap < 1e-6);
        CHECK(LateGap > 1e-3);

        printf("  **인과 마스크가 블록 전체에서 지켜진다.**\n");
        printf("  이 검사가 없으면 뒤를 훔쳐보는 버그를 못 잡는다.\n\n");
    }

    return ReportResult();
}
