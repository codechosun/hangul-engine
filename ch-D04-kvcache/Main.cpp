// ch-D04-kvcache/Main.cpp
//
// D4. 추론 완성 — KV 캐시
//
// 저장소 루트에서 실행할 것.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Model.hpp"
#include "Nn.hpp"
#include "Random.h"
#include "Tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECTED_PATH "data/transformer_expected.csv"
#define BOOK_SEED 20260914ull

// 정답표와 같은 설정.
#define TINY_VOCAB  11
#define TINY_MODEL   8
#define TINY_HEADS   2
#define TINY_HIDDEN 16
#define TINY_LAYERS  2
#define TINY_MAX    12

// 속도를 재볼 크기.
#define BIG_VOCAB  512
#define BIG_MODEL  128
#define BIG_HEADS    4
#define BIG_HIDDEN 512
#define BIG_LAYERS   4
#define BIG_MAX    128

namespace
{

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

Real FormulaValue(size_t Index, int Multiplier, int Offset, int Modulus,
                  int Half, double Divisor)
{
    long long Raw = (long long)Index * Multiplier + Offset;
    long long Mod = Raw % Modulus;
    return (Real)(((double)(Mod - Half)) / Divisor);
}

void FillFormula(FTensor& T, int Multiplier, int Offset, int Modulus,
                 int Half, double Divisor)
{
    for (size_t i = 0; i < T.Count(); i++)
    {
        T.At(i) = FormulaValue(i, Multiplier, Offset, Modulus, Half, Divisor);
    }
}

} // namespace

int main(void)
{
    printf("D4. 추론 완성 — KV 캐시\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ---- D4-1. 파이썬 참조 구현과 대조 ----
    printf("[D4-1] NumPy 참조 구현과 점수 맞추기\n\n");

    FModelConfig TinyConfig;
    TinyConfig.Vocab = TINY_VOCAB;
    TinyConfig.Model = TINY_MODEL;
    TinyConfig.Heads = TINY_HEADS;
    TinyConfig.Hidden = TINY_HIDDEN;
    TinyConfig.Layers = TINY_LAYERS;
    TinyConfig.MaxLength = TINY_MAX;

    FTransformer Tiny(TinyConfig);

    {
        FillFormula(Tiny.TokenEmbedding, 31, 17, 41, 20, 40.0);
        FillFormula(Tiny.PositionEmbedding, 23, 11, 37, 18, 36.0);
        FillFormula(Tiny.Head.Weight, 19, 7, 43, 21, 42.0);

        for (size_t L = 0; L < TINY_LAYERS; L++)
        {
            const int Base = (int)(L + 1) * 7;
            FBlock& B = Tiny.Blocks[L];

            FillFormula(B.Query.Weight,  13 + Base,  5, 29, 14, 28.0);
            FillFormula(B.Key.Weight,    17 + Base,  3, 31, 15, 30.0);
            FillFormula(B.Value.Weight,  11 + Base,  9, 37, 18, 36.0);
            FillFormula(B.Project.Weight, 23 + Base, 1, 41, 20, 40.0);
            FillFormula(B.Up.Weight,     29 + Base, 13, 43, 21, 42.0);
            FillFormula(B.Down.Weight,   31 + Base,  7, 47, 23, 46.0);
        }

        const uint32_t Tokens[6] = { 3, 7, 1, 9, 0, 4 };
        FTensor Logits = Tiny.Forward(Tokens, 1, 6);

        printf("  어휘 %d, 모델차원 %d, 헤드 %d, 층 %d, 토큰 6개\n",
               TINY_VOCAB, TINY_MODEL, TINY_HEADS, TINY_LAYERS);
        printf("  가중치 %zu개\n\n", Tiny.ParameterCount());

        CHECK(Logits.Rank() == 3);
        CHECK(Logits.Size(1) == 6);
        CHECK(Logits.Size(2) == TINY_VOCAB);

        FILE* File = fopen(EXPECTED_PATH, "rb");
        if (!CHECK(File != NULL))
        {
            printf("  %s 가 없다. tools/make_transformer_expected.py 를 돌릴 것.\n",
                   EXPECTED_PATH);
            return ReportResult();
        }

        char Line[256];
        (void)fgets(Line, sizeof(Line), File);

        int Rows = 0;
        int Bad = 0;
        double WorstGap = 0.0;

        const double Tolerance = (sizeof(Real) == 8) ? 1e-11 : 1e-4;

        while (fgets(Line, sizeof(Line), File) != NULL)
        {
            int Position = 0;
            int Token = 0;
            double Value = 0.0;

            if (sscanf(Line, "%d,%d,%lf", &Position, &Token, &Value) != 3)
            {
                continue;
            }

            double Mine = (double)Logits(0, (size_t)Position, (size_t)Token);
            double Gap = std::fabs(Mine - Value);

            if (Gap > WorstGap) WorstGap = Gap;
            if (Gap > Tolerance) Bad++;

            Rows++;
        }

        fclose(File);

        printf("  정답표 %d줄 대조 -> 어긋난 줄 %d개 (허용 %.0e)\n", Rows, Bad,
               Tolerance);
        printf("  최대 차이 = %.3e\n\n", WorstGap);

        CHECK(Rows == 66);
        CHECK(Bad == 0);

        printf("  마지막 위치의 점수\n   ");
        for (size_t v = 0; v < TINY_VOCAB; v++)
        {
            printf("%+8.4f", (double)Logits(0, 5, v));
        }
        printf("\n\n");
    }

    // ---- D4-2. 캐시가 같은 답을 내는가 ----
    printf("[D4-2] 한 번에 vs 한 글자씩\n\n");

    {
        const uint32_t Tokens[6] = { 3, 7, 1, 9, 0, 4 };

        FTensor Whole = Tiny.Forward(Tokens, 1, 6);

        FKvCache Cache(TinyConfig);
        double WorstGap = 0.0;

        for (size_t t = 0; t < 6; t++)
        {
            FTensor Step = Tiny.Step(Tokens[t], Cache);

            for (size_t v = 0; v < TINY_VOCAB; v++)
            {
                double Gap = std::fabs((double)Step.At(v)
                                     - (double)Whole(0, t, v));
                if (Gap > WorstGap) WorstGap = Gap;
            }
        }

        printf("  통째로 넣은 것과 하나씩 넣은 것의 최대 차이 = %.3e\n",
               WorstGap);
        printf("  캐시에 담긴 토큰 수 = %zu\n\n", Cache.Length);

        const double Tolerance = (sizeof(Real) == 8) ? 1e-12 : 1e-4;
        CHECK(WorstGap < Tolerance);
        CHECK(Cache.Length == 6);

        printf("  **같은 답이 나온다.** 이 검사가 없으면 캐시 버그를 못 잡는다.\n");
        printf("  캐시가 한 칸 밀리거나 위치 임베딩을 잘못 더해도\n");
        printf("  생성은 그럴듯하게 돌아간다.\n\n");
    }

    // ---- D4-3. 속도 ----
    printf("[D4-3] 캐시가 있고 없고\n\n");

    FModelConfig BigConfig;
    BigConfig.Vocab = BIG_VOCAB;
    BigConfig.Model = BIG_MODEL;
    BigConfig.Heads = BIG_HEADS;
    BigConfig.Hidden = BIG_HIDDEN;
    BigConfig.Layers = BIG_LAYERS;
    BigConfig.MaxLength = BIG_MAX;

    FTransformer Big(BigConfig);

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);
        Big.Init(Rng);

        printf("  어휘 %d, 모델차원 %d, 헤드 %d, 층 %d\n",
               BIG_VOCAB, BIG_MODEL, BIG_HEADS, BIG_LAYERS);
        printf("  가중치 %zu개\n\n", Big.ParameterCount());

        printf("  ");
        PrintPaddedRight("생성 길이", 12);
        PrintPaddedRight("캐시 없이", 14);
        PrintPaddedRight("캐시로", 14);
        PrintPaddedRight("배속", 10);
        printf("\n");

        const size_t Lengths[4] = { 16, 32, 64, 128 };

        double FirstRatio = 0.0;
        double LastRatio = 0.0;

        for (int s = 0; s < 4; s++)
        {
            const size_t Length = Lengths[s];

            std::vector<uint32_t> Tokens(Length, 0);
            for (size_t i = 0; i < Length; i++)
            {
                Tokens[i] = (uint32_t)(i % BIG_VOCAB);
            }

            // (가) 캐시 없이. 매번 처음부터 다시 계산한다.
            auto Begin = std::chrono::steady_clock::now();
            for (size_t t = 1; t <= Length; t++)
            {
                FTensor Logits = Big.Forward(Tokens.data(), 1, t);
                if (Logits.Count() == 0) printf("never\n");
            }
            double WithoutSeconds = Seconds(Begin);

            // (나) 캐시로. 한 칸씩만 계산한다.
            Begin = std::chrono::steady_clock::now();
            {
                FKvCache Cache(BigConfig);
                for (size_t t = 0; t < Length; t++)
                {
                    FTensor Logits = Big.Step(Tokens[t], Cache);
                    if (Logits.Count() == 0) printf("never\n");
                }
            }
            double WithSeconds = Seconds(Begin);

            double Ratio = WithoutSeconds / WithSeconds;
            if (s == 0) FirstRatio = Ratio;
            LastRatio = Ratio;

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%zu", Length);
            PrintPaddedRight(Buffer, 12);
            snprintf(Buffer, sizeof(Buffer), "%.3f 초", WithoutSeconds);
            PrintPaddedRight(Buffer, 14);
            snprintf(Buffer, sizeof(Buffer), "%.3f 초", WithSeconds);
            PrintPaddedRight(Buffer, 14);
            snprintf(Buffer, sizeof(Buffer), "%.1f 배", Ratio);
            PrintPaddedRight(Buffer, 10);
            printf("\n");
        }

        printf("\n  길이가 길어질수록 격차가 벌어진다 (%.1f 배 -> %.1f 배).\n",
               FirstRatio, LastRatio);
        printf("  캐시 없이는 T 글자에 T^3, 캐시로는 T^2 이 든다.\n\n");

        CHECK(LastRatio > FirstRatio);
        CHECK(LastRatio > 2.0);
    }

    // ---- D4-4. 캐시가 먹는 메모리 ----
    printf("[D4-4] 캐시의 값\n\n");

    {
        FKvCache Cache(BigConfig);

        printf("  층 %d, 헤드 %d, 최대길이 %d, 헤드차원 %d\n",
               BIG_LAYERS, BIG_HEADS, BIG_MAX, BIG_MODEL / BIG_HEADS);
        printf("  캐시 크기 = %.0f KB (가중치는 %.0f KB)\n\n",
               (double)Cache.Bytes() / 1024.0,
               (double)Big.ParameterCount() * sizeof(Real) / 1024.0);

        CHECK(Cache.Bytes() > 0);

        // 실제 크기로 환산해본다.
        const double Layers = 32;
        const double Heads = 32;
        const double HeadDim = 128;
        const double Context = 8192;
        const double Batch = 32;

        double Bytes = 2.0 * Layers * Heads * HeadDim * Context * Batch * 2.0;

        printf("  같은 계산을 요즘 크기로 해보면\n");
        printf("    층 %.0f, 헤드 %.0f, 헤드차원 %.0f, 문맥 %.0f, 동시 요청 %.0f\n",
               Layers, Heads, HeadDim, Context, Batch);
        printf("    캐시 = %.1f GB (fp16 기준)\n\n",
               Bytes / (1024.0 * 1024.0 * 1024.0));

        printf("  **추론 서버 메모리의 대부분이 이것**이다. 가중치가 아니라.\n");
        printf("  동시 요청 수를 못 늘리는 이유이기도 하다.\n\n");
    }

    // ---- D4-5. 글자를 뽑는다 ----
    printf("[D4-5] 아직 아무것도 안 배운 모델이 뽑는 글자\n\n");

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        FKvCache Cache(BigConfig);

        uint32_t Token = 0;
        double LossSum = 0.0;
        int Count = 0;

        printf("  뽑은 토큰 번호: ");

        for (int i = 0; i < 20; i++)
        {
            FTensor Logits = Big.Step(Token, Cache);

            // 점수를 확률로 바꿔 뽑는다. A3 의 샘플러와 같은 일이다.
            FVector Scores((size_t)BIG_VOCAB);
            for (size_t v = 0; v < BIG_VOCAB; v++)
            {
                Scores[v] = Logits.At(v);
            }

            FVector Probabilities = Softmax(Scores);

            double Target = RandomUnit(&Rng);
            double Running = 0.0;
            uint32_t Picked = 0;

            for (size_t v = 0; v < BIG_VOCAB; v++)
            {
                Running += (double)Probabilities[v];
                if (Running > Target)
                {
                    Picked = (uint32_t)v;
                    break;
                }
            }

            LossSum += (double)CrossEntropy(Probabilities, (size_t)Picked);
            Count++;

            printf("%u ", Picked);
            Token = Picked;
        }

        printf("\n\n  평균 손실 = %.4f, 어휘의 로그 = %.4f\n",
               LossSum / (double)Count, std::log((double)BIG_VOCAB));
        printf("  퍼플렉서티 = %.1f\n\n", std::exp(LossSum / (double)Count));

        // 훈련을 안 했으므로 거의 균등해야 한다. C1 에서 세운 검사.
        double Ppl = std::exp(LossSum / (double)Count);
        CHECK(Ppl > (double)BIG_VOCAB * 0.5);
        CHECK(Ppl < (double)BIG_VOCAB * 2.0);

        printf("  **추론 경로가 완결됐다.** 이제 훈련만 남았다.\n\n");
    }

    return ReportResult();
}
