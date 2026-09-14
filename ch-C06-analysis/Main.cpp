// ch-C06-analysis/Main.cpp
//
// C6. 결과 분석
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]
//
// 만드는 것
//     data/embedding.npy   훈련한 임베딩 (어휘 x 16)
//     data/vocab.csv       번호 -> 글자
//     data/loss_curve.csv  에폭별 손실
//     data/pca_cpp.csv     C++ 이 계산한 2차원 좌표

#include "Test.h"
#include "Pretty.h"

#include "Ngram.h"
#include "Npy.hpp"
#include "Nplm.hpp"
#include "Pca.hpp"
#include "Scan.h"
#include "Utf8.h"
#include "Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define DEFAULT_CORPUS "data/corpus.txt"

#define EMBEDDING_NPY "data/embedding.npy"
#define VOCAB_CSV     "data/vocab.csv"
#define LOSS_CSV      "data/loss_curve.csv"
#define PCA_CSV       "data/pca_cpp.csv"

#define BOOK_SEED 20260914ull

// C3 와 같은 설정. 에폭만 줄였다.
#define VOCAB_SIZE 512
#define TOKEN_UNK  0u
#define TOKEN_END  1u
#define TOKEN_PAD  2u
#define FIRST_CHAR 3u

#define CONTEXT_SIZE 3
#define EMBED_DIM    16
#define HIDDEN_DIM   64

#define TRAIN_LINES  20000
#define VALID_LINES   2000
#define MAX_TRAIN_EXAMPLES 150000
#define BATCH_SIZE   64
#define EPOCH_COUNT  6
#define LEARN_RATE   Real(0.5)

namespace
{

struct FExample
{
    uint32_t Context[CONTEXT_SIZE];
    uint32_t Target;
};

std::vector<uint32_t> ReadTokens(const char* Path, uint64_t SkipLines,
                                 uint64_t TakeLines,
                                 const std::vector<int>& MapOf)
{
    std::vector<uint32_t> Tokens;

    FScanner Scanner;
    if (!ScanOpen(&Scanner, Path))
    {
        return Tokens;
    }

    uint64_t Seen = 0;
    uint32_t Code = 0;

    while (ScanNext(&Scanner, &Code))
    {
        if (Code == (uint32_t)'\n')
        {
            if (Seen >= SkipLines)
            {
                Tokens.push_back(TOKEN_END);
            }
            Seen++;
            if (Seen >= SkipLines + TakeLines)
            {
                break;
            }
            continue;
        }

        if (Seen < SkipLines)
        {
            continue;
        }

        int Mapped = (Code < VOCAB_LIMIT) ? MapOf[(size_t)Code] : -1;
        Tokens.push_back((Mapped >= 0) ? (uint32_t)Mapped : TOKEN_UNK);
    }

    ScanClose(&Scanner);
    return Tokens;
}

std::vector<FExample> MakeExamples(const std::vector<uint32_t>& Tokens,
                                   size_t Limit)
{
    std::vector<FExample> Examples;

    uint32_t Window[CONTEXT_SIZE];
    for (int i = 0; i < CONTEXT_SIZE; i++)
    {
        Window[i] = TOKEN_PAD;
    }

    for (size_t i = 0; i < Tokens.size() && Examples.size() < Limit; i++)
    {
        FExample Example;
        for (int k = 0; k < CONTEXT_SIZE; k++)
        {
            Example.Context[k] = Window[k];
        }
        Example.Target = Tokens[i];
        Examples.push_back(Example);

        if (Tokens[i] == TOKEN_END)
        {
            for (int k = 0; k < CONTEXT_SIZE; k++)
            {
                Window[k] = TOKEN_PAD;
            }
        }
        else
        {
            for (int k = 0; k < CONTEXT_SIZE - 1; k++)
            {
                Window[k] = Window[k + 1];
            }
            Window[CONTEXT_SIZE - 1] = Tokens[i];
        }
    }

    return Examples;
}

double MeanLoss(const FNplm& Model, const std::vector<FExample>& Examples,
                size_t Limit)
{
    double Sum = 0.0;
    size_t Count = 0;

    for (size_t i = 0; i < Examples.size() && Count < Limit; i++)
    {
        Sum += (double)Model.Forward(Examples[i].Context, Examples[i].Target).Loss;
        Count++;
    }

    return (Count > 0) ? (Sum / (double)Count) : 0.0;
}

} // namespace

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("C6. 결과 분석\n\n");

    // ---- 어휘 ----
    std::vector<int> MapOf((size_t)VOCAB_LIMIT, -1);
    std::vector<uint32_t> CodeOf((size_t)VOCAB_SIZE, 0);
    std::vector<uint64_t> CountOf((size_t)VOCAB_SIZE, 0);

    {
        FNgram Unigram;
        if (!CHECK(NgramBuild(&Unigram, CorpusPath, 1, TRAIN_LINES)))
        {
            printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
            return ReportResult();
        }

        std::vector<std::pair<uint64_t, uint32_t>> ByCount;
        for (uint64_t i = 0; i < Unigram.GramCount; i++)
        {
            uint64_t Before = (i == 0) ? 0 : Unigram.Cumulative[i - 1];
            uint32_t Code = Unigram.Grams[i];
            if (Code >= VOCAB_LIMIT || Code == TOKEN_EOS || Code == TOKEN_BOS)
            {
                continue;
            }
            ByCount.emplace_back(Unigram.Cumulative[i] - Before, Code);
        }

        std::sort(ByCount.begin(), ByCount.end(),
                  [](const std::pair<uint64_t, uint32_t>& L,
                     const std::pair<uint64_t, uint32_t>& R)
                  {
                      if (L.first != R.first) return L.first > R.first;
                      return L.second < R.second;
                  });

        uint32_t Next = FIRST_CHAR;
        for (const auto& Row : ByCount)
        {
            if (Next >= (uint32_t)VOCAB_SIZE) break;
            MapOf[(size_t)Row.second] = (int)Next;
            CodeOf[(size_t)Next] = Row.second;
            CountOf[(size_t)Next] = Row.first;
            Next++;
        }

        NgramFree(&Unigram);
    }

    std::vector<uint32_t> TrainTokens =
        ReadTokens(CorpusPath, 0, TRAIN_LINES, MapOf);
    std::vector<uint32_t> ValidTokens =
        ReadTokens(CorpusPath, TRAIN_LINES, VALID_LINES, MapOf);

    std::vector<FExample> TrainSet = MakeExamples(TrainTokens, MAX_TRAIN_EXAMPLES);
    std::vector<FExample> ValidSet = MakeExamples(ValidTokens, 20000);

    CHECK(TrainSet.size() > 1000);

    // ---- C6-1. 훈련하며 손실 곡선을 남긴다 ----
    printf("[C6-1] 훈련 (에폭 %d)\n\n", EPOCH_COUNT);

    FNplm Model(VOCAB_SIZE, CONTEXT_SIZE, EMBED_DIM, HIDDEN_DIM);

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);
        Model.Init(Rng);

        FNplmGrad Grad(VOCAB_SIZE, CONTEXT_SIZE, EMBED_DIM, HIDDEN_DIM);

        std::vector<size_t> Order(TrainSet.size());
        for (size_t i = 0; i < Order.size(); i++)
        {
            Order[i] = i;
        }

        FRandom Shuffle;
        RandomSeed(&Shuffle, BOOK_SEED + 1);

        FILE* Curve = fopen(LOSS_CSV, "wb");
        if (!CHECK(Curve != NULL))
        {
            return ReportResult();
        }
        fprintf(Curve, "Epoch,TrainLoss,ValidLoss\n");

        double TrainLoss = MeanLoss(Model, TrainSet, 5000);
        double ValidLoss = MeanLoss(Model, ValidSet, 5000);
        fprintf(Curve, "0,%.9f,%.9f\n", TrainLoss, ValidLoss);

        printf("  ");
        PrintPaddedRight("에폭", 8);
        PrintPaddedRight("훈련 손실", 14);
        PrintPaddedRight("검증 손실", 14);
        printf("\n");
        printf("  %7d %13.4f %13.4f\n", 0, TrainLoss, ValidLoss);

        for (int Epoch = 1; Epoch <= EPOCH_COUNT; Epoch++)
        {
            for (size_t i = Order.size() - 1; i > 0; i--)
            {
                uint64_t j = RandomBelow(&Shuffle, (uint64_t)i + 1);
                std::swap(Order[i], Order[(size_t)j]);
            }

            for (size_t Start = 0; Start < Order.size(); Start += BATCH_SIZE)
            {
                Grad.Zero();

                size_t Stop = Start + BATCH_SIZE;
                if (Stop > Order.size()) Stop = Order.size();

                for (size_t i = Start; i < Stop; i++)
                {
                    const FExample& Example = TrainSet[Order[i]];
                    FNplmTrace Trace =
                        Model.Forward(Example.Context, Example.Target);
                    Model.Backward(Example.Context, Example.Target, Trace, Grad);
                }

                Model.Step(Grad, LEARN_RATE, Real(1) / (Real)(Stop - Start));
            }

            TrainLoss = MeanLoss(Model, TrainSet, 5000);
            ValidLoss = MeanLoss(Model, ValidSet, 5000);

            fprintf(Curve, "%d,%.9f,%.9f\n", Epoch, TrainLoss, ValidLoss);
            printf("  %7d %13.4f %13.4f\n", Epoch, TrainLoss, ValidLoss);
        }

        fclose(Curve);

        printf("\n  %s 에 적었다\n\n", LOSS_CSV);
        CHECK(ValidLoss < 4.0);
    }

    // ---- C6-2. 임베딩을 내보낸다 ----
    printf("[C6-2] 임베딩을 .npy 로 내보낸다\n\n");

    if (!CHECK(NpySaveMatrix(EMBEDDING_NPY, Model.Embedding)))
    {
        return ReportResult();
    }

    {
        FILE* File = fopen(VOCAB_CSV, "wb");
        if (!CHECK(File != NULL))
        {
            return ReportResult();
        }

        fprintf(File, "Id,Codepoint,Count,Kind\n");
        for (uint32_t t = 0; t < (uint32_t)VOCAB_SIZE; t++)
        {
            const char* Kind = "char";
            uint32_t Code = CodeOf[(size_t)t];

            if (t == TOKEN_UNK)      { Kind = "unk";  Code = 0; }
            else if (t == TOKEN_END) { Kind = "eos";  Code = 0; }
            else if (t == TOKEN_PAD) { Kind = "pad";  Code = 0; }
            else if (Code == ' ')    { Kind = "space"; }
            else if (Code >= '0' && Code <= '9') { Kind = "digit"; }
            else if (Code < 0x80u)   { Kind = "ascii"; }
            else if (Code >= 0xAC00u && Code <= 0xD7A3u) { Kind = "hangul"; }
            else                     { Kind = "other"; }

            fprintf(File, "%u,%u,%llu,%s\n", t, Code,
                    (unsigned long long)CountOf[(size_t)t], Kind);
        }

        fclose(File);
    }

    printf("  %s  (%zu x %zu)\n", EMBEDDING_NPY, Model.Embedding.Rows(),
           Model.Embedding.Cols());
    printf("  %s  (%d줄)\n\n", VOCAB_CSV, VOCAB_SIZE);

    // 다시 읽어 왕복을 확인한다. C4 에서 만든 도구가 바로 쓰인다.
    {
        FMatrix Again;
        CHECK(NpyLoadMatrix(EMBEDDING_NPY, Again));
        CHECK(Again.Rows() == Model.Embedding.Rows());
        CHECK(Again.Cols() == Model.Embedding.Cols());

        int Same = 1;
        for (size_t i = 0; i < Again.Count(); i++)
        {
            if (Again.Data()[i] != Model.Embedding.Data()[i])
            {
                Same = 0;
                break;
            }
        }
        CHECK(Same == 1);
    }

    // ---- C6-3. 16차원을 2차원으로 ----
    printf("[C6-3] 주성분 분석\n\n");

    // 특수 토큰 셋은 빼고 진짜 글자만 본다.
    FMatrix Real2D;
    std::vector<uint32_t> Ids;

    {
        FMatrix Rows(VOCAB_SIZE - FIRST_CHAR, EMBED_DIM);
        size_t At = 0;
        for (uint32_t t = FIRST_CHAR; t < (uint32_t)VOCAB_SIZE; t++)
        {
            for (size_t d = 0; d < EMBED_DIM; d++)
            {
                Rows(At, d) = Model.Embedding((size_t)t, d);
            }
            Ids.push_back(t);
            At++;
        }

        FPca Pca = PcaFit(Rows, 2, 200);
        Real2D = PcaTransform(Rows, Pca);

        printf("  표본 %zu개, %zu차원 -> 2차원\n", Rows.Rows(), Rows.Cols());
        printf("  1주성분이 설명하는 분산 = %.2f%%\n",
               (double)Pca.Variance[0] * 100.0 / (double)Pca.TotalVariance);
        printf("  2주성분까지 = %.2f%%\n\n",
               ((double)Pca.Variance[0] + (double)Pca.Variance[1]) * 100.0
               / (double)Pca.TotalVariance);

        // 주성분은 길이가 1 이고 서로 직각이어야 한다.
        double LengthOne = 0.0;
        double LengthTwo = 0.0;
        double Dot = 0.0;

        for (size_t d = 0; d < EMBED_DIM; d++)
        {
            double A = (double)Pca.Components(0, d);
            double B = (double)Pca.Components(1, d);
            LengthOne += A * A;
            LengthTwo += B * B;
            Dot += A * B;
        }

        printf("  1주성분의 길이 = %.9f (1 이어야 한다)\n", std::sqrt(LengthOne));
        printf("  2주성분의 길이 = %.9f\n", std::sqrt(LengthTwo));
        printf("  둘의 내적      = %.9f (0 이어야 한다)\n\n", Dot);

        CHECK_NEAR(std::sqrt(LengthOne), 1.0, 1e-5);
        CHECK_NEAR(std::sqrt(LengthTwo), 1.0, 1e-5);
        CHECK_NEAR(Dot, 0.0, 1e-4);

        // 분산이 큰 순서여야 한다.
        CHECK((double)Pca.Variance[0] >= (double)Pca.Variance[1]);

        FILE* File = fopen(PCA_CSV, "wb");
        if (!CHECK(File != NULL))
        {
            return ReportResult();
        }

        fprintf(File, "Id,X,Y\n");
        for (size_t i = 0; i < Real2D.Rows(); i++)
        {
            fprintf(File, "%u,%.9f,%.9f\n", Ids[i],
                    (double)Real2D(i, 0), (double)Real2D(i, 1));
        }
        fclose(File);

        printf("  %s 에 적었다\n\n", PCA_CSV);
    }

    // ---- C6-4. 눌러도 남는가 ----
    printf("[C6-4] 2차원으로 눌러도 이웃이 남는가\n\n");

    {
        auto Row16 = [&](uint32_t Token)
        {
            FVector V(EMBED_DIM);
            for (size_t d = 0; d < EMBED_DIM; d++)
            {
                V[d] = Model.Embedding((size_t)Token, d);
            }
            return V;
        };

        auto Row2 = [&](size_t Index)
        {
            FVector V(2);
            V[0] = Real2D(Index, 0);
            V[1] = Real2D(Index, 1);
            return V;
        };

        // 숫자끼리, 한글끼리 평균 유사도를 16차원과 2차원에서 각각 잰다.
        auto MeanSimilarity = [&](const char* Kind, bool bTwoD)
        {
            double Sum = 0.0;
            int Count = 0;

            for (size_t a = 0; a < Ids.size(); a++)
            {
                uint32_t CodeA = CodeOf[(size_t)Ids[a]];
                bool bDigitA = (CodeA >= '0' && CodeA <= '9');
                bool bHangulA = (CodeA >= 0xAC00u && CodeA <= 0xD7A3u);

                bool bWantA = (strcmp(Kind, "digit") == 0) ? bDigitA : bHangulA;
                if (!bWantA) continue;

                for (size_t b = a + 1; b < Ids.size(); b++)
                {
                    uint32_t CodeB = CodeOf[(size_t)Ids[b]];
                    bool bDigitB = (CodeB >= '0' && CodeB <= '9');
                    bool bHangulB = (CodeB >= 0xAC00u && CodeB <= 0xD7A3u);

                    bool bWantB =
                        (strcmp(Kind, "digit") == 0) ? bDigitB : bHangulB;
                    if (!bWantB) continue;

                    double S = bTwoD
                        ? (double)CosineSimilarity(Row2(a), Row2(b))
                        : (double)CosineSimilarity(Row16(Ids[a]), Row16(Ids[b]));

                    Sum += S;
                    Count++;
                }
            }

            return (Count > 0) ? (Sum / (double)Count) : 0.0;
        };

        double DigitIn16 = MeanSimilarity("digit", false);
        double DigitIn2 = MeanSimilarity("digit", true);
        double HangulIn16 = MeanSimilarity("hangul", false);
        double HangulIn2 = MeanSimilarity("hangul", true);

        printf("  ");
        PrintPadded("무리", 14);
        PrintPaddedRight("16차원", 12);
        PrintPaddedRight("2차원", 12);
        printf("\n");

        printf("  ");
        PrintPadded("숫자끼리", 14);
        printf("%11.4f %11.4f\n", DigitIn16, DigitIn2);

        printf("  ");
        PrintPadded("한글끼리", 14);
        printf("%11.4f %11.4f\n\n", HangulIn16, HangulIn2);

        printf("  숫자끼리가 한글끼리보다 훨씬 가깝다. 눌러도 그대로다.\n\n");

        CHECK(DigitIn16 > HangulIn16);
    }

    printf("이제 파이썬 차례다.\n");
    printf("    .venv/Scripts/python tools/plot_embedding.py\n\n");

    return ReportResult();
}
