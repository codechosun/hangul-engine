// ch-C03-nplm/Main.cpp
//
// C3. NPLM 통합
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Backoff.h"
#include "GradCheck.hpp"
#include "Ngram.h"
#include "Nplm.hpp"
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
#define EXPECTED_PATH  "data/nplm_expected.csv"

#define BOOK_SEED 20260914ull

// ---- 어휘 ----
//
// 전체 5,534종을 다 쓰면 출력층이 그만큼 커져서 이 장의 예제가 너무 느려진다.
// 자주 나오는 것만 남기고 나머지는 UNK 로 몰아넣는다.
// B2 연습문제 2 에서 예고한 그 방법이다.
#define VOCAB_SIZE 512
#define TOKEN_UNK  0u
#define TOKEN_END  1u
#define TOKEN_PAD  2u
#define FIRST_CHAR 3u

// ---- 모델 크기 ----
#define CONTEXT_SIZE 3
#define EMBED_DIM    16
#define HIDDEN_DIM   64

// ---- 훈련 ----
#define TRAIN_LINES  20000
#define VALID_LINES   2000
#define MAX_TRAIN_EXAMPLES 200000
#define BATCH_SIZE   64
#define EPOCH_COUNT  10
#define LEARN_RATE   Real(0.5)

namespace
{

struct FExample
{
    uint32_t Context[CONTEXT_SIZE];
    uint32_t Target;
};

// 코퍼스를 읽어 토큰 배열로 바꾼다. 줄 끝마다 TOKEN_END 를 넣는다.
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

// 토큰 배열에서 (문맥, 정답) 쌍을 뽑는다.
std::vector<FExample> MakeExamples(const std::vector<uint32_t>& Tokens,
                                   size_t Limit)
{
    std::vector<FExample> Examples;

    uint32_t Window[CONTEXT_SIZE];
    for (int i = 0; i < CONTEXT_SIZE; i++)
    {
        Window[i] = TOKEN_PAD;
    }

    for (size_t i = 0; i < Tokens.size(); i++)
    {
        if (Examples.size() >= Limit)
        {
            break;
        }

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
        FNplmTrace Trace = Model.Forward(Examples[i].Context, Examples[i].Target);
        Sum += (double)Trace.Loss;
        Count++;
    }

    return (Count > 0) ? (Sum / (double)Count) : 0.0;
}

// 정수 나눗셈으로 정해지는 값. 파이썬 참조 구현과 똑같이 채운다.
Real FormulaValue(size_t Index, int Multiplier, int Offset, int Modulus,
                  int Half, double Divisor)
{
    long long Raw = (long long)Index * Multiplier + Offset;
    long long Mod = Raw % Modulus;
    return (Real)(((double)(Mod - Half)) / Divisor);
}

} // namespace

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("C3. NPLM 통합\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ---- C3-1. 파이썬이 계산한 그래디언트와 대조 ----
    printf("[C3-1] 파이썬 참조 구현과 대조\n\n");

    {
        const size_t Vocab = 7;
        const size_t Context = 2;
        const size_t Embed = 3;
        const size_t HiddenSize = 4;

        FNplm Tiny(Vocab, Context, Embed, HiddenSize);

        for (size_t i = 0; i < Tiny.Embedding.Count(); i++)
        {
            Tiny.Embedding.Data()[i] = FormulaValue(i, 31, 17, 41, 20, 40.0);
        }
        for (size_t i = 0; i < Tiny.Hidden.Weight.Count(); i++)
        {
            Tiny.Hidden.Weight.Data()[i] = FormulaValue(i, 23, 11, 37, 18, 36.0);
        }
        for (size_t i = 0; i < Tiny.Hidden.Bias.Size(); i++)
        {
            Tiny.Hidden.Bias[i] = FormulaValue(i, 13, 5, 29, 14, 28.0);
        }
        for (size_t i = 0; i < Tiny.Output.Weight.Count(); i++)
        {
            Tiny.Output.Weight.Data()[i] = FormulaValue(i, 19, 7, 43, 21, 42.0);
        }
        for (size_t i = 0; i < Tiny.Output.Bias.Size(); i++)
        {
            Tiny.Output.Bias[i] = FormulaValue(i, 29, 3, 31, 15, 30.0);
        }

        uint32_t TinyContext[2] = { 2, 5 };
        const size_t TinyTarget = 3;

        FNplmTrace Trace = Tiny.Forward(TinyContext, TinyTarget);

        FNplmGrad Grad(Vocab, Context, Embed, HiddenSize);
        Grad.Zero();
        Tiny.Backward(TinyContext, TinyTarget, Trace, Grad);

        // 정답표를 읽어 하나씩 맞춰본다.
        FILE* File = fopen(EXPECTED_PATH, "rb");
        if (!CHECK(File != NULL))
        {
            printf("  %s 가 없다. tools/make_nplm_expected.py 를 돌릴 것.\n",
                   EXPECTED_PATH);
            return ReportResult();
        }

        char Line[256];
        (void)fgets(Line, sizeof(Line), File);   // 헤더

        const Real* Arrays[6] = {
            NULL,
            Grad.Embedding.Data(),
            Grad.Hidden.Weight.Data(),
            &Grad.Hidden.Bias[0],
            Grad.Output.Weight.Data(),
            &Grad.Output.Bias[0],
        };

        int Rows = 0;
        int Bad = 0;
        double WorstGap = 0.0;
        double PythonLoss = 0.0;

        const double Tolerance = (sizeof(Real) == 8) ? 1e-12 : 1e-5;

        while (fgets(Line, sizeof(Line), File) != NULL)
        {
            int Kind = 0;
            int Index = 0;
            double Value = 0.0;

            if (sscanf(Line, "%d,%d,%lf", &Kind, &Index, &Value) != 3)
            {
                continue;
            }

            double Mine = 0.0;
            if (Kind == 0)
            {
                Mine = (double)Trace.Loss;
                PythonLoss = Value;
            }
            else
            {
                Mine = (double)Arrays[Kind][(size_t)Index];
            }

            double Gap = std::fabs(Mine - Value);
            if (Gap > WorstGap)
            {
                WorstGap = Gap;
            }
            if (Gap > Tolerance)
            {
                Bad++;
            }

            Rows++;
        }

        fclose(File);

        printf("  어휘 7, 문맥 2, 임베딩 3, 은닉 4 짜리 망\n");
        printf("  손실      C++ %.15f\n", (double)Trace.Loss);
        printf("         파이썬 %.15f\n", PythonLoss);
        printf("  정답표 %d줄 대조 -> 어긋난 줄 %d개 (허용 %.0e)\n", Rows, Bad,
               Tolerance);
        printf("  최대 차이 = %.3e\n\n", WorstGap);

        CHECK(Rows == 85);
        CHECK(Bad == 0);
    }

    // ---- C3-2. gradcheck ----
    printf("[C3-2] 수치미분으로 한 번 더\n\n");

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        FNplm Small(11, CONTEXT_SIZE, 4, 5);
        Small.Init(Rng);

        uint32_t Context[CONTEXT_SIZE] = { 2, 7, 2 };   // 같은 토큰이 두 번
        const size_t Target = 4;

        FNplmGrad Grad(11, CONTEXT_SIZE, 4, 5);
        Grad.Zero();

        FNplmTrace Trace = Small.Forward(Context, Target);
        Small.Backward(Context, Target, Trace, Grad);

        auto LossOf = [&]()
        {
            return (double)Small.Forward(Context, Target).Loss;
        };

        struct FPart { const char* Name; Real* Values; const Real* Grad; int Count; };

        FPart Parts[5] = {
            { "임베딩", Small.Embedding.Data(), Grad.Embedding.Data(),
              (int)Small.Embedding.Count() },
            { "은닉 W", Small.Hidden.Weight.Data(), Grad.Hidden.Weight.Data(),
              (int)Small.Hidden.Weight.Count() },
            { "은닉 b", &Small.Hidden.Bias[0], &Grad.Hidden.Bias[0],
              (int)Small.Hidden.Bias.Size() },
            { "출력 W", Small.Output.Weight.Data(), Grad.Output.Weight.Data(),
              (int)Small.Output.Weight.Count() },
            { "출력 b", &Small.Output.Bias[0], &Grad.Output.Bias[0],
              (int)Small.Output.Bias.Size() },
        };

        printf("  ");
        PrintPadded("대상", 12);
        PrintPaddedRight("개수", 8);
        PrintPaddedRight("최대 상대 오차", 18);
        printf("\n");

        double Worst = 0.0;
        for (int i = 0; i < 5; i++)
        {
            // float 에서는 손실 자체가 7자리뿐이라 1e-4 만큼 흔들어봐야
            // 변화가 반올림에 묻힌다. 크게 흔들어 그나마 재어지게 한다.
            const double Step = (sizeof(Real) == 8) ? 1e-4 : 1e-2;

            FGradCheckResult Check = GradCheck(LossOf, Parts[i].Values,
                                               Parts[i].Grad, Parts[i].Count,
                                               Step);
            if (Check.WorstRelative > Worst)
            {
                Worst = Check.WorstRelative;
            }

            char Buffer[32];
            snprintf(Buffer, sizeof(Buffer), "%.3e", Check.WorstRelative);

            printf("  ");
            PrintPadded(Parts[i].Name, 12);
            printf("%7d ", Check.Count);
            PrintPaddedRight(Buffer, 18);
            printf("\n");
        }

        const double Threshold = (sizeof(Real) == 8) ? 1e-6 : 5e-1;
        printf("\n  전체 %.3e (기준 %.0e)\n\n", Worst, Threshold);

        CHECK(Worst < Threshold);

        printf("  문맥에 같은 토큰(2번)을 두 번 넣었다. 임베딩 그래디언트가\n");
        printf("  두 곳에서 와서 더해지는 경우를 일부러 만든 것이다.\n\n");

        if (sizeof(Real) != 8)
        {
            printf("  float 에서는 이 검사가 **의미가 없다.** 기준을 5e-1 까지\n");
            printf("  풀어야 통과하는데, 그 정도면 두 배 틀린 것도 통과한다.\n");
            printf("  USE_DOUBLE 로 다시 빌드하면 4e-08 이 나온다. C2 참고.\n\n");
        }
    }

    // ---- C3-3. 데이터 ----
    printf("[C3-3] 어휘를 %d종으로 줄인다\n\n", VOCAB_SIZE);

    std::vector<int> MapOf((size_t)VOCAB_LIMIT, -1);
    std::vector<uint32_t> CodeOf((size_t)VOCAB_SIZE, 0);

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

        uint64_t Covered = 0;
        uint64_t Total = 0;
        for (const auto& Row : ByCount)
        {
            Total += Row.first;
        }

        uint32_t Next = FIRST_CHAR;
        for (const auto& Row : ByCount)
        {
            if (Next >= (uint32_t)VOCAB_SIZE)
            {
                break;
            }
            MapOf[(size_t)Row.second] = (int)Next;
            CodeOf[(size_t)Next] = Row.second;
            Covered += Row.first;
            Next++;
        }

        printf("  전체 %llu종 중 %u종을 남겼다\n",
               (unsigned long long)ByCount.size(), Next - FIRST_CHAR);
        printf("  남긴 것이 전체 글자의 %.3f%% 를 덮는다\n",
               (double)Covered * 100.0 / (double)Total);
        printf("  나머지는 전부 UNK 한 칸으로 몰아넣는다\n\n");

        CHECK(Covered * 100 > Total * 95);   // 95% 는 넘어야 한다

        NgramFree(&Unigram);
    }

    std::vector<uint32_t> TrainTokens = ReadTokens(CorpusPath, 0, TRAIN_LINES, MapOf);
    std::vector<uint32_t> ValidTokens =
        ReadTokens(CorpusPath, TRAIN_LINES, VALID_LINES, MapOf);

    std::vector<FExample> TrainSet = MakeExamples(TrainTokens, MAX_TRAIN_EXAMPLES);
    std::vector<FExample> ValidSet = MakeExamples(ValidTokens, 20000);

    printf("  훈련 예제 %zu개, 검증 예제 %zu개\n\n", TrainSet.size(), ValidSet.size());

    CHECK(TrainSet.size() > 1000);
    CHECK(ValidSet.size() > 1000);

    // ---- C3-4. 훈련 ----
    printf("[C3-4] 훈련\n\n");

    FNplm Model(VOCAB_SIZE, CONTEXT_SIZE, EMBED_DIM, HIDDEN_DIM);

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);
        Model.Init(Rng);

        printf("  어휘 %d, 문맥 %d, 임베딩 %d, 은닉 %d\n",
               VOCAB_SIZE, CONTEXT_SIZE, EMBED_DIM, HIDDEN_DIM);
        printf("  가중치 %zu개\n", Model.ParameterCount());
        printf("  배치 %d, 에폭 %d, 학습률 %.2f\n\n",
               BATCH_SIZE, EPOCH_COUNT, (double)LEARN_RATE);

        double StartTrain = MeanLoss(Model, TrainSet, 5000);
        double StartValid = MeanLoss(Model, ValidSet, 5000);

        printf("  ");
        PrintPaddedRight("에폭", 8);
        PrintPaddedRight("훈련 손실", 14);
        PrintPaddedRight("검증 손실", 14);
        PrintPaddedRight("검증 PPL", 14);
        PrintPaddedRight("시간", 10);
        printf("\n");

        printf("  %7d %13.4f %13.4f %13.1f %9s\n", 0, StartTrain, StartValid,
               std::exp(StartValid), "-");

        // 시작 손실이 log(어휘) 근처여야 한다. C1 에서 본 그 검사다.
        CHECK(std::fabs(StartValid - std::log((double)VOCAB_SIZE)) < 0.5);

        FNplmGrad Grad(VOCAB_SIZE, CONTEXT_SIZE, EMBED_DIM, HIDDEN_DIM);

        std::vector<size_t> Order(TrainSet.size());
        for (size_t i = 0; i < Order.size(); i++)
        {
            Order[i] = i;
        }

        FRandom Shuffle;
        RandomSeed(&Shuffle, BOOK_SEED + 1);

        double LastValid = StartValid;
        double BestValid = StartValid;

        for (int Epoch = 1; Epoch <= EPOCH_COUNT; Epoch++)
        {
            clock_t Begin = clock();

            // 에폭마다 순서를 섞는다.
            for (size_t i = Order.size() - 1; i > 0; i--)
            {
                uint64_t j = RandomBelow(&Shuffle, (uint64_t)i + 1);
                std::swap(Order[i], Order[(size_t)j]);
            }

            for (size_t Start = 0; Start < Order.size(); Start += BATCH_SIZE)
            {
                Grad.Zero();

                size_t Stop = Start + BATCH_SIZE;
                if (Stop > Order.size())
                {
                    Stop = Order.size();
                }

                for (size_t i = Start; i < Stop; i++)
                {
                    const FExample& Example = TrainSet[Order[i]];

                    FNplmTrace Trace =
                        Model.Forward(Example.Context, Example.Target);
                    Model.Backward(Example.Context, Example.Target, Trace, Grad);
                }

                Model.Step(Grad, LEARN_RATE, Real(1) / (Real)(Stop - Start));
            }

            double Seconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

            double TrainLoss = MeanLoss(Model, TrainSet, 5000);
            double ValidLoss = MeanLoss(Model, ValidSet, 5000);

            printf("  %7d %13.4f %13.4f %13.1f %8.1fs\n", Epoch, TrainLoss,
                   ValidLoss, std::exp(ValidLoss), Seconds);

            LastValid = ValidLoss;
            if (ValidLoss < BestValid)
            {
                BestValid = ValidLoss;
            }
        }

        printf("\n  검증 손실 %.4f -> %.4f\n", StartValid, LastValid);
        printf("  검증 퍼플렉서티 %.1f -> %.1f\n\n",
               std::exp(StartValid), std::exp(LastValid));

        CHECK(LastValid < StartValid);
        CHECK(std::exp(LastValid) < (double)VOCAB_SIZE / 4.0);
    }

    // ---- C3-5. 같은 데이터로 만든 N-그램과 비교 ----
    printf("[C3-5] 같은 어휘, 같은 데이터의 4그램 백오프와 비교\n\n");

    {
        FBackoff Backoff;
        memset(&Backoff, 0, sizeof(Backoff));
        Backoff.MaxOrder = CONTEXT_SIZE + 1;
        Backoff.DiscountNum = BACKOFF_NUM;
        Backoff.DiscountDen = BACKOFF_DEN;

        // lib/Ngram 은 줄 끝을 TOKEN_EOS 로 안다. 우리 토큰을 그 규약에 맞춘다.
        std::vector<uint32_t> Converted = TrainTokens;
        for (size_t i = 0; i < Converted.size(); i++)
        {
            if (Converted[i] == TOKEN_END)
            {
                Converted[i] = TOKEN_EOS;
            }
        }

        int Built = 1;
        for (int Order = 1; Order <= Backoff.MaxOrder; Order++)
        {
            if (!NgramBuildFromTokens(&Backoff.Orders[Order], Converted.data(),
                                      Converted.size(), Order))
            {
                Built = 0;
                break;
            }
        }

        if (CHECK(Built == 1))
        {
            double Sum = 0.0;
            size_t Count = 0;

            for (size_t i = 0; i < ValidSet.size() && Count < 5000; i++)
            {
                uint32_t Context[CONTEXT_SIZE];
                for (int k = 0; k < CONTEXT_SIZE; k++)
                {
                    uint32_t Token = ValidSet[i].Context[k];
                    Context[k] = (Token == TOKEN_PAD) ? (uint32_t)TOKEN_BOS : Token;
                }

                uint32_t Target = ValidSet[i].Target;
                if (Target == TOKEN_END)
                {
                    Target = (uint32_t)TOKEN_EOS;
                }

                double Prob = BackoffProb(&Backoff, Context, CONTEXT_SIZE, Target);
                if (Prob < 1e-12)
                {
                    Prob = 1e-12;
                }

                Sum += -std::log(Prob);
                Count++;
            }

            double NgramLoss = Sum / (double)Count;
            double NplmLoss = MeanLoss(Model, ValidSet, 5000);

            printf("  ");
            PrintPadded("모델", 20);
            PrintPaddedRight("검증 손실", 14);
            PrintPaddedRight("퍼플렉서티", 14);
            printf("\n");

            printf("  ");
            PrintPadded("4그램 백오프", 20);
            printf("%13.4f %13.1f\n", NgramLoss, std::exp(NgramLoss));

            printf("  ");
            PrintPadded("NPLM", 20);
            printf("%13.4f %13.1f\n", NplmLoss, std::exp(NplmLoss));

            printf("\n  가중치 개수 : N-그램 %llu 항목, NPLM %zu 개\n\n",
                   (unsigned long long)Backoff.Orders[CONTEXT_SIZE + 1].GramCount,
                   Model.ParameterCount());
        }

        BackoffFree(&Backoff);
    }

    // ---- C3-6. 배운 임베딩 ----
    printf("[C3-6] 훈련이 만든 임베딩\n\n");

    {
        auto RowOf = [&](uint32_t Token)
        {
            FVector V(EMBED_DIM);
            const Real* Row = Model.Embedding.RowData((size_t)Token);
            for (size_t d = 0; d < EMBED_DIM; d++)
            {
                V[d] = Row[d];
            }
            return V;
        };

        auto Describe = [&](uint32_t Token) -> std::string
        {
            if (Token == TOKEN_UNK) return "UNK";
            if (Token == TOKEN_END) return "EOS";
            if (Token == TOKEN_PAD) return "PAD";

            uint32_t Code = CodeOf[(size_t)Token];
            if (Code == ' ') return "공백";

            char Utf8[8] = {};
            int Bytes = Utf8Encode(Code, Utf8);
            return std::string(Utf8, (size_t)Bytes);
        };

        const uint32_t Queries[4] = { 0xC744u, 0x0031u, 0xD55Cu, 0x002Eu };

        for (uint32_t Code : Queries)
        {
            if (MapOf[(size_t)Code] < 0)
            {
                continue;
            }
            uint32_t Self = (uint32_t)MapOf[(size_t)Code];

            std::vector<std::pair<Real, uint32_t>> Scores;
            for (uint32_t t = FIRST_CHAR; t < (uint32_t)VOCAB_SIZE; t++)
            {
                if (t == Self)
                {
                    continue;
                }
                Scores.emplace_back(CosineSimilarity(RowOf(Self), RowOf(t)), t);
            }

            std::sort(Scores.begin(), Scores.end(),
                      [](const std::pair<Real, uint32_t>& L,
                         const std::pair<Real, uint32_t>& R)
                      {
                          if (L.first != R.first) return L.first > R.first;
                          return L.second < R.second;
                      });

            char Label[32];
            snprintf(Label, sizeof(Label), "'%s'", Describe(Self).c_str());

            printf("  ");
            PrintPadded(Label, 8);
            printf("→ ");
            for (int k = 0; k < 5 && k < (int)Scores.size(); k++)
            {
                printf("%s(%.3f) ", Describe(Scores[(size_t)k].second).c_str(),
                       (double)Scores[(size_t)k].first);
            }
            printf("\n");
        }

        printf("\n  B3 에서는 바이그램 표를 그대로 벡터로 썼다. 여기서는\n");
        printf("  **훈련이 만든** 16차원 벡터다. 5,534차원이 16차원이 됐다.\n\n");
    }

    return ReportResult();
}
