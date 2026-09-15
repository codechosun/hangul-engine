// ch-E03-wall/Main.cpp
//
// E3. 규모의 벽
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// 몇 분 걸린다. /openmp 를 켜면 빠르다.

#include "Test.h"
#include "Pretty.h"

#include "Backward.hpp"
#include "Engine.hpp"
#include "Optimizer.hpp"
#include "Random.h"
#include "Tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define BOOK_SEED 20260914ull

#define CORPUS_PATH "data/corpus.txt"
#define CURVE_PATH  "data/scaling.csv"

#define VOCAB   256
#define LENGTH   32
#define BATCH     8

#ifndef STEPS
#define STEPS 600
#endif

namespace
{

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

struct FRun
{
    size_t Width = 0;
    size_t Layers = 0;
    size_t Parameters = 0;
    double Loss = 0.0;
    double StepSeconds = 0.0;
};

// 최소제곱으로 y = a + b*x 를 맞춘다. C6 에서 쓴 것과 같은 계산이다.
void FitLine(const std::vector<double>& X, const std::vector<double>& Y,
             double* OutA, double* OutB)
{
    const double N = (double)X.size();

    double SumX = 0.0, SumY = 0.0, SumXX = 0.0, SumXY = 0.0;
    for (size_t i = 0; i < X.size(); i++)
    {
        SumX += X[i];
        SumY += Y[i];
        SumXX += X[i] * X[i];
        SumXY += X[i] * Y[i];
    }

    const double Bottom = N * SumXX - SumX * SumX;

    *OutB = (N * SumXY - SumX * SumY) / Bottom;
    *OutA = (SumY - (*OutB) * SumX) / N;
}

} // namespace

int main(void)
{
    printf("E3. 규모의 벽\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ---- 자료 ----
    FILE* File = fopen(CORPUS_PATH, "rb");
    if (File == NULL)
    {
        printf("  %s 가 없다. tools/download_corpus.py 를 돌릴 것.\n",
               CORPUS_PATH);
        return 1;
    }

    std::vector<char> Raw(1u << 21);   // 2MB
    const size_t Got = fread(Raw.data(), 1, Raw.size(), File);
    fclose(File);

    FTokenizer Tokenizer;
    Tokenizer.BuildFromText(Raw.data(), Got, VOCAB, 0);

    std::vector<uint32_t> Stream;
    {
        std::string Text(Raw.data(), Got);
        Tokenizer.Encode(Text.c_str(), Stream);
    }

    printf("  코퍼스 앞 %zuKB -> 어휘 %zu자로 %.2f%% 를 덮었다\n",
           Got / 1024, Tokenizer.Size(), Tokenizer.Coverage() * 100.0);
    printf("  토큰 %zu개\n\n", Stream.size());

    CHECK(Stream.size() > 100000);

    // ================================================================
    // E3-1. 모델을 키우면 손실이 얼마나 내려가는가
    // ================================================================
    printf("[E3-1] 같은 자료, 같은 걸음 수, 크기만 바꿔서\n\n");

    const size_t Widths[5] = { 24, 32, 48, 64, 96 };

    std::vector<FRun> Runs;

    for (int w = 0; w < 5; w++)
    {
        FModelConfig Config;
        Config.Vocab = Tokenizer.Size();
        Config.Model = Widths[w];
        Config.Heads = 4;
        Config.Hidden = Widths[w] * 4;
        Config.Layers = 2;
        Config.MaxLength = LENGTH;

        FTransformer Model(Config);
        {
            FRandom Init;
            RandomSeed(&Init, BOOK_SEED);
            Model.Init(Init);
        }

        FTransformerGrad Grad;
        Grad.Init(Model);

        std::vector<Real*> Parameters, Gradients;
        CollectParameters(Model, Parameters);
        CollectGradients(Grad, Gradients);

        FAdamW Adam;
        FAdamWConfig AdamConfig;
        AdamConfig.Rate = 3e-3;
        AdamConfig.WeightDecay = 0.01;
        Adam.Init(Parameters.size(), AdamConfig);

        // 묶음은 크기와 무관하게 같은 것을 쓴다. 같은 씨앗이면 같은 순서다.
        FRandom Pick;
        RandomSeed(&Pick, BOOK_SEED + 1);

        std::vector<uint32_t> Inputs(BATCH * LENGTH, 0);
        std::vector<uint32_t> Targets(BATCH * LENGTH, 0);

        auto Begin = std::chrono::steady_clock::now();

        double Tail = 0.0;
        int TailCount = 0;

        for (int Step = 0; Step < STEPS; Step++)
        {
            for (size_t b = 0; b < BATCH; b++)
            {
                const uint64_t Start =
                    RandomBelow(&Pick, Stream.size() - LENGTH - 1);

                for (size_t t = 0; t < LENGTH; t++)
                {
                    Inputs[b * LENGTH + t] = Stream[Start + t];
                    Targets[b * LENGTH + t] = Stream[Start + t + 1];
                }
            }

            Grad.Zero();

            FModelTrace Trace;
            FTensor Logits = TransformerForwardTrace(Model, Inputs.data(),
                                                     BATCH, LENGTH, Trace);

            const double Loss = CrossEntropyLoss(Logits, Targets.data());

            if (Step >= STEPS - 20) { Tail += Loss; TailCount++; }

            FTensor Upstream = CrossEntropyBackward(Logits, Targets.data());
            TransformerBackward(Model, Trace, Upstream, Grad);

            ClipGradients(Gradients.data(), Gradients.size(), 1.0);

            const double Rate = ScheduleRate(AdamConfig.Rate, (size_t)Step, 40,
                                             (size_t)STEPS, 0.1);

            Adam.Step(Parameters.data(), (const Real* const*)Gradients.data(),
                      Parameters.size(), Rate);
        }

        const double Elapsed = Seconds(Begin);

        FRun One;
        One.Width = Config.Model;
        One.Layers = Config.Layers;
        One.Parameters = Model.ParameterCount();
        One.Loss = Tail / (double)TailCount;
        One.StepSeconds = Elapsed / (double)STEPS;

        Runs.push_back(One);

        printf("  차원 %3zu  파라미터 %7zu  손실 %.4f  한 걸음 %.1fms\n",
               One.Width, One.Parameters, One.Loss,
               One.StepSeconds * 1000.0);
        fflush(stdout);
    }

    printf("\n");

    printf("  ");
    PrintPaddedRight("차원", 8);
    PrintPaddedRight("파라미터", 12);
    PrintPaddedRight("손실", 10);
    PrintPaddedRight("글자당 비트", 14);
    printf("  그림\n");

    for (size_t i = 0; i < Runs.size(); i++)
    {
        char A[32];
        printf("  ");
        snprintf(A, sizeof(A), "%zu", Runs[i].Width);
        PrintPaddedRight(A, 8);
        snprintf(A, sizeof(A), "%zu", Runs[i].Parameters);
        PrintPaddedRight(A, 12);
        snprintf(A, sizeof(A), "%.4f", Runs[i].Loss);
        PrintPaddedRight(A, 10);
        snprintf(A, sizeof(A), "%.3f", Runs[i].Loss / std::log(2.0));
        PrintPaddedRight(A, 14);

        printf("  ");
        const int Bars = (int)(Runs[i].Loss / std::log((double)VOCAB)
                             * 50.0 + 0.5);
        for (int k = 0; k < Bars; k++) printf("#");
        printf("\n");
    }

    printf("\n");

    // 커질수록 손실이 내려가야 한다.
    CHECK(Runs[4].Loss < Runs[0].Loss);

    {
        FILE* Out = fopen(CURVE_PATH, "wb");
        if (Out != NULL)
        {
            fprintf(Out, "Width,Parameters,Loss,StepSeconds\n");
            for (size_t i = 0; i < Runs.size(); i++)
            {
                fprintf(Out, "%zu,%zu,%.9g,%.9g\n", Runs[i].Width,
                        Runs[i].Parameters, Runs[i].Loss,
                        Runs[i].StepSeconds);
            }
            fclose(Out);
            printf("  곡선은 %s 에 남겼다.\n\n", CURVE_PATH);
        }
    }

    // ================================================================
    // E3-2. 기울기를 재고, 외삽해 본다
    // ================================================================
    printf("[E3-2] 파라미터를 10배로 하면 손실이 얼마나 내려가나\n\n");

    std::vector<double> LogN, Loss;
    for (size_t i = 0; i < Runs.size(); i++)
    {
        LogN.push_back(std::log10((double)Runs[i].Parameters));
        Loss.push_back(Runs[i].Loss);
    }

    double A = 0.0, B = 0.0;
    FitLine(LogN, Loss, &A, &B);

    printf("  손실 = %.4f + (%.4f) x log10(파라미터)\n\n", A, B);
    printf("  **파라미터를 10배로 하면 손실이 %.4f 내려간다.**\n", -B);
    printf("  글자당 %.3f 비트다.\n\n", -B / std::log(2.0));

    CHECK(B < 0.0);

    printf("  ");
    PrintPaddedRight("파라미터", 14);
    PrintPaddedRight("이 식의 예측", 14);
    PrintPaddedRight("글자당 비트", 14);
    printf("\n");

    const double Sizes[6] =
    {
        1e5, 1e6, 1e7, 1e8, 7e9, 1e12
    };

    for (int i = 0; i < 6; i++)
    {
        const double Predicted = A + B * std::log10(Sizes[i]);

        char X[32], Y[32], Z[32];
        snprintf(X, sizeof(X), "%.0e", Sizes[i]);
        snprintf(Y, sizeof(Y), "%.4f", Predicted);
        snprintf(Z, sizeof(Z), "%.3f", Predicted / std::log(2.0));

        printf("  ");
        PrintPaddedRight(X, 14);
        PrintPaddedRight(Y, 14);
        PrintPaddedRight(Z, 14);
        printf("%s\n", (Predicted < 0.0) ? "   <- 말이 안 된다" : "");
    }

    printf("\n");

    // 외삽한 값이 언젠가 음수가 된다. 손실은 음수가 될 수 없다.
    const double AtTrillion = A + B * std::log10(1e12);
    CHECK(AtTrillion < Runs[4].Loss);

    printf("  **이 직선은 틀렸다.** 손실은 0 아래로 못 간다.\n");
    printf("  글자를 완벽히 맞혀도 한국어 자체의 불확실성이 남는다.\n\n");

    printf("  실제 스케일링 법칙은 이렇게 생겼다.\n\n");
    printf("      손실 = 바닥 + 계수 / 파라미터^알파\n\n");
    printf("  직선이 아니라 **바닥을 향해 눕는 곡선**이다. 우리가 잰\n");
    printf("  다섯 점은 그 곡선의 아주 왼쪽 끝이라 직선으로 보인다.\n\n");

    printf("  다섯 점으로 12자릿수를 외삽하는 것은 **과학이 아니다.**\n");
    printf("  이 표의 값은 \"이 정도 방향\"이지 예측이 아니다.\n\n");

    // ================================================================
    // E3-3. 시간의 벽
    // ================================================================
    printf("[E3-3] 이 코드로 70억을 훈련하면\n\n");

    // 한 걸음 시간이 파라미터에 비례하는지 본다.
    printf("  ");
    PrintPaddedRight("파라미터", 12);
    PrintPaddedRight("한 걸음", 12);
    PrintPaddedRight("파라미터당", 16);
    printf("\n");

    for (size_t i = 0; i < Runs.size(); i++)
    {
        char X[32], Y[32], Z[32];
        snprintf(X, sizeof(X), "%zu", Runs[i].Parameters);
        snprintf(Y, sizeof(Y), "%.1fms", Runs[i].StepSeconds * 1000.0);
        snprintf(Z, sizeof(Z), "%.1fns",
                 Runs[i].StepSeconds / (double)Runs[i].Parameters * 1e9);

        printf("  ");
        PrintPaddedRight(X, 12);
        PrintPaddedRight(Y, 12);
        PrintPaddedRight(Z, 16);
        printf("\n");
    }

    printf("\n");

    const FRun& Big = Runs[4];
    const double PerParameter = Big.StepSeconds / (double)Big.Parameters;

    printf("  가장 큰 판 기준 파라미터당 %.1fns.\n\n", PerParameter * 1e9);

    // 70억 파라미터, 토큰 1조 개를 이 속도로.
    const double Seven = 7e9;
    const double TokensPerStep = (double)(BATCH * LENGTH);
    const double StepSecondsAt7B = PerParameter * Seven;

    printf("  70억 파라미터면 한 걸음에 %.1f초 (묶음 %zu x %d = 토큰 %.0f개)\n",
           StepSecondsAt7B, (size_t)BATCH, LENGTH, TokensPerStep);

    const double TotalTokens = 1e12;
    const double Steps = TotalTokens / TokensPerStep;
    const double TotalSeconds = Steps * StepSecondsAt7B;
    const double Years = TotalSeconds / (365.25 * 24.0 * 3600.0);

    printf("  토큰 1조 개를 보려면 걸음이 %.2e번.\n", Steps);
    printf("  **%.2e 년이다.**\n\n", Years);

    CHECK(Years > 1000.0);

    printf("  기록된 인류 역사가 5천 년쯤이다. 그 스무 배다.\n\n");

    printf("  이 계산에서 정직해야 할 것 둘.\n");
    printf("    1. 실제로는 묶음을 훨씬 크게 잡는다. 그래도 자릿수가 안 준다\n");
    printf("    2. GPU 는 이 CPU 코드보다 1000배쯤 빠르고, 카드를 수천 장\n");
    printf("       쓴다. 그래서 %.2e년이 몇 달이 된다\n\n",
           Years);

    printf("  **1000 x 수천 = 백만 배.** 그 백만 배가 규모의 벽이다.\n\n");

    // ================================================================
    // E3-4. 메모리의 벽
    // ================================================================
    printf("[E3-4] 메모리는 더 단단한 벽이다\n\n");

    printf("  D6~D7 에서 잰 비율 그대로 계산한다.\n\n");

    printf("  ");
    PrintPadded("무엇", 20);
    PrintPaddedRight("우리 것", 14);
    PrintPaddedRight("70억", 14);
    printf("\n");

    struct FItem { const char* Name; double Ratio; };
    const FItem Items[4] =
    {
        { "파라미터",       1.0 },
        { "그래디언트",     1.0 },
        { "AdamW 상태",     2.0 },
        { "합",             4.0 },
    };

    for (int i = 0; i < 4; i++)
    {
        const double Ours = (double)Big.Parameters * 4.0 * Items[i].Ratio;
        const double Seven2 = Seven * 4.0 * Items[i].Ratio;

        char X[32], Y[32];
        snprintf(X, sizeof(X), "%.1f KB", Ours / 1024.0);
        snprintf(Y, sizeof(Y), "%.0f GB", Seven2 / (1024.0 * 1024.0 * 1024.0));

        printf("  ");
        PrintPadded(Items[i].Name, 20);
        PrintPaddedRight(X, 14);
        PrintPaddedRight(Y, 14);
        printf("\n");
    }

    printf("\n");
    printf("  여기에 자취(D6) 가 더 붙는다. 배치와 길이에 비례한다.\n\n");

    printf("  **시간은 기다리면 되지만 메모리는 안 그렇다.**\n");
    printf("  안 들어가면 안 도는 것이다. 그래서 훈련 쪽 기법의 대부분이\n");
    printf("  메모리 이야기다 — 혼합 정밀도, 8비트 옵티마이저, ZeRO,\n");
    printf("  gradient checkpointing. D5 와 D7 에서 본 그대로다.\n\n");

    // ================================================================
    // E3-5. 그래서 어디서 갈아타는가
    // ================================================================
    printf("[E3-5] 이 코드가 맞는 자리와 아닌 자리\n\n");

    printf("  ");
    PrintPadded("무엇을", 30);
    PrintPadded("이 엔진", 14);
    printf("왜\n");

    struct FCase { const char* What; const char* Verdict; const char* Why; };
    const FCase Cases[8] =
    {
        { "글자·낱말 N-그램",       "맞다",   "A파트로 끝난다" },
        { "100만 이하 모델 추론",   "맞다",   "953KB, 의존성 0" },
        { "100만 이하 모델 훈련",   "맞다",   "이 장이 그 증거" },
        { "1억 모델 추론",          "된다",   "느리지만 돈다. 양자화 필요" },
        { "1억 모델 훈련",          "아니다", "CPU 로 몇 주" },
        { "70억 모델 추론",         "아니다", "28GB. 메모리부터 안 된다" },
        { "70억 모델 훈련",         "아니다", "백만 배 모자란다" },
        { "남이 훈련한 모델 싣기",  "맞다",   "형식 변환만 하면 된다" },
    };

    for (int i = 0; i < 8; i++)
    {
        printf("  ");
        PrintPadded(Cases[i].What, 30);
        PrintPadded(Cases[i].Verdict, 14);
        printf("%s\n", Cases[i].Why);
    }

    printf("\n");

    printf("  **경계는 대략 백만에서 1억 사이**다. 그 위로는 이 코드가\n");
    printf("  틀린 게 아니라 **도구가 다르다.**\n\n");

    printf("  그리고 마지막 줄이 중요하다. 남이 훈련한 가중치를 싣는 것은\n");
    printf("  이 엔진이 가장 잘하는 일이다. 훈련은 파이썬과 GPU 가 하고,\n");
    printf("  **제품에 실려 도는 것은 이쪽이 낫다.** 의존성 0, 오프라인,\n");
    printf("  프로세스 안에서 도는 953KB.\n\n");

    // ================================================================
    // E3-6. 남은 것
    // ================================================================
    printf("[E3-6] 30장을 지나 남은 것\n\n");

    printf("  숫자로 남은 것\n");
    printf("    UTF-8 글자 세기부터 트랜스포머 훈련까지 C/C++ 로\n");
    printf("    엔진 %d개 파일, 외부 의존성 0\n", 43);
    printf("    gradcheck 6.787e-07, NumPy 곡선 대조 6.661e-15\n");
    printf("    한국어 글자당 %.3f 비트\n\n", Runs[4].Loss / std::log(2.0));

    printf("  숫자가 아닌 것\n");
    printf("    **재보지 않으면 모른다** — C5 의 블록 크기, D5 의 양자화 속도\n");
    printf("    **조용히 틀리는 것이 무섭다** — A6 의 창 초기화, D6 의 잔차\n");
    printf("    **못 재는 것과 틀린 것은 다르다** — D6 의 float gradcheck\n");
    printf("    **도구를 먼저 맞춰놓는다** — D8 이 그 값을 회수한 자리\n\n");

    printf("  이 교재가 한 일은 **판단할 근거를 만든 것**이다.\n");
    printf("  무엇이 되고 무엇이 안 되는지를 남의 말이 아니라\n");
    printf("  자기가 잰 숫자로 아는 것. 그게 여기까지 온 값이다.\n\n");

    return ReportResult();
}
