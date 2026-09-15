// ch-D08-grokking/Main.cpp
//
// D8. 그로킹 재현
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// 몇 분 걸린다. **오래 보는 것이 이 장의 내용이다.**
//
// /openmp 를 켜고 빌드하면 훨씬 빠르다. 켜든 안 켜든 숫자는 같다 —
// MatMul 이 출력 줄 단위로만 나뉘므로 더하는 순서가 안 바뀐다.

#include "Test.h"
#include "Pretty.h"

#include "Backward.hpp"
#include "Block.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"
#include "Random.h"
#include "Tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define BOOK_SEED 20260914ull

#define CURVE_PATH "data/grokking_curve.csv"

// (a + b) mod P. P 는 소수로 잡는다.
#define P 37

// 어휘는 0..P-1 과 '=' 하나.
#define EQUALS P
#define VOCAB  (P + 1)

#define LENGTH 3     // a, b, =

#define MODEL   64
#define HEADS    4
#ifndef HIDDEN
#define HIDDEN 128
#endif
#define LAYERS   1

// 훈련에 쓸 비율. 적을수록 외우기는 쉽고 이해하기는 어렵다.
#ifndef TRAIN_PERCENT
#define TRAIN_PERCENT 35
#endif

#ifndef STEPS
#define STEPS  8000
#endif
#define REPORT   200

namespace
{

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

// 마지막 자리만 채점한다.
//
// a 와 b 를 보기 전에는 답을 알 수 없으므로 앞의 두 자리는 뜻이 없다.
// 거기까지 손실에 넣으면 **알 수 없는 것을 못 맞혔다고 벌하는 것**이라
// 신호가 묻힌다.
//
// E1 에서 이것을 임의의 자리로 일반화한다. 여기서는 마지막 하나면 된다.
double LastLoss(const FTensor& Logits, const uint32_t* Answers)
{
    const size_t Batch = Logits.Size(0);
    const size_t Last = Logits.Size(1) - 1;

    double Total = 0.0;

    for (size_t b = 0; b < Batch; b++)
    {
        const Real* Row = Logits.Data() + (b * Logits.Size(1) + Last) * VOCAB;

        double Biggest = (double)Row[0];
        for (size_t v = 1; v < VOCAB; v++)
        {
            if ((double)Row[v] > Biggest) Biggest = (double)Row[v];
        }

        double Sum = 0.0;
        for (size_t v = 0; v < VOCAB; v++)
        {
            Sum += std::exp((double)Row[v] - Biggest);
        }

        Total -= ((double)Row[Answers[b]] - Biggest) - std::log(Sum);
    }

    return Total / (double)Batch;
}

FTensor LastBackward(const FTensor& Logits, const uint32_t* Answers)
{
    const size_t Batch = Logits.Size(0);
    const size_t Last = Logits.Size(1) - 1;

    FTensor Result(Logits.GetShape());
    Result.Fill(Real(0));

    for (size_t b = 0; b < Batch; b++)
    {
        const size_t Base = (b * Logits.Size(1) + Last) * VOCAB;
        const Real* Row = Logits.Data() + Base;
        Real* Out = Result.Data() + Base;

        double Biggest = (double)Row[0];
        for (size_t v = 1; v < VOCAB; v++)
        {
            if ((double)Row[v] > Biggest) Biggest = (double)Row[v];
        }

        double Sum = 0.0;
        for (size_t v = 0; v < VOCAB; v++)
        {
            Sum += std::exp((double)Row[v] - Biggest);
        }

        for (size_t v = 0; v < VOCAB; v++)
        {
            const double Prob = std::exp((double)Row[v] - Biggest) / Sum;
            Out[v] = (Real)((Prob - ((size_t)Answers[b] == v ? 1.0 : 0.0))
                          / (double)Batch);
        }
    }

    return Result;
}

double Accuracy(const FTensor& Logits, const uint32_t* Answers)
{
    const size_t Batch = Logits.Size(0);
    const size_t Last = Logits.Size(1) - 1;

    size_t Right = 0;

    for (size_t b = 0; b < Batch; b++)
    {
        const Real* Row = Logits.Data() + (b * Logits.Size(1) + Last) * VOCAB;

        size_t Best = 0;
        for (size_t v = 1; v < VOCAB; v++)
        {
            if (Row[v] > Row[Best]) Best = v;
        }

        if (Best == (size_t)Answers[b]) Right++;
    }

    return (double)Right / (double)Batch;
}

void PrintBar(double Value, int Width)
{
    const int Bars = (int)(Value * Width + 0.5);
    for (int i = 0; i < Width; i++)
    {
        printf("%s", (i < Bars) ? "#" : ".");
    }
}

} // namespace

int main(void)
{
    printf("D8. 그로킹 재현\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ---- D8-1. 문제 ----
    printf("[D8-1] (a + b) mod %d 를 외우게 한다\n\n", P);

    printf("  입력  [a, b, =]     세 자리\n");
    printf("  정답  (a + b) %% %d  마지막 자리에서만 채점한다\n\n", P);

    printf("  쌍은 모두 %d x %d = %d 개다.\n", P, P, P * P);

    // 모든 쌍을 만들고 섞는다.
    std::vector<uint32_t> AllTokens;
    std::vector<uint32_t> AllAnswers;

    for (int a = 0; a < P; a++)
    {
        for (int b = 0; b < P; b++)
        {
            AllTokens.push_back((uint32_t)a);
            AllTokens.push_back((uint32_t)b);
            AllTokens.push_back((uint32_t)EQUALS);
            AllAnswers.push_back((uint32_t)((a + b) % P));
        }
    }

    {
        FRandom Shuffle;
        RandomSeed(&Shuffle, BOOK_SEED);

        for (size_t i = AllAnswers.size(); i-- > 1; )
        {
            const size_t j = (size_t)RandomBelow(&Shuffle, i + 1);

            for (int k = 0; k < LENGTH; k++)
            {
                const uint32_t Keep = AllTokens[i * LENGTH + k];
                AllTokens[i * LENGTH + k] = AllTokens[j * LENGTH + k];
                AllTokens[j * LENGTH + k] = Keep;
            }

            const uint32_t Keep = AllAnswers[i];
            AllAnswers[i] = AllAnswers[j];
            AllAnswers[j] = Keep;
        }
    }

    const size_t TrainCount = AllAnswers.size() * TRAIN_PERCENT / 100;
    const size_t ValidCount = AllAnswers.size() - TrainCount;

    std::vector<uint32_t> TrainTokens(AllTokens.begin(),
                                      AllTokens.begin() + TrainCount * LENGTH);
    std::vector<uint32_t> TrainAnswers(AllAnswers.begin(),
                                       AllAnswers.begin() + TrainCount);

    std::vector<uint32_t> ValidTokens(AllTokens.begin() + TrainCount * LENGTH,
                                      AllTokens.end());
    std::vector<uint32_t> ValidAnswers(AllAnswers.begin() + TrainCount,
                                       AllAnswers.end());

    printf("  그중 %d%% 인 %zu 개로만 훈련하고,\n", TRAIN_PERCENT, TrainCount);
    printf("  나머지 %zu 개는 **한 번도 안 보여준다.**\n\n", ValidCount);

    printf("  외우기만 하면 훈련 정확도는 100%% 가 된다.\n");
    printf("  안 본 쌍을 맞히려면 **규칙을 알아야 한다.**\n\n");

    CHECK(TrainCount + ValidCount == (size_t)(P * P));

    // ---- D8-2. 모델 ----
    FModelConfig Config;
    Config.Vocab = VOCAB;
    Config.Model = MODEL;
    Config.Heads = HEADS;
    Config.Hidden = HIDDEN;
    Config.Layers = LAYERS;
    Config.MaxLength = LENGTH;

    FTransformer Model(Config);
    {
        FRandom Init;
        RandomSeed(&Init, BOOK_SEED + 7);
        Model.Init(Init);
    }

    FTransformerGrad Grad;
    Grad.Init(Model);

    std::vector<Real*> Parameters, Gradients;
    CollectParameters(Model, Parameters);
    CollectGradients(Grad, Gradients);

    FAdamW Adam;
    FAdamWConfig AdamConfig;
    AdamConfig.Rate = 1e-3;
    AdamConfig.Beta1 = 0.9;
    AdamConfig.Beta2 = 0.98;
    AdamConfig.WeightDecay = 1.0;
    Adam.Init(Parameters.size(), AdamConfig);

    printf("[D8-2] 아주 작은 모델, 아주 센 가중치 감쇠\n\n");
    printf("  층 %d, 모델차원 %d, 헤드 %d, 앞먹임 %d\n", LAYERS, MODEL, HEADS,
           HIDDEN);
    printf("  파라미터 %zu개\n\n", Model.ParameterCount());
    printf("  학습률 %.0e, 가중치 감쇠 **%.1f**, 묶음 = 훈련 집합 전체\n\n",
           AdamConfig.Rate, AdamConfig.WeightDecay);

    printf("  감쇠 %.1f 은 보통 쓰는 값(0.01)의 100배다. 이것이 핵심이다.\n",
           AdamConfig.WeightDecay);
    printf("  **외운 해답은 무겁고, 규칙은 가볍다.** 감쇠가 무거운 쪽을\n");
    printf("  계속 밀어낸다. D8 은 그 밀어냄이 언제 이기는지를 보는 장이다.\n\n");

    // ---- D8-3. 오래 본다 ----
    printf("[D8-3] %d 걸음\n\n", STEPS);

    printf("  ");
    PrintPaddedRight("걸음", 8);
    PrintPaddedRight("훈련 손실", 12);
    PrintPaddedRight("검증 손실", 12);
    PrintPaddedRight("훈련", 8);
    PrintPaddedRight("검증", 8);
    printf("  검증 정확도\n");

    FILE* Out = fopen(CURVE_PATH, "wb");
    if (Out != NULL)
    {
        fprintf(Out, "Step,TrainLoss,ValidLoss,TrainAccuracy,ValidAccuracy\n");
    }

    int MemorizedAt = -1;     // 훈련 정확도가 처음 99% 를 넘은 걸음
    int GrokkedAt = -1;       // 검증 정확도가 처음 90% 를 넘은 걸음

    double FirstValidAfterMemorize = 0.0;

    auto Begin = std::chrono::steady_clock::now();

    for (int Step = 0; Step <= STEPS; Step++)
    {
        Grad.Zero();

        FModelTrace Trace;
        FTensor Logits = TransformerForwardTrace(Model, TrainTokens.data(),
                                                 TrainCount, LENGTH, Trace);

        const double TrainLoss = LastLoss(Logits, TrainAnswers.data());
        const double TrainAccuracy = Accuracy(Logits, TrainAnswers.data());

        if (MemorizedAt < 0 && TrainAccuracy > 0.99)
        {
            MemorizedAt = Step;
        }

        // 검증은 채점만 한다. 역전파하지 않는다.
        double ValidLoss = 0.0;
        double ValidAccuracy = 0.0;

        if (Step % REPORT == 0 || Step == STEPS)
        {
            FModelTrace Scratch;
            FTensor V = TransformerForwardTrace(Model, ValidTokens.data(),
                                                ValidCount, LENGTH, Scratch);

            ValidLoss = LastLoss(V, ValidAnswers.data());
            ValidAccuracy = Accuracy(V, ValidAnswers.data());

            if (MemorizedAt >= 0 && FirstValidAfterMemorize == 0.0)
            {
                FirstValidAfterMemorize = ValidAccuracy;
            }

            if (GrokkedAt < 0 && ValidAccuracy > 0.90)
            {
                GrokkedAt = Step;
            }

            char A[32];
            printf("  ");
            snprintf(A, sizeof(A), "%d", Step);
            PrintPaddedRight(A, 8);
            snprintf(A, sizeof(A), "%.4f", TrainLoss);
            PrintPaddedRight(A, 12);
            snprintf(A, sizeof(A), "%.4f", ValidLoss);
            PrintPaddedRight(A, 12);
            snprintf(A, sizeof(A), "%.0f%%", TrainAccuracy * 100.0);
            PrintPaddedRight(A, 8);
            snprintf(A, sizeof(A), "%.0f%%", ValidAccuracy * 100.0);
            PrintPaddedRight(A, 8);
            printf("  ");
            PrintBar(ValidAccuracy, 40);
            printf("\n");
            fflush(stdout);

            if (Out != NULL)
            {
                fprintf(Out, "%d,%.9g,%.9g,%.9g,%.9g\n", Step, TrainLoss,
                        ValidLoss, TrainAccuracy, ValidAccuracy);
            }
        }

        if (Step == STEPS) break;

        FTensor Upstream = LastBackward(Logits, TrainAnswers.data());
        TransformerBackward(Model, Trace, Upstream, Grad);

        ClipGradients(Gradients.data(), Gradients.size(), 1.0);

        const double Rate = ScheduleRate(AdamConfig.Rate, (size_t)Step, 100,
                                         (size_t)STEPS * 10, 1.0);

        Adam.Step(Parameters.data(), (const Real* const*)Gradients.data(),
                  Parameters.size(), Rate);
    }

    const double Elapsed = Seconds(Begin);
    if (Out != NULL) fclose(Out);

    printf("\n  %d 걸음에 %.1f초\n\n", STEPS, Elapsed);

    // ---- D8-4. 무엇을 봤는가 ----
    printf("[D8-4] 무엇을 본 것인가\n\n");

    if (MemorizedAt >= 0)
    {
        printf("  훈련 정확도가 99%% 를 넘은 걸음 = %d\n", MemorizedAt);
    }
    else
    {
        printf("  훈련 정확도가 99%% 를 못 넘었다.\n");
    }

    if (GrokkedAt >= 0)
    {
        printf("  검증 정확도가 90%% 를 넘은 걸음 = %d\n\n", GrokkedAt);
        printf("  **외우고 나서 %d 걸음 뒤에 이해했다.**\n\n",
               GrokkedAt - MemorizedAt);
    }
    else
    {
        printf("  검증 정확도가 90%% 를 못 넘었다. 더 돌려야 한다.\n\n");
    }

    CHECK(MemorizedAt >= 0);
    CHECK(GrokkedAt > MemorizedAt);

    printf("  이 사이가 그로킹이다. 훈련 곡선만 보면 %d 걸음에서\n",
           MemorizedAt);
    printf("  이미 끝난 것처럼 보인다. 손실도 정확도도 더 좋아질 데가 없다.\n\n");

    printf("  **그런데 안에서는 계속 무언가가 일어나고 있었다.**\n\n");

    printf("  그 무언가가 가중치 감쇠다. 외운 해답은 큰 가중치를 쓰고,\n");
    printf("  규칙은 작은 가중치로도 된다. 훈련 손실이 이미 0 이라\n");
    printf("  그래디언트가 거의 없으니, 감쇠만 남아서 계속 민다.\n\n");

    printf("  곡선은 %s 에 남겼다.\n\n", CURVE_PATH);

    // ---- D8-5. 이 장이 D 파트의 마지막인 이유 ----
    printf("[D8-5] 이 장이 마지막인 이유\n\n");

    printf("  그로킹을 관측하려면 세 가지가 필요했다.\n\n");

    printf("    1. **오래 본다.** %d 걸음에서 껐으면 아무것도 못 봤다\n",
           MemorizedAt + REPORT);
    printf("    2. **훈련과 검증을 따로 본다.** 훈련만 보면 %d 걸음부터\n",
           MemorizedAt);
    printf("       그냥 평평한 선이다\n");
    printf("    3. **버그가 아니라는 확신.** 손실이 한참 안 움직이면\n");
    printf("       보통은 코드를 의심한다\n\n");

    printf("  세 번째가 이 교재의 값이다. D6 의 gradcheck 이 통과했고,\n");
    printf("  D7 의 손실 곡선이 NumPy 와 1e-15 까지 맞았다.\n");
    printf("  **그래서 \"이건 원래 이런 것\"이라고 말할 수 있다.**\n\n");

    printf("  검증 도구가 없으면 이 현상은 관측되지 않는다. 관측되기 전에\n");
    printf("  꺼진다. 그로킹이 2022년에야 이름을 얻은 이유이기도 하다.\n\n");

    return ReportResult();
}
