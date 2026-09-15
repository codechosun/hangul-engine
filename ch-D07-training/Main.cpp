// ch-D07-training/Main.cpp
//
// D7. 훈련 루프
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// USE_DOUBLE 을 켜고 빌드해야 정답표와 끝까지 맞는다. D7-3 참고.

#include "Test.h"
#include "Pretty.h"

#include "Backward.hpp"
#include "Block.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"
#include "Random.h"
#include "Tensor.hpp"
#include "Utf8.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define BOOK_SEED 20260914ull

#define EXPECTED_PATH "data/training_expected.csv"
#define CORPUS_PATH   "data/corpus.txt"
#define CURVE_PATH    "data/training_curve.csv"

// 정답표와 같은 작은 모델.
#define TINY_VOCAB   11
#define TINY_MODEL    8
#define TINY_HEADS    2
#define TINY_HIDDEN  16
#define TINY_LAYERS   2
#define TINY_MAX     12

#define TINY_STEPS   30

namespace
{

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

// D4 와 같은 가중치를 채운다. 파이썬 쪽과 비트까지 같다.
void BuildTiny(FTransformer& Model)
{
    FillFormula(Model.TokenEmbedding, 31, 17, 41, 20, 40.0);
    FillFormula(Model.PositionEmbedding, 23, 11, 37, 18, 36.0);
    FillFormula(Model.Head.Weight, 19, 7, 43, 21, 42.0);

    Model.FinalGain.Fill(Real(1));

    for (size_t L = 0; L < Model.Blocks.size(); L++)
    {
        FBlock& B = Model.Blocks[L];
        const int Base = (int)(L + 1) * 7;

        FillFormula(B.Query.Weight,    13 + Base,  5, 29, 14, 28.0);
        FillFormula(B.Key.Weight,      17 + Base,  3, 31, 15, 30.0);
        FillFormula(B.Value.Weight,    11 + Base,  9, 37, 18, 36.0);
        FillFormula(B.Project.Weight,  23 + Base,  1, 41, 20, 40.0);
        FillFormula(B.Up.Weight,       29 + Base, 13, 43, 21, 42.0);
        FillFormula(B.Down.Weight,     31 + Base,  7, 47, 23, 46.0);

        B.AttentionGain.Fill(Real(1));
        B.FeedGain.Fill(Real(1));
    }
}

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

// 한 걸음. 손실을 돌려주고 Grad 를 채운다.
double OneStep(const FTransformer& Model, const uint32_t* Tokens,
               const uint32_t* Targets, size_t Batch, size_t Length,
               FTransformerGrad& Grad)
{
    Grad.Zero();

    FModelTrace Trace;
    FTensor Logits = TransformerForwardTrace(Model, Tokens, Batch, Length,
                                             Trace);

    const double Loss = CrossEntropyLoss(Logits, Targets);

    FTensor Upstream = CrossEntropyBackward(Logits, Targets);
    TransformerBackward(Model, Trace, Upstream, Grad);

    return Loss;
}

} // namespace

int main(void)
{
    printf("D7. 훈련 루프\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ================================================================
    // D7-1. 맨손 경사하강으로는 어디까지 가는가
    // ================================================================
    printf("[D7-1] w = w - lr * g 로는 어디까지 가는가\n\n");

    FModelConfig TinyConfig;
    TinyConfig.Vocab = TINY_VOCAB;
    TinyConfig.Model = TINY_MODEL;
    TinyConfig.Heads = TINY_HEADS;
    TinyConfig.Hidden = TINY_HIDDEN;
    TinyConfig.Layers = TINY_LAYERS;
    TinyConfig.MaxLength = TINY_MAX;

    const uint32_t TinyTokens[6] = { 3, 7, 1, 9, 0, 4 };
    const uint32_t TinyTargets[6] = { 7, 1, 9, 0, 4, 2 };
    const size_t TinyLength = 6;

    {
        printf("  ");
        PrintPaddedRight("걸음", 8);
        PrintPaddedRight("SGD 0.05", 14);
        PrintPaddedRight("SGD 0.5", 14);
        PrintPaddedRight("AdamW 0.05", 14);
        printf("\n");

        double Curves[3][TINY_STEPS + 1];

        for (int Which = 0; Which < 3; Which++)
        {
            FTransformer Model(TinyConfig);
            BuildTiny(Model);

            FTransformerGrad Grad;
            Grad.Init(Model);

            std::vector<Real*> Parameters, Gradients;
            CollectParameters(Model, Parameters);
            CollectGradients(Grad, Gradients);

            FAdamW Adam;
            FAdamWConfig AdamConfig;
            AdamConfig.Rate = 0.05;
            Adam.Init(Parameters.size(), AdamConfig);

            for (int Step = 0; Step < TINY_STEPS; Step++)
            {
                Curves[Which][Step] = OneStep(Model, TinyTokens, TinyTargets,
                                              1, TinyLength, Grad);

                if (Which == 2)
                {
                    Adam.Step(Parameters.data(),
                              (const Real* const*)Gradients.data(),
                              Parameters.size());
                }
                else
                {
                    const double Rate = (Which == 0) ? 0.05 : 0.5;
                    for (size_t i = 0; i < Parameters.size(); i++)
                    {
                        *Parameters[i] = (Real)((double)(*Parameters[i])
                                              - Rate * (double)(*Gradients[i]));
                    }
                }
            }

            Curves[Which][TINY_STEPS] = OneStep(Model, TinyTokens, TinyTargets,
                                                1, TinyLength, Grad);
        }

        const int Show[6] = { 0, 1, 5, 10, 20, TINY_STEPS };
        for (int i = 0; i < 6; i++)
        {
            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%d", Show[i]);
            PrintPaddedRight(Buffer, 8);

            for (int Which = 0; Which < 3; Which++)
            {
                snprintf(Buffer, sizeof(Buffer), "%.6f", Curves[Which][Show[i]]);
                PrintPaddedRight(Buffer, 14);
            }
            printf("\n");
        }

        printf("\n");
        printf("  같은 학습률에서 AdamW 가 %.0f배 낮은 손실에 가 있다.\n",
               Curves[0][TINY_STEPS] / Curves[2][TINY_STEPS]);
        printf("  SGD 의 학습률을 10배 올리면 따라오는가. %.6f — 아니다.\n\n",
               Curves[1][TINY_STEPS]);

        CHECK(Curves[2][TINY_STEPS] < Curves[0][TINY_STEPS]);
        CHECK(Curves[2][TINY_STEPS] < Curves[1][TINY_STEPS]);

        printf("  **보폭 하나를 전부에게 강요하는 것이 문제다.**\n");
        printf("  임베딩과 정규화 이득은 그래디언트 크기가 자릿수로 다르다.\n");
        printf("  한쪽에 맞추면 다른 쪽이 못 가거나 튄다.\n\n");
    }

    // ================================================================
    // D7-2. AdamW 를 한 칸만 손으로
    // ================================================================
    printf("[D7-2] 파라미터 하나를 손으로 따라가기\n\n");

    {
        Real Weight = Real(1.0);
        Real Gradient = Real(0.1);

        Real* ParameterList[1] = { &Weight };
        const Real* GradientList[1] = { &Gradient };

        FAdamW Adam;
        FAdamWConfig AdamConfig;
        AdamConfig.Rate = 0.1;
        AdamConfig.WeightDecay = 0.0;
        Adam.Init(1, AdamConfig);

        printf("  그래디언트가 계속 0.1 로 같다면?\n\n");

        printf("  ");
        PrintPaddedRight("걸음", 8);
        PrintPaddedRight("가중치", 16);
        PrintPaddedRight("실제 보폭", 16);
        printf("\n");

        Real Before = Weight;
        double LastMove = 0.0;

        for (int Step = 1; Step <= 5; Step++)
        {
            Adam.Step(ParameterList, GradientList, 1);
            LastMove = (double)Weight - (double)Before;

            char Number[32], Value[32], Move[32];
            snprintf(Number, sizeof(Number), "%d", Step);
            snprintf(Value, sizeof(Value), "%.6f", (double)Weight);
            snprintf(Move, sizeof(Move), "%+.6f", LastMove);

            printf("  ");
            PrintPaddedRight(Number, 8);
            PrintPaddedRight(Value, 16);
            PrintPaddedRight(Move, 16);
            printf("\n");

            Before = Weight;
        }

        printf("\n");
        printf("  보폭이 학습률 0.1 에 딱 붙는다. 그래디언트가 0.1 이든\n");
        printf("  100 이든 마찬가지다. **m / sqrt(v) 가 크기를 지우기 때문**이다.\n\n");

        // 그래디언트가 일정하면 m^ / sqrt(v^) = 1 이므로 보폭이 정확히 학습률이다.
        CHECK_NEAR(LastMove, -0.1, 1e-6);
        CHECK_NEAR((double)Weight, 0.5, 1e-6);

        printf("  이것이 SGD 와의 결정적 차이다. SGD 의 보폭은 lr * g 이고\n");
        printf("  AdamW 의 보폭은 **lr 에 가깝다.** 크기를 안 보고 방향만 본다.\n\n");
    }

    // ================================================================
    // D7-3. 정답표와 손실 곡선 대조
    // ================================================================
    printf("[D7-3] NumPy 참조 구현과 손실 곡선 맞추기\n\n");

    {
        FILE* File = fopen(EXPECTED_PATH, "rb");
        if (File == NULL)
        {
            printf("  %s 가 없다. tools/make_training_expected.py 를 돌릴 것.\n",
                   EXPECTED_PATH);
            return 1;
        }

        char Line[512];
        (void)fgets(Line, sizeof(Line), File);

        std::vector<double> WantLoss, WantNorm, WantRate;

        while (fgets(Line, sizeof(Line), File) != NULL)
        {
            int Step = 0;
            double Loss = 0.0, Norm = 0.0, Rate = 0.0;

            if (sscanf(Line, "%d,%lf,%lf,%lf", &Step, &Loss, &Norm, &Rate) != 4)
            {
                continue;
            }

            WantLoss.push_back(Loss);
            WantNorm.push_back(Norm);
            WantRate.push_back(Rate);
        }

        fclose(File);

        printf("  정답표 %zu줄을 읽었다.\n\n", WantLoss.size());
        CHECK(WantLoss.size() == TINY_STEPS + 1);

        FTransformer Model(TinyConfig);
        BuildTiny(Model);

        FTransformerGrad Grad;
        Grad.Init(Model);

        std::vector<Real*> Parameters, Gradients;
        CollectParameters(Model, Parameters);
        CollectGradients(Grad, Gradients);

        FAdamW Adam;
        FAdamWConfig AdamConfig;
        AdamConfig.Rate = 0.05;
        AdamConfig.WeightDecay = 0.01;
        Adam.Init(Parameters.size(), AdamConfig);

        double WorstLoss = 0.0, WorstNorm = 0.0, WorstRate = 0.0;
        std::vector<double> GotLoss;

        for (int Step = 0; Step < TINY_STEPS; Step++)
        {
            const double Loss = OneStep(Model, TinyTokens, TinyTargets, 1,
                                        TinyLength, Grad);
            GotLoss.push_back(Loss);

            const double Norm = ClipGradients(Gradients.data(),
                                              Gradients.size(), 1.0);

            const double Rate = ScheduleRate(0.05, (size_t)Step, 3,
                                             TINY_STEPS, 0.1);

            Adam.Step(Parameters.data(),
                      (const Real* const*)Gradients.data(),
                      Parameters.size(), Rate);

            const double A = std::fabs(Loss - WantLoss[Step]);
            const double B = std::fabs(Norm - WantNorm[Step]);
            const double C = std::fabs(Rate - WantRate[Step]);

            if (A > WorstLoss) WorstLoss = A;
            if (B > WorstNorm) WorstNorm = B;
            if (C > WorstRate) WorstRate = C;
        }

        const double Last = OneStep(Model, TinyTokens, TinyTargets, 1,
                                    TinyLength, Grad);
        GotLoss.push_back(Last);

        const double LastGap = std::fabs(Last - WantLoss[TINY_STEPS]);
        if (LastGap > WorstLoss) WorstLoss = LastGap;

        printf("  ");
        PrintPaddedRight("걸음", 8);
        PrintPaddedRight("우리", 14);
        PrintPaddedRight("NumPy", 14);
        PrintPaddedRight("차이", 14);
        printf("\n");

        const int Show[7] = { 0, 1, 2, 3, 10, 20, TINY_STEPS };
        for (int i = 0; i < 7; i++)
        {
            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%d", Show[i]);
            PrintPaddedRight(Buffer, 8);

            snprintf(Buffer, sizeof(Buffer), "%.6f", GotLoss[Show[i]]);
            PrintPaddedRight(Buffer, 14);

            snprintf(Buffer, sizeof(Buffer), "%.6f", WantLoss[Show[i]]);
            PrintPaddedRight(Buffer, 14);

            snprintf(Buffer, sizeof(Buffer), "%.2e",
                     std::fabs(GotLoss[Show[i]] - WantLoss[Show[i]]));
            PrintPaddedRight(Buffer, 14);

            printf("\n");
        }

        printf("\n");
        printf("  %d걸음 전체에서 가장 큰 차이\n", TINY_STEPS + 1);
        printf("    손실            %.3e\n", WorstLoss);
        printf("    그래디언트 크기 %.3e\n", WorstNorm);
        printf("    학습률          %.3e\n\n", WorstRate);

        const double Allow = (sizeof(Real) == 8) ? 1e-9 : 1e-2;

        CHECK(WorstLoss < Allow);
        CHECK(WorstNorm < Allow);
        CHECK(WorstRate < 1e-12);

        printf("  손실 %.6f -> %.6f. **순전파, 역전파, 옵티마이저가\n",
               GotLoss[0], GotLoss[TINY_STEPS]);
        printf("  전부 맞아야만 %d걸음이 끝까지 붙는다.**\n\n", TINY_STEPS);
    }

    // ================================================================
    // D7-4. 클리핑
    // ================================================================
    printf("[D7-4] 어쩌다 튀는 한 걸음\n\n");

    {
        // 크기가 제각각인 가짜 그래디언트를 만든다.
        std::vector<Real> Raw(1000, Real(0));
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        for (size_t i = 0; i < Raw.size(); i++)
        {
            Raw[i] = (Real)RandomRange(&Rng, 0.01);
        }

        // 한 칸만 크게 튄다.
        Raw[500] = Real(50.0);

        std::vector<Real*> Pointers;
        for (size_t i = 0; i < Raw.size(); i++) Pointers.push_back(&Raw[i]);

        const Real BeforeBig = Raw[500];
        const Real BeforeSmall = Raw[0];

        const double Norm = ClipGradients(Pointers.data(), Pointers.size(), 1.0);

        printf("  자르기 전 전체 크기 = %.4f  (한계 1.0)\n", Norm);
        printf("  줄인 비율 = %.6f\n\n", 1.0 / Norm);

        printf("  ");
        PrintPadded("어디", 16);
        PrintPaddedRight("전", 14);
        PrintPaddedRight("후", 14);
        printf("\n");

        char A[32], B[32];
        printf("  ");
        PrintPadded("튄 칸", 16);
        snprintf(A, sizeof(A), "%.6f", (double)BeforeBig);
        snprintf(B, sizeof(B), "%.6f", (double)Raw[500]);
        PrintPaddedRight(A, 14);
        PrintPaddedRight(B, 14);
        printf("\n");

        printf("  ");
        PrintPadded("보통 칸", 16);
        snprintf(A, sizeof(A), "%.6f", (double)BeforeSmall);
        snprintf(B, sizeof(B), "%.6f", (double)Raw[0]);
        PrintPaddedRight(A, 14);
        PrintPaddedRight(B, 14);
        printf("\n\n");

        // 자른 뒤의 크기는 정확히 1 이어야 한다.
        double Square = 0.0;
        for (size_t i = 0; i < Raw.size(); i++)
        {
            Square += (double)Raw[i] * (double)Raw[i];
        }

        printf("  자른 뒤 전체 크기 = %.9f\n\n", std::sqrt(Square));
        CHECK_NEAR(std::sqrt(Square), 1.0, 1e-6);

        printf("  **방향은 그대로다.** 모든 칸에 같은 비율을 곱했으니\n");
        printf("  서로의 비도 그대로다. 크기만 잘렸다.\n\n");

        printf("  칸마다 따로 자르면(예: 각 칸을 -1~1 로) 방향이 바뀐다.\n");
        printf("  그건 다른 방향으로 걷는 것이지 조심히 걷는 게 아니다.\n\n");
    }

    // ================================================================
    // D7-5. 스케줄
    // ================================================================
    printf("[D7-5] 학습률을 어떻게 움직이는가\n\n");

    {
        printf("  워밍업 3, 전체 30, 바닥 비율 0.1, 기준 학습률 0.05\n\n");

        printf("  ");
        PrintPaddedRight("걸음", 8);
        PrintPaddedRight("학습률", 14);
        printf("  그림\n");

        for (int Step = 0; Step <= 30; Step += 3)
        {
            const double Rate = ScheduleRate(0.05, (size_t)Step, 3, 30, 0.1);

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%d", Step);
            PrintPaddedRight(Buffer, 8);
            snprintf(Buffer, sizeof(Buffer), "%.6f", Rate);
            PrintPaddedRight(Buffer, 14);

            printf("  ");
            const int Bars = (int)(Rate / 0.05 * 40.0 + 0.5);
            for (int i = 0; i < Bars; i++) printf("#");
            printf("\n");
        }

        printf("\n");

        CHECK(ScheduleRate(0.05, 0, 3, 30, 0.1) < 0.05);
        CHECK_NEAR(ScheduleRate(0.05, 2, 3, 30, 0.1), 0.05, 1e-12);
        CHECK(ScheduleRate(0.05, 29, 3, 30, 0.1) < 0.05 * 0.2);

        printf("  **처음에 천천히 가는 이유.** 훈련 시작 직후에는 Adam 의\n");
        printf("  v 가 아직 아무것도 모른다. 한두 걸음의 그래디언트만 보고\n");
        printf("  보폭을 정하니 엉뚱한 크기가 나온다.\n\n");

        printf("  **끝에서 줄이는 이유.** 보폭이 크면 최저점 주위를 맴돈다.\n");
        printf("  줄여야 안으로 들어간다.\n\n");
    }

    // ================================================================
    // D7-6. 진짜 한국어로
    // ================================================================
    printf("[D7-6] 한국어 글자를 실제로 배우게 하기\n\n");

    {
        // 코퍼스 앞부분만 읽는다. 글자 단위다.
        FILE* File = fopen(CORPUS_PATH, "rb");
        if (File == NULL)
        {
            printf("  %s 가 없다. tools/download_corpus.py 를 돌릴 것.\n\n",
                   CORPUS_PATH);
            return ReportResult();
        }

        std::vector<char> Raw(1u << 21);   // 2MB
        const size_t Got = fread(Raw.data(), 1, Raw.size(), File);
        fclose(File);

        // UTF-8 을 코드포인트로 푼다.
        std::vector<uint32_t> Points;
        size_t At = 0;
        while (At < Got)
        {
            uint32_t Code = 0;
            const int Used = Utf8Decode(Raw.data() + At, &Code);
            if (Used <= 0) break;
            At += (size_t)Used;
            Points.push_back(Code);
        }

        // 자주 나오는 글자만 어휘로 삼는다. 나머지는 버린다.
        const size_t Vocab = 256;

        std::vector<uint32_t> Unique;
        std::vector<uint64_t> Counts;
        {
            // 작은 자료라 그냥 센다. A2 의 FreqCountFile 은 파일을 받으므로
            // 여기서는 손으로 센다.
            std::vector<uint32_t> Seen(0x11000, 0);
            std::vector<uint64_t> Tally(0x11000, 0);
            for (size_t i = 0; i < Points.size(); i++)
            {
                if (Points[i] < 0x11000) Tally[Points[i]]++;
            }
            (void)Seen;

            for (size_t Rank = 0; Rank < Vocab; Rank++)
            {
                uint64_t Best = 0;
                size_t Where = 0;
                for (size_t c = 0; c < Tally.size(); c++)
                {
                    if (Tally[c] > Best) { Best = Tally[c]; Where = c; }
                }
                if (Best == 0) break;
                Unique.push_back((uint32_t)Where);
                Counts.push_back(Best);
                Tally[Where] = 0;
            }
        }

        // 코드포인트 -> 어휘 번호
        std::vector<int> ToId(0x11000, -1);
        for (size_t i = 0; i < Unique.size(); i++)
        {
            ToId[Unique[i]] = (int)i;
        }

        std::vector<uint32_t> Stream;
        uint64_t Kept = 0, Dropped = 0;
        for (size_t i = 0; i < Points.size(); i++)
        {
            const int Id = (Points[i] < 0x11000) ? ToId[Points[i]] : -1;
            if (Id < 0) { Dropped++; continue; }
            Stream.push_back((uint32_t)Id);
            Kept++;
        }

        printf("  코퍼스 앞 %zuKB -> 글자 %zu개\n", Got / 1024, Points.size());
        printf("  어휘 %zu자로 덮은 비율 = %.2f%%\n\n", Unique.size(),
               100.0 * (double)Kept / (double)(Kept + Dropped));

        // ---- 모델 ----
        FModelConfig Config;
        Config.Vocab = Unique.size();
        Config.Model = 64;
        Config.Heads = 4;
        Config.Hidden = 256;
        Config.Layers = 2;
        Config.MaxLength = 32;

        const size_t Batch = 8;
        const size_t Length = 32;
        const int Steps = 1500;

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

        printf("  모델 %zu 파라미터, 묶음 %zu x 길이 %zu, %d걸음\n\n",
               Model.ParameterCount(), Batch, Length, Steps);

        FRandom Pick;
        RandomSeed(&Pick, BOOK_SEED + 1);

        std::vector<uint32_t> Inputs(Batch * Length, 0);
        std::vector<uint32_t> Targets(Batch * Length, 0);

        std::vector<double> Curve;
        int Clipped = 0;

        FILE* Out = fopen(CURVE_PATH, "wb");
        if (Out != NULL) fprintf(Out, "Step,Loss,GradNorm,Rate\n");

        auto Begin = std::chrono::steady_clock::now();

        for (int Step = 0; Step < Steps; Step++)
        {
            // 묶음을 뽑는다.
            for (size_t b = 0; b < Batch; b++)
            {
                const uint64_t Start =
                    RandomBelow(&Pick, Stream.size() - Length - 1);

                for (size_t t = 0; t < Length; t++)
                {
                    Inputs[b * Length + t] = Stream[Start + t];
                    Targets[b * Length + t] = Stream[Start + t + 1];
                }
            }

            const double Loss = OneStep(Model, Inputs.data(), Targets.data(),
                                        Batch, Length, Grad);

            const double Norm = ClipGradients(Gradients.data(),
                                              Gradients.size(), 1.0);
            if (Norm > 1.0) Clipped++;

            const double Rate = ScheduleRate(AdamConfig.Rate, (size_t)Step, 75,
                                             (size_t)Steps, 0.1);

            Adam.Step(Parameters.data(),
                      (const Real* const*)Gradients.data(),
                      Parameters.size(), Rate);

            Curve.push_back(Loss);
            if (Out != NULL)
            {
                fprintf(Out, "%d,%.9g,%.9g,%.9g\n", Step, Loss, Norm, Rate);
            }
        }

        const double Elapsed = Seconds(Begin);
        if (Out != NULL) fclose(Out);

        printf("  ");
        PrintPaddedRight("걸음", 8);
        PrintPaddedRight("손실", 12);
        printf("  그림\n");

        for (int i = 0; i <= 10; i++)
        {
            const int Step = (i * (Steps - 1)) / 10;

            // 주위 열 걸음의 평균. 한 걸음은 묶음마다 튄다.
            double Sum = 0.0;
            int Used = 0;
            for (int k = Step - 5; k <= Step + 5; k++)
            {
                if (k < 0 || k >= Steps) continue;
                Sum += Curve[k];
                Used++;
            }
            const double Mean = Sum / Used;

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%d", Step);
            PrintPaddedRight(Buffer, 8);
            snprintf(Buffer, sizeof(Buffer), "%.4f", Mean);
            PrintPaddedRight(Buffer, 12);

            printf("  ");
            const int Bars = (int)(Mean / std::log((double)Config.Vocab)
                                 * 50.0 + 0.5);
            for (int k = 0; k < Bars; k++) printf("#");
            printf("\n");
        }

        printf("\n");

        const double Uniform = std::log((double)Config.Vocab);
        double FirstTen = 0.0, LastTen = 0.0;
        for (int i = 0; i < 10; i++) FirstTen += Curve[i];
        for (int i = 0; i < 10; i++) LastTen += Curve[Steps - 1 - i];
        FirstTen /= 10.0;
        LastTen /= 10.0;

        printf("  아무것도 모르는 모델의 손실 = ln %zu = %.4f\n",
               Config.Vocab, Uniform);
        printf("  처음 열 걸음 평균 = %.4f\n", FirstTen);
        printf("  마지막 열 걸음 평균 = %.4f\n", LastTen);
        printf("  글자당 %.3f 비트 (%.4f / ln2)\n\n", LastTen / std::log(2.0),
               LastTen);

        printf("  %d걸음에 %.1f초. 한 걸음 %.1fms\n", Steps, Elapsed,
               Elapsed / Steps * 1000.0);
        printf("  그래디언트가 잘린 걸음 = %d / %d\n\n", Clipped, Steps);

        CHECK(LastTen < FirstTen);
        CHECK(LastTen < Uniform);

        // ---- 생성해 본다 ----
        printf("  뽑아보기 (앞 글자를 주고 이어 쓰게 한다)\n\n");

        const char* Seeds[3] = { "한국", "그는", "19" };

        FRandom Tail;
        RandomSeed(&Tail, BOOK_SEED + 2);

        // Mode 0 = 늘 1등, Mode 1 = 온도 0.8 로 뽑기
        for (int Mode = 0; Mode < 2; Mode++)
        {
            printf("    %s\n", (Mode == 0) ? "늘 1등을 고르면"
                                           : "온도 0.8 로 뽑으면");

            for (int s = 0; s < 3; s++)
            {
                FKvCache Cache(Config);
                Cache.Clear();

                printf("      \"%s\" -> \"%s", Seeds[s], Seeds[s]);

                // 씨앗을 밀어넣는다.
                size_t Where = 0;
                const size_t SeedBytes = strlen(Seeds[s]);
                FTensor Logits;

                while (Where < SeedBytes)
                {
                    uint32_t Code = 0;
                    const int Used = Utf8Decode(Seeds[s] + Where, &Code);
                    if (Used <= 0) break;
                    Where += (size_t)Used;

                    const int Id = (Code < 0x11000) ? ToId[Code] : -1;
                    if (Id < 0) continue;

                    Logits = Model.Step((uint32_t)Id, Cache);
                }

                for (int i = 0; i < 24 && Cache.Length < Config.MaxLength; i++)
                {
                    size_t Pick = 0;

                    if (Mode == 0)
                    {
                        for (size_t v = 1; v < Config.Vocab; v++)
                        {
                            if (Logits(v) > Logits(Pick)) Pick = v;
                        }
                    }
                    else
                    {
                        // A5 에서 만든 것과 같은 방식. 온도로 나눈 뒤
                        // softmax 를 걸고 누적합에서 하나를 집는다.
                        const double Temperature = 0.8;

                        double Biggest = (double)Logits(0);
                        for (size_t v = 1; v < Config.Vocab; v++)
                        {
                            if ((double)Logits(v) > Biggest)
                            {
                                Biggest = (double)Logits(v);
                            }
                        }

                        double Total = 0.0;
                        std::vector<double> Weight(Config.Vocab, 0.0);
                        for (size_t v = 0; v < Config.Vocab; v++)
                        {
                            Weight[v] = std::exp(
                                ((double)Logits(v) - Biggest) / Temperature);
                            Total += Weight[v];
                        }

                        double Target = RandomUnit(&Tail) * Total;
                        for (size_t v = 0; v < Config.Vocab; v++)
                        {
                            Target -= Weight[v];
                            if (Target <= 0.0) { Pick = v; break; }
                        }
                    }

                    char Buffer[8] = { 0 };
                    const int Bytes = Utf8Encode(Unique[Pick], Buffer);
                    Buffer[Bytes] = '\0';
                    printf("%s", Buffer);

                    Logits = Model.Step((uint32_t)Pick, Cache);
                }

                printf("\"\n");
            }

            printf("\n");
        }

        printf("  **말이 되는 문장은 아니다.** %d걸음, 2MB, 파라미터 %zu개로\n",
               Steps, Model.ParameterCount());
        printf("  되는 일이 아니다. 그래도 아무것도 모르던 상태에서\n");
        printf("  글자당 %.2f 비트까지는 왔다. E3 에서 이 벽을 다룬다.\n\n",
               LastTen / std::log(2.0));

        printf("  1등만 고르면 같은 말을 되풀이한다. **A5 에서 본 그대로다.**\n");
        printf("  모델이 나빠서가 아니라 뽑는 방법이 그런 것이다.\n\n");

        printf("  손실 곡선은 %s 에 남겼다.\n\n", CURVE_PATH);

        // ---- 메모리 ----
        printf("  훈련에 든 메모리\n\n");

        const size_t ParamBytes = Model.ParameterCount() * sizeof(Real);

        printf("    파라미터        %8.2f KB\n", ParamBytes / 1024.0);
        printf("    그래디언트      %8.2f KB\n", ParamBytes / 1024.0);
        printf("    AdamW 상태      %8.2f KB  (파라미터당 double 둘)\n",
               Adam.Bytes() / 1024.0);
        printf("    합              %8.2f KB  (파라미터의 %.1f배)\n\n",
               (ParamBytes * 2 + Adam.Bytes()) / 1024.0,
               (double)(ParamBytes * 2 + Adam.Bytes()) / (double)ParamBytes);

        printf("  여기에 D6 의 자취가 더 붙는다. **추론만 할 때의 몇 배**다.\n");
        printf("  70억짜리 모델을 fp32 로 훈련하면 파라미터 28GB 에\n");
        printf("  그래디언트 28GB, 옵티마이저 상태 56GB 다. 카드 하나에 못 넣는다.\n\n");

        CHECK(Adam.Bytes() == Model.ParameterCount() * 2 * sizeof(double));
    }

    return ReportResult();
}
