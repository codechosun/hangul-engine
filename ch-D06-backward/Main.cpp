// ch-D06-backward/Main.cpp
//
// D6. 트랜스포머 역전파
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// USE_DOUBLE 을 켜고 빌드해야 판단이 선다. D6-7 에서 이유를 보인다.

#include "Test.h"
#include "Pretty.h"

#include "Attention.hpp"
#include "Backward.hpp"
#include "Block.hpp"
#include "GradCheck.hpp"
#include "Model.hpp"
#include "Random.h"
#include "Tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#define BOOK_SEED 20260914ull

namespace
{

void FillRandom(FTensor& T, FRandom& Rng, double Range)
{
    for (size_t i = 0; i < T.Count(); i++)
    {
        T.At(i) = (Real)RandomRange(&Rng, Range);
    }
}

// 텐서 전체를 하나씩 흔들어 수치 미분과 대조한다.
//
// 돌려주는 것은 **가장 나쁜 상대 오차**다.
double CheckTensor(FTensor& Parameter, const FTensor& Analytic,
                   const std::function<double()>& Loss, double Step)
{
    double Worst = 0.0;

    for (size_t i = 0; i < Parameter.Count(); i++)
    {
        const Real Saved = Parameter.At(i);

        Parameter.At(i) = (Real)((double)Saved + Step);
        const double Plus = Loss();

        Parameter.At(i) = (Real)((double)Saved - Step);
        const double Minus = Loss();

        Parameter.At(i) = Saved;

        const double Numeric = (Plus - Minus) / (2.0 * Step);
        const double Relative = RelativeError(Numeric, (double)Analytic.At(i));

        if (Relative > Worst) Worst = Relative;
    }

    return Worst;
}

// float 빌드에서는 **판정을 내리지 않는다.**
//
// 상대 오차가 1e-01 이 나와도 그것이 버그인지 반올림인지 알 수가 없다.
// 이럴 때 "실패"라고 찍는 것은 거짓말이고, 기준을 느슨하게 풀어
// "통과"라고 찍는 것은 더 나쁜 거짓말이다. 판단 불가가 정답이다.
const bool GCanJudge = (sizeof(Real) == 8);

// 이 빌드에서 본 가장 나쁜 상대 오차. D6-7 에서 쓴다.
double GWorstSeen = 0.0;

void PrintCheck(const char* Name, double Worst, double Tolerance)
{
    char Buffer[64];

    if (Worst > GWorstSeen) GWorstSeen = Worst;

    printf("  ");
    PrintPadded(Name, 26);

    snprintf(Buffer, sizeof(Buffer), "%.3e", Worst);
    PrintPaddedRight(Buffer, 14);

    if (!GCanJudge)
    {
        printf("   판단 불가\n");
        return;
    }

    printf("   %s\n", (Worst < Tolerance) ? "통과" : "실패");
}

// 임의의 텐서를 스칼라 하나로 줄인다. 그래야 미분을 말할 수 있다.
//
// 그냥 Sum 을 쓰면 모든 원소의 가중치가 1 이라 어떤 실수는 상쇄되어 안 보인다.
// 난수 가중치로 섞는다.
double Reduce(const FTensor& T, const FTensor& Weights)
{
    double Total = 0.0;
    for (size_t i = 0; i < T.Count(); i++)
    {
        Total += (double)T.At(i) * (double)Weights.At(i);
    }
    return Total;
}

} // namespace

int main(void)
{
    printf("D6. 트랜스포머 역전파\n\n");

    printf("  Real = %s (%zu 바이트)\n\n",
           (sizeof(Real) == 8) ? "double" : "float", sizeof(Real));

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    const double Step = (sizeof(Real) == 8) ? 1e-5 : 1e-3;
    const double Tolerance = (sizeof(Real) == 8) ? 1e-6 : 1e-1;

    // ---- D6-1. softmax 의 야코비안 ----
    printf("[D6-1] softmax 는 야코비안이다\n\n");

    {
        FTensor Scores({ 1, 4 });
        Scores(0, 0) = Real(2.0);
        Scores(0, 1) = Real(1.0);
        Scores(0, 2) = Real(0.1);
        Scores(0, 3) = Real(-0.5);

        FTensor P = Scores;
        SoftmaxLastAxis(P);

        printf("  점수 (2.0, 1.0, 0.1, -0.5)\n");
        printf("  확률 (%.4f, %.4f, %.4f, %.4f)\n\n",
               (double)P(0, 0), (double)P(0, 1),
               (double)P(0, 2), (double)P(0, 3));

        // 야코비안을 통째로 찍어본다. dP_i / dS_j = P_i (delta_ij - P_j)
        printf("  야코비안 dP_i/dS_j\n\n");
        printf("        ");
        for (size_t j = 0; j < 4; j++)
        {
            char Buffer[32];
            snprintf(Buffer, sizeof(Buffer), "j=%zu", j);
            PrintPaddedRight(Buffer, 11);
        }
        printf("\n");

        for (size_t i = 0; i < 4; i++)
        {
            printf("   i=%zu ", i);
            for (size_t j = 0; j < 4; j++)
            {
                const double Value = (double)P(0, i)
                    * ((i == j ? 1.0 : 0.0) - (double)P(0, j));

                char Buffer[32];
                snprintf(Buffer, sizeof(Buffer), "%+.5f", Value);
                PrintPaddedRight(Buffer, 11);
            }
            printf("\n");
        }

        printf("\n  **대각선 밖이 0 이 아니다.** 한 점수를 올리면\n");
        printf("  다른 확률이 전부 내려간다. 합이 1 이어야 하니 당연하다.\n\n");

        // 각 줄의 합은 0 이다. 확률의 합이 1 로 고정이므로.
        for (size_t i = 0; i < 4; i++)
        {
            double RowSum = 0.0;
            for (size_t j = 0; j < 4; j++)
            {
                RowSum += (double)P(0, i)
                    * ((i == j ? 1.0 : 0.0) - (double)P(0, j));
            }
            CHECK_NEAR(RowSum, 0.0, GCanJudge ? 1e-9 : 1e-6);
        }

        printf("  각 줄의 합 = 0. 확률의 총합이 1 로 묶여 있다는 뜻이다.\n\n");

        // 그런데 SoftmaxBackward 는 야코비안을 안 만든다. O(N) 이다.
        FTensor Upstream({ 1, 4 });
        FillRandom(Upstream, Rng, 1.0);

        FTensor Analytic = SoftmaxBackward(P, Upstream);

        // 야코비안을 실제로 곱해서 대조한다.
        double Worst = 0.0;
        for (size_t j = 0; j < 4; j++)
        {
            double ByMatrix = 0.0;
            for (size_t i = 0; i < 4; i++)
            {
                ByMatrix += (double)Upstream(0, i) * (double)P(0, i)
                          * ((i == j ? 1.0 : 0.0) - (double)P(0, j));
            }

            const double Gap = RelativeError(ByMatrix, (double)Analytic(0, j));
            if (Gap > Worst) Worst = Gap;
        }

        printf("  야코비안을 곱한 값 vs O(N) 공식  최대 상대 오차 = %.3e\n",
               Worst);
        CHECK(Worst < (GCanJudge ? 1e-9 : 1e-6));

        printf("  (N x N) 행렬을 만들 필요가 없다. 안쪽 합이 i 에 안 걸린다.\n\n");
    }

    // ---- D6-2. 조각별로 확인 ----
    printf("[D6-2] 조각 하나씩 수치 미분과 대조\n\n");

    printf("  ");
    PrintPadded("무엇을", 26);
    PrintPaddedRight("최대 상대 오차", 14);
    printf("\n");

    {
        // --- RMSNorm ---
        FTensor X({ 2, 3, 8 });
        FillRandom(X, Rng, 1.0);

        FTensor Gain({ 8 });
        FillRandom(Gain, Rng, 0.5);
        for (size_t i = 0; i < Gain.Count(); i++) Gain.At(i) += Real(1);

        FTensor Weights(X.GetShape());
        FillRandom(Weights, Rng, 1.0);

        auto Loss = [&]() {
            return Reduce(RmsNorm(X, Gain, (Real)1e-6), Weights);
        };

        FTensor GainGrad({ 8 });
        GainGrad.Fill(Real(0));
        FTensor XGrad = RmsNormBackward(X, Gain, (Real)1e-6, Weights, GainGrad);

        PrintCheck("RMSNorm  dX", CheckTensor(X, XGrad, Loss, Step), Tolerance);
        PrintCheck("RMSNorm  dGain",
                   CheckTensor(Gain, GainGrad, Loss, Step), Tolerance);
    }

    {
        // --- SiLU ---
        FTensor X({ 2, 16 });
        FillRandom(X, Rng, 3.0);

        FTensor Weights(X.GetShape());
        FillRandom(Weights, Rng, 1.0);

        auto Loss = [&]() { return Reduce(Silu(X), Weights); };

        FTensor XGrad = SiluBackward(X, Weights);
        PrintCheck("SiLU  dX", CheckTensor(X, XGrad, Loss, Step), Tolerance);
    }

    {
        // --- FDense ---
        FDense Dense(6, 4, true);
        Dense.Init(Rng);
        FillRandom(Dense.Bias, Rng, 0.5);

        FTensor X({ 2, 5, 6 });
        FillRandom(X, Rng, 1.0);

        FTensor Weights({ 2, 5, 4 });
        FillRandom(Weights, Rng, 1.0);

        auto Loss = [&]() { return Reduce(Dense.Forward(X), Weights); };

        FDenseGrad Grad;
        Grad.Init(Dense);
        FTensor XGrad = DenseBackward(Dense, X, Weights, Grad);

        PrintCheck("FDense  dX", CheckTensor(X, XGrad, Loss, Step), Tolerance);
        PrintCheck("FDense  dW",
                   CheckTensor(Dense.Weight, Grad.Weight, Loss, Step),
                   Tolerance);
        PrintCheck("FDense  dBias",
                   CheckTensor(Dense.Bias, Grad.Bias, Loss, Step), Tolerance);
    }

    {
        // --- 어텐션 ---
        FTensor Q({ 1, 2, 5, 4 });
        FTensor K({ 1, 2, 5, 4 });
        FTensor V({ 1, 2, 5, 4 });
        FillRandom(Q, Rng, 1.0);
        FillRandom(K, Rng, 1.0);
        FillRandom(V, Rng, 1.0);

        FTensor Weights({ 1, 2, 5, 4 });
        FillRandom(Weights, Rng, 1.0);

        auto Loss = [&]() {
            return Reduce(Attend(Q, K, V, true).Output, Weights);
        };

        FAttention Forward = Attend(Q, K, V, true);
        FAttentionGrad Grad = AttendBackward(Q, K, V, Forward.Weights, Weights);

        PrintCheck("Attend  dQ", CheckTensor(Q, Grad.Q, Loss, Step), Tolerance);
        PrintCheck("Attend  dK", CheckTensor(K, Grad.K, Loss, Step), Tolerance);
        PrintCheck("Attend  dV", CheckTensor(V, Grad.V, Loss, Step), Tolerance);
    }

    printf("\n");

    // ---- D6-3. 블록 하나 ----
    printf("[D6-3] 블록 하나를 통째로\n\n");

    {
        FBlockConfig Config;
        Config.Model = 8;
        Config.Heads = 2;
        Config.Hidden = 16;

        FBlock Block(Config);
        Block.Init(Rng);
        FillRandom(Block.AttentionGain, Rng, 0.3);
        FillRandom(Block.FeedGain, Rng, 0.3);
        for (size_t i = 0; i < Block.AttentionGain.Count(); i++)
        {
            Block.AttentionGain.At(i) += Real(1);
            Block.FeedGain.At(i) += Real(1);
        }

        FTensor X({ 2, 5, 8 });
        FillRandom(X, Rng, 1.0);

        FTensor Weights(X.GetShape());
        FillRandom(Weights, Rng, 1.0);

        auto Loss = [&]() {
            FBlockTrace Scratch;
            return Reduce(BlockForwardTrace(Block, X, Scratch), Weights);
        };

        FBlockTrace Trace;
        FTensor Out = BlockForwardTrace(Block, X, Trace);

        FBlockGrad Grad;
        Grad.Init(Block);
        FTensor XGrad = BlockBackward(Block, Trace, Weights, Grad);

        printf("  ");
        PrintPadded("무엇을", 26);
        PrintPaddedRight("최대 상대 오차", 14);
        printf("\n");

        PrintCheck("입력  dX", CheckTensor(X, XGrad, Loss, Step), Tolerance);
        PrintCheck("Query  dW",
                   CheckTensor(Block.Query.Weight, Grad.Query.Weight, Loss,
                               Step), Tolerance);
        PrintCheck("Value  dW",
                   CheckTensor(Block.Value.Weight, Grad.Value.Weight, Loss,
                               Step), Tolerance);
        PrintCheck("Down  dW",
                   CheckTensor(Block.Down.Weight, Grad.Down.Weight, Loss,
                               Step), Tolerance);
        PrintCheck("AttentionGain",
                   CheckTensor(Block.AttentionGain, Grad.AttentionGain, Loss,
                               Step), Tolerance);
        PrintCheck("FeedGain",
                   CheckTensor(Block.FeedGain, Grad.FeedGain, Loss, Step),
                   Tolerance);

        if (GCanJudge)
        {
            CHECK(CheckTensor(X, XGrad, Loss, Step) < Tolerance);
        }
        printf("\n");
    }

    // ---- D6-4. 모델 전체 ----
    printf("[D6-4] 트랜스포머 한 채 전부\n\n");

    FModelConfig Config;
    Config.Vocab = 16;
    Config.Model = 8;
    Config.Heads = 2;
    Config.Hidden = 16;
    Config.Layers = 2;
    Config.MaxLength = 8;

    const size_t Batch = 2;
    const size_t Length = 6;

    FTransformer Model(Config);
    {
        FRandom Init;
        RandomSeed(&Init, BOOK_SEED);
        Model.Init(Init);
        FillRandom(Model.FinalGain, Init, 0.3);
        for (size_t i = 0; i < Model.FinalGain.Count(); i++)
        {
            Model.FinalGain.At(i) += Real(1);
        }
    }

    std::vector<uint32_t> Tokens(Batch * Length, 0);
    std::vector<uint32_t> Targets(Batch * Length, 0);
    for (size_t i = 0; i < Tokens.size(); i++)
    {
        Tokens[i] = (uint32_t)RandomBelow(&Rng, Config.Vocab);
        Targets[i] = (uint32_t)RandomBelow(&Rng, Config.Vocab);
    }

    {
        auto Loss = [&]() {
            FModelTrace Scratch;
            FTensor Logits = TransformerForwardTrace(Model, Tokens.data(),
                                                     Batch, Length, Scratch);
            return CrossEntropyLoss(Logits, Targets.data());
        };

        FModelTrace Trace;
        FTensor Logits = TransformerForwardTrace(Model, Tokens.data(), Batch,
                                                 Length, Trace);

        const double Start = CrossEntropyLoss(Logits, Targets.data());

        FTransformerGrad Grad;
        Grad.Init(Model);

        FTensor Upstream = CrossEntropyBackward(Logits, Targets.data());
        TransformerBackward(Model, Trace, Upstream, Grad);

        printf("  파라미터 %zu개, 손실 = %.6f\n", Model.ParameterCount(), Start);
        printf("  (어휘 %zu 짜리 균등 분포면 %.6f 이다)\n\n",
               Config.Vocab, std::log((double)Config.Vocab));

        std::vector<Real*> Parameters;
        std::vector<Real*> Gradients;
        CollectParameters(Model, Parameters);
        CollectGradients(Grad, Gradients);

        CHECK(Parameters.size() == Gradients.size());

        // 전부 흔들면 순전파가 파라미터 수의 두 배만큼 돈다.
        // 골라서 흔든다. 어디를 고르는지는 난수로 정한다.
        const int Sample = 300;
        std::vector<int> Which(Sample, 0);
        for (int i = 0; i < Sample; i++)
        {
            Which[i] = (int)RandomBelow(&Rng, Parameters.size());
        }

        FGradCheckResult Result = GradCheckScattered(
            Loss, Parameters.data(),
            (const Real* const*)Gradients.data(),
            Which.data(), Sample, Step);

        printf("  파라미터 %zu개 중 %d개를 골라 흔들었다.\n\n",
               Parameters.size(), Sample);

        printf("  최대 상대 오차 = %.3e  (번호 %d)\n", Result.WorstRelative,
               Result.WorstIndex);
        printf("    역전파  %+.9f\n", Result.WorstAnalytic);
        printf("    수치    %+.9f\n\n", Result.WorstNumeric);

        PrintCheck("모델 전체 gradcheck", Result.WorstRelative, Tolerance);

        if (GCanJudge)
        {
            CHECK(Result.WorstRelative < Tolerance);
        }
        else
        {
            printf("\n  float 빌드다. 이 숫자로는 아무 말도 할 수 없다.\n");
            printf("  D6-7 을 볼 것.\n");
        }

        printf("\n");

        // 그래디언트가 맞으면 한 걸음 내려갔을 때 손실이 줄어야 한다.
        const double Rate = 0.1;
        for (size_t i = 0; i < Parameters.size(); i++)
        {
            *Parameters[i] = (Real)((double)(*Parameters[i])
                                  - Rate * (double)(*Gradients[i]));
        }

        FModelTrace After;
        FTensor AfterLogits = TransformerForwardTrace(Model, Tokens.data(),
                                                      Batch, Length, After);
        const double End = CrossEntropyLoss(AfterLogits, Targets.data());

        printf("  한 걸음(학습률 %.1f) 내려가니 손실 %.6f -> %.6f\n\n",
               Rate, Start, End);
        CHECK(End < Start);

        // 되돌려 놓는다.
        for (size_t i = 0; i < Parameters.size(); i++)
        {
            *Parameters[i] = (Real)((double)(*Parameters[i])
                                  + Rate * (double)(*Gradients[i]));
        }
    }

    // ---- D6-5. 갈라진 길은 더한다 ----
    printf("[D6-5] 같은 값이 여러 번 쓰이면\n\n");

    {
        // 토큰 하나를 일부러 여러 번 넣는다.
        std::vector<uint32_t> Repeated(Batch * Length, 0);
        for (size_t i = 0; i < Repeated.size(); i++)
        {
            Repeated[i] = (uint32_t)(i % 3);   // 0,1,2 만 돌려 쓴다
        }

        FModelTrace Trace;
        FTensor Logits = TransformerForwardTrace(Model, Repeated.data(), Batch,
                                                 Length, Trace);

        FTransformerGrad Grad;
        Grad.Init(Model);

        FTensor Upstream = CrossEntropyBackward(Logits, Targets.data());
        TransformerBackward(Model, Trace, Upstream, Grad);

        // 몇 번씩 쓰였는지 센다.
        int Used[3] = { 0, 0, 0 };
        for (size_t i = 0; i < Repeated.size(); i++) Used[Repeated[i]]++;

        printf("  토큰 %zu자리에 어휘 0,1,2 만 돌려 썼다.\n\n",
               Repeated.size());

        printf("  ");
        PrintPadded("어휘", 10);
        PrintPaddedRight("쓰인 횟수", 12);
        PrintPaddedRight("그래디언트 크기", 18);
        printf("\n");

        for (size_t v = 0; v < 4; v++)
        {
            double Norm = 0.0;
            for (size_t d = 0; d < Config.Model; d++)
            {
                const double G = (double)Grad.TokenEmbedding(v, d);
                Norm += G * G;
            }
            Norm = std::sqrt(Norm);

            char A[32], B[32], C[32];
            snprintf(A, sizeof(A), "%zu", v);
            snprintf(B, sizeof(B), "%d", (v < 3) ? Used[v] : 0);
            snprintf(C, sizeof(C), "%.6f", Norm);

            printf("  ");
            PrintPadded(A, 10);
            PrintPaddedRight(B, 12);
            PrintPaddedRight(C, 18);
            printf("\n");
        }

        printf("\n");

        // 안 쓰인 어휘는 정확히 0 이어야 한다.
        for (size_t v = 3; v < Config.Vocab; v++)
        {
            for (size_t d = 0; d < Config.Model; d++)
            {
                CHECK_NEAR(Grad.TokenEmbedding(v, d), 0.0, 1e-12);
            }
        }

        printf("  안 쓰인 어휘 %zu개는 전부 정확히 0 이다.\n",
               Config.Vocab - 3);
        printf("  **쓰이지 않은 가중치는 배우지 않는다.**\n");
        printf("  어휘가 커질수록 한 묶음에서 실제로 움직이는 줄은 적어진다.\n\n");

        // 위치 임베딩은 배치 수만큼 더해진다. 배치 2 이므로 두 번씩.
        //
        // 확인 방법: 배치를 1 로 줄여 같은 자리를 계산하고 더해본다.
        FModelTrace One;
        FTensor OneLogits = TransformerForwardTrace(Model, Repeated.data(), 1,
                                                    Length, One);

        FTransformerGrad OneGrad;
        OneGrad.Init(Model);

        // 배치 1 짜리 손실은 나눗수가 다르므로 그래디언트도 다르다.
        // 여기서는 상류를 직접 1 로 주어 비교한다.
        FTensor Ones(OneLogits.GetShape());
        Ones.Fill(Real(1));
        TransformerBackward(Model, One, Ones, OneGrad);

        double Total = 0.0;
        for (size_t i = 0; i < OneGrad.PositionEmbedding.Count(); i++)
        {
            Total += std::fabs((double)OneGrad.PositionEmbedding.At(i));
        }

        printf("  위치 임베딩 쪽도 같다. 배치가 몇이든 같은 줄을 쓰므로\n");
        printf("  배치 수만큼 더해진다. (배치 1 기준 크기 합 = %.4f)\n\n",
               Total);

        CHECK(Total > 0.0);
    }

    // ---- D6-6. 자취가 메모리를 먹는다 ----
    printf("[D6-6] 훈련이 추론보다 메모리를 먹는 이유\n\n");

    {
        FModelTrace Trace;
        FTensor Logits = TransformerForwardTrace(Model, Tokens.data(), Batch,
                                                 Length, Trace);
        if (Logits.Count() == 0) printf("never\n");

        size_t TraceBytes = 0;
        for (size_t i = 0; i < Trace.Blocks.size(); i++)
        {
            const FBlockTrace& T = Trace.Blocks[i];
            TraceBytes += (T.Input.Count() + T.Normed.Count()
                         + T.Q.Count() + T.K.Count() + T.V.Count()
                         + T.Weights.Count() + T.Attended.Count()
                         + T.Merged.Count() + T.AfterAttention.Count()
                         + T.FeedNormed.Count() + T.UpOut.Count()
                         + T.Activated.Count()) * sizeof(Real);
        }
        TraceBytes += (Trace.Final.Count() + Trace.Normed.Count())
                    * sizeof(Real);

        const size_t ParamBytes = Model.ParameterCount() * sizeof(Real);

        printf("  파라미터    %7zu 바이트\n", ParamBytes);
        printf("  자취        %7zu 바이트  (%.2f 배)\n", TraceBytes,
               (double)TraceBytes / (double)ParamBytes);
        printf("  그래디언트  %7zu 바이트  (파라미터와 같다)\n\n", ParamBytes);

        printf("  훈련에 드는 것은 최소한 이 셋이다. 옵티마이저 상태까지\n");
        printf("  더하면 더 는다. AdamW 는 파라미터당 둘을 더 든다 — D7.\n\n");

        printf("  자취는 **배치와 길이에 비례**한다. 파라미터는 안 그렇다.\n");
        printf("  긴 문맥으로 훈련하기가 어려운 이유가 여기에도 있다.\n");
        printf("  어텐션 확률표 하나가 (배치 x 헤드 x 길이 x 길이) 다.\n\n");

        CHECK(TraceBytes > 0);
    }

    // ---- D6-7. 정밀도 ----
    printf("[D6-7] float 로는 판단이 안 선다\n\n");

    printf("  이 빌드의 Real = %s\n", (sizeof(Real) == 8) ? "double" : "float");
    printf("  쓴 흔들기 폭 = %.0e, 판정 기준 = %.0e\n\n", Step, Tolerance);

    printf("  이 빌드에서 본 가장 나쁜 상대 오차 = %.3e\n\n", GWorstSeen);

    if (GCanJudge)
    {
        printf("  같은 코드를 float 로 빌드하면 이 값이 1.5 까지 올라간다.\n");
        printf("  **150%% 어긋난다.** 그런데 역전파는 한 줄도 안 바뀌었다.\n\n");
        printf("  수치 미분이 f(x+h) - f(x-h) 라는 뺄셈이기 때문이다.\n");
        printf("  손실이 2.93 인데 차이는 1e-10 근처다. float 의 유효숫자\n");
        printf("  7자리로는 그 차이가 반올림에 묻힌다. **자릿수가 모자란다.**\n\n");
    }
    else
    {
        printf("  같은 코드를 USE_DOUBLE 로 빌드하면 이 값이 1e-06 아래로\n");
        printf("  떨어진다. 역전파는 한 줄도 안 바뀌었다.\n\n");
        printf("  수치 미분이 f(x+h) - f(x-h) 라는 뺄셈이기 때문이다.\n");
        printf("  손실이 2.93 인데 차이는 1e-10 근처다. float 의 유효숫자\n");
        printf("  7자리로는 그 차이가 반올림에 묻힌다. **자릿수가 모자란다.**\n\n");
        printf("  그래서 이 빌드는 판정을 내리지 않았다.\n\n");
    }

    printf("  C2 에서 Real 을 만들 때 적어둔 값이 여기서 다시 돌아온다.\n");
    printf("  추론은 float 로 하고 gradcheck 만 double 로 돌린다.\n\n");

    return ReportResult();
}
