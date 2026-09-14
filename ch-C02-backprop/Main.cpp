// ch-C02-backprop/Main.cpp
//
// C2. 로스와 수동 역전파
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// Real 이 float 이면 gradcheck 판정이 애매해진다.
// 전처리기 정의에 USE_DOUBLE 을 넣고 다시 빌드해 비교해볼 것.

#include "Test.h"
#include "Pretty.h"

#include "GradCheck.hpp"
#include "Matrix.hpp"
#include "Nn.hpp"
#include "Random.h"
#include "Vector.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#define BOOK_SEED 20260914ull

// 작은 망 하나. 손으로 따라갈 수 있는 크기.
#define IN_SIZE 6
#define HIDDEN_SIZE 5
#define OUT_SIZE 4

namespace
{

// 순전파에서 나온 중간값들. 역전파가 이걸 다시 본다.
struct FTrace
{
    FVector Input;
    FVector HiddenPre;     // W1 x + b1
    FVector HiddenOut;     // tanh(HiddenPre)
    FVector Scores;        // W2 h + b2
    FVector Probabilities; // softmax(Scores)
    Real Loss = Real(0);
};

FTrace Forward(const FLinear& First, const FLinear& Second,
               const FVector& X, size_t Target)
{
    FTrace Trace;

    Trace.Input = X;
    Trace.HiddenPre = First.Forward(X);
    Trace.HiddenOut = Tanh(Trace.HiddenPre);
    Trace.Scores = Second.Forward(Trace.HiddenOut);
    Trace.Probabilities = Softmax(Trace.Scores);
    Trace.Loss = CrossEntropy(Trace.Probabilities, Target);

    return Trace;
}

// 순전파의 정확한 역순이다. 줄을 거꾸로 읽으면 대응이 보인다.
void Backward(const FLinear& First, const FLinear& Second,
              const FTrace& Trace, size_t Target,
              FLinearGrad& FirstGrad, FLinearGrad& SecondGrad)
{
    FVector GradScores = SoftmaxCrossEntropyBackward(Trace.Probabilities, Target);

    FVector GradHiddenOut =
        LinearBackward(Second, Trace.HiddenOut, GradScores, SecondGrad);

    FVector GradHiddenPre = TanhBackward(Trace.HiddenOut, GradHiddenOut);

    LinearBackward(First, Trace.Input, GradHiddenPre, FirstGrad);
}

} // namespace

int main(void)
{
    printf("C2. 로스와 수동 역전파\n\n");

    // gradcheck 판정 기준.
    //
    // double 이면 1e-6 아래가 나온다. float 이면 1e-1 까지 풀어야 통과한다.
    // 그 차이가 이 장의 마지막 이야기다.
    const double Threshold = (sizeof(Real) == 8) ? 1e-6 : 1e-1;

    printf("  Real = %s (%zu 바이트), gradcheck 판정 기준 = %.0e\n\n",
           (sizeof(Real) == 8) ? "double" : "float", sizeof(Real), Threshold);

    // ---- C2-1. 미분을 수치로 구하기 ----
    printf("[C2-1] 흔들어 보면 기울기를 알 수 있다\n\n");

    {
        // f(x) = x^3 을 x = 2 에서 미분하면 3x^2 = 12 다.
        auto F = [](double X) { return X * X * X; };
        const double At = 2.0;
        const double Exact = 12.0;

        printf("  f(x) = x^3, f'(2) = 12 을 수치로 구해본다\n\n");
        printf("  ");
        PrintPaddedRight("h", 10);
        PrintPaddedRight("앞으로만", 18);
        PrintPaddedRight("오차", 12);
        PrintPaddedRight("양쪽으로", 18);
        PrintPaddedRight("오차", 12);
        printf("\n");

        const double Steps[6] = { 1e-1, 1e-2, 1e-3, 1e-5, 1e-8, 1e-12 };
        double BestCentral = 1e30;
        double BestForward = 1e30;

        for (int i = 0; i < 6; i++)
        {
            double H = Steps[i];

            double Forward = (F(At + H) - F(At)) / H;
            double Central = (F(At + H) - F(At - H)) / (2.0 * H);

            double ForwardError = std::fabs(Forward - Exact);
            double CentralError = std::fabs(Central - Exact);

            if (ForwardError < BestForward) BestForward = ForwardError;
            if (CentralError < BestCentral) BestCentral = CentralError;

            char Buffer[32];
            printf("  ");
            snprintf(Buffer, sizeof(Buffer), "%.0e", H);
            PrintPaddedRight(Buffer, 10);
            printf("%17.9f ", Forward);
            snprintf(Buffer, sizeof(Buffer), "%.2e", ForwardError);
            PrintPaddedRight(Buffer, 12);
            printf("%17.9f ", Central);
            snprintf(Buffer, sizeof(Buffer), "%.2e", CentralError);
            PrintPaddedRight(Buffer, 12);
            printf("\n");
        }

        printf("\n  양쪽으로 흔드는 쪽이 훨씬 정확하다. gradcheck 는 이쪽을 쓴다.\n");
        printf("  그리고 h 를 계속 줄인다고 좋아지지 않는다. 1e-12 에서 다시 나빠진다.\n\n");

        CHECK(BestCentral < BestForward);
    }

    // ---- C2-2. 연쇄법칙 한 단계 ----
    printf("[C2-2] 연쇄법칙을 손으로 따라가기\n\n");

    {
        // s = 2x + 1,  y = tanh(s),  L = y^2
        // dL/dx = dL/dy * dy/ds * ds/dx = 2y * (1 - y^2) * 2
        const double X = 0.5;
        const double S = 2.0 * X + 1.0;         // 2.0
        const double Y = std::tanh(S);          // 0.9640275800758169
        const double L = Y * Y;

        const double dLdy = 2.0 * Y;
        const double dyds = 1.0 - Y * Y;
        const double dsdx = 2.0;
        const double dLdx = dLdy * dyds * dsdx;

        printf("  s = 2x + 1,  y = tanh(s),  L = y^2,  x = 0.5\n");
        printf("    s = %.9f,  y = %.9f,  L = %.9f\n", S, Y, L);
        printf("    dL/dy = 2y      = %.9f\n", dLdy);
        printf("    dy/ds = 1 - y^2 = %.9f\n", dyds);
        printf("    ds/dx = 2       = %.9f\n", dsdx);
        printf("    dL/dx = 곱하면  = %.9f\n", dLdx);

        // 수치로 확인한다.
        auto Loss = [](double Xv)
        {
            double Sv = 2.0 * Xv + 1.0;
            double Yv = std::tanh(Sv);
            return Yv * Yv;
        };

        const double H = 1e-6;
        double Numeric = (Loss(X + H) - Loss(X - H)) / (2.0 * H);

        printf("    수치로 구하면   = %.9f\n\n", Numeric);

        CHECK_NEAR(dLdx, Numeric, 1e-6);

        printf("  **곱하기만 하면 된다.** 각 단계가 자기 몫의 미분을 알고 있으면\n");
        printf("  전체 미분은 그것들을 이어 곱한 것이다. 이게 연쇄법칙이고,\n");
        printf("  역전파는 이 곱셈을 뒤에서 앞으로 한 번에 처리하는 것뿐이다.\n\n");
    }

    // ---- C2-3. softmax + 교차 엔트로피가 왜 이렇게 간단한가 ----
    printf("[C2-3] dL/ds = p - y\n\n");

    {
        FVector Scores(3);
        Scores[0] = Real(1); Scores[1] = Real(2); Scores[2] = Real(3);

        const size_t Target = 1;

        FVector Probs = Softmax(Scores);
        FVector Grad = SoftmaxCrossEntropyBackward(Probs, Target);

        printf("  점수 (1, 2, 3), 정답은 1번\n");
        printf("    확률      = (%.6f, %.6f, %.6f)\n",
               (double)Probs[0], (double)Probs[1], (double)Probs[2]);
        printf("    dL/ds     = (%.6f, %.6f, %.6f)\n",
               (double)Grad[0], (double)Grad[1], (double)Grad[2]);
        printf("    손계산    = (0.090031, -0.755272, 0.665241)\n");

        CHECK_NEAR(Grad[0], 0.090030573, 1e-6);
        CHECK_NEAR(Grad[1], 0.244728471 - 1.0, 1e-6);
        CHECK_NEAR(Grad[2], 0.665240956, 1e-6);

        // 그래디언트의 합은 항상 0 이다. 확률의 합이 1 이고 정답에서 1 을 빼므로.
        double Sum = 0.0;
        for (size_t i = 0; i < Grad.Size(); i++)
        {
            Sum += (double)Grad[i];
        }
        printf("    합        = %.9f  (항상 0 이다)\n\n", Sum);

        CHECK_NEAR(Sum, 0.0, 1e-6);

        // 수치미분으로도 확인한다.
        auto Loss = [&Scores, Target]()
        {
            return (double)CrossEntropy(Softmax(Scores), Target);
        };

        Real Analytic[3] = { Grad[0], Grad[1], Grad[2] };
        Real Raw[3] = { Scores[0], Scores[1], Scores[2] };

        FGradCheckResult Check = GradCheck(
            [&Scores, &Raw, Target]()
            {
                for (size_t i = 0; i < 3; i++) Scores[i] = Raw[i];
                return (double)CrossEntropy(Softmax(Scores), Target);
            },
            Raw, Analytic, 3, 1e-4);

        printf("  gradcheck 최대 상대 오차 = %.3e\n\n", Check.WorstRelative);
        CHECK(Check.WorstRelative < Threshold);

        (void)Loss;
    }

    // ---- C2-4. 망 전체의 gradcheck ----
    printf("[C2-4] 2층 망 전체를 검사한다\n\n");

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    FLinear First(IN_SIZE, HIDDEN_SIZE);
    FLinear Second(HIDDEN_SIZE, OUT_SIZE);
    First.InitUniform(Rng, Real(0.8));
    Second.InitUniform(Rng, Real(0.8));

    for (size_t i = 0; i < First.Bias.Size(); i++)
    {
        First.Bias[i] = (Real)RandomRange(&Rng, 0.5);
    }
    for (size_t i = 0; i < Second.Bias.Size(); i++)
    {
        Second.Bias[i] = (Real)RandomRange(&Rng, 0.5);
    }

    FVector X(IN_SIZE);
    for (size_t i = 0; i < X.Size(); i++)
    {
        X[i] = (Real)RandomRange(&Rng, 1.0);
    }

    const size_t Target = 2;

    FLinearGrad FirstGrad(IN_SIZE, HIDDEN_SIZE);
    FLinearGrad SecondGrad(HIDDEN_SIZE, OUT_SIZE);
    FirstGrad.Zero();
    SecondGrad.Zero();

    {
        FTrace Trace = Forward(First, Second, X, Target);
        Backward(First, Second, Trace, Target, FirstGrad, SecondGrad);

        printf("  손실 = %.9f\n", (double)Trace.Loss);
        printf("  가중치 %zu개 + 치우침 %zu개 = %zu개를 전부 검사한다\n\n",
               First.Weight.Count() + Second.Weight.Count(),
               First.Bias.Size() + Second.Bias.Size(),
               First.Weight.Count() + Second.Weight.Count()
               + First.Bias.Size() + Second.Bias.Size());
    }

    auto LossOf = [&]()
    {
        return (double)Forward(First, Second, X, Target).Loss;
    };

    printf("  ");
    PrintPadded("대상", 16);
    PrintPaddedRight("개수", 8);
    PrintPaddedRight("최대 상대 오차", 18);
    printf("\n");

    struct FPart
    {
        const char* Name;
        Real* Values;
        const Real* Grad;
        int Count;
    };

    FPart Parts[4] = {
        { "W1", First.Weight.Data(),  FirstGrad.Weight.Data(),
          (int)First.Weight.Count() },
        { "b1", &First.Bias[0],       &FirstGrad.Bias[0],
          (int)First.Bias.Size() },
        { "W2", Second.Weight.Data(), SecondGrad.Weight.Data(),
          (int)Second.Weight.Count() },
        { "b2", &Second.Bias[0],      &SecondGrad.Bias[0],
          (int)Second.Bias.Size() },
    };

    double WorstOverall = 0.0;

    for (int i = 0; i < 4; i++)
    {
        FGradCheckResult Check = GradCheck(LossOf, Parts[i].Values,
                                           Parts[i].Grad, Parts[i].Count, 1e-4);

        if (Check.WorstRelative > WorstOverall)
        {
            WorstOverall = Check.WorstRelative;
        }

        char Buffer[32];
        snprintf(Buffer, sizeof(Buffer), "%.3e", Check.WorstRelative);

        printf("  ");
        PrintPadded(Parts[i].Name, 16);
        printf("%7d ", Check.Count);
        PrintPaddedRight(Buffer, 18);
        printf("\n");
    }

    printf("\n  전체 최대 상대 오차 = %.3e\n", WorstOverall);

    printf("  판정 기준 = %.0e\n\n", Threshold);

    CHECK(WorstOverall < Threshold);

    if (sizeof(Real) != 8)
    {
        printf("  지금은 float 이라 기준을 1e-1 까지 풀어야 통과한다.\n");
        printf("  이 정도 여유로는 **부호가 뒤집힌 버그도 통과**할 수 있다.\n");
        printf("  USE_DOUBLE 을 정의하고 다시 빌드해볼 것.\n\n");
    }

    // ---- C2-5. 일부러 틀려본다 ----
    printf("[C2-5] 버그를 하나 심으면 잡히는가\n\n");

    {
        // tanh 의 미분을 (1 - y^2) 대신 (1 - y) 로 잘못 쓴 경우를 흉내낸다.
        FLinearGrad BadFirst(IN_SIZE, HIDDEN_SIZE);
        FLinearGrad BadSecond(HIDDEN_SIZE, OUT_SIZE);
        BadFirst.Zero();
        BadSecond.Zero();

        FTrace Trace = Forward(First, Second, X, Target);

        FVector GradScores =
            SoftmaxCrossEntropyBackward(Trace.Probabilities, Target);
        FVector GradHiddenOut =
            LinearBackward(Second, Trace.HiddenOut, GradScores, BadSecond);

        // 틀린 줄
        FVector BadGradPre(Trace.HiddenOut.Size());
        for (size_t i = 0; i < BadGradPre.Size(); i++)
        {
            BadGradPre[i] = GradHiddenOut[i] * (Real(1) - Trace.HiddenOut[i]);
        }

        LinearBackward(First, Trace.Input, BadGradPre, BadFirst);

        FGradCheckResult Check = GradCheck(LossOf, First.Weight.Data(),
                                           BadFirst.Weight.Data(),
                                           (int)First.Weight.Count(), 1e-4);

        printf("  tanh 의 미분을 (1 - y^2) 대신 (1 - y) 로 쓰면\n");
        printf("    W1 의 최대 상대 오차 = %.3e\n", Check.WorstRelative);
        printf("    가장 어긋난 자리 %d 번: 역전파 %.9f, 수치 %.9f\n\n",
               Check.WorstIndex, Check.WorstAnalytic, Check.WorstNumeric);

        // 확실히 잡힌다.
        CHECK(Check.WorstRelative > 0.01);

        printf("  **이 버그로도 손실은 내려간다.** 방향이 대충 맞기 때문이다.\n");
        printf("  gradcheck 가 없으면 '느리게 배우는 모델' 로 보일 뿐이다.\n\n");
    }

    // ---- C2-6. 실제로 손실이 내려가는가 ----
    printf("[C2-6] 그래디언트 반대 방향으로 조금씩 움직이면\n\n");

    {
        FRandom Local;
        RandomSeed(&Local, BOOK_SEED);

        FLinear A(IN_SIZE, HIDDEN_SIZE);
        FLinear B(HIDDEN_SIZE, OUT_SIZE);
        A.InitUniform(Local, Real(0.8));
        B.InitUniform(Local, Real(0.8));

        FLinearGrad AGrad(IN_SIZE, HIDDEN_SIZE);
        FLinearGrad BGrad(HIDDEN_SIZE, OUT_SIZE);

        const Real Rate = Real(0.5);

        printf("  ");
        PrintPaddedRight("걸음", 8);
        PrintPaddedRight("손실", 14);
        PrintPaddedRight("정답 확률", 14);
        printf("\n");

        double FirstLoss = 0.0;
        double LastLoss = 0.0;

        for (int Step = 0; Step <= 50; Step++)
        {
            AGrad.Zero();
            BGrad.Zero();

            FTrace Trace = Forward(A, B, X, Target);
            Backward(A, B, Trace, Target, AGrad, BGrad);

            if (Step == 0)  FirstLoss = (double)Trace.Loss;
            LastLoss = (double)Trace.Loss;

            if (Step % 10 == 0)
            {
                printf("  %7d %13.6f %13.6f\n", Step, (double)Trace.Loss,
                       (double)Trace.Probabilities[Target]);
            }

            // 경사하강 한 걸음.
            for (size_t i = 0; i < A.Weight.Count(); i++)
            {
                A.Weight.Data()[i] -= Rate * AGrad.Weight.Data()[i];
            }
            for (size_t i = 0; i < A.Bias.Size(); i++)
            {
                A.Bias[i] -= Rate * AGrad.Bias[i];
            }
            for (size_t i = 0; i < B.Weight.Count(); i++)
            {
                B.Weight.Data()[i] -= Rate * BGrad.Weight.Data()[i];
            }
            for (size_t i = 0; i < B.Bias.Size(); i++)
            {
                B.Bias[i] -= Rate * BGrad.Bias[i];
            }
        }

        printf("\n  손실 %.6f -> %.6f\n\n", FirstLoss, LastLoss);

        CHECK(LastLoss < FirstLoss);
        CHECK(LastLoss < 0.01);

        printf("  예제 하나를 완전히 외운 것이다. 이건 '배웠다' 가 아니다.\n");
        printf("  여러 예제를 한꺼번에 맞히게 하는 것이 C3 의 일이다.\n\n");
    }

    return ReportResult();
}
