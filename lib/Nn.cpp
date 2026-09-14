// lib/Nn.cpp

#include "Nn.hpp"

#include <cassert>
#include <cmath>

Real Tanh(Real X)
{
    return (Real)std::tanh((double)X);
}

Real Relu(Real X)
{
    return (X > Real(0)) ? X : Real(0);
}

FVector Tanh(const FVector& V)
{
    FVector Result(V.Size());
    for (size_t i = 0; i < V.Size(); i++)
    {
        Result[i] = Tanh(V[i]);
    }
    return Result;
}

FVector Relu(const FVector& V)
{
    FVector Result(V.Size());
    for (size_t i = 0; i < V.Size(); i++)
    {
        Result[i] = Relu(V[i]);
    }
    return Result;
}

FVector Softmax(const FVector& Scores)
{
    FVector Result(Scores.Size());
    if (Scores.Size() == 0)
    {
        return Result;
    }

    // 1단계. 최댓값을 찾는다.
    //
    // exp(1000) 은 double 에서 무한대다. 최댓값을 빼면 지수가 0 이하가 되어
    // exp 가 절대 넘치지 않는다. 그리고 최댓값 항이 정확히 1 이 되므로
    // 분모가 0 이 되는 일도 없다.
    double Biggest = (double)Scores[0];
    for (size_t i = 1; i < Scores.Size(); i++)
    {
        if ((double)Scores[i] > Biggest)
        {
            Biggest = (double)Scores[i];
        }
    }

    // 2단계. exp 를 취해 더한다.
    double Sum = 0.0;
    std::vector<double> Exps(Scores.Size());

    for (size_t i = 0; i < Scores.Size(); i++)
    {
        Exps[i] = std::exp((double)Scores[i] - Biggest);
        Sum += Exps[i];
    }

    // 3단계. 나눈다.
    for (size_t i = 0; i < Scores.Size(); i++)
    {
        Result[i] = (Real)(Exps[i] / Sum);
    }

    return Result;
}

Real CrossEntropy(const FVector& Probabilities, size_t Target)
{
    assert(Target < Probabilities.Size());

    double P = (double)Probabilities[Target];

    // 확률이 0 이면 로그가 음의 무한대다. B2 에서 본 그 문제다.
    // 여기서는 아주 작은 값으로 받쳐둔다.
    if (P < 1e-300)
    {
        P = 1e-300;
    }

    return (Real)(-std::log(P));
}

FVector FLinear::Forward(const FVector& Input) const
{
    assert(Input.Size() == InSize());

    FVector Result = Weight * Input;
    Result += Bias;
    return Result;
}

void FLinear::InitUniform(FRandom& Rng, Real Range)
{
    for (size_t Row = 0; Row < Weight.Rows(); Row++)
    {
        for (size_t Col = 0; Col < Weight.Cols(); Col++)
        {
            Weight(Row, Col) = (Real)RandomRange(&Rng, (double)Range);
        }
    }

    for (size_t i = 0; i < Bias.Size(); i++)
    {
        Bias[i] = Real(0);
    }
}

void FLinear::InitScaled(FRandom& Rng)
{
    Real Range = (Real)(1.0 / std::sqrt((double)InSize()));
    InitUniform(Rng, Range);
}
