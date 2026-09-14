// lib/Block.cpp

#include "Block.hpp"
#include "Attention.hpp"

#include <cassert>
#include <cmath>

FTensor RmsNorm(const FTensor& X, const FTensor& Gain, Real Epsilon)
{
    const size_t Last = X.Size(X.Rank() - 1);
    const size_t Rows = X.Count() / Last;

    assert(Gain.Count() == Last);

    FTensor Result(X.GetShape());

    for (size_t r = 0; r < Rows; r++)
    {
        const Real* Row = X.Data() + r * Last;
        Real* Out = Result.Data() + r * Last;

        // 누적은 double 로. B3 에서 세운 규칙이다.
        double SquareSum = 0.0;
        for (size_t i = 0; i < Last; i++)
        {
            SquareSum += (double)Row[i] * (double)Row[i];
        }

        // Epsilon 은 0 으로 나누는 것을 막는다. 값이 전부 0 인 줄이 있을 수 있다.
        double Scale = 1.0 / std::sqrt(SquareSum / (double)Last + (double)Epsilon);

        for (size_t i = 0; i < Last; i++)
        {
            Out[i] = (Real)((double)Row[i] * Scale * (double)Gain.At(i));
        }
    }

    return Result;
}

Real Sigmoid(Real X)
{
    // 지수가 커지면 넘치므로 부호에 따라 식을 바꾼다.
    // C1 의 softmax 에서 최댓값을 뺀 것과 같은 요령이다.
    double V = (double)X;

    if (V >= 0.0)
    {
        return (Real)(1.0 / (1.0 + std::exp(-V)));
    }

    double E = std::exp(V);
    return (Real)(E / (1.0 + E));
}

Real Silu(Real X)
{
    return X * Sigmoid(X);
}

FTensor Silu(const FTensor& X)
{
    FTensor Result(X.GetShape());
    for (size_t i = 0; i < X.Count(); i++)
    {
        Result.At(i) = Silu(X.At(i));
    }
    return Result;
}

FDense::FDense(size_t InSize, size_t OutSize, bool bInUseBias)
    : Weight({ InSize, OutSize }), Bias({ OutSize }), bUseBias(bInUseBias)
{
}

void FDense::Init(FRandom& Rng)
{
    // C3 의 InitScaled 와 같은 규칙. 입력 개수의 제곱근에 반비례한다.
    const double Range = 1.0 / std::sqrt((double)InSize());

    for (size_t i = 0; i < Weight.Count(); i++)
    {
        Weight.At(i) = (Real)RandomRange(&Rng, Range);
    }

    Bias.Fill(Real(0));
}

FTensor FDense::Forward(const FTensor& X) const
{
    assert(X.Size(X.Rank() - 1) == InSize());

    const size_t In = InSize();
    const size_t Out = OutSize();
    const size_t Rows = X.Count() / In;

    // 앞쪽 축이 무엇이든 (줄 수, In) 으로 눕혀서 한 번에 곱한다.
    FTensor Flat = X.Reshaped({ Rows, In });
    FTensor Product = MatMul(Flat, Weight);

    if (bUseBias)
    {
        for (size_t r = 0; r < Rows; r++)
        {
            Real* Row = Product.Data() + r * Out;
            for (size_t i = 0; i < Out; i++)
            {
                Row[i] += Bias.At(i);
            }
        }
    }

    std::vector<size_t> Shape = X.GetShape();
    Shape[Shape.size() - 1] = Out;

    return Product.Reshaped(Shape);
}

FBlock::FBlock(const FBlockConfig& InConfig)
    : Config(InConfig),
      Query(InConfig.Model, InConfig.Model, false),
      Key(InConfig.Model, InConfig.Model, false),
      Value(InConfig.Model, InConfig.Model, false),
      Project(InConfig.Model, InConfig.Model, false),
      Up(InConfig.Model, InConfig.Hidden, false),
      Down(InConfig.Hidden, InConfig.Model, false),
      AttentionGain({ InConfig.Model }),
      FeedGain({ InConfig.Model }),
      QueryGain({ InConfig.Model / InConfig.Heads }),
      KeyGain({ InConfig.Model / InConfig.Heads })
{
    AttentionGain.Fill(Real(1));
    FeedGain.Fill(Real(1));
    QueryGain.Fill(Real(1));
    KeyGain.Fill(Real(1));
}

void FBlock::Init(FRandom& Rng)
{
    Query.Init(Rng);
    Key.Init(Rng);
    Value.Init(Rng);
    Project.Init(Rng);
    Up.Init(Rng);
    Down.Init(Rng);
}

FTensor FBlock::Forward(const FTensor& X) const
{
    assert(X.Rank() == 3);
    assert(X.Size(2) == Config.Model);

    // ---- 어텐션 쪽 ----
    //
    // Pre-Norm 이면 정규화를 먼저 하고, Post-Norm 이면 나중에 한다.
    FTensor Input = X;

    FTensor Normed = (Config.bNorm && Config.bPreNorm)
        ? RmsNorm(Input, AttentionGain, (Real)1e-6)
        : Input;

    FTensor Q = SplitHeads(Query.Forward(Normed), Config.Heads);
    FTensor K = SplitHeads(Key.Forward(Normed), Config.Heads);
    FTensor V = SplitHeads(Value.Forward(Normed), Config.Heads);

    if (Config.bQkNorm)
    {
        // Q 와 K 를 각각 정규화한다. 그러면 내적의 크기가
        // 가중치가 커지는 것과 무관해진다. sqrt(d) 로도 모자랄 때의 대비책.
        Q = RmsNorm(Q, QueryGain, (Real)1e-6);
        K = RmsNorm(K, KeyGain, (Real)1e-6);
    }

    FAttention Attended = Attend(Q, K, V, Config.bCausal);
    FTensor AttentionOut = Project.Forward(MergeHeads(Attended.Output));

    FTensor AfterAttention = Config.bResidual
        ? (std::move(AttentionOut) + Input)
        : std::move(AttentionOut);

    if (Config.bNorm && !Config.bPreNorm)
    {
        AfterAttention = RmsNorm(AfterAttention, AttentionGain, (Real)1e-6);
    }

    // ---- 앞먹임 쪽 ----
    FTensor FeedInput = AfterAttention;

    FTensor FeedNormed = (Config.bNorm && Config.bPreNorm)
        ? RmsNorm(FeedInput, FeedGain, (Real)1e-6)
        : FeedInput;

    FTensor FeedOut = Down.Forward(Silu(Up.Forward(FeedNormed)));

    FTensor Result = Config.bResidual
        ? (std::move(FeedOut) + FeedInput)
        : std::move(FeedOut);

    if (Config.bNorm && !Config.bPreNorm)
    {
        Result = RmsNorm(Result, FeedGain, (Real)1e-6);
    }

    return Result;
}
