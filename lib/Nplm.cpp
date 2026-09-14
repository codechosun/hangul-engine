// lib/Nplm.cpp

#include "Nplm.hpp"

#include <cassert>
#include <cmath>

void FNplmGrad::Zero()
{
    Embedding.Fill(Real(0));
    Hidden.Zero();
    Output.Zero();
}

size_t FNplm::ParameterCount() const
{
    return Embedding.Count()
         + Hidden.Weight.Count() + Hidden.Bias.Size()
         + Output.Weight.Count() + Output.Bias.Size();
}

void FNplm::Init(FRandom& Rng)
{
    // 임베딩은 작게 흩뿌린다. 0 으로 두면 모든 토큰이 구별되지 않아
    // 그래디언트도 똑같이 흘러서 영원히 같은 값으로 남는다.
    for (size_t i = 0; i < Embedding.Count(); i++)
    {
        Embedding.Data()[i] = (Real)RandomRange(&Rng, 0.1);
    }

    Hidden.InitScaled(Rng);
    Output.InitScaled(Rng);
}

FNplmTrace FNplm::Forward(const uint32_t* ContextTokens, size_t Target) const
{
    assert(ContextTokens != NULL);
    assert(Target < VocabSize);

    FNplmTrace Trace;

    // 1. 임베딩을 꺼내 이어 붙인다.
    //
    // C1 에서 본 "원핫 x 행렬 = 열 하나 꺼내기" 가 이것이다.
    // 곱셈을 하나도 안 한다. 그냥 표에서 줄을 읽어 온다.
    Trace.Embedded = FVector(ContextSize * EmbedDim);

    for (size_t k = 0; k < ContextSize; k++)
    {
        uint32_t Token = ContextTokens[k];
        assert((size_t)Token < VocabSize);

        const Real* Row = Embedding.RowData((size_t)Token);
        for (size_t d = 0; d < EmbedDim; d++)
        {
            Trace.Embedded[k * EmbedDim + d] = Row[d];
        }
    }

    // 2. 나머지는 C1 의 2층 망 그대로다.
    Trace.HiddenPre = Hidden.Forward(Trace.Embedded);
    Trace.HiddenOut = Tanh(Trace.HiddenPre);
    Trace.Scores = Output.Forward(Trace.HiddenOut);
    Trace.Probabilities = Softmax(Trace.Scores);
    Trace.Loss = CrossEntropy(Trace.Probabilities, Target);

    return Trace;
}

void FNplm::Backward(const uint32_t* ContextTokens, size_t Target,
                     const FNplmTrace& Trace, FNplmGrad& Grad) const
{
    // 순전파를 거꾸로 읽는다.
    FVector GradScores = SoftmaxCrossEntropyBackward(Trace.Probabilities, Target);

    FVector GradHiddenOut =
        LinearBackward(Output, Trace.HiddenOut, GradScores, Grad.Output);

    FVector GradHiddenPre = TanhBackward(Trace.HiddenOut, GradHiddenOut);

    FVector GradEmbedded =
        LinearBackward(Hidden, Trace.Embedded, GradHiddenPre, Grad.Hidden);

    // 임베딩의 역전파. 꺼내온 줄에 **되돌려 더한다.**
    //
    // 같은 토큰이 문맥 안에 두 번 나오면 그 줄에 두 번 더해진다.
    // 그래서 여기는 반드시 누적이어야 한다. (C2 연습문제 2)
    for (size_t k = 0; k < ContextSize; k++)
    {
        uint32_t Token = ContextTokens[k];
        Real* Row = Grad.Embedding.RowData((size_t)Token);

        for (size_t d = 0; d < EmbedDim; d++)
        {
            Row[d] += GradEmbedded[k * EmbedDim + d];
        }
    }
}

void FNplm::Step(const FNplmGrad& Grad, Real Rate, Real Scale)
{
    const Real Factor = Rate * Scale;

    for (size_t i = 0; i < Embedding.Count(); i++)
    {
        Embedding.Data()[i] -= Factor * Grad.Embedding.Data()[i];
    }

    for (size_t i = 0; i < Hidden.Weight.Count(); i++)
    {
        Hidden.Weight.Data()[i] -= Factor * Grad.Hidden.Weight.Data()[i];
    }
    for (size_t i = 0; i < Hidden.Bias.Size(); i++)
    {
        Hidden.Bias[i] -= Factor * Grad.Hidden.Bias[i];
    }

    for (size_t i = 0; i < Output.Weight.Count(); i++)
    {
        Output.Weight.Data()[i] -= Factor * Grad.Output.Weight.Data()[i];
    }
    for (size_t i = 0; i < Output.Bias.Size(); i++)
    {
        Output.Bias[i] -= Factor * Grad.Output.Bias[i];
    }
}
