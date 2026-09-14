// lib/Nplm.hpp
//
// 신경 확률 언어모델 (Neural Probabilistic Language Model).
// 2003년 벤지오가 제안한 구조를 그대로 만든다.
//
//     앞 C개 토큰 -> 각각 임베딩 -> 이어 붙이기 -> 은닉층(tanh) -> 출력층 -> softmax
//
// A파트의 N-그램과 **입력이 똑같다.** 앞 C개를 보고 다음 하나를 맞힌다.
// 다른 것은 표를 찾는 대신 계산한다는 것뿐이다.

#ifndef NPLM_HPP
#define NPLM_HPP

#include "Matrix.hpp"
#include "Nn.hpp"
#include "Random.h"
#include "Types.h"
#include "Vector.hpp"

#include <cstdint>
#include <vector>

struct FNplmTrace
{
    FVector Embedded;       // 이어 붙인 임베딩 (ContextSize * EmbedDim)
    FVector HiddenPre;      // W1 e + b1
    FVector HiddenOut;      // tanh(HiddenPre)
    FVector Scores;         // W2 h + b2
    FVector Probabilities;  // softmax(Scores)
    Real Loss = Real(0);
};

struct FNplmGrad
{
    FNplmGrad() = default;

    FNplmGrad(size_t VocabSize, size_t ContextSize, size_t EmbedDim,
              size_t HiddenDim)
        : Embedding(VocabSize, EmbedDim),
          Hidden(ContextSize * EmbedDim, HiddenDim),
          Output(HiddenDim, VocabSize)
    {
    }

    void Zero();

    FMatrix Embedding;
    FLinearGrad Hidden;
    FLinearGrad Output;
};

class FNplm
{
public:
    FNplm() = default;

    FNplm(size_t InVocabSize, size_t InContextSize, size_t InEmbedDim,
          size_t InHiddenDim)
        : Embedding(InVocabSize, InEmbedDim),
          Hidden(InContextSize * InEmbedDim, InHiddenDim),
          Output(InHiddenDim, InVocabSize),
          VocabSize(InVocabSize),
          ContextSize(InContextSize),
          EmbedDim(InEmbedDim),
          HiddenDim(InHiddenDim)
    {
    }

    size_t Vocab() const { return VocabSize; }
    size_t Context() const { return ContextSize; }
    size_t Embed() const { return EmbedDim; }
    size_t Hidden_() const { return HiddenDim; }

    size_t ParameterCount() const;

    // 가중치를 무작위로 채운다.
    void Init(FRandom& Rng);

    // Context 는 토큰 번호 ContextSize 개.
    FNplmTrace Forward(const uint32_t* ContextTokens, size_t Target) const;

    // 그래디언트를 Grad 에 **누적**한다.
    void Backward(const uint32_t* ContextTokens, size_t Target,
                  const FNplmTrace& Trace, FNplmGrad& Grad) const;

    // 경사하강 한 걸음. Scale 은 보통 1/배치크기 다.
    void Step(const FNplmGrad& Grad, Real Rate, Real Scale);

    FMatrix Embedding;   // VocabSize x EmbedDim. 한 줄이 토큰 하나의 벡터다
    FLinear Hidden;
    FLinear Output;

private:
    size_t VocabSize = 0;
    size_t ContextSize = 0;
    size_t EmbedDim = 0;
    size_t HiddenDim = 0;
};

#endif
