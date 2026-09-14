// lib/Attention.cpp

#include "Attention.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace
{

// 마스크에 쓸 값. -무한대 대신 아주 작은 유한값을 쓴다.
//
// -무한대를 넣으면 한 줄이 통째로 막혔을 때 softmax 의 분모가 0 이 되어
// 0/0 = NaN 이 된다. 인과 마스크에서는 자기 자신이 늘 열려 있어서 그럴 일이
// 없지만, 패딩 마스크와 겹치면 생긴다. 유한값이면 최악에도 균등 분포가 된다.
const Real GMaskValue = (Real)(-1e30);

} // namespace

void SoftmaxLastAxis(FTensor& T)
{
    assert(T.Rank() >= 1);

    const size_t Last = T.Size(T.Rank() - 1);
    const size_t Rows = T.Count() / Last;

    for (size_t r = 0; r < Rows; r++)
    {
        Real* Row = T.Data() + r * Last;

        double Biggest = (double)Row[0];
        for (size_t i = 1; i < Last; i++)
        {
            if ((double)Row[i] > Biggest)
            {
                Biggest = (double)Row[i];
            }
        }

        double Sum = 0.0;
        for (size_t i = 0; i < Last; i++)
        {
            double Value = std::exp((double)Row[i] - Biggest);
            Row[i] = (Real)Value;
            Sum += Value;
        }

        for (size_t i = 0; i < Last; i++)
        {
            Row[i] = (Real)((double)Row[i] / Sum);
        }
    }
}

void ApplyCausalMask(FTensor& Scores)
{
    assert(Scores.Rank() >= 2);

    const size_t Keys = Scores.Size(Scores.Rank() - 1);
    const size_t Queries = Scores.Size(Scores.Rank() - 2);

    const size_t Batches = Scores.Count() / (Queries * Keys);

    for (size_t b = 0; b < Batches; b++)
    {
        Real* Block = Scores.Data() + b * Queries * Keys;

        for (size_t q = 0; q < Queries; q++)
        {
            // 열쇠가 질의보다 뒤에 있으면 막는다.
            for (size_t k = q + 1; k < Keys; k++)
            {
                Block[q * Keys + k] = GMaskValue;
            }
        }
    }
}

FAttention Attend(const FTensor& Q, const FTensor& K, const FTensor& V,
                  bool bCausal)
{
    assert(Q.Rank() == 4 && K.Rank() == 4 && V.Rank() == 4);
    assert(Q.Size(3) == K.Size(3));
    assert(K.Size(2) == V.Size(2));

    const size_t Dim = Q.Size(3);

    // 1. Q K^T
    //
    // K 의 마지막 두 축을 바꿔 (배치, 헤드, 차원, 위치) 로 만든 뒤 곱한다.
    FTensor Scores = MatMul(Q, K.Transposed(2, 3));

    // 2. sqrt(d) 로 나눈다
    const Real Scale = (Real)(1.0 / std::sqrt((double)Dim));
    for (size_t i = 0; i < Scores.Count(); i++)
    {
        Scores.At(i) *= Scale;
    }

    // 3. 마스크
    if (bCausal)
    {
        ApplyCausalMask(Scores);
    }

    // 4. softmax
    SoftmaxLastAxis(Scores);

    // 5. V 를 섞어 온다
    FAttention Result;
    Result.Output = MatMul(Scores, V);
    Result.Weights = std::move(Scores);

    return Result;
}

FTensor SplitHeads(const FTensor& X, size_t HeadCount)
{
    assert(X.Rank() == 3);

    const size_t Batch = X.Size(0);
    const size_t Length = X.Size(1);
    const size_t Model = X.Size(2);

    assert(Model % HeadCount == 0);
    const size_t Head = Model / HeadCount;

    // (배치, 위치, 모델) -> (배치, 위치, 헤드, 차원) -> 축을 바꿔
    // (배치, 헤드, 위치, 차원)
    FTensor Reshaped = X.Reshaped({ Batch, Length, HeadCount, Head });
    return Reshaped.Transposed(1, 2);
}

FTensor MergeHeads(const FTensor& X)
{
    assert(X.Rank() == 4);

    const size_t Batch = X.Size(0);
    const size_t HeadCount = X.Size(1);
    const size_t Length = X.Size(2);
    const size_t Head = X.Size(3);

    FTensor Swapped = X.Transposed(1, 2);   // (배치, 위치, 헤드, 차원)
    return Swapped.Reshaped({ Batch, Length, HeadCount * Head });
}
