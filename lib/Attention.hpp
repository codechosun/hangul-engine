// lib/Attention.hpp
//
// 어텐션. 트랜스포머의 심장.
//
//     Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
//
// 한 줄인데 여기 전부가 들어 있다.
//
//   Q K^T      모든 위치가 모든 위치를 본다. (위치 x 위치) 점수표가 나온다
//   / sqrt(d)  차원이 크면 내적이 커진다. 나눠서 softmax 가 뾰족해지는 걸 막는다
//   softmax    점수를 확률로. 각 줄의 합이 1 이 된다
//   V 를 곱함  주목한 만큼 섞어 온다
//
// 모양 규약
//   Q, K, V 는 전부 (배치, 헤드, 위치, 차원).
//   앞의 두 축은 묶음일 뿐이고 실제 계산은 뒤 두 축에서 일어난다.

#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "Tensor.hpp"
#include "Types.h"

// 마지막 축을 따라 softmax 를 건다. 각 줄의 합이 1 이 된다.
//
// C1 의 Softmax 와 같은 계산이다. 최댓값을 먼저 빼는 것도 그대로다.
void SoftmaxLastAxis(FTensor& T);

// 뒤를 못 보게 막는다.
//
// 마지막 두 축을 (질의 위치, 열쇠 위치) 로 보고, 열쇠가 질의보다
// 뒤에 있으면 점수를 아주 작은 값으로 눌러버린다. softmax 를 통과하면
// 그 자리가 0 이 된다.
//
// 언어모델은 다음 글자를 맞히는 것이 일이다. 뒤를 보면 답을 베끼는 것이다.
void ApplyCausalMask(FTensor& Scores);

struct FAttention
{
    FTensor Weights;   // (배치, 헤드, 위치, 위치) — 누가 누구를 봤는가
    FTensor Output;    // (배치, 헤드, 위치, 차원)
};

// bCausal 이 참이면 인과 마스크를 건다.
FAttention Attend(const FTensor& Q, const FTensor& K, const FTensor& V,
                  bool bCausal);

// (배치, 위치, 모델차원) -> (배치, 헤드, 위치, 모델차원/헤드)
//
// 헤드를 나누는 일은 **값을 새로 계산하는 것이 아니라 보는 방법을 바꾸는 것**이다.
// 다만 메모리 배치가 달라지므로 실제로 옮긴다.
FTensor SplitHeads(const FTensor& X, size_t HeadCount);

// 그 반대.
FTensor MergeHeads(const FTensor& X);

#endif
