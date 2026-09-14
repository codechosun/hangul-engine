// lib/Nn.hpp
//
// 뉴럴 네트워크의 조각들. 아직 순전파만 있다.
//
// 층 하나가 하는 일은 이것뿐이다.
//
//     y = f(W x + b)
//
// W 는 가중치 행렬, b 는 치우침(bias), f 는 비선형 함수다.
// 이 셋이 왜 다 필요한지는 C1 본문에서 하나씩 확인한다.

#ifndef NN_HPP
#define NN_HPP

#include "Matrix.hpp"
#include "Random.h"
#include "Types.h"
#include "Vector.hpp"

// ---- 활성화 함수 ----

Real Tanh(Real X);
Real Relu(Real X);

FVector Tanh(const FVector& V);
FVector Relu(const FVector& V);

// 점수를 확률로 바꾼다. 합이 1 이 된다.
//
// 최댓값을 먼저 빼고 계산한다. 안 그러면 exp 가 넘친다.
// B2 의 LogSumExp 와 같은 요령이다.
FVector Softmax(const FVector& Scores);

// 정답 하나에 대한 교차 엔트로피. -log(확률[정답])
// C2 에서 손실 함수로 쓴다.
Real CrossEntropy(const FVector& Probabilities, size_t Target);

// ---- 완전연결 층 ----

class FLinear
{
public:
    FLinear() = default;

    FLinear(size_t InSize, size_t OutSize)
        : Weight(OutSize, InSize), Bias(OutSize)
    {
    }

    size_t InSize() const { return Weight.Cols(); }
    size_t OutSize() const { return Weight.Rows(); }

    // W x + b
    FVector Forward(const FVector& Input) const;

    // 가중치를 [-Range, Range) 에서 균등하게 채운다.
    // 치우침은 0 으로 둔다.
    void InitUniform(FRandom& Rng, Real Range);

    // 입력 개수에 맞춰 범위를 정한다. sqrt(1/InSize) 를 쓴다.
    // 왜 이 값인지는 C3 에서 다룬다.
    void InitScaled(FRandom& Rng);

    // 공개해 둔다. C2 에서 그래디언트를 여기에 직접 더해야 한다.
    FMatrix Weight;
    FVector Bias;
};

#endif
