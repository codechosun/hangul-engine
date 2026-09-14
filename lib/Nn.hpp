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

// ---- 역전파 (C2 에서 추가) ----
//
// 순전파가 값을 앞으로 보내는 일이라면, 역전파는 **책임을 뒤로 보내는** 일이다.
//
//     "출력이 이만큼 틀렸다" -> "그럼 내 가중치는 이만큼 움직여야 한다"
//                           -> "그리고 내 입력은 이만큼 틀린 것이었다"
//
// 마지막 줄이 앞 층으로 넘어간다. 그래서 층을 거꾸로 한 번만 훑으면
// 모든 가중치의 그래디언트가 나온다.

// 층 하나의 그래디언트를 담는다. 모양은 가중치와 똑같다.
struct FLinearGrad
{
    FLinearGrad() = default;

    FLinearGrad(size_t InSize, size_t OutSize)
        : Weight(OutSize, InSize), Bias(OutSize)
    {
    }

    void Zero();

    FMatrix Weight;
    FVector Bias;
};

// y = Wx + b 의 역전파.
//
// GradOutput 은 dL/dy 다. 여기서 dL/dW, dL/db 를 구해 Grad 에 **누적**하고,
// dL/dx 를 돌려준다. 누적하는 이유는 같은 가중치가 여러 번 쓰일 수 있어서다.
// (D6 의 가중치 공유에서 이 성질이 결정적이 된다)
FVector LinearBackward(const FLinear& Layer, const FVector& Input,
                       const FVector& GradOutput, FLinearGrad& Grad);

// y = tanh(s) 의 역전파. dL/ds = dL/dy * (1 - y^2)
//
// **출력**을 받는다. tanh 의 미분이 출력만으로 구해지기 때문이다.
// 입력을 따로 들고 있을 필요가 없어서 메모리를 아낀다.
FVector TanhBackward(const FVector& Output, const FVector& GradOutput);

// y = relu(s) 의 역전파. dL/ds = dL/dy * (s > 0 ? 1 : 0)
//
// 이쪽은 **입력**을 받아야 한다. 출력이 0 일 때 입력이 음수였는지
// 정확히 0 이었는지 구별할 수 없기 때문이다.
FVector ReluBackward(const FVector& PreActivation, const FVector& GradOutput);

// softmax 와 교차 엔트로피를 한꺼번에 미분한 것. dL/ds = p - onehot(target)
//
// 둘을 따로 미분하면 야코비안 행렬이 나와서 복잡한데, 붙여서 미분하면
// 이렇게 짧아진다. 왜 그런지는 C2 본문에 있다.
FVector SoftmaxCrossEntropyBackward(const FVector& Probabilities, size_t Target);

#endif
