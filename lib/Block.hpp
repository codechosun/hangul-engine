// lib/Block.hpp
//
// 트랜스포머 블록 하나와, 그 안에 들어가는 현대적 장치들.
//
// 여기 들어가는 넷은 전부 **훈련 안정성 장치**다.
//
//     잔차 연결   층을 건너뛰는 지름길
//     RMSNorm     층마다 크기를 1 근처로 되돌린다
//     Pre-Norm    정규화를 층 앞에 둔다
//     QK-Norm     어텐션 점수가 커지는 것을 한 번 더 막는다
//
// 왜 필요한지는 D6~D7 에서 그래디언트가 실제로 폭주해봐야 들어온다.
// **D3 은 설치, D7 이 납득.**

#ifndef BLOCK_HPP
#define BLOCK_HPP

#include "Random.h"
#include "Tensor.hpp"
#include "Types.h"

// ---- 정규화 ----
//
// RMS = sqrt(평균(x^2)). 평균을 빼지 않는다.
//
//     LayerNorm  (x - 평균) / 표준편차 * 이득 + 치우침
//     RMSNorm    x / RMS * 이득
//
// 평균 빼기와 치우침을 없앴는데 성능이 거의 같다는 것이 2019년에 밝혀졌고,
// 그 뒤로 큰 모델은 대개 이쪽을 쓴다. 계산이 적고 구현도 짧다.
FTensor RmsNorm(const FTensor& X, const FTensor& Gain, Real Epsilon);

// ---- 활성화 ----
//
// SiLU(x) = x * sigmoid(x). Swish 라고도 부른다.
// ReLU 와 달리 음수 쪽이 완전히 0 이 아니라 그래디언트가 조금 흐른다.
Real Silu(Real X);
FTensor Silu(const FTensor& X);

Real Sigmoid(Real X);

// ---- 완전연결 층 (텐서 판) ----
//
// C1 의 FLinear 는 FVector 만 받았다. 이쪽은 (..., In) 모양이면 무엇이든
// 받아 마지막 축만 바꾼다. 앞쪽 축이 배치든 위치든 상관하지 않는다.
struct FDense
{
    FDense() = default;
    FDense(size_t InSize, size_t OutSize, bool bInUseBias = true);

    FTensor Forward(const FTensor& X) const;
    void Init(FRandom& Rng);

    size_t InSize() const { return Weight.Size(0); }
    size_t OutSize() const { return Weight.Size(1); }

    FTensor Weight;   // (In, Out)
    FTensor Bias;     // (Out)
    bool bUseBias = true;
};

// ---- 블록 ----

struct FBlockConfig
{
    size_t Model = 64;
    size_t Heads = 4;
    size_t Hidden = 256;   // 보통 Model 의 4배

    bool bResidual = true;   // 잔차 연결을 쓸 것인가
    bool bNorm = true;       // 정규화를 쓸 것인가
    bool bPreNorm = true;    // 정규화를 층 앞에 둘 것인가
    bool bQkNorm = false;    // Q, K 를 따로 정규화할 것인가
    bool bCausal = true;
};

class FBlock
{
public:
    FBlock() = default;
    explicit FBlock(const FBlockConfig& InConfig);

    void Init(FRandom& Rng);

    // X 는 (배치, 위치, 모델차원). 결과도 같은 모양이다.
    FTensor Forward(const FTensor& X) const;

    const FBlockConfig& GetConfig() const { return Config; }

    FBlockConfig Config;

    FDense Query;
    FDense Key;
    FDense Value;
    FDense Project;

    FDense Up;
    FDense Down;

    FTensor AttentionGain;   // (Model)
    FTensor FeedGain;        // (Model)
    FTensor QueryGain;       // (Model / Heads)
    FTensor KeyGain;         // (Model / Heads)
};

#endif
