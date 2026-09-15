// lib/Backward.hpp
//
// 트랜스포머 역전파.
//
// C2 에서 세 층짜리 신경망의 미분을 손으로 썼다. 여기서는 같은 일을
// 트랜스포머 한 채에 한다. 규칙은 하나도 안 바뀐다. 연쇄 법칙뿐이다.
//
// 다만 두 가지가 새로 나온다.
//
//   1. softmax 의 야코비안
//      C2 에서는 softmax 뒤에 교차 엔트로피가 바로 붙어 있어서
//      p - y 로 깔끔하게 정리됐다. 어텐션 안의 softmax 는 그렇지 않다.
//      야코비안을 제대로 써야 한다.
//
//   2. 갈라진 길의 그래디언트는 **더해진다**
//      잔차 연결은 같은 값을 두 길로 보낸다. 임베딩 하나가 여러 자리에
//      쓰이기도 한다. 이럴 때 덮어쓰면 틀린다.
//
// 순전파 때 쓴 중간값이 역전파에 필요하므로 **자취(trace)** 를 남긴다.
// 이것이 훈련이 추론보다 메모리를 훨씬 많이 먹는 이유다.

#ifndef BACKWARD_HPP
#define BACKWARD_HPP

#include "Block.hpp"
#include "Model.hpp"
#include "Tensor.hpp"
#include "Types.h"

#include <cstdint>
#include <vector>

// ---- 조각별 역전파 ----

// softmax 의 미분. P 는 **순전파 결과**(확률)다. 입력이 아니다.
//
//     dS[i] = P[i] * ( G[i] - sum_j P[j] G[j] )
//
// 마스크로 막힌 자리는 P 가 0 이므로 dS 도 저절로 0 이 된다.
// 역전파에서 마스크를 따로 챙길 필요가 없다.
FTensor SoftmaxBackward(const FTensor& Probabilities, const FTensor& Upstream);

// SiLU 의 미분.
//
//     d/dx [ x * sigmoid(x) ] = sigmoid(x) * ( 1 + x * (1 - sigmoid(x)) )
FTensor SiluBackward(const FTensor& X, const FTensor& Upstream);

// RMSNorm 의 미분.
//
// y_i = g_i * x_i * s,   s = 1 / sqrt( mean(x^2) + eps )
//
//     dL/dg_i = sum_rows( u_i * x_i * s )
//     dL/dx_k = g_k * s * u_k  -  (s^3 * x_k / N) * sum_i( u_i * g_i * x_i )
//
// 뒤쪽 항이 핵심이다. **한 칸을 흔들면 그 줄의 RMS 가 같이 움직인다.**
// 그래서 모든 칸이 서로 얽힌다. GainGrad 에는 **더한다.**
FTensor RmsNormBackward(const FTensor& X, const FTensor& Gain, Real Epsilon,
                        const FTensor& Upstream, FTensor& GainGrad);

// ---- 완전연결 층 ----

struct FDenseGrad
{
    FTensor Weight;
    FTensor Bias;

    void Init(const FDense& Source);
    void Zero();
};

// Y = X W (+ b) 의 역전파. X 에 대한 그래디언트를 돌려주고
// Grad 에는 **더한다.**
//
//     dW = X^T U     dX = U W^T     db = sum_rows U
FTensor DenseBackward(const FDense& Dense, const FTensor& X,
                      const FTensor& Upstream, FDenseGrad& Grad);

// ---- 어텐션 ----

struct FAttentionGrad
{
    FTensor Q;
    FTensor K;
    FTensor V;
};

// Attend 의 역전파. Weights 는 순전파가 남긴 확률표다.
FAttentionGrad AttendBackward(const FTensor& Q, const FTensor& K,
                              const FTensor& V, const FTensor& Weights,
                              const FTensor& Upstream);

// ---- 블록 ----

// 순전파의 중간값. 역전파가 이걸 먹고 돈다.
struct FBlockTrace
{
    FTensor Input;
    FTensor Normed;
    FTensor Q, K, V;
    FTensor Weights;
    FTensor Attended;
    FTensor Merged;
    FTensor AfterAttention;
    FTensor FeedNormed;
    FTensor UpOut;
    FTensor Activated;
};

struct FBlockGrad
{
    FDenseGrad Query, Key, Value, Project, Up, Down;
    FTensor AttentionGain;
    FTensor FeedGain;

    void Init(const FBlock& Source);
    void Zero();
};

// FBlock::Forward 와 같은 계산을 하면서 중간값을 남긴다.
//
// 지원하는 설정은 **Pre-Norm + 잔차 + 인과 마스크** 하나다.
// D3 에서 만든 스위치들을 여기서 전부 다루면 코드가 네 배가 되고
// 배우는 것은 안 는다. 껐다 켜보는 것은 D3 의 일이었다.
FTensor BlockForwardTrace(const FBlock& Block, const FTensor& X,
                          FBlockTrace& Trace);

// 블록 하나의 역전파. 입력 X 에 대한 그래디언트를 돌려준다.
FTensor BlockBackward(const FBlock& Block, const FBlockTrace& Trace,
                      const FTensor& Upstream, FBlockGrad& Grad);

// ---- 모델 ----

struct FModelTrace
{
    std::vector<uint32_t> Tokens;
    size_t Batch = 0;
    size_t Length = 0;

    std::vector<FBlockTrace> Blocks;
    FTensor Final;     // 마지막 정규화의 입력
    FTensor Normed;    // 출력층의 입력
};

struct FTransformerGrad
{
    FTensor TokenEmbedding;
    FTensor PositionEmbedding;
    FTensor FinalGain;
    FDenseGrad Head;
    std::vector<FBlockGrad> Blocks;

    void Init(const FTransformer& Source);
    void Zero();
};

FTensor TransformerForwardTrace(const FTransformer& Model,
                                const uint32_t* Tokens, size_t Batch,
                                size_t Length, FModelTrace& Trace);

void TransformerBackward(const FTransformer& Model, const FModelTrace& Trace,
                         const FTensor& UpstreamLogits, FTransformerGrad& Grad);

// ---- 손실 ----

// 점수 (배치, 위치, 어휘) 와 정답 토큰 (배치 * 위치) 으로 평균 교차 엔트로피.
double CrossEntropyLoss(const FTensor& Logits, const uint32_t* Targets);

// 그 미분. (p - y) / N 이다. C2 에서 유도한 그대로다.
FTensor CrossEntropyBackward(const FTensor& Logits, const uint32_t* Targets);

// 마스크가 1 인 자리만 채점한다. (E1 에서 추가)
//
// 나눗수가 **채점하는 자리 수**라는 점이 중요하다. 전체 자리 수로 나누면
// 마스킹된 자리가 많을수록 손실이 작아 보여서, 서로 다른 묶음의 손실을
// 비교할 수 없게 된다.
//
// 채점하는 자리가 하나도 없으면 손실 0, 그래디언트 0 을 돌려준다.
double CrossEntropyLossMasked(const FTensor& Logits, const uint32_t* Targets,
                              const uint8_t* Mask);

FTensor CrossEntropyBackwardMasked(const FTensor& Logits,
                                   const uint32_t* Targets,
                                   const uint8_t* Mask);

// ---- gradcheck 을 위한 배선 ----
//
// 모델의 파라미터는 텐서 여러 개에 흩어져 있다. C2 의 GradCheck 는
// 한 줄로 이어진 배열을 받으므로 그대로는 못 쓴다. 포인터를 모아준다.
//
// GradCheck.hpp 를 여기서 포함하지 않는다. 쓰는 쪽이 직접 포함한다.
// 검증 도구는 **엔진에 실리지 않아야** 하기 때문이다. (E2)
//
// 짝이 되는 CollectParameters 는 **Model.hpp** 에 있다. 파라미터를 세는
// 일은 역전파의 일이 아니라 모델의 일이기 때문이다. (E2 에서 옮겼다)
//
// 두 함수가 **같은 순서**로 모은다는 것이 이 배선의 전부다.
void CollectGradients(FTransformerGrad& Grad, std::vector<Real*>& Out);

#endif
