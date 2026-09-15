// lib/Optimizer.hpp
//
// 그래디언트를 받아 가중치를 실제로 움직이는 부분.
//
// D6 끝에서 한 것은 이것뿐이었다.
//
//     w = w - 학습률 * g
//
// 이것으로도 손실은 내려간다. 그런데 실제로 돌려보면 세 가지가 필요해진다.
//
//   AdamW      파라미터마다 보폭을 다르게 가져간다
//   클리핑     어쩌다 튀는 묶음 하나가 모델을 망가뜨리는 것을 막는다
//   스케줄     처음엔 천천히, 나중엔 줄인다
//
// 셋 다 "없으면 안 되는 것"이 아니라 **"없으면 훨씬 나쁜 것"**이다.
// D7 에서 실제로 껐다 켜보며 숫자로 확인한다.

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Types.h"

#include <cstddef>
#include <vector>

struct FAdamWConfig
{
    double Rate = 1e-3;
    double Beta1 = 0.9;      // 1차 관성. 방향을 기억한다
    double Beta2 = 0.999;    // 2차 관성. 크기를 기억한다
    double Epsilon = 1e-8;
    double WeightDecay = 0.0;
};

// AdamW.
//
//     m = b1 m + (1 - b1) g            방향의 이동평균
//     v = b2 v + (1 - b2) g^2          크기의 이동평균
//     m^ = m / (1 - b1^t)              처음에 0 에서 시작한 것을 보정
//     v^ = v / (1 - b2^t)
//     w = w - lr * m^ / (sqrt(v^) + eps)  -  lr * wd * w
//                                          ~~~~~~~~~~~~~ 이 항이 W 다
//
// Adam 과 AdamW 의 차이는 마지막 항 하나다. Adam 은 가중치 감쇠를
// 그래디언트에 더해 넣고(그러면 v 에도 섞인다), AdamW 는 갱신 식에서
// 따로 뺀다. 뒤쪽이 맞다는 것이 2017년에 밝혀졌다.
//
// 상태는 double 로 든다. v 가 g^2 의 평균이라 아주 작아지기 때문이다.
class FAdamW
{
public:
    void Init(size_t Count, const FAdamWConfig& InConfig);

    // Parameters[i] 와 Gradients[i] 가 같은 것을 가리켜야 한다.
    // 학습률을 따로 주면 그 값을 쓴다. 0 보다 작으면 Config.Rate 를 쓴다.
    void Step(Real* const* Parameters, const Real* const* Gradients,
              size_t Count, double Rate = -1.0);

    void Reset();

    size_t StepCount() const { return Steps; }

    // 옵티마이저가 들고 있는 바이트. 파라미터당 둘이다.
    size_t Bytes() const;

    FAdamWConfig Config;

private:
    std::vector<double> Moment;
    std::vector<double> Velocity;
    size_t Steps = 0;
};

// 그래디언트 전체의 크기(L2 노름)를 재고, Limit 를 넘으면 줄인다.
//
// **재기 전의 노름**을 돌려준다. 잘랐는지 아닌지는 그 값과 Limit 를
// 비교하면 안다.
//
// 왜 필요한가. 훈련 중 어쩌다 한 묶음이 유난히 어려우면 그래디언트가
// 수십 배로 튄다. 그 한 걸음이 모델을 망가뜨리고, 그 뒤로는 회복이 안 된다.
// 방향은 그대로 두고 **크기만** 자르는 것이 클리핑이다.
double ClipGradients(Real* const* Gradients, size_t Count, double Limit);

// 워밍업 + 코사인 감쇠.
//
//     0 ~ Warmup       0 에서 Base 까지 곧게 올린다
//     Warmup ~ Total   Base 에서 Base*MinRatio 까지 코사인으로 내린다
//
// 처음에 천천히 가는 이유. 훈련 시작 직후에는 Adam 의 v 가 아직
// 아무것도 모르는 상태다. 그때 큰 걸음을 내디디면 엉뚱한 데로 간다.
double ScheduleRate(double Base, size_t Step, size_t Warmup, size_t Total,
                    double MinRatio);

#endif
