// lib/GradCheck.hpp
//
// 역전파가 맞는지 증명하는 도구.
//
// 왜 필요한가
//   추론 버그는 출력이 쓰레기가 되어 **명백하다.**
//   훈련 버그는 손실이 *조금* 내려간다. **조용히 틀린다.**
//
// 손실이 내려가는 것만 보고 "잘 되고 있다"고 판단하면 안 된다.
// 그래디언트가 절반쯤 틀려도 손실은 내려간다. 다만 훨씬 느리게,
// 그리고 어느 지점부터 안 내려간다.
//
// 방법은 단순하다. 가중치 하나를 아주 조금 흔들어 손실이
// 얼마나 변하는지 본다. 그게 곧 미분이다.
//
//     df/dx ~= ( f(x+h) - f(x-h) ) / 2h
//
// 이 값과 역전파가 내놓은 값을 비교한다.

#ifndef GRADCHECK_HPP
#define GRADCHECK_HPP

#include "Types.h"

#include <cstddef>
#include <functional>

struct FGradCheckResult
{
    double WorstRelative = 0.0;   // 가장 큰 상대 오차
    double WorstAbsolute = 0.0;
    int WorstIndex = -1;
    double WorstAnalytic = 0.0;
    double WorstNumeric = 0.0;
    int Count = 0;
};

// Parameters 를 하나씩 흔들어 Loss() 를 다시 계산하고,
// Analytic 에 담긴 해석적 그래디언트와 비교한다.
//
// Loss 는 Parameters 를 읽어 손실을 돌려주는 함수여야 한다.
// 흔든 값은 이 함수가 원래대로 되돌려 놓는다.
FGradCheckResult GradCheck(const std::function<double()>& Loss,
                           Real* Parameters, const Real* Analytic,
                           int Count, double Step);

// 상대 오차. 둘 다 0 에 가까우면 0 을 돌려준다.
double RelativeError(double A, double B);

#endif
