// lib/LogMath.h
//
// 로그 공간에서 더하기.
//
// 확률을 여러 개 곱하면 double 에서도 0 이 된다. 그래서 로그로 다룬다.
// 로그로 다루면 곱하기는 더하기가 되어 편한데, **더하기가 어려워진다.**
//
//     log(a) 와 log(b) 를 알 때 log(a + b) 를 어떻게 구하는가
//
// 그냥 exp 로 되돌리면 그 순간 0 이 되어 애써 로그로 바꾼 의미가 없다.
// 큰 쪽을 뽑아내면 그 문제가 사라진다. B2 본문 참고.
//
// C파트의 softmax 와 D파트의 어텐션에서 같은 기법을 다시 쓴다.

#ifndef LOGMATH_H
#define LOGMATH_H

#ifdef __cplusplus
extern "C" {
#endif

// log(exp(LogA) + exp(LogB)) 를 넘치지도 사라지지도 않게 구한다.
double LogAdd(double LogA, double LogB);

// 확률 배열의 로그합. 비어 있으면 -inf.
double LogSumExp(const double* LogValues, int Count);

#ifdef __cplusplus
}
#endif
#endif
