// common/Types.h
//
// 교재 전체가 쓰는 숫자 타입.
// 기본은 float 이고, USE_DOUBLE 을 정의하면 double 로 바뀐다.
//
// 왜 이렇게 두는가:
//   실행할 때는 float 가 빠르고 메모리를 절반만 쓴다.
//   그런데 나중에 역전파를 수치미분으로 검증할 때는
//   float 의 오차가 너무 커서 통과인지 아닌지 판정이 안 된다.
//   그때 이 한 줄을 바꿔서 double 로 돌린다.

#ifndef TYPES_H
#define TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef USE_DOUBLE
typedef double Real;
#else
typedef float Real;
#endif

#ifdef __cplusplus
}
#endif
#endif
