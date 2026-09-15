// lib/Types.h
//
// 엔진 전체가 쓰는 숫자 타입.
// 기본은 float 이고, USE_DOUBLE 을 정의하면 double 로 바뀐다.
//
// 왜 이렇게 두는가:
//   실행할 때는 float 가 빠르고 메모리를 절반만 쓴다.
//   그런데 나중에 역전파를 수치미분으로 검증할 때는
//   float 의 오차가 너무 커서 통과인지 아닌지 판정이 안 된다.
//   그때 이 한 줄을 바꿔서 double 로 돌린다.
//
// E2 에서 common/ 에서 lib/ 로 옮겼다.
//   엔진 헤더 15개가 이 파일을 포함하고 있었는데, common/ 은 **교재 도구**다.
//   CHECK 매크로와 표 찍는 함수가 사는 곳이다. 엔진이 거기 기대면
//   엔진만 떼어 배포할 수가 없다. 경계를 거꾸로 밟고 있었던 것이다.

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
