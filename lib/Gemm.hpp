// lib/Gemm.hpp
//
// 행렬 곱셈 여러 판.
//
//     C = A * B      A 는 M x K, B 는 K x N, C 는 M x N.  전부 행 우선.
//
// 같은 계산을 여섯 가지 방법으로 한다. 결과는 전부 같아야 하고,
// 속도는 전부 달라야 한다. 그 차이를 재는 것이 C5 의 일이다.
//
// 이름이 GEMM 인 이유
//   BLAS 라는 오래된 규약에서 온 이름이다. GEneral Matrix Multiply.
//   요즘 신경망 계산 시간의 대부분이 이 함수 하나에 들어간다.

#ifndef GEMM_HPP
#define GEMM_HPP

#include "Types.h"

#include <cstddef>

// 이 CPU 가 AVX2 와 FMA 를 쓸 수 있는가. 프로그램이 돌기 시작한 뒤에 묻는다.
bool CpuHasAvx2();

// 논리 코어 수.
int CpuThreadCount();

// 1. 교과서 그대로. i, j, k 순서.
void GemmNaive(const Real* A, const Real* B, Real* C,
               size_t M, size_t N, size_t K);

// 2. 루프 순서만 바꿨다. i, k, j.
void GemmReorder(const Real* A, const Real* B, Real* C,
                 size_t M, size_t N, size_t K);

// 3. 블록으로 쪼개 캐시 안에서 돌게 한다.
void GemmBlocked(const Real* A, const Real* B, Real* C,
                 size_t M, size_t N, size_t K);

// 3-1. 블록 크기를 인자로 받는 판. BlockN 이 0 이면 N 은 안 쪼갠다.
//      C5 에서 어느 축을 쪼개야 하는지 재보려고 열어둔 문이다.
void GemmBlockedTuned(const Real* A, const Real* B, Real* C,
                      size_t M, size_t N, size_t K,
                      size_t BlockM, size_t BlockK, size_t BlockN);

// 4. AVX2 로 한 번에 여덟 개씩. CpuHasAvx2() 가 거짓이면 3번으로 넘긴다.
void GemmAvx2(const Real* A, const Real* B, Real* C,
              size_t M, size_t N, size_t K);

// 5. 4번을 코어 수만큼 나눠 돌린다.
void GemmParallel(const Real* A, const Real* B, Real* C,
                  size_t M, size_t N, size_t K);

// 이 기계가 낼 수 있는 이론상 최대 성능(초당 부동소수점 연산 수)을 잰다.
// FMA 명령만 줄지어 돌려서 측정한다.
double MeasurePeakGflops(int ThreadCount);

#endif
