// lib/Pca.hpp
//
// 주성분 분석. 높은 차원을 낮은 차원으로 누른다.
//
// 하는 일은 한 줄로 말할 수 있다.
//
//     **데이터가 가장 넓게 퍼져 있는 방향**을 찾아, 그 방향으로 재어 적는다.
//
// 16차원 임베딩을 2차원으로 눌러 그림을 그리려고 만든다.
// 눌러도 남는 것이 원래 데이터에서 가장 중요한 성질이라는 가정 위에 서 있다.
//
// 고유값 분해 라이브러리를 안 쓰고 거듭제곱법으로 직접 구한다.
// 차원이 16이라 몇 줄이면 되고, 무엇을 하는지도 눈에 보인다.

#ifndef PCA_HPP
#define PCA_HPP

#include "Matrix.hpp"
#include "Types.h"
#include "Vector.hpp"

#include <cstddef>

struct FPca
{
    FMatrix Components;   // ComponentCount x D. 한 줄이 방향 하나
    FVector Mean;         // D
    FVector Variance;     // ComponentCount. 그 방향으로 퍼진 정도
    Real TotalVariance = Real(0);
};

// Data 는 N x D (한 줄이 표본 하나).
// Iterations 는 거듭제곱법을 몇 번 돌릴지. 100 이면 충분하다.
FPca PcaFit(const FMatrix& Data, size_t ComponentCount, int Iterations);

// 표본들을 주성분 방향으로 재어 N x ComponentCount 로 만든다.
FMatrix PcaTransform(const FMatrix& Data, const FPca& Pca);

#endif
