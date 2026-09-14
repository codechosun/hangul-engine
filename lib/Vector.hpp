// lib/Vector.hpp
//
// 실수 벡터 하나. 이 교재의 첫 C++ 클래스다.
//
// 확장자가 .hpp 인 이유
//   lib/*.h 는 C 에서도 C++ 에서도 읽을 수 있게 extern "C" 창구를 냈다.
//   이 파일은 클래스와 연산자 오버로딩을 쓰므로 **C++ 전용**이다.
//   확장자로 그 차이를 드러낸다.
//
// D1 의 FTensor 로 가는 예비 훈련이기도 하다.
// 거기서는 차원이 여러 개가 되고 이동 시맨틱이 들어온다.

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "Types.h"

#include <cstddef>
#include <vector>

class FVector
{
public:
    FVector() = default;

    // explicit 를 붙이는 이유는 아래 주석에 있다.
    explicit FVector(size_t Size) : Values(Size, Real(0)) {}

    explicit FVector(std::vector<Real> InValues) : Values(std::move(InValues)) {}

    size_t Size() const { return Values.size(); }

    Real& operator[](size_t Index) { return Values[Index]; }
    const Real& operator[](size_t Index) const { return Values[Index]; }

    const Real* Data() const { return Values.data(); }

    FVector& operator+=(const FVector& Other);
    FVector& operator-=(const FVector& Other);
    FVector& operator*=(Real Scale);

    // 길이의 제곱. 제곱근을 안 씌우므로 빠르고 오차도 덜하다.
    Real LengthSquared() const;

    Real Length() const;

    // 길이를 1 로 맞춘 새 벡터. 길이가 0 이면 그대로 돌려준다.
    FVector Normalized() const;

private:
    std::vector<Real> Values;
};

// ---- 이항 연산자 ----
//
// 왼쪽을 **값으로** 받는다. 그러면 함수 안에서 마음대로 고칠 수 있고,
// 부르는 쪽이 임시 객체를 넘겼다면 복사조차 일어나지 않는다.
// A + B + C 처럼 이어 쓸 때 중간 결과가 그대로 다음 연산으로 넘어간다.
FVector operator+(FVector Left, const FVector& Right);
FVector operator-(FVector Left, const FVector& Right);
FVector operator*(FVector Left, Real Scale);
FVector operator*(Real Scale, FVector Right);

// 벡터 곱하기 벡터는 **내적**으로 정의한다. 결과가 벡터가 아니라 실수다.
//
// 이 선택은 논쟁거리다. B3 본문에서 다룬다.
Real operator*(const FVector& Left, const FVector& Right);

// 두 벡터가 이루는 각의 코사인. -1 에서 1 사이다.
// 둘 중 하나라도 길이가 0 이면 0 을 돌려준다.
Real CosineSimilarity(const FVector& A, const FVector& B);

Real EuclideanDistance(const FVector& A, const FVector& B);

#endif
