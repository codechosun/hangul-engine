// lib/Vector.cpp

#include "Vector.hpp"

#include <cassert>
#include <cmath>

FVector& FVector::operator+=(const FVector& Other)
{
    assert(Size() == Other.Size());

    for (size_t i = 0; i < Values.size(); i++)
    {
        Values[i] += Other[i];
    }

    return *this;
}

FVector& FVector::operator-=(const FVector& Other)
{
    assert(Size() == Other.Size());

    for (size_t i = 0; i < Values.size(); i++)
    {
        Values[i] -= Other[i];
    }

    return *this;
}

FVector& FVector::operator*=(Real Scale)
{
    for (size_t i = 0; i < Values.size(); i++)
    {
        Values[i] *= Scale;
    }

    return *this;
}

Real FVector::LengthSquared() const
{
    // 누적은 double 로 한다.
    //
    // Real 이 float 일 때, 5천 개를 float 에 더하면 뒤쪽 작은 값들이
    // 앞쪽 큰 합에 묻혀 사라진다. 누적만 double 로 해도 크게 나아진다.
    double Sum = 0.0;

    for (size_t i = 0; i < Values.size(); i++)
    {
        Sum += (double)Values[i] * (double)Values[i];
    }

    return (Real)Sum;
}

Real FVector::Length() const
{
    return (Real)std::sqrt((double)LengthSquared());
}

FVector FVector::Normalized() const
{
    Real Len = Length();
    if (Len == Real(0))
    {
        return *this;
    }

    FVector Result = *this;
    Result *= Real(1) / Len;
    return Result;
}

FVector operator+(FVector Left, const FVector& Right)
{
    Left += Right;
    return Left;
}

FVector operator-(FVector Left, const FVector& Right)
{
    Left -= Right;
    return Left;
}

FVector operator*(FVector Left, Real Scale)
{
    Left *= Scale;
    return Left;
}

FVector operator*(Real Scale, FVector Right)
{
    Right *= Scale;
    return Right;
}

Real operator*(const FVector& Left, const FVector& Right)
{
    assert(Left.Size() == Right.Size());

    double Sum = 0.0;

    for (size_t i = 0; i < Left.Size(); i++)
    {
        Sum += (double)Left[i] * (double)Right[i];
    }

    return (Real)Sum;
}

Real CosineSimilarity(const FVector& A, const FVector& B)
{
    assert(A.Size() == B.Size());

    // 길이를 각각 구해 곱하지 않고, 제곱의 곱에 제곱근을 한 번만 씌운다.
    //
    //     Dot / (sqrt(AA) * sqrt(BB))   <- 제곱근 두 번, 나눗셈 한 번
    //     Dot / sqrt(AA * BB)           <- 제곱근 한 번
    //
    // 연산이 줄면 반올림이 끼어들 자리도 준다. B3 본문 참고.
    double Dot = 0.0;
    double AA = 0.0;
    double BB = 0.0;

    for (size_t i = 0; i < A.Size(); i++)
    {
        double Left = (double)A[i];
        double Right = (double)B[i];

        Dot += Left * Right;
        AA += Left * Left;
        BB += Right * Right;
    }

    if (AA == 0.0 || BB == 0.0)
    {
        return Real(0);
    }

    return (Real)(Dot / std::sqrt(AA * BB));
}

Real EuclideanDistance(const FVector& A, const FVector& B)
{
    assert(A.Size() == B.Size());

    double Sum = 0.0;

    for (size_t i = 0; i < A.Size(); i++)
    {
        double Diff = (double)A[i] - (double)B[i];
        Sum += Diff * Diff;
    }

    return (Real)std::sqrt(Sum);
}
