// lib/Quant.cpp

#include "Quant.hpp"

#include <cassert>
#include <cmath>

namespace
{

// 반올림. round() 는 0.5 를 0 에서 먼 쪽으로 보낸다.
int32_t RoundToInt(double Value)
{
    return (int32_t)std::lround(Value);
}

int32_t Clamp(int32_t Value, int32_t Low, int32_t High)
{
    if (Value < Low)  return Low;
    if (Value > High) return High;
    return Value;
}

} // namespace

size_t FQuant8::Bytes() const
{
    return Values.size() * sizeof(int8_t) + Scales.size() * sizeof(Real);
}

size_t FQuant4::Bytes() const
{
    return Packed.size() * sizeof(uint8_t) + Scales.size() * sizeof(Real);
}

FQuant8 Quantize8(const FTensor& T, bool bPerRow)
{
    assert(T.Rank() == 2);

    FQuant8 Result;
    Result.Rows = T.Size(0);
    Result.Cols = T.Size(1);
    Result.bPerRow = bPerRow;
    Result.Values.resize(Result.Rows * Result.Cols);

    const size_t Groups = bPerRow ? Result.Rows : 1;
    Result.Scales.resize(Groups);

    for (size_t g = 0; g < Groups; g++)
    {
        const size_t From = bPerRow ? (g * Result.Cols) : 0;
        const size_t To = bPerRow ? (From + Result.Cols) : T.Count();

        // 대칭 양자화. 절대값의 최대치를 127 에 맞춘다.
        //
        // 0 이 정확히 0 으로 남는다는 점이 중요하다. 비대칭(0 을 옮기는)
        // 방식이 더 촘촘하지만, 0 이 틀어지면 마스킹 같은 곳에서 사고가 난다.
        double Biggest = 0.0;
        for (size_t i = From; i < To; i++)
        {
            double V = std::fabs((double)T.At(i));
            if (V > Biggest) Biggest = V;
        }

        double Scale = (Biggest > 0.0) ? (Biggest / 127.0) : 1.0;
        Result.Scales[g] = (Real)Scale;

        for (size_t i = From; i < To; i++)
        {
            int32_t Q = RoundToInt((double)T.At(i) / Scale);
            Result.Values[i] = (int8_t)Clamp(Q, -127, 127);
        }
    }

    return Result;
}

FTensor Dequantize8(const FQuant8& Q)
{
    FTensor Result({ Q.Rows, Q.Cols });

    for (size_t r = 0; r < Q.Rows; r++)
    {
        const Real Scale = Q.bPerRow ? Q.Scales[r] : Q.Scales[0];

        for (size_t c = 0; c < Q.Cols; c++)
        {
            Result(r, c) = (Real)((double)Q.Values[r * Q.Cols + c]
                                * (double)Scale);
        }
    }

    return Result;
}

FQuant4 Quantize4(const FTensor& T)
{
    assert(T.Rank() == 2);

    FQuant4 Result;
    Result.Rows = T.Size(0);
    Result.Cols = T.Size(1);
    Result.Scales.resize(Result.Rows);
    Result.Packed.resize((Result.Rows * Result.Cols + 1) / 2, 0);

    for (size_t r = 0; r < Result.Rows; r++)
    {
        double Biggest = 0.0;
        for (size_t c = 0; c < Result.Cols; c++)
        {
            double V = std::fabs((double)T(r, c));
            if (V > Biggest) Biggest = V;
        }

        // 4비트 부호 있는 정수의 범위는 -8 ~ 7 이다.
        // 대칭으로 쓰려면 -7 ~ 7 만 쓴다. -8 을 버리는 대가로 0 이 가운데 온다.
        double Scale = (Biggest > 0.0) ? (Biggest / 7.0) : 1.0;
        Result.Scales[r] = (Real)Scale;

        for (size_t c = 0; c < Result.Cols; c++)
        {
            int32_t Q = Clamp(RoundToInt((double)T(r, c) / Scale), -7, 7);

            // -7..7 을 0..14 로 옮겨 담는다. 4비트에 음수를 그대로 넣으면
            // 꺼낼 때 부호 확장을 손으로 해야 해서 번거롭다.
            uint8_t Nibble = (uint8_t)(Q + 8);

            const size_t Flat = r * Result.Cols + c;
            const size_t Byte = Flat / 2;

            if (Flat % 2 == 0)
            {
                // 아래 4비트
                Result.Packed[Byte] = (uint8_t)((Result.Packed[Byte] & 0xF0u)
                                              | (Nibble & 0x0Fu));
            }
            else
            {
                // 위 4비트
                Result.Packed[Byte] = (uint8_t)((Result.Packed[Byte] & 0x0Fu)
                                              | (uint8_t)(Nibble << 4));
            }
        }
    }

    return Result;
}

FTensor Dequantize4(const FQuant4& Q)
{
    FTensor Result({ Q.Rows, Q.Cols });

    for (size_t r = 0; r < Q.Rows; r++)
    {
        const double Scale = (double)Q.Scales[r];

        for (size_t c = 0; c < Q.Cols; c++)
        {
            const size_t Flat = r * Q.Cols + c;
            const size_t Byte = Flat / 2;

            uint8_t Nibble = (Flat % 2 == 0)
                ? (uint8_t)(Q.Packed[Byte] & 0x0Fu)
                : (uint8_t)(Q.Packed[Byte] >> 4);

            int32_t Value = (int32_t)Nibble - 8;
            Result(r, c) = (Real)((double)Value * Scale);
        }
    }

    return Result;
}

FTensor QuantizedForward8(const FQuant8& Q, const FTensor& X)
{
    assert(X.Size(X.Rank() - 1) == Q.Rows);

    const size_t In = Q.Rows;
    const size_t Out = Q.Cols;
    const size_t Lines = X.Count() / In;

    std::vector<size_t> Shape = X.GetShape();
    Shape[Shape.size() - 1] = Out;

    FTensor Result(Shape);

    // C5 의 i-k-j. 안쪽 고리가 열을 훑으므로 양쪽 다 순서대로 읽는다.
    //
    // scale 이 k 축에 붙어 있다는 점이 여기서 값을 한다.
    // X[i][k] 와 Scale[k] 를 미리 곱해두면 안쪽 고리에는 곱셈이 하나뿐이다.
    // **줄마다 scale 을 두는 값이 공짜인 것**은 이 순서일 때만이다.
    for (size_t line = 0; line < Lines; line++)
    {
        const Real* Row = X.Data() + line * In;
        Real* OutRow = Result.Data() + line * Out;

        for (size_t k = 0; k < In; k++)
        {
            const Real Scale = Q.bPerRow ? Q.Scales[k] : Q.Scales[0];
            const Real Factor = Row[k] * Scale;
            const int8_t* Weights = Q.Values.data() + k * Out;

            for (size_t o = 0; o < Out; o++)
            {
                OutRow[o] += Factor * (Real)Weights[o];
            }
        }
    }

    return Result;
}

FTensor QuantizedForward8Dot(const FQuant8& Q, const FTensor& X)
{
    assert(X.Size(X.Rank() - 1) == Q.Rows);

    const size_t In = Q.Rows;
    const size_t Out = Q.Cols;
    const size_t Lines = X.Count() / In;

    std::vector<size_t> Shape = X.GetShape();
    Shape[Shape.size() - 1] = Out;

    FTensor Result(Shape);

    // 내적 순서. 사람이 수식을 그대로 옮기면 대개 이렇게 쓴다.
    //
    // 안쪽 고리가 가중치를 Out 칸씩 건너뛰며 읽고, scale 도 안쪽에서
    // 매번 다시 가져온다. C5 에서 본 그 문제가 그대로 나온다.
    for (size_t line = 0; line < Lines; line++)
    {
        const Real* Row = X.Data() + line * In;
        Real* OutRow = Result.Data() + line * Out;

        for (size_t o = 0; o < Out; o++)
        {
            Real Sum = Real(0);
            for (size_t i = 0; i < In; i++)
            {
                const Real Scale = Q.bPerRow ? Q.Scales[i] : Q.Scales[0];
                Sum += Row[i] * (Real)Q.Values[i * Out + o] * Scale;
            }

            OutRow[o] = Sum;
        }
    }

    return Result;
}

FError CompareTensors(const FTensor& A, const FTensor& B)
{
    assert(A.Count() == B.Count());

    FError Result;

    double Total = 0.0;
    double SquareError = 0.0;
    double SquareOriginal = 0.0;

    for (size_t i = 0; i < A.Count(); i++)
    {
        double Gap = std::fabs((double)A.At(i) - (double)B.At(i));

        if (Gap > Result.MaxAbsolute) Result.MaxAbsolute = Gap;

        Total += Gap;
        SquareError += Gap * Gap;
        SquareOriginal += (double)A.At(i) * (double)A.At(i);
    }

    Result.MeanAbsolute = Total / (double)A.Count();
    Result.RelativeRms = (SquareOriginal > 0.0)
        ? std::sqrt(SquareError / SquareOriginal)
        : 0.0;

    return Result;
}
