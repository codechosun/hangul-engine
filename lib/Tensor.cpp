// lib/Tensor.cpp

#include "Tensor.hpp"

#include <cassert>
#include <cstring>

namespace
{

FTensor::FCounters GCounters;

size_t ProductOf(const std::vector<size_t>& Shape)
{
    size_t Total = 1;
    for (size_t i = 0; i < Shape.size(); i++)
    {
        Total *= Shape[i];
    }
    return Total;
}

} // namespace

const FTensor::FCounters& FTensor::Counters()
{
    return GCounters;
}

void FTensor::ResetCounters()
{
    GCounters = FCounters();
}

void FTensor::Allocate(const std::vector<size_t>& InShape)
{
    Shape = InShape;
    Total = ProductOf(InShape);

    Values = new Real[Total];
    std::memset(Values, 0, Total * sizeof(Real));

    GCounters.Allocation++;
}

FTensor::FTensor(std::initializer_list<size_t> InShape)
{
    Allocate(std::vector<size_t>(InShape));
}

FTensor::FTensor(const std::vector<size_t>& InShape)
{
    Allocate(InShape);
}

// ---- 복사 생성 ----
//
// 새 버퍼를 잡고 내용을 통째로 옮겨 적는다. 원본은 그대로 산다.
FTensor::FTensor(const FTensor& Other)
{
    if (Other.Values == nullptr)
    {
        return;
    }

    Allocate(Other.Shape);
    std::memcpy(Values, Other.Values, Total * sizeof(Real));

    GCounters.Copy++;
    GCounters.ElementsCopied += Total;
}

// ---- 복사 대입 ----
//
// 자기 자신을 대입하는 경우를 먼저 걸러야 한다.
// 안 그러면 내 버퍼를 지운 뒤 그 버퍼에서 읽으려 든다.
FTensor& FTensor::operator=(const FTensor& Other)
{
    if (this == &Other)
    {
        return *this;
    }

    delete[] Values;
    Values = nullptr;
    Shape.clear();
    Total = 0;

    if (Other.Values != nullptr)
    {
        Allocate(Other.Shape);
        std::memcpy(Values, Other.Values, Total * sizeof(Real));

        GCounters.Copy++;
        GCounters.ElementsCopied += Total;
    }

    return *this;
}

// ---- 이동 생성 ----
//
// 버퍼를 **훔쳐 온다.** 복사가 한 번도 안 일어난다.
// 원본은 빈 상태로 남겨야 한다. 안 그러면 둘이 같은 버퍼를 해제한다.
//
// noexcept 를 붙이는 이유: std::vector 가 커질 때 원소를 옮기는데,
// 이동이 예외를 던질 수 있으면 안전을 위해 **복사**를 택한다.
// noexcept 를 안 붙이면 애써 만든 이동 생성자가 안 불린다.
FTensor::FTensor(FTensor&& Other) noexcept
    : Values(Other.Values), Shape(std::move(Other.Shape)), Total(Other.Total)
{
    Other.Values = nullptr;
    Other.Total = 0;

    GCounters.Move++;
}

// ---- 이동 대입 ----
FTensor& FTensor::operator=(FTensor&& Other) noexcept
{
    if (this == &Other)
    {
        return *this;
    }

    delete[] Values;

    Values = Other.Values;
    Shape = std::move(Other.Shape);
    Total = Other.Total;

    Other.Values = nullptr;
    Other.Total = 0;

    GCounters.Move++;

    return *this;
}

// ---- 소멸 ----
FTensor::~FTensor()
{
    delete[] Values;
}

Real& FTensor::operator()(size_t A)
{
    assert(Rank() == 1 && A < Shape[0]);
    return Values[A];
}

Real& FTensor::operator()(size_t A, size_t B)
{
    assert(Rank() == 2 && A < Shape[0] && B < Shape[1]);
    return Values[A * Shape[1] + B];
}

Real& FTensor::operator()(size_t A, size_t B, size_t C)
{
    assert(Rank() == 3 && A < Shape[0] && B < Shape[1] && C < Shape[2]);
    return Values[(A * Shape[1] + B) * Shape[2] + C];
}

Real& FTensor::operator()(size_t A, size_t B, size_t C, size_t D)
{
    assert(Rank() == 4);
    assert(A < Shape[0] && B < Shape[1] && C < Shape[2] && D < Shape[3]);
    return Values[((A * Shape[1] + B) * Shape[2] + C) * Shape[3] + D];
}

const Real& FTensor::operator()(size_t A) const
{
    assert(Rank() == 1 && A < Shape[0]);
    return Values[A];
}

const Real& FTensor::operator()(size_t A, size_t B) const
{
    assert(Rank() == 2 && A < Shape[0] && B < Shape[1]);
    return Values[A * Shape[1] + B];
}

const Real& FTensor::operator()(size_t A, size_t B, size_t C) const
{
    assert(Rank() == 3 && A < Shape[0] && B < Shape[1] && C < Shape[2]);
    return Values[(A * Shape[1] + B) * Shape[2] + C];
}

const Real& FTensor::operator()(size_t A, size_t B, size_t C, size_t D) const
{
    assert(Rank() == 4);
    assert(A < Shape[0] && B < Shape[1] && C < Shape[2] && D < Shape[3]);
    return Values[((A * Shape[1] + B) * Shape[2] + C) * Shape[3] + D];
}

void FTensor::Fill(Real Value)
{
    for (size_t i = 0; i < Total; i++)
    {
        Values[i] = Value;
    }
}

FTensor FTensor::Reshaped(const std::vector<size_t>& NewShape) const
{
    assert(ProductOf(NewShape) == Total);

    FTensor Result(NewShape);
    std::memcpy(Result.Values, Values, Total * sizeof(Real));

    return Result;
}

FTensor FTensor::Transposed(size_t DimA, size_t DimB) const
{
    assert(DimA < Rank() && DimB < Rank());

    std::vector<size_t> NewShape = Shape;
    std::swap(NewShape[DimA], NewShape[DimB]);

    FTensor Result(NewShape);

    // 일반 차원에서 축을 맞바꾸려면 첨자를 풀었다 다시 조립해야 한다.
    std::vector<size_t> Index(Rank(), 0);

    for (size_t Flat = 0; Flat < Total; Flat++)
    {
        // 평평한 번호를 각 축의 첨자로 푼다.
        size_t Remaining = Flat;
        for (size_t d = Rank(); d-- > 0; )
        {
            Index[d] = Remaining % Shape[d];
            Remaining /= Shape[d];
        }

        // 두 축을 바꿔 새 자리를 계산한다.
        std::swap(Index[DimA], Index[DimB]);

        size_t Target = 0;
        for (size_t d = 0; d < Rank(); d++)
        {
            Target = Target * NewShape[d] + Index[d];
        }

        Result.Values[Target] = Values[Flat];

        std::swap(Index[DimA], Index[DimB]);   // 원래대로 돌려둔다
    }

    return Result;
}

FTensor operator+(FTensor Left, const FTensor& Right)
{
    assert(Left.Count() == Right.Count());

    for (size_t i = 0; i < Left.Count(); i++)
    {
        Left.At(i) += Right.At(i);
    }

    return Left;
}

FTensor operator*(FTensor Left, const FTensor& Right)
{
    assert(Left.Count() == Right.Count());

    for (size_t i = 0; i < Left.Count(); i++)
    {
        Left.At(i) *= Right.At(i);
    }

    return Left;
}

FTensor operator*(FTensor Left, Real Scale)
{
    for (size_t i = 0; i < Left.Count(); i++)
    {
        Left.At(i) *= Scale;
    }
    return Left;
}

FTensor operator*(Real Scale, FTensor Right)
{
    return std::move(Right) * Scale;
}

FTensor MatMul(const FTensor& A, const FTensor& B)
{
    assert(A.Rank() >= 2 && B.Rank() >= 2);

    const size_t M = A.Size(A.Rank() - 2);
    const size_t K = A.Size(A.Rank() - 1);
    const size_t N = B.Size(B.Rank() - 1);

    assert(K == B.Size(B.Rank() - 2));

    // 앞쪽 축(배치·헤드)은 몇 묶음인지만 센다.
    size_t Batches = 1;
    for (size_t d = 0; d + 2 < A.Rank(); d++)
    {
        Batches *= A.Size(d);
    }

    std::vector<size_t> Shape = A.GetShape();
    Shape[Shape.size() - 1] = N;

    FTensor Result(Shape);

    // 묶음과 출력 줄을 한 줄로 펴서 나눈다. (D8 에서 추가)
    //
    // 나누는 축이 **출력 줄**이라는 점이 중요하다. 한 줄의 합을 쪼개지
    // 않으므로 더하는 순서가 안 바뀐다. 스레드 수를 바꿔도 결과가 같다.
    // D7 의 정답표 대조가 그대로 유지되는 이유다.
    //
    // 작은 행렬에서는 스레드를 깨우는 값이 더 비싸다. 그래서 if 를 단다.
    const int Rows = (int)(Batches * M);
    const long long Work = (long long)Rows * (long long)K * (long long)N;

    #pragma omp parallel for schedule(static) if (Work > 200000)
    for (int Flat = 0; Flat < Rows; Flat++)
    {
        const size_t b = (size_t)Flat / M;
        const size_t i = (size_t)Flat % M;

        const Real* Left = A.Data() + b * M * K;
        const Real* Right = B.Data() + ((B.Rank() > 2) ? (b * K * N) : 0);
        Real* OutRow = Result.Data() + b * M * N + i * N;

        for (size_t k = 0; k < K; k++)
        {
            Real Scale = Left[i * K + k];
            const Real* Row = Right + k * N;

            for (size_t j = 0; j < N; j++)
            {
                OutRow[j] += Scale * Row[j];
            }
        }
    }

    return Result;
}

Real Sum(const FTensor& T)
{
    double Total = 0.0;
    for (size_t i = 0; i < T.Count(); i++)
    {
        Total += (double)T.At(i);
    }
    return (Real)Total;
}
