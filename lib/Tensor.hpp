// lib/Tensor.hpp
//
// N차원 실수 배열. 이 교재의 C++ 교육이 도달하는 정점이다.
//
// 왜 FMatrix 로는 안 되는가
//   트랜스포머의 어텐션은 (배치, 헤드, 위치, 차원) 네 축을 동시에 다룬다.
//   FMatrix 로 하려면 첨자 계산을 손으로 써야 한다.
//
//       ((n * HeadCount + h) * Length + t) * Dim + d
//
//   이 식이 코드 곳곳에 흩어지면 반드시 한 군데를 틀린다. 그리고
//   차원 크기가 우연히 같으면 **테스트를 통과한다.**
//
// 왜 std::vector 를 멤버로 안 쓰는가
//   B3 의 FVector 는 std::vector 하나만 들고 있어서 복사·이동·소멸을
//   컴파일러가 알아서 해줬다("0의 규칙"). 여기서는 일부러 날 포인터를
//   들고 **다섯 개를 직접 쓴다**("5의 규칙"). 그래야 이동 시맨틱이
//   무엇을 하는지가 코드에 보이고, 그 효과를 숫자로 잴 수 있다.
//
//   실무에서는 std::vector 를 쓰는 편이 옳다. 여기서는 배우려고 만든다.

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "Types.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

class FTensor
{
public:
    // ---- 만들고 없애기 ----
    FTensor() = default;

    explicit FTensor(std::initializer_list<size_t> InShape);
    explicit FTensor(const std::vector<size_t>& InShape);

    // ---- 5의 규칙 ----
    //
    // 자원을 직접 들고 있으면 이 다섯을 전부 써야 한다.
    // 하나라도 빠뜨리면 이중 해제나 누수가 난다.
    FTensor(const FTensor& Other);                 // 복사 생성
    FTensor& operator=(const FTensor& Other);      // 복사 대입
    FTensor(FTensor&& Other) noexcept;             // 이동 생성
    FTensor& operator=(FTensor&& Other) noexcept;  // 이동 대입
    ~FTensor();                                    // 소멸

    // ---- 모양 ----
    size_t Rank() const { return Shape.size(); }
    size_t Count() const { return Total; }
    size_t Size(size_t Dim) const { return Shape[Dim]; }
    const std::vector<size_t>& GetShape() const { return Shape; }

    // ---- 값 ----
    Real* Data() { return Values; }
    const Real* Data() const { return Values; }

    Real& At(size_t Index) { return Values[Index]; }
    const Real& At(size_t Index) const { return Values[Index]; }

    // 첨자 계산이 사는 유일한 자리.
    Real& operator()(size_t A);
    Real& operator()(size_t A, size_t B);
    Real& operator()(size_t A, size_t B, size_t C);
    Real& operator()(size_t A, size_t B, size_t C, size_t D);

    const Real& operator()(size_t A) const;
    const Real& operator()(size_t A, size_t B) const;
    const Real& operator()(size_t A, size_t B, size_t C) const;
    const Real& operator()(size_t A, size_t B, size_t C, size_t D) const;

    void Fill(Real Value);

    // ---- 모양 바꾸기 ----
    //
    // 원소 개수가 같으면 모양만 바꾼다. 값은 하나도 안 움직인다.
    FTensor Reshaped(const std::vector<size_t>& NewShape) const;

    // 두 축을 맞바꾼다. 이쪽은 **값을 실제로 옮긴다.**
    FTensor Transposed(size_t DimA, size_t DimB) const;

    // ---- 통계 ----
    //
    // 복사가 몇 번 일어났는지 세어둔다. D1 에서 이동 시맨틱의 효과를
    // 숫자로 보이려고 만든 것이다. 실무 코드에는 안 들어간다.
    struct FCounters
    {
        uint64_t Allocation = 0;
        uint64_t Copy = 0;
        uint64_t Move = 0;
        uint64_t ElementsCopied = 0;
    };

    static const FCounters& Counters();
    static void ResetCounters();

private:
    void Allocate(const std::vector<size_t>& InShape);

    Real* Values = nullptr;
    std::vector<size_t> Shape;
    size_t Total = 0;
};

// 원소별 덧셈. 모양이 같아야 한다.
FTensor operator+(FTensor Left, const FTensor& Right);

// **원소별** 곱셈이다. 내적이 아니다.
//
// B3 의 FVector 에서는 operator* 를 내적으로 뒀다. 그때 적어둔 대로
// 텐서에서는 원소별 곱이 훨씬 자주 나오므로 여기서는 반대로 정한다.
// 내적이 필요하면 이름 있는 함수를 쓴다.
FTensor operator*(FTensor Left, const FTensor& Right);

FTensor operator*(FTensor Left, Real Scale);
FTensor operator*(Real Scale, FTensor Right);

// 마지막 두 축을 행렬로 보고 곱한다. (D2 에서 쓴다)
FTensor MatMul(const FTensor& A, const FTensor& B);

// 모든 원소의 합.
Real Sum(const FTensor& T);

#endif
