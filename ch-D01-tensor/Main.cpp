// ch-D01-tensor/Main.cpp
//
// D1. Tensor 타입 설계
//
// 저장소 루트에서 실행할 것.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Random.h"
#include "Tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#define BOOK_SEED 20260914ull

// 어텐션 크기. (배치, 헤드, 위치, 차원)
#define BATCH  2
#define HEADS  4
#define LENGTH 8
#define DIM    16

namespace
{

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

// 텐서를 값으로 돌려주는 함수. 이동이 없으면 여기서 복사가 일어난다.
FTensor MakeTensor(size_t Rows, size_t Cols, Real Value)
{
    FTensor Result({ Rows, Cols });
    Result.Fill(Value);
    return Result;
}

void PrintCounters(const char* Name)
{
    const FTensor::FCounters& C = FTensor::Counters();

    char Buffer[64];

    printf("  ");
    PrintPadded(Name, 26);

    snprintf(Buffer, sizeof(Buffer), "%llu", (unsigned long long)C.Allocation);
    PrintPaddedRight(Buffer, 10);

    snprintf(Buffer, sizeof(Buffer), "%llu", (unsigned long long)C.Copy);
    PrintPaddedRight(Buffer, 10);

    snprintf(Buffer, sizeof(Buffer), "%llu", (unsigned long long)C.Move);
    PrintPaddedRight(Buffer, 10);

    snprintf(Buffer, sizeof(Buffer), "%llu",
             (unsigned long long)C.ElementsCopied);
    PrintPaddedRight(Buffer, 16);

    printf("\n");
}

} // namespace

int main(void)
{
    printf("D1. Tensor 타입 설계\n\n");

    // ---- D1-1. 첨자를 손으로 쓰면 ----
    printf("[D1-1] 4차원 첨자를 손으로 쓰면\n\n");

    {
        const size_t N = BATCH, H = HEADS, T = LENGTH, D = DIM;

        std::vector<Real> Raw(N * H * T * D, Real(0));

        // 맞는 식
        auto Right = [&](size_t n, size_t h, size_t t, size_t d)
        {
            return ((n * H + h) * T + t) * D + d;
        };

        // 자주 나오는 실수. H 와 T 를 바꿔 썼다.
        auto Wrong = [&](size_t n, size_t h, size_t t, size_t d)
        {
            return ((n * T + h) * H + t) * D + d;
        };

        printf("  모양 = (%zu, %zu, %zu, %zu)\n", N, H, T, D);
        printf("  맞는 식   ((n*H + h)*T + t)*D + d\n");
        printf("  틀린 식   ((n*T + h)*H + t)*D + d   <- H 와 T 를 바꿔 썼다\n\n");

        int Different = 0;
        size_t WorstIndex = 0;

        for (size_t n = 0; n < N; n++)
        for (size_t h = 0; h < H; h++)
        for (size_t t = 0; t < T; t++)
        for (size_t d = 0; d < D; d++)
        {
            if (Right(n, h, t, d) != Wrong(n, h, t, d))
            {
                Different++;
                WorstIndex = Right(n, h, t, d);
            }
        }

        printf("  %zu개 자리 중 %d개가 어긋난다\n", Raw.size(), Different);
        printf("  (마지막으로 어긋난 자리 %zu)\n\n", WorstIndex);

        CHECK(Different > 0);

        // 그런데 H == T 이면 두 식이 **같아진다.**
        int SameWhenEqual = 1;
        const size_t Square = 8;
        for (size_t n = 0; n < 2; n++)
        for (size_t h = 0; h < Square; h++)
        for (size_t t = 0; t < Square; t++)
        for (size_t d = 0; d < 4; d++)
        {
            size_t A = ((n * Square + h) * Square + t) * 4 + d;
            size_t B = ((n * Square + h) * Square + t) * 4 + d;
            if (A != B)
            {
                SameWhenEqual = 0;
            }
        }

        printf("  **헤드 수와 위치 수가 같으면 두 식이 같은 답을 낸다.**\n");
        printf("  테스트를 (2, 8, 8, 4) 로 짜면 이 버그가 안 잡힌다.\n\n");

        CHECK(SameWhenEqual == 1);
    }

    // ---- D1-2. 첨자를 한 군데로 ----
    printf("[D1-2] FTensor 는 그 식을 한 군데에 가둔다\n\n");

    {
        FTensor T({ BATCH, HEADS, LENGTH, DIM });

        printf("  차원 수   = %zu\n", T.Rank());
        printf("  원소 수   = %zu\n", T.Count());
        printf("  모양      = (%zu, %zu, %zu, %zu)\n\n",
               T.Size(0), T.Size(1), T.Size(2), T.Size(3));

        CHECK(T.Rank() == 4);
        CHECK(T.Count() == (size_t)(BATCH * HEADS * LENGTH * DIM));

        // 모든 자리에 서로 다른 값을 넣고 되찾아본다.
        for (size_t n = 0; n < BATCH; n++)
        for (size_t h = 0; h < HEADS; h++)
        for (size_t t = 0; t < LENGTH; t++)
        for (size_t d = 0; d < DIM; d++)
        {
            T(n, h, t, d) = (Real)(((n * HEADS + h) * LENGTH + t) * DIM + d);
        }

        int AllRight = 1;
        for (size_t n = 0; n < BATCH; n++)
        for (size_t h = 0; h < HEADS; h++)
        for (size_t t = 0; t < LENGTH; t++)
        for (size_t d = 0; d < DIM; d++)
        {
            Real Want = (Real)(((n * HEADS + h) * LENGTH + t) * DIM + d);
            if (T(n, h, t, d) != Want)
            {
                AllRight = 0;
            }
        }

        printf("  모든 자리에 값을 넣고 되찾았는가 = %s\n\n",
               AllRight ? "예" : "아니오");
        CHECK(AllRight == 1);
    }

    // ---- D1-3. 복사는 비싸다 ----
    printf("[D1-3] 복사가 몇 번 일어나는가\n\n");

    {
        printf("  ");
        PrintPadded("", 26);
        PrintPaddedRight("할당", 10);
        PrintPaddedRight("복사", 10);
        PrintPaddedRight("이동", 10);
        PrintPaddedRight("복사한 원소", 16);
        printf("\n");

        // (가) 값으로 돌려받기
        FTensor::ResetCounters();
        {
            FTensor A = MakeTensor(256, 256, Real(1));
            if (A.Count() == 0) printf("never\n");
        }
        PrintCounters("함수에서 값으로 받기");

        uint64_t ReturnCopies = FTensor::Counters().Copy;

        // (나) 일부러 복사
        FTensor::ResetCounters();
        {
            FTensor A = MakeTensor(256, 256, Real(1));
            FTensor B = A;                  // 복사 생성
            if (B.Count() == 0) printf("never\n");
        }
        PrintCounters("A 를 B 에 복사");

        uint64_t ExplicitCopies = FTensor::Counters().Copy;

        // (다) 명시적으로 이동
        FTensor::ResetCounters();
        {
            FTensor A = MakeTensor(256, 256, Real(1));
            FTensor B = std::move(A);       // 이동 생성
            if (B.Count() == 0) printf("never\n");
        }
        PrintCounters("A 를 B 로 이동");

        uint64_t MoveCopies = FTensor::Counters().Copy;

        // (라) vector 에 담기
        FTensor::ResetCounters();
        {
            std::vector<FTensor> Bag;
            for (int i = 0; i < 8; i++)
            {
                Bag.push_back(MakeTensor(256, 256, Real(1)));
            }
        }
        PrintCounters("vector 에 8개 담기");

        uint64_t BagCopies = FTensor::Counters().Copy;

        printf("\n");

        // 값으로 돌려받는 데 복사가 없어야 한다.
        CHECK(ReturnCopies == 0);

        // 명시적 복사는 한 번.
        CHECK(ExplicitCopies == 1);

        // std::move 를 쓰면 복사가 없다.
        CHECK(MoveCopies == 0);

        // vector 가 커질 때도 복사가 없어야 한다. noexcept 덕분이다.
        CHECK(BagCopies == 0);

        printf("  **함수에서 텐서를 값으로 돌려줘도 복사가 없다.**\n");
        printf("  이동 생성자가 버퍼를 훔쳐 오기 때문이다.\n\n");

        printf("  vector 가 커질 때도 복사가 0 인 것에 주목할 것.\n");
        printf("  이동 생성자에 noexcept 를 안 붙이면 **복사로 되돌아간다.**\n\n");
    }

    // ---- D1-4. 복사와 이동의 값 차이 ----
    printf("[D1-4] 복사와 이동의 값 차이\n\n");

    {
        const size_t Size = 1024;
        const int Count = 8;
        const int Rounds = 50;

        // 1024 x 1024 텐서 여덟 개. 통틀어 32MB.
        std::vector<FTensor> Bag;
        for (int i = 0; i < Count; i++)
        {
            Bag.push_back(FTensor({ Size, Size }));
        }

        FTensor::ResetCounters();

        auto Begin = std::chrono::steady_clock::now();
        for (int r = 0; r < Rounds; r++)
        {
            std::vector<FTensor> Copied = Bag;          // 여덟 개를 전부 복사
            if (Copied.size() != Count) printf("never\n");
        }
        double CopySeconds = Seconds(Begin);

        uint64_t CopiedElements = FTensor::Counters().ElementsCopied;

        FTensor::ResetCounters();

        Begin = std::chrono::steady_clock::now();
        for (int r = 0; r < Rounds; r++)
        {
            std::vector<FTensor> Source = std::move(Bag);   // 통째로 이동
            Bag = std::move(Source);                        // 다시 가져온다
            if (Bag.size() != Count) printf("never\n");
        }
        double MoveSeconds = Seconds(Begin);

        uint64_t MovedElements = FTensor::Counters().ElementsCopied;

        printf("  %zu x %zu 텐서 %d개 (%.0f MB) 를 %d번 주고받기\n\n",
               Size, Size, Count,
               (double)Count * Size * Size * sizeof(Real) / (1024.0 * 1024.0),
               Rounds);

        printf("  복사 = %.3f 초, 옮긴 원소 %llu개\n", CopySeconds,
               (unsigned long long)CopiedElements);
        printf("  이동 = %.3f 초, 옮긴 원소 %llu개\n", MoveSeconds,
               (unsigned long long)MovedElements);
        if (MoveSeconds < 1e-4)
        {
            printf("  이동 쪽은 아예 재어지지 않는다 (시계 해상도 아래)\n\n");
        }
        else
        {
            printf("  %.0f 배 빠르다\n\n", CopySeconds / MoveSeconds);
        }

        CHECK(MoveSeconds < CopySeconds);
        CHECK(MovedElements == 0);
        CHECK(CopiedElements > 0);

        printf("  이동은 포인터를 옮기는 일이라 **크기와 무관**하다.\n");
        printf("  복사는 %llu개를 옮긴다.\n\n",
               (unsigned long long)CopiedElements);
    }

    // ---- D1-5. 모양 바꾸기 ----
    printf("[D1-5] 같은 값을 다른 모양으로\n\n");

    {
        FTensor T({ 2, 3, 4 });
        for (size_t i = 0; i < T.Count(); i++)
        {
            T.At(i) = (Real)i;
        }

        FTensor Flat = T.Reshaped({ 24 });
        FTensor Wide = T.Reshaped({ 6, 4 });
        FTensor Deep = T.Reshaped({ 2, 12 });

        printf("  (2,3,4) -> (24), (6,4), (2,12)\n");
        printf("  원소 수 = %zu, %zu, %zu, %zu\n\n",
               T.Count(), Flat.Count(), Wide.Count(), Deep.Count());

        CHECK(Flat.Count() == T.Count());
        CHECK(Wide.Count() == T.Count());

        // 모양만 바뀌고 값의 순서는 그대로다.
        int SameOrder = 1;
        for (size_t i = 0; i < T.Count(); i++)
        {
            if (Flat.At(i) != T.At(i) || Wide.At(i) != T.At(i))
            {
                SameOrder = 0;
            }
        }

        printf("  값의 순서가 그대로인가 = %s\n", SameOrder ? "예" : "아니오");
        CHECK(SameOrder == 1);

        // (1,2) 자리는 두 모양에서 같은 원소를 가리킨다.
        printf("  T(1,2,3) = %.0f, Wide(5,3) = %.0f (같은 원소)\n\n",
               (double)T(1, 2, 3), (double)Wide(5, 3));

        CHECK(T(1, 2, 3) == Wide(5, 3));
    }

    // ---- D1-6. 전치는 값을 옮긴다 ----
    printf("[D1-6] 축 바꾸기\n\n");

    {
        FTensor T({ BATCH, HEADS, LENGTH, DIM });
        for (size_t i = 0; i < T.Count(); i++)
        {
            T.At(i) = (Real)i;
        }

        // 어텐션에서 실제로 하는 일. (배치, 위치, 헤드, 차원) 으로 본다.
        FTensor Swapped = T.Transposed(1, 2);

        printf("  (%zu,%zu,%zu,%zu) -> (%zu,%zu,%zu,%zu)\n",
               T.Size(0), T.Size(1), T.Size(2), T.Size(3),
               Swapped.Size(0), Swapped.Size(1), Swapped.Size(2),
               Swapped.Size(3));

        CHECK(Swapped.Size(1) == LENGTH);
        CHECK(Swapped.Size(2) == HEADS);
        CHECK(Swapped.Count() == T.Count());

        int AllRight = 1;
        for (size_t n = 0; n < BATCH; n++)
        for (size_t h = 0; h < HEADS; h++)
        for (size_t t = 0; t < LENGTH; t++)
        for (size_t d = 0; d < DIM; d++)
        {
            if (Swapped(n, t, h, d) != T(n, h, t, d))
            {
                AllRight = 0;
            }
        }

        printf("  Swapped(n,t,h,d) == T(n,h,t,d) 인가 = %s\n",
               AllRight ? "예" : "아니오");
        CHECK(AllRight == 1);

        // 두 번 바꾸면 제자리다.
        FTensor Back = Swapped.Transposed(1, 2);
        int SameAgain = 1;
        for (size_t i = 0; i < T.Count(); i++)
        {
            if (Back.At(i) != T.At(i))
            {
                SameAgain = 0;
            }
        }

        printf("  두 번 바꾸면 제자리인가 = %s\n\n", SameAgain ? "예" : "아니오");
        CHECK(SameAgain == 1);

        printf("  모양 바꾸기는 값을 안 옮기고, 축 바꾸기는 옮긴다.\n");
        printf("  이름이 비슷해 보여도 비용이 전혀 다르다.\n\n");
    }

    // ---- D1-7. 연산자 ----
    printf("[D1-7] 연산자 — B3 와 반대로 정했다\n\n");

    {
        FTensor A({ 2, 2 });
        A(0, 0) = Real(1); A(0, 1) = Real(2);
        A(1, 0) = Real(3); A(1, 1) = Real(4);

        FTensor B({ 2, 2 });
        B(0, 0) = Real(5); B(0, 1) = Real(6);
        B(1, 0) = Real(7); B(1, 1) = Real(8);

        FTensor Added = A + B;
        FTensor Multiplied = A * B;          // **원소별** 곱
        FTensor Scaled = A * Real(10);
        FTensor Product = MatMul(A, B);      // 행렬 곱은 이름 있는 함수

        printf("  A = [1 2; 3 4], B = [5 6; 7 8]\n\n");
        printf("  A + B      = [%.0f %.0f; %.0f %.0f]  (손계산 [6 8; 10 12])\n",
               (double)Added(0,0), (double)Added(0,1),
               (double)Added(1,0), (double)Added(1,1));
        printf("  A * B      = [%.0f %.0f; %.0f %.0f]  (원소별. 손계산 [5 12; 21 32])\n",
               (double)Multiplied(0,0), (double)Multiplied(0,1),
               (double)Multiplied(1,0), (double)Multiplied(1,1));
        printf("  A * 10     = [%.0f %.0f; %.0f %.0f]\n",
               (double)Scaled(0,0), (double)Scaled(0,1),
               (double)Scaled(1,0), (double)Scaled(1,1));
        printf("  MatMul(A,B)= [%.0f %.0f; %.0f %.0f]  (손계산 [19 22; 43 50])\n\n",
               (double)Product(0,0), (double)Product(0,1),
               (double)Product(1,0), (double)Product(1,1));

        CHECK_NEAR(Added(0, 0), 6.0, 1e-6);
        CHECK_NEAR(Added(1, 1), 12.0, 1e-6);
        CHECK_NEAR(Multiplied(0, 0), 5.0, 1e-6);
        CHECK_NEAR(Multiplied(1, 1), 32.0, 1e-6);
        CHECK_NEAR(Scaled(1, 0), 30.0, 1e-6);
        CHECK_NEAR(Product(0, 0), 19.0, 1e-6);
        CHECK_NEAR(Product(0, 1), 22.0, 1e-6);
        CHECK_NEAR(Product(1, 0), 43.0, 1e-6);
        CHECK_NEAR(Product(1, 1), 50.0, 1e-6);
        CHECK_NEAR(Sum(Added), 36.0, 1e-6);

        // 묶음(배치)이 있는 행렬 곱도 된다.
        FTensor BatchA({ 3, 2, 4 });
        FTensor BatchB({ 3, 4, 5 });
        BatchA.Fill(Real(1));
        BatchB.Fill(Real(2));

        FTensor BatchProduct = MatMul(BatchA, BatchB);

        printf("  (3,2,4) x (3,4,5) = (%zu,%zu,%zu)\n",
               BatchProduct.Size(0), BatchProduct.Size(1),
               BatchProduct.Size(2));
        printf("  모든 값이 4*1*2 = 8 이어야 한다 -> %.0f\n\n",
               (double)BatchProduct(0, 0, 0));

        CHECK(BatchProduct.Rank() == 3);
        CHECK(BatchProduct.Size(2) == 5);
        CHECK_NEAR(BatchProduct(2, 1, 4), 8.0, 1e-6);
    }

    return ReportResult();
}
