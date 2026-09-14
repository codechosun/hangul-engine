// ch-A00-skeleton/Main.c
//
// A0-4. 검증 도구가 제대로 동작하는지 확인한다.
// 마지막 검사는 일부러 실패시켜서 실패 출력이 어떻게 생겼는지 본다.

#include "Test.h"
#include "Types.h"

#include <stdio.h>

int Add(int A, int B)
{
    return A + B;
}

int main(void)
{
    printf("A0. 프로젝트 골격과 검증 도구\n\n");

    // 통과하는 검사
    CHECK(Add(2, 3) == 5);
    CHECK(Add(-1, 1) == 0);

    // 실수 비교. 0.1 을 열 번 더해도 정확히 1 이 되지 않는다.
    Real Sum = (Real)0;
    for (int i = 0; i < 10; i++)
    {
        Sum += (Real)0.1;
    }
    CHECK_NEAR(Sum, 1.0, 1e-6);

    // 일부러 실패시킨다. 이 줄을 지우면 종료 코드가 0 이 된다.
    CHECK(Add(2, 2) == 5);

    return ReportResult();
}
