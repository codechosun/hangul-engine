// common/Test.c

#include "Test.h"

#include <stdio.h>
#include <math.h>

int GCheckCount = 0;
int GFailCount = 0;

int CheckImpl(int Ok, const char* Expr, const char* File, int Line)
{
    GCheckCount++;

    if (Ok)
    {
        return 1;
    }

    GFailCount++;
    printf("[실패] %s(%d)\n", File, Line);
    printf("       CHECK(%s)\n", Expr);
    return 0;
}

int CheckNearImpl(double A, double B, double Tol,
                  const char* ExprA, const char* ExprB,
                  const char* File, int Line)
{
    GCheckCount++;

    double Diff = fabs(A - B);
    if (Diff <= Tol)
    {
        return 1;
    }

    GFailCount++;
    printf("[실패] %s(%d)\n", File, Line);
    printf("       CHECK_NEAR(%s, %s)\n", ExprA, ExprB);
    printf("       %.17g 와 %.17g 의 차이 %.17g (허용 %.17g)\n",
           A, B, Diff, Tol);
    return 0;
}

int ReportResult(void)
{
    int PassCount = GCheckCount - GFailCount;

    printf("\n검사 %d개 중 %d개 통과", GCheckCount, PassCount);
    if (GFailCount > 0)
    {
        printf(", %d개 실패", GFailCount);
    }
    printf("\n");

    return (GFailCount == 0) ? 0 : 1;
}
