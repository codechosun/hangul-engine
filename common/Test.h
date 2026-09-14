// common/Test.h
//
// 교재 전체에서 쓰는 최소 검증 도구.
// A0에서 만들고 마지막 장까지 이것만 쓴다.

#ifndef TEST_H
#define TEST_H

// 지금까지 실행한 검사 수와 그중 실패한 수.
extern int GCheckCount;
extern int GFailCount;

// 참이어야 하는 식을 검사한다.
// 실패해도 멈추지 않고 다음 검사를 계속한다.
#define CHECK(Expr) \
    CheckImpl((Expr) ? 1 : 0, #Expr, __FILE__, __LINE__)

// 두 실수가 허용 오차 Tol 안에 있는지 검사한다.
// 실수는 == 로 비교하면 안 되기 때문에 따로 둔다.
#define CHECK_NEAR(A, B, Tol) \
    CheckNearImpl((double)(A), (double)(B), (double)(Tol), \
                  #A, #B, __FILE__, __LINE__)

int CheckImpl(int Ok, const char* Expr, const char* File, int Line);

int CheckNearImpl(double A, double B, double Tol,
                  const char* ExprA, const char* ExprB,
                  const char* File, int Line);

// 결과를 출력하고 종료 코드를 돌려준다. 전부 통과면 0.
int ReportResult(void);

#endif
