// common/Golden.h
//
// 정답표(골든 파일)와 대조하는 도구.
//
// 파이썬이 미리 만들어 저장소에 커밋해둔 CSV 를 읽어들인다.
// 덕분에 독자는 파이썬을 깔지 않고도 자기 C 코드를 검증할 수 있다.

#ifndef GOLDEN_H
#define GOLDEN_H

#include <stdint.h>

// 정답표 한 줄. "키,값" 두 칸짜리 CSV 를 담는다.
typedef struct
{
    uint32_t Key;
    uint64_t Value;
} FGoldenRow;

// "키,값" 형식의 CSV 를 읽는다. 첫 줄은 헤더로 보고 건너뛴다.
//
// OutRows 에 malloc 된 배열을 담아준다. 부른 쪽이 free 해야 한다.
// 파일 순서를 그대로 유지한다. 정렬하지 않는다.
//
// 성공하면 1, 파일을 못 열거나 형식이 깨졌으면 0.
int GoldenLoad(const char* Path, FGoldenRow** OutRows, int* OutCount);

// 정답표 한 줄, 세 칸짜리. "앞,뒤,횟수" 형식에 쓴다. (A5 에서 추가)
typedef struct
{
    uint32_t First;
    uint32_t Second;
    uint64_t Value;
} FGoldenTriple;

// "값,값,값" 형식의 CSV 를 읽는다. 나머지는 GoldenLoad 와 같다.
int GoldenLoadTriple(const char* Path, FGoldenTriple** OutRows, int* OutCount);

#endif
