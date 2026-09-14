// lib/Freq.h
//
// 텍스트 파일에서 코드포인트별 빈도를 센다.

#ifndef FREQ_H
#define FREQ_H

#include <stdint.h>

#include "Utf8.h"

#ifdef __cplusplus
extern "C" {
#endif

// 글자 하나와 그 등장 횟수.
typedef struct
{
    uint32_t Codepoint;
    uint64_t Count;
} FCharCount;

// 파일 전체를 읽어 코드포인트별 빈도를 센다.
//
// OutTable 에 malloc 된 배열을 담아준다. 부른 쪽이 free 해야 한다.
// 정렬은 하지 않는다. FreqSort 를 따로 부를 것.
//
// 성공하면 1, 파일을 못 열거나 메모리가 부족하면 0.
int FreqCountFile(const char* Path,
                  FCharCount** OutTable, int* OutCount, uint64_t* OutTotal);

// 빈도 내림차순, 동점이면 코드포인트 오름차순으로 정렬한다.
//
// 2순위가 없으면 동점인 글자들의 순서가 정렬 구현에 따라 달라져서
// 정답표와 한 줄씩 비교할 수 없게 된다.
void FreqSort(FCharCount* Table, int Count);

// qsort 에 넘기는 비교 함수. 본문에서 따로 다루므로 밖으로 꺼내 두었다.
int FreqCompare(const void* A, const void* B);

#ifdef __cplusplus
}
#endif
#endif
