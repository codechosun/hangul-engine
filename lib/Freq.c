// lib/Freq.c

#include "Freq.h"
#include "Scan.h"
#include "Utf8.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int FreqCompare(const void* A, const void* B)
{
    const FCharCount* Left = (const FCharCount*)A;
    const FCharCount* Right = (const FCharCount*)B;

    // 1순위: 빈도 내림차순.
    // 빼서 돌려주면 안 된다. uint64 끼리 빼면 넘쳐서 부호가 뒤집힌다.
    if (Left->Count > Right->Count) return -1;
    if (Left->Count < Right->Count) return 1;

    // 2순위: 코드포인트 오름차순.
    if (Left->Codepoint < Right->Codepoint) return -1;
    if (Left->Codepoint > Right->Codepoint) return 1;

    return 0;
}

void FreqSort(FCharCount* Table, int Count)
{
    assert(Table != NULL);
    qsort(Table, (size_t)Count, sizeof(FCharCount), FreqCompare);
}

int FreqCountFile(const char* Path,
                  FCharCount** OutTable, int* OutCount, uint64_t* OutTotal)
{
    assert(Path != NULL);
    assert(OutTable != NULL);
    assert(OutCount != NULL);

    FScanner Scanner;
    if (!ScanOpen(&Scanner, Path))
    {
        return 0;
    }

    // 코드포인트를 그대로 인덱스로 쓴다. 8.9MB 정도다.
    uint64_t* Counts = (uint64_t*)calloc(CODEPOINT_LIMIT, sizeof(uint64_t));
    if (Counts == NULL)
    {
        ScanClose(&Scanner);
        return 0;
    }

    uint64_t Total = 0;
    uint32_t Code = 0;

    while (ScanNext(&Scanner, &Code))
    {
        Counts[Code]++;
        Total++;
    }

    ScanClose(&Scanner);

    // 실제로 나온 글자만 추려 담는다.
    int Distinct = 0;
    for (uint32_t i = 0; i < CODEPOINT_LIMIT; i++)
    {
        if (Counts[i] > 0)
        {
            Distinct++;
        }
    }

    FCharCount* Table = (FCharCount*)malloc((size_t)Distinct * sizeof(FCharCount));
    if (Table == NULL)
    {
        free(Counts);
        return 0;
    }

    int Index = 0;
    for (uint32_t i = 0; i < CODEPOINT_LIMIT; i++)
    {
        if (Counts[i] > 0)
        {
            Table[Index].Codepoint = i;
            Table[Index].Count = Counts[i];
            Index++;
        }
    }

    free(Counts);

    *OutTable = Table;
    *OutCount = Distinct;
    if (OutTotal != NULL)
    {
        *OutTotal = Total;
    }

    return 1;
}
