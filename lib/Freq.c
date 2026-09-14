// lib/Freq.c

#include "Freq.h"
#include "Utf8.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 한 번에 읽어들일 크기. 뒤쪽 여유 8바이트는 잘린 글자를 담아두는 자리.
#define CHUNK_SIZE (1u << 20)

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

    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0;
    }

    // 코드포인트를 그대로 인덱스로 쓴다. 8.9MB 정도다.
    uint64_t* Counts = (uint64_t*)calloc(CODEPOINT_LIMIT, sizeof(uint64_t));
    char* Buffer = (char*)malloc(CHUNK_SIZE + 8);

    if (Counts == NULL || Buffer == NULL)
    {
        free(Counts);
        free(Buffer);
        fclose(File);
        return 0;
    }

    uint64_t Total = 0;
    size_t Leftover = 0;

    while (1)
    {
        size_t Read = fread(Buffer + Leftover, 1, CHUNK_SIZE, File);
        size_t Filled = Leftover + Read;

        if (Filled == 0)
        {
            break;
        }

        size_t Pos = 0;
        while (Pos < Filled)
        {
            unsigned char First = (unsigned char)Buffer[Pos];
            int Need = Utf8SequenceLength(First);

            if (Need == 0)
            {
                Pos++;          // 선행 바이트가 아니다. 한 바이트 버리고 계속.
                continue;
            }

            if (Pos + (size_t)Need > Filled)
            {
                break;          // 청크 경계에서 잘렸다. 다음 번에 이어서 읽는다.
            }

            uint32_t Code = 0;
            if (Utf8Decode(Buffer + Pos, &Code) == Need && Code < CODEPOINT_LIMIT)
            {
                Counts[Code]++;
                Total++;
            }

            Pos += (size_t)Need;
        }

        // 못 읽고 남은 꼬리를 앞으로 당겨 다음 청크와 이어 붙인다.
        Leftover = Filled - Pos;
        memmove(Buffer, Buffer + Pos, Leftover);

        if (Read == 0)
        {
            break;              // 파일 끝. 남은 꼬리는 깨진 바이트이므로 버린다.
        }
    }

    fclose(File);
    free(Buffer);

    // 실제로 나온 글자만 추려 담는다.
    int Distinct = 0;
    for (uint32_t Code = 0; Code < CODEPOINT_LIMIT; Code++)
    {
        if (Counts[Code] > 0)
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
    for (uint32_t Code = 0; Code < CODEPOINT_LIMIT; Code++)
    {
        if (Counts[Code] > 0)
        {
            Table[Index].Codepoint = Code;
            Table[Index].Count = Counts[Code];
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
