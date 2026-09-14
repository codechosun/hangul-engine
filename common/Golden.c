// common/Golden.c

#include "Golden.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// 처음에 잡아둘 칸 수. 모자라면 두 배씩 늘린다.
#define INITIAL_CAPACITY 1024

int GoldenLoad(const char* Path, FGoldenRow** OutRows, int* OutCount)
{
    assert(Path != NULL);
    assert(OutRows != NULL);
    assert(OutCount != NULL);

    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0;
    }

    int Capacity = INITIAL_CAPACITY;
    int Count = 0;

    FGoldenRow* Rows = (FGoldenRow*)malloc((size_t)Capacity * sizeof(FGoldenRow));
    if (Rows == NULL)
    {
        fclose(File);
        return 0;
    }

    char Line[256];

    // 첫 줄은 헤더다. 읽고 버린다.
    if (fgets(Line, sizeof(Line), File) == NULL)
    {
        free(Rows);
        fclose(File);
        return 0;
    }

    while (fgets(Line, sizeof(Line), File) != NULL)
    {
        unsigned long long Key = 0;
        unsigned long long Value = 0;

        // 줄 끝이 LF 든 CRLF 든 %llu 가 알아서 멈춘다.
        if (sscanf(Line, "%llu,%llu", &Key, &Value) != 2)
        {
            continue;   // 빈 줄이나 깨진 줄은 건너뛴다.
        }

        if (Count == Capacity)
        {
            int NewCapacity = Capacity * 2;
            FGoldenRow* Grown =
                (FGoldenRow*)realloc(Rows, (size_t)NewCapacity * sizeof(FGoldenRow));

            if (Grown == NULL)
            {
                free(Rows);
                fclose(File);
                return 0;
            }

            Rows = Grown;
            Capacity = NewCapacity;
        }

        Rows[Count].Key = (uint32_t)Key;
        Rows[Count].Value = (uint64_t)Value;
        Count++;
    }

    fclose(File);

    *OutRows = Rows;
    *OutCount = Count;
    return 1;
}
