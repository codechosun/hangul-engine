// ch-A02-frequency/Main.c
//
// A2. 빈도 집계와 정렬
//
// 저장소 루트에서 실행할 것. 경로를 인자로 넘길 수도 있다.
//     Main.exe [코퍼스경로] [정답표경로]

#include "Test.h"
#include "Golden.h"
#include "Freq.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_CORPUS "data/corpus.txt"
#define DEFAULT_GOLDEN "data/expected.csv"

// 코드포인트를 사람이 읽을 수 있는 형태로 바꾼다.
static void DescribeCode(uint32_t Code, char* Out, int OutSize)
{
    if (Code == ' ')  { snprintf(Out, OutSize, "공백");   return; }
    if (Code == '\n') { snprintf(Out, OutSize, "줄바꿈"); return; }
    if (Code == '\t') { snprintf(Out, OutSize, "탭");     return; }

    if (Code < 0x20u)
    {
        snprintf(Out, OutSize, "제어문자");
        return;
    }

    char Utf8[8] = { 0 };
    int Len = Utf8Encode(Code, Utf8);
    Utf8[Len] = '\0';
    snprintf(Out, OutSize, "%s", Utf8);
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;
    const char* GoldenPath = (argc > 2) ? argv[2] : DEFAULT_GOLDEN;

    printf("A2. 빈도 집계와 정렬\n\n");
    printf("  코퍼스 : %s\n", CorpusPath);
    printf("  정답표 : %s\n\n", GoldenPath);

    // ---- 1. 센다 ----
    FCharCount* Table = NULL;
    int Distinct = 0;
    uint64_t Total = 0;

    printf("세는 중...\n");
    if (!CHECK(FreqCountFile(CorpusPath, &Table, &Distinct, &Total)))
    {
        printf("\n코퍼스를 못 읽었다. 먼저 준비할 것:\n");
        printf("    .venv/Scripts/python tools/download_corpus.py\n");
        return ReportResult();
    }

    // ---- 2. 정렬 ----
    FreqSort(Table, Distinct);

    printf("\n  서로 다른 글자 %d종\n", Distinct);
    printf("  전체 글자 %llu개\n\n", (unsigned long long)Total);

    // ---- 3. 상위 10종 ----
    printf("  상위 10종\n");
    for (int Rank = 0; Rank < 10 && Rank < Distinct; Rank++)
    {
        char Name[32];
        DescribeCode(Table[Rank].Codepoint, Name, sizeof(Name));

        double Share = (double)Table[Rank].Count * 100.0 / (double)Total;
        printf("    %2d. %-8s U+%04X  %12llu  %5.2f%%\n",
               Rank + 1, Name, Table[Rank].Codepoint,
               (unsigned long long)Table[Rank].Count, Share);
    }
    printf("\n");

    // ---- 4. 정답표와 대조 ----
    FGoldenRow* Rows = NULL;
    int RowCount = 0;

    if (!CHECK(GoldenLoad(GoldenPath, &Rows, &RowCount)))
    {
        free(Table);
        return ReportResult();
    }

    CHECK(RowCount == Distinct);

    int Mismatch = 0;
    int Limit = (RowCount < Distinct) ? RowCount : Distinct;

    for (int i = 0; i < Limit; i++)
    {
        if (Rows[i].Key != Table[i].Codepoint || Rows[i].Value != Table[i].Count)
        {
            if (Mismatch < 3)   // 처음 몇 개만 보여준다
            {
                printf("  [다름] %d번째 줄\n", i + 1);
                printf("         정답표 U+%04X %llu\n",
                       Rows[i].Key, (unsigned long long)Rows[i].Value);
                printf("         내 결과 U+%04X %llu\n",
                       Table[i].Codepoint, (unsigned long long)Table[i].Count);
            }
            Mismatch++;
        }
    }

    CHECK(Mismatch == 0);

    // ---- 5. 정렬 규칙이 지켜졌는지 ----
    int OrderBroken = 0;
    for (int i = 0; i + 1 < Distinct; i++)
    {
        if (FreqCompare(&Table[i], &Table[i + 1]) > 0)
        {
            OrderBroken++;
        }
    }
    CHECK(OrderBroken == 0);

    free(Rows);
    free(Table);
    return ReportResult();
}
