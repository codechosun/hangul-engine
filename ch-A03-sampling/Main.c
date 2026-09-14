// ch-A03-sampling/Main.c
//
// A3. 확률과 샘플링
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Freq.h"
#include "Random.h"
#include "Sample.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_CORPUS "data/corpus.txt"

// 뽑을 횟수. 경험 분포가 이론 분포에 얼마나 가까워지는지 본다.
#define DRAW_COUNT 10000000

// 시험 삼아 뽑아보는 횟수.
#define TRY_COUNT 100000

// 교재 전체에서 쓰는 고정 씨앗. 결과가 항상 같아야 하므로 시간을 쓰지 않는다.
#define BOOK_SEED 20260914ull

// 코드포인트를 사람이 읽을 수 있는 짧은 이름으로.
static void DescribeCode(uint32_t Code, char* Out, int OutSize)
{
    if (Code == ' ')  { snprintf(Out, OutSize, "공백");   return; }
    if (Code == '\n') { snprintf(Out, OutSize, "줄바꿈"); return; }
    if (Code < 0x20u) { snprintf(Out, OutSize, "제어");   return; }

    char Utf8[8] = { 0 };
    int Len = Utf8Encode(Code, Utf8);
    Utf8[Len] = '\0';
    snprintf(Out, OutSize, "%s", Utf8);
}

static void PrintCode(uint32_t Code)
{
    char Utf8[8] = { 0 };
    int Len = Utf8Encode(Code, Utf8);
    Utf8[Len] = '\0';
    printf("%s", Utf8);
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A3. 확률과 샘플링\n\n");

    // ---- 빈도표 준비 (A2 재사용) ----
    FCharCount* Table = NULL;
    int Distinct = 0;
    uint64_t Total = 0;

    printf("빈도표를 만드는 중...\n\n");
    if (!CHECK(FreqCountFile(CorpusPath, &Table, &Distinct, &Total)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        return ReportResult();
    }
    FreqSort(Table, Distinct);

    // ---- 누적 분포 ----
    FSampler Sampler;
    if (!CHECK(SamplerInit(&Sampler, Table, Distinct)))
    {
        free(Table);
        return ReportResult();
    }

    // 누적합의 마지막 칸은 전체 개수와 같아야 한다.
    CHECK(Sampler.Total == Total);

    // ---- A3-1. rand() 로 뽑으면 어떻게 되는가 ----
    printf("[A3-1] rand() 로 %d 번 뽑아보기\n", TRY_COUNT);
    printf("  RAND_MAX           = %d\n", RAND_MAX);
    printf("  뽑아야 할 범위     = 0 ~ %llu\n", (unsigned long long)Total - 1);
    printf("  1등 글자가 가진 칸 = 0 ~ %llu\n",
           (unsigned long long)Sampler.Cumulative[0] - 1);

    uint64_t RandBiggest = 0;
    int RandDistinct = 0;
    uint32_t RandSeen = 0;

    for (int i = 0; i < TRY_COUNT; i++)
    {
        uint64_t Target = (uint64_t)rand() % Total;
        if (Target > RandBiggest)
        {
            RandBiggest = Target;
        }

        uint32_t Code = SamplerFind(&Sampler, Target);
        if (i == 0 || Code != RandSeen)
        {
            RandSeen = Code;
            RandDistinct++;
        }
    }

    printf("  뽑은 값의 최대값   = %llu (전체의 %.5f%%)\n",
           (unsigned long long)RandBiggest,
           (double)RandBiggest * 100.0 / (double)Total);

    char SeenName[32];
    DescribeCode(RandSeen, SeenName, sizeof(SeenName));
    printf("  나온 글자 종류     = %d 가지 (%s)\n\n", RandDistinct, SeenName);

    // 첫 칸도 못 벗어난다. 천만 번을 뽑아도 결과는 같다.
    CHECK(RandBiggest < Sampler.Cumulative[0]);
    CHECK(RandDistinct == 1);

    // ---- A3-2. 직접 만든 난수는 전 범위에 닿는다 ----
    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    uint64_t MyBiggest = 0;
    for (int i = 0; i < TRY_COUNT; i++)
    {
        uint64_t Pick = RandomBelow(&Rng, Total);
        if (Pick > MyBiggest)
        {
            MyBiggest = Pick;
        }
    }

    printf("[A3-2] xorshift64 로 %d 번 뽑아보기\n", TRY_COUNT);
    printf("  뽑은 값의 최대값   = %llu (전체의 %.2f%%)\n\n",
           (unsigned long long)MyBiggest,
           (double)MyBiggest * 100.0 / (double)Total);

    CHECK(MyBiggest > Total / 2);   // 절반은 넘어야 정상

    // ---- A3-3. 같은 씨앗은 같은 수열 ----
    FRandom A, B;
    RandomSeed(&A, BOOK_SEED);
    RandomSeed(&B, BOOK_SEED);

    int Same = 1;
    for (int i = 0; i < 1000; i++)
    {
        if (RandomNext(&A) != RandomNext(&B))
        {
            Same = 0;
            break;
        }
    }
    CHECK(Same == 1);

    // ---- A3-4. 확률대로 뽑기 ----
    uint64_t* Drawn = (uint64_t*)calloc((size_t)Distinct, sizeof(uint64_t));
    if (!CHECK(Drawn != NULL))
    {
        SamplerFree(&Sampler);
        free(Table);
        return ReportResult();
    }

    // 코드포인트 -> 표에서의 자리. 뽑은 글자를 되찾기 편하게 만들어둔다.
    int* IndexOf = (int*)malloc((size_t)CODEPOINT_LIMIT * sizeof(int));
    if (!CHECK(IndexOf != NULL))
    {
        free(Drawn);
        SamplerFree(&Sampler);
        free(Table);
        return ReportResult();
    }
    for (uint32_t i = 0; i < CODEPOINT_LIMIT; i++)
    {
        IndexOf[i] = -1;
    }
    for (int i = 0; i < Distinct; i++)
    {
        IndexOf[Table[i].Codepoint] = i;
    }

    RandomSeed(&Rng, BOOK_SEED);

    printf("[A3-4] %d 번 뽑는 중...\n\n", DRAW_COUNT);
    for (int i = 0; i < DRAW_COUNT; i++)
    {
        uint32_t Code = SamplerPick(&Sampler, &Rng);
        Drawn[IndexOf[Code]]++;
    }

    // 글자 칸을 맨 뒤에 둔다. printf 의 폭 지정은 바이트를 세기 때문에
    // 3바이트짜리 한글이 들어가면 앞쪽 칸이 어긋난다. (A1 참고)
    printf("  순위      이론      실제      차이  글자\n");

    double WorstGap = 0.0;
    for (int Rank = 0; Rank < 10; Rank++)
    {
        char Name[32];
        DescribeCode(Table[Rank].Codepoint, Name, sizeof(Name));

        double Expected = (double)Table[Rank].Count * 100.0 / (double)Total;
        double Actual = (double)Drawn[Rank] * 100.0 / (double)DRAW_COUNT;
        double Gap = Expected - Actual;
        if (Gap < 0.0)
        {
            Gap = -Gap;
        }
        if (Gap > WorstGap)
        {
            WorstGap = Gap;
        }

        printf("  %4d %8.3f%% %8.3f%% %7.4f%%p  %s\n",
               Rank + 1, Expected, Actual, Gap, Name);
    }
    printf("\n  가장 큰 차이 = %.4f%%p\n\n", WorstGap);

    // 천만 번 뽑았으면 상위 글자는 0.05%p 안으로 들어와야 한다.
    CHECK(WorstGap < 0.05);

    // ---- A3-5. 뽑은 글자를 이어 붙이면 ----
    RandomSeed(&Rng, BOOK_SEED);
    printf("[A3-5] 뽑은 글자를 그대로 이어 붙이면\n\n  ");
    for (int i = 0; i < 60; i++)
    {
        uint32_t Code = SamplerPick(&Sampler, &Rng);
        if (Code == '\n')
        {
            Code = ' ';   // 줄바꿈은 한 줄로 보기 위해 공백으로
        }
        PrintCode(Code);
    }
    printf("\n\n");

    free(IndexOf);
    free(Drawn);
    SamplerFree(&Sampler);
    free(Table);
    return ReportResult();
}
