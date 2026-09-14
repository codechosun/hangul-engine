// ch-A05-bigram/Main.c
//
// A5. 바이그램과 특수 토큰
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Golden.h"
#include "Pretty.h"

#include "Bigram.h"
#include "Random.h"
#include "Sample.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CORPUS  "data/corpus.txt"
#define SUMMARY_PATH    "data/bigram_summary.csv"
#define EXPECTED_PATH   "data/bigram_expected.csv"

#define BOOK_SEED 20260914ull

// 한 문장의 최대 길이. 모델이 EOS 를 안 뽑고 계속 갈 수 있다.
#define MAX_SENTENCE 200

static void DescribeCode(uint32_t Code, char* Out, int OutSize)
{
    if (Code == TOKEN_BOS) { snprintf(Out, OutSize, "BOS");    return; }
    if (Code == TOKEN_EOS) { snprintf(Out, OutSize, "EOS");    return; }
    if (Code == ' ')       { snprintf(Out, OutSize, "공백");   return; }
    if (Code == '\n')      { snprintf(Out, OutSize, "줄바꿈"); return; }
    if (Code < 0x20u)      { snprintf(Out, OutSize, "제어");   return; }

    char Utf8[8] = { 0 };
    int Length = Utf8Encode(Code, Utf8);
    Utf8[Length] = '\0';
    snprintf(Out, OutSize, "%s", Utf8);
}

static void PrintCode(uint32_t Code)
{
    char Utf8[8] = { 0 };
    int Length = Utf8Encode(Code, Utf8);
    Utf8[Length] = '\0';
    printf("%s", Utf8);
}

// 앞 글자 Prev 다음에 올 글자 상위 Top 개를 찍는다.
static void PrintTopNext(const FBigram* Model, uint32_t Prev, int Top)
{
    uint32_t From = Model->Start[Prev];
    uint32_t To = Model->Start[Prev + 1];
    uint64_t Total = BigramContextTotal(Model, Prev);

    char Name[32];
    DescribeCode(Prev, Name, sizeof(Name));

    printf("  '%s' 다음 (후보 %u종, 총 %llu번)\n",
           Name, (unsigned)(To - From), (unsigned long long)Total);

    uint64_t Limit = UINT64_MAX;
    uint32_t LastPick = 0;

    for (int Rank = 0; Rank < Top; Rank++)
    {
        uint64_t Best = 0;
        uint32_t BestCode = 0;

        for (uint32_t i = From; i < To; i++)
        {
            uint64_t Before = (i == From) ? 0 : Model->Cumulative[i - 1];
            uint64_t Count = Model->Cumulative[i] - Before;

            int Allowed = (Count < Limit) ||
                          (Count == Limit && Model->Next[i] > LastPick);

            if (Allowed && Count > Best)
            {
                Best = Count;
                BestCode = Model->Next[i];
            }
        }

        if (Best == 0)
        {
            break;
        }

        char Next[32];
        DescribeCode(BestCode, Next, sizeof(Next));

        printf("    %2d. ", Rank + 1);
        PrintPadded(Next, 8);
        printf("%10llu  %6.2f%%\n",
               (unsigned long long)Best, (double)Best * 100.0 / (double)Total);

        Limit = Best;
        LastPick = BestCode;
    }

    printf("\n");
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A5. 바이그램과 특수 토큰\n\n");

    // ---- 모델 만들기 ----
    printf("모델을 만드는 중...\n\n");

    clock_t Begin = clock();

    FBigram Model;
    if (!CHECK(BigramBuild(&Model, CorpusPath)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        return ReportResult();
    }

    double BuildSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

    double Bytes = (double)Model.PairCount * 12.0
                 + (double)(VOCAB_LIMIT + 1) * 4.0;

    printf("[A5-1] 만들어진 모델\n");
    printf("  문장 수        = %llu\n", (unsigned long long)Model.LineCount);
    printf("  쌍의 개수      = %llu\n", (unsigned long long)Model.Total);
    printf("  서로 다른 쌍   = %llu\n", (unsigned long long)Model.PairCount);
    printf("  메모리         = %.0f MB\n", Bytes / (1024.0 * 1024.0));
    printf("  걸린 시간      = %.2f 초\n\n", BuildSeconds);

    // ---- A5-2. 정답표와 대조 ----
    printf("[A5-2] 파이썬 참조 구현과 대조\n");

    // 요약표는 "이름,값" 형식이라 Golden 도구를 못 쓴다. 직접 읽는다.
    {
        FILE* File = fopen(SUMMARY_PATH, "rb");
        uint64_t Chars = 0;
        uint64_t Lines = 0;
        uint64_t Pairs = 0;

        if (CHECK(File != NULL))
        {
            char Line[128];
            while (fgets(Line, sizeof(Line), File) != NULL)
            {
                unsigned long long Value = 0;
                if (sscanf(Line, "Chars,%llu", &Value) == 1) Chars = Value;
                else if (sscanf(Line, "Lines,%llu", &Value) == 1) Lines = Value;
                else if (sscanf(Line, "Pairs,%llu", &Value) == 1) Pairs = Value;
            }
            fclose(File);
        }

        printf("  ");
        PrintPadded("문장 수", 12);
        printf("파이썬 %12llu   C %12llu\n",
               (unsigned long long)Lines, (unsigned long long)Model.LineCount);
        printf("  ");
        PrintPadded("쌍의 개수", 12);
        printf("파이썬 %12llu   C %12llu\n",
               (unsigned long long)Pairs, (unsigned long long)Model.Total);

        CHECK(Lines == Model.LineCount);
        CHECK(Pairs == Model.Total);

        // 이 규칙에서는 쌍의 개수가 글자 수와 같아야 한다.
        CHECK(Chars == Model.Total);
    }

    // ---- 쌍 정답표와 한 줄씩 대조 ----
    FGoldenTriple* Expected = NULL;
    int ExpectedCount = 0;

    if (!CHECK(GoldenLoadTriple(EXPECTED_PATH, &Expected, &ExpectedCount)))
    {
        printf("%s 가 없다. tools/make_bigram_expected.py 를 돌릴 것.\n",
               EXPECTED_PATH);
        BigramFree(&Model);
        return ReportResult();
    }

    int Mismatch = 0;
    int FirstBadRow = -1;

    for (int i = 0; i < ExpectedCount; i++)
    {
        uint64_t Mine = BigramCount(&Model, Expected[i].First, Expected[i].Second);
        if (Mine != Expected[i].Value)
        {
            Mismatch++;
            if (FirstBadRow < 0)
            {
                FirstBadRow = i;
            }
        }
    }

    printf("  쌍 정답표 %d줄 대조 → 어긋난 줄 %d개\n", ExpectedCount, Mismatch);

    if (Mismatch > 0)
    {
        printf("    처음 어긋난 줄: 앞 %u, 뒤 %u, 파이썬 %llu\n",
               Expected[FirstBadRow].First, Expected[FirstBadRow].Second,
               (unsigned long long)Expected[FirstBadRow].Value);
    }

    CHECK(Mismatch == 0);

    // 정답표에 없는 쌍은 0 이어야 한다. 없는 조합을 하나 고른다.
    CHECK(BigramCount(&Model, TOKEN_EOS, TOKEN_BOS) == 0);

    // 앞 글자별 합계도 맞아야 한다.
    {
        uint32_t Contexts[2] = { (uint32_t)' ', TOKEN_BOS };
        for (int k = 0; k < 2; k++)
        {
            uint64_t Sum = 0;
            for (int i = 0; i < ExpectedCount; i++)
            {
                if (Expected[i].First == Contexts[k])
                {
                    Sum += Expected[i].Value;
                }
            }
            CHECK(Sum == BigramContextTotal(&Model, Contexts[k]));
        }
    }

    free(Expected);
    printf("\n");

    // ---- A5-3. 시작이 다르다 ----
    // 바이그램 표에서 "뒤 글자" 쪽만 모으면 글자 빈도가 된다.
    uint64_t* Unigram = (uint64_t*)calloc(VOCAB_LIMIT, sizeof(uint64_t));
    if (!CHECK(Unigram != NULL))
    {
        BigramFree(&Model);
        return ReportResult();
    }

    for (uint32_t Prev = 0; Prev < VOCAB_LIMIT; Prev++)
    {
        uint32_t From = Model.Start[Prev];
        uint32_t To = Model.Start[Prev + 1];

        for (uint32_t i = From; i < To; i++)
        {
            uint64_t Before = (i == From) ? 0 : Model.Cumulative[i - 1];
            Unigram[Model.Next[i]] += Model.Cumulative[i] - Before;
        }
    }

    printf("[A5-3] 문장의 첫 글자는 아무 글자가 아니다\n\n");
    PrintTopNext(&Model, TOKEN_BOS, 5);

    printf("  글자 전체 빈도 상위 5종\n");
    {
        uint64_t Limit = UINT64_MAX;
        uint32_t LastPick = 0;
        uint64_t GrandTotal = 0;

        for (uint32_t i = 0; i < VOCAB_LIMIT; i++)
        {
            GrandTotal += Unigram[i];
        }

        for (int Rank = 0; Rank < 5; Rank++)
        {
            uint64_t Best = 0;
            uint32_t BestCode = 0;

            for (uint32_t i = 0; i < VOCAB_LIMIT; i++)
            {
                int Allowed = (Unigram[i] < Limit) ||
                              (Unigram[i] == Limit && i > LastPick);

                if (Allowed && Unigram[i] > Best)
                {
                    Best = Unigram[i];
                    BestCode = i;
                }
            }

            char Name[32];
            DescribeCode(BestCode, Name, sizeof(Name));

            printf("    %2d. ", Rank + 1);
            PrintPadded(Name, 8);
            printf("%10llu  %6.2f%%\n", (unsigned long long)Best,
                   (double)Best * 100.0 / (double)GrandTotal);

            Limit = Best;
            LastPick = BestCode;
        }
    }
    printf("\n");

    // 두 분포가 얼마나 다른가를 숫자 하나로 만든다.
    //
    // 총변동거리(total variation distance).
    //     TVD = 0.5 * sum |p_i - q_i|
    // 0 이면 완전히 같고, 1 이면 겹치는 데가 하나도 없다.
    {
        uint64_t FirstTotal = BigramContextTotal(&Model, TOKEN_BOS);
        uint64_t GrandTotal = 0;
        for (uint32_t i = 0; i < VOCAB_LIMIT; i++)
        {
            GrandTotal += Unigram[i];
        }

        double Distance = 0.0;
        for (uint32_t i = 0; i < VOCAB_LIMIT; i++)
        {
            double P = (double)BigramCount(&Model, TOKEN_BOS, i)
                     / (double)FirstTotal;
            double Q = (double)Unigram[i] / (double)GrandTotal;
            Distance += (P > Q) ? (P - Q) : (Q - P);
        }
        Distance *= 0.5;

        printf("  두 분포의 총변동거리 = %.3f\n", Distance);
        printf("  (0 이면 같은 분포, 1 이면 전혀 다른 분포)\n\n");

        // 첫 글자 분포가 글자 전체 분포와 확연히 달라야 한다.
        // 같다면 BOS 를 붙인 의미가 없다.
        CHECK(Distance > 0.2);
    }

    // ---- A5-4. 조건부 확률 ----
    printf("[A5-4] 앞 글자가 정해지면 분포가 달라진다\n\n");
    PrintTopNext(&Model, 0xC758u, 5);   // 의
    PrintTopNext(&Model, 0xB2E4u, 5);   // 다
    PrintTopNext(&Model, 0xD55Cu, 5);   // 한

    // ---- A5-5. 문장 만들기 ----
    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    printf("[A5-5] 바이그램이 쓴 문장 열 개\n\n");

    uint32_t Buffer[MAX_SENTENCE];
    int TotalLength = 0;

    for (int i = 0; i < 10; i++)
    {
        int Length = BigramGenerate(&Model, &Rng, Buffer, MAX_SENTENCE);
        TotalLength += Length;

        printf("  %2d. ", i + 1);
        for (int k = 0; k < Length; k++)
        {
            PrintCode(Buffer[k]);
        }
        printf("\n");
    }
    printf("\n  평균 길이 %.1f 글자\n\n", (double)TotalLength / 10.0);

    CHECK(TotalLength > 0);

    // ---- A5-6. A3 방식과 나란히 ----
    // 글자 빈도만 보고 뽑으면 어떤 글이 나오는지 다시 본다.
    {
        int Distinct = 0;
        for (uint32_t i = 0; i < CODEPOINT_LIMIT; i++)
        {
            if (Unigram[i] > 0)
            {
                Distinct++;
            }
        }

        FCharCount* Table =
            (FCharCount*)malloc((size_t)Distinct * sizeof(FCharCount));

        int Index = 0;
        for (uint32_t i = 0; i < CODEPOINT_LIMIT; i++)
        {
            if (Unigram[i] > 0)
            {
                Table[Index].Codepoint = i;
                Table[Index].Count = Unigram[i];
                Index++;
            }
        }

        FreqSort(Table, Distinct);

        FSampler Sampler;
        SamplerInit(&Sampler, Table, Distinct);

        RandomSeed(&Rng, BOOK_SEED);

        printf("[A5-6] 같은 씨앗, 앞 글자를 안 보면 (A3 방식)\n\n");
        for (int i = 0; i < 3; i++)
        {
            printf("  %2d. ", i + 1);
            for (int k = 0; k < 40; k++)
            {
                uint32_t Code = SamplerPick(&Sampler, &Rng);
                PrintCode((Code == '\n') ? (uint32_t)' ' : Code);
            }
            printf("\n");
        }
        printf("\n");

        SamplerFree(&Sampler);
        free(Table);
    }

    free(Unigram);
    BigramFree(&Model);

    return ReportResult();
}
