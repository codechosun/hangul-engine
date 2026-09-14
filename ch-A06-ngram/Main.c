// ch-A06-ngram/Main.c
//
// A6. N 일반화
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Golden.h"
#include "Pretty.h"

#include "Ngram.h"
#include "GramMap.h"
#include "Random.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CORPUS "data/corpus.txt"
#define SUMMARY_PATH   "data/ngram_summary.csv"
#define EXPECTED_PATH  "data/ngram_expected.csv"
#define CONTEXT_PATH   "data/ngram_context.csv"

#define BOOK_SEED 20260914ull

// 정답표를 만든 파이썬과 반드시 같아야 한다.
#define MAX_LINES 200000
#define MAX_ORDER 6

#define MAX_SENTENCE 300

// " 대한민국". N 에 따라 뒤에서 N-1 글자를 문맥으로 쓴다.
static const uint32_t GContextSource[5] =
{
    0x0020u, 0xB300u, 0xD55Cu, 0xBBFCu, 0xAD6Du
};

static void PrintCode(uint32_t Code)
{
    char Utf8[8] = { 0 };
    int Length = Utf8Encode(Code, Utf8);
    Utf8[Length] = '\0';
    printf("%s", Utf8);
}

static void DescribeCode(uint32_t Code, char* Out, int OutSize)
{
    if (Code == TOKEN_BOS) { snprintf(Out, OutSize, "BOS");    return; }
    if (Code == TOKEN_EOS) { snprintf(Out, OutSize, "EOS");    return; }
    if (Code == ' ')       { snprintf(Out, OutSize, "공백");   return; }
    if (Code < 0x20u)      { snprintf(Out, OutSize, "제어");   return; }

    char Utf8[8] = { 0 };
    int Length = Utf8Encode(Code, Utf8);
    Utf8[Length] = '\0';
    snprintf(Out, OutSize, "%s", Utf8);
}

// "이름,값" 형식의 요약표에서 값 하나를 찾는다.
static uint64_t ReadSummary(const char* Path, const char* Key)
{
    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0;
    }

    char Wanted[64];
    snprintf(Wanted, sizeof(Wanted), "%s,", Key);
    size_t WantedLength = strlen(Wanted);

    char Line[128];
    uint64_t Value = 0;

    while (fgets(Line, sizeof(Line), File) != NULL)
    {
        if (strncmp(Line, Wanted, WantedLength) == 0)
        {
            Value = strtoull(Line + WantedLength, NULL, 10);
            break;
        }
    }

    fclose(File);
    return Value;
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A6. N 일반화\n\n");

    // ---- A6-1. 키가 64비트에 안 들어간다 ----
    printf("[A6-1] 글자 N 개를 64비트 키 하나에 담을 수 있는가\n");
    printf("  코드포인트 하나 = 21비트 (0x10FFFF 까지)\n\n");
    printf("  ");
    PrintPadded("N", 4);
    PrintPaddedRight("필요한 비트", 14);
    PrintPaddedRight("64비트에", 12);
    printf("\n");

    for (int Order = 1; Order <= MAX_ORDER; Order++)
    {
        int Bits = Order * 21;
        printf("  ");
        printf("%-4d", Order);
        printf("%13d ", Bits);
        PrintPaddedRight(Bits <= 64 ? "들어간다" : "넘친다", 12);
        printf("\n");
    }
    printf("\n");

    // ---- A6-2. 정답표 읽기 ----
    uint64_t WantLines = ReadSummary(SUMMARY_PATH, "Lines");
    uint64_t WantChars = ReadSummary(SUMMARY_PATH, "Chars");
    uint64_t WantGrams = ReadSummary(SUMMARY_PATH, "Grams");

    if (!CHECK(WantGrams > 0))
    {
        printf("%s 가 없다. tools/make_ngram_expected.py 를 돌릴 것.\n",
               SUMMARY_PATH);
        return ReportResult();
    }

    FGoldenTriple* Expected = NULL;
    int ExpectedCount = 0;
    FGoldenTriple* Context = NULL;
    int ContextCount = 0;

    CHECK(GoldenLoadTriple(EXPECTED_PATH, &Expected, &ExpectedCount));
    CHECK(GoldenLoadTriple(CONTEXT_PATH, &Context, &ContextCount));

    printf("[A6-2] N=1 부터 %d 까지 같은 데이터로 만든다 (앞 %d줄)\n\n",
           MAX_ORDER, MAX_LINES);

    printf("  ");
    PrintPadded("N", 4);
    PrintPaddedRight("서로 다른 그램", 16);
    PrintPaddedRight("한 번만", 12);
    PrintPaddedRight("비율", 8);
    PrintPaddedRight("문맥 수", 12);
    PrintPaddedRight("메모리", 10);
    PrintPaddedRight("시간", 8);
    printf("\n");

    char Sentences[MAX_ORDER + 1][2][1024];
    uint64_t ChoiceCount[MAX_ORDER + 1];
    uint64_t ChoiceTotal[MAX_ORDER + 1];

    memset(Sentences, 0, sizeof(Sentences));
    memset(ChoiceCount, 0, sizeof(ChoiceCount));
    memset(ChoiceTotal, 0, sizeof(ChoiceTotal));

    int GoldenMismatch = 0;
    int ContextMismatch = 0;

    for (int Order = 1; Order <= MAX_ORDER; Order++)
    {
        clock_t Begin = clock();

        FNgram Model;
        if (!CHECK(NgramBuild(&Model, CorpusPath, Order, MAX_LINES)))
        {
            printf("코퍼스를 못 읽었다.\n");
            return ReportResult();
        }

        double Seconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

        double Bytes = (double)Model.GramCount
                     * ((double)Order * 4.0 + 8.0)
                     + (double)(Model.ContextCount + 1) * 8.0;

        printf("  %-4d", Order);
        printf("%15llu ", (unsigned long long)Model.GramCount);
        printf("%11llu ", (unsigned long long)Model.OnceCount);
        printf("%7.1f%% ",
               (double)Model.OnceCount * 100.0 / (double)Model.GramCount);
        printf("%11llu ", (unsigned long long)Model.ContextCount);
        printf("%7.0f MB ", Bytes / (1024.0 * 1024.0));
        printf("%6.2fs\n", Seconds);

        // ---- 불변식 ----
        CHECK(Model.LineCount == WantLines);
        CHECK(Model.CharCount == WantChars);
        CHECK(Model.Total == WantGrams);

        // ---- 정답표와 대조 ----
        char Key[32];

        snprintf(Key, sizeof(Key), "Distinct%d", Order);
        CHECK(Model.GramCount == ReadSummary(SUMMARY_PATH, Key));

        snprintf(Key, sizeof(Key), "Once%d", Order);
        CHECK(Model.OnceCount == ReadSummary(SUMMARY_PATH, Key));

        snprintf(Key, sizeof(Key), "Contexts%d", Order);
        CHECK(Model.ContextCount == ReadSummary(SUMMARY_PATH, Key));

        // 문맥이 전부 BOS 인 그램들
        uint32_t Gram[NGRAM_MAX_ORDER];
        for (int i = 0; i < Order - 1; i++)
        {
            Gram[i] = TOKEN_BOS;
        }

        for (int i = 0; i < ExpectedCount; i++)
        {
            if ((int)Expected[i].First != Order)
            {
                continue;
            }

            Gram[Order - 1] = Expected[i].Second;
            if (NgramCount(&Model, Gram) != Expected[i].Value)
            {
                GoldenMismatch++;
            }
        }

        // 문맥이 " 대한민국" 의 꼬리인 그램들
        for (int i = 0; i < Order - 1; i++)
        {
            Gram[i] = GContextSource[5 - (Order - 1) + i];
        }

        for (int i = 0; i < ContextCount; i++)
        {
            if ((int)Context[i].First != Order)
            {
                continue;
            }

            Gram[Order - 1] = Context[i].Second;
            if (NgramCount(&Model, Gram) != Context[i].Value)
            {
                ContextMismatch++;
            }
        }

        ChoiceCount[Order] = NgramChoiceCount(&Model, Gram);
        ChoiceTotal[Order] = NgramContextTotal(&Model, Gram);

        // ---- 문장 두 개 ----
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        uint32_t Buffer[MAX_SENTENCE];
        for (int k = 0; k < 2; k++)
        {
            int Length = NgramGenerate(&Model, &Rng, Buffer, MAX_SENTENCE);

            int Used = 0;
            for (int i = 0; i < Length && Used < 1000; i++)
            {
                char Utf8[8] = { 0 };
                int Bytes8 = Utf8Encode(Buffer[i], Utf8);
                memcpy(Sentences[Order][k] + Used, Utf8, (size_t)Bytes8);
                Used += Bytes8;
            }
            Sentences[Order][k][Used] = '\0';
        }

        NgramFree(&Model);
    }

    printf("\n");
    printf("  BOS 문맥 정답표 %d줄, '대한민국' 문맥 정답표 %d줄\n",
           ExpectedCount, ContextCount);
    printf("  어긋난 줄 = %d + %d\n\n", GoldenMismatch, ContextMismatch);

    CHECK(GoldenMismatch == 0);
    CHECK(ContextMismatch == 0);

    free(Expected);
    free(Context);

    // ---- A6-3. 문맥이 길어지면 선택지가 줄어든다 ----
    printf("[A6-3] 문맥이 길어지면 다음 글자가 정해진다\n\n");
    printf("  ");
    PrintPadded("N", 4);
    PrintPadded("문맥", 14);
    PrintPaddedRight("다음 후보", 12);
    PrintPaddedRight("본 횟수", 12);
    printf("\n");

    for (int Order = 2; Order <= MAX_ORDER; Order++)
    {
        char Text[64] = { 0 };
        int Used = 0;

        for (int i = 0; i < Order - 1; i++)
        {
            char Utf8[8] = { 0 };
            int Bytes8 = Utf8Encode(GContextSource[5 - (Order - 1) + i], Utf8);
            memcpy(Text + Used, Utf8, (size_t)Bytes8);
            Used += Bytes8;
        }
        Text[Used] = '\0';

        printf("  %-4d", Order);
        PrintPadded(Text, 14);
        printf("%11llu ", (unsigned long long)ChoiceCount[Order]);
        printf("%11llu\n", (unsigned long long)ChoiceTotal[Order]);
    }
    printf("\n");

    CHECK(ChoiceCount[MAX_ORDER] < ChoiceCount[2]);

    // ---- A6-4. N 을 올리면 글이 어떻게 바뀌는가 ----
    printf("[A6-4] 같은 씨앗으로 N 만 바꿔 쓴 문장\n\n");

    for (int Order = 1; Order <= MAX_ORDER; Order++)
    {
        printf("  N=%d\n", Order);
        printf("    1. %s\n", Sentences[Order][0]);
        printf("    2. %s\n\n", Sentences[Order][1]);
    }

    return ReportResult();
}
