// ch-A08-backoff/Main.c
//
// A8. 백오프
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Backoff.h"
#include "Ngram.h"
#include "Random.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CORPUS "data/corpus.txt"
#define BOOK_SEED 20260914ull

// A6 과 같은 데이터를 쓴다.
#define MAX_LINES 200000
#define TOP_ORDER 6

// "이 조각이 코퍼스에 그대로 있는가"를 물어볼 자
#define PROBE_ORDER 8

#define SENTENCE_COUNT 200
#define MAX_SENTENCE 300

// 백오프 없이 뽑는다. 문맥을 못 찾으면 그 자리에서 끝난다.
static int PlainGenerate(const FNgram* Model, FRandom* Rng,
                         uint32_t* Out, int MaxLength, int* OutBlocked)
{
    int Length = Model->Order - 1;

    uint32_t Context[NGRAM_MAX_ORDER];
    for (int i = 0; i < Length; i++)
    {
        Context[i] = TOKEN_BOS;
    }

    int Written = 0;
    *OutBlocked = 0;

    while (Written < MaxLength)
    {
        if (NgramFindContext(Model, Context) < 0)
        {
            *OutBlocked = 1;      // 본 적 없는 문맥. 여기서 막혔다
            break;
        }

        uint32_t Token = NgramPick(Model, Context, Rng);
        if (Token == TOKEN_EOS)
        {
            break;
        }

        Out[Written++] = Token;

        for (int i = 0; i < Length - 1; i++)
        {
            Context[i] = Context[i + 1];
        }
        if (Length >= 1)
        {
            Context[Length - 1] = Token;
        }
    }

    return Written;
}

// 문장 안의 PROBE_ORDER 글자 조각 중 몇 개가 코퍼스에 그대로 있는가.
static void MeasureCopy(const FNgram* Probe, const uint32_t* Tokens, int Length,
                        uint64_t* OutWindows, uint64_t* OutCopied,
                        int* OutLongestRun)
{
    int Order = Probe->Order;
    int Run = 0;
    int Longest = 0;

    for (int i = 0; i + Order <= Length; i++)
    {
        (*OutWindows)++;

        if (NgramCount(Probe, Tokens + i) > 0)
        {
            (*OutCopied)++;
            Run++;
            if (Run > Longest)
            {
                Longest = Run;
            }
        }
        else
        {
            Run = 0;
        }
    }

    // 연속으로 Run 개의 창이 전부 원문이면 Order + Run - 1 글자가 이어진다.
    *OutLongestRun = (Longest > 0) ? (Order + Longest - 1) : 0;
}

// 바깥에서 받은 문맥. 5글자씩이므로 6그램의 문맥이 된다.
#define PROMPT_LENGTH (TOP_ORDER - 1)
#define PROMPT_COUNT 4

static const uint32_t GPrompts[PROMPT_COUNT][PROMPT_LENGTH] =
{
    { 0x0020u, 0xB300u, 0xD55Cu, 0xBBFCu, 0xAD6Du },   // " 대한민국"
    { 0xC5ECu, 0xC131u, 0xC5D0u, 0xAC8Cu, 0xB3C4u },   // "여성에게도"
    { 0xAD6Du, 0xC758u, 0x0020u, 0xC5EDu, 0xC0ACu },   // "국의 역사"
    { 0xD7A3u, 0xD7A3u, 0xD7A3u, 0xD7A3u, 0xD7A3u },   // "힣힣힣힣힣"
};

static void PrintSentence(const uint32_t* Tokens, int Length)
{
    for (int i = 0; i < Length; i++)
    {
        char Utf8[8] = { 0 };
        int Bytes = Utf8Encode(Tokens[i], Utf8);
        Utf8[Bytes] = '\0';
        printf("%s", Utf8);
    }
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A8. 백오프\n\n");
    printf("모델을 만드는 중 (앞 %d줄)...\n\n", MAX_LINES);

    clock_t Begin = clock();

    // 자를 먼저 만든다. 메모리 봉우리를 낮추려고 이것부터 세운다.
    FNgram Probe;
    if (!CHECK(NgramBuild(&Probe, CorpusPath, PROBE_ORDER, MAX_LINES)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        return ReportResult();
    }

    FBackoff Backoff;
    if (!CHECK(BackoffBuild(&Backoff, CorpusPath, TOP_ORDER, MAX_LINES)))
    {
        printf("모델을 못 만들었다. 메모리가 모자랐을 수 있다.\n");
        NgramFree(&Probe);
        return ReportResult();
    }

    double BuildSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

    printf("  1그램부터 %d그램까지 + 자로 쓸 %d그램, %.1f초\n\n",
           TOP_ORDER, PROBE_ORDER, BuildSeconds);

    const FNgram* Top = &Backoff.Orders[TOP_ORDER];

    FRandom Rng;

    // ---- A8-1. 바깥에서 문맥을 받으면 막힌다 ----
    printf("[A8-1] 문맥을 바깥에서 주면\n\n");
    printf("  ");
    PrintPadded("문맥", 14);
    PrintPaddedRight("아는 최고 차수", 16);
    char TopLabel[32];
    snprintf(TopLabel, sizeof(TopLabel), "%d그램 단독", TOP_ORDER);
    PrintPaddedRight(TopLabel, 14);
    printf("  백오프가 이어 쓴 글\n");

    int BlockedPrompts = 0;

    for (int p = 0; p < PROMPT_COUNT; p++)
    {
        char Text[64] = { 0 };
        int Used = 0;
        for (int i = 0; i < PROMPT_LENGTH; i++)
        {
            char Utf8[8] = { 0 };
            int Bytes = Utf8Encode(GPrompts[p][i], Utf8);
            memcpy(Text + Used, Utf8, (size_t)Bytes);
            Used += Bytes;
        }
        Text[Used] = '\0';

        // 이 문맥을 아는 가장 높은 차수를 찾는다.
        int Known = 0;
        for (int Order = TOP_ORDER; Order >= 2; Order--)
        {
            const uint32_t* Context =
                GPrompts[p] + (PROMPT_LENGTH - (Order - 1));

            if (NgramFindContext(&Backoff.Orders[Order], Context) >= 0)
            {
                Known = Order;
                break;
            }
        }

        int Blocked = (Known < TOP_ORDER);
        if (Blocked)
        {
            BlockedPrompts++;
        }

        RandomSeed(&Rng, BOOK_SEED);

        uint32_t Context[NGRAM_MAX_ORDER];
        memcpy(Context, GPrompts[p], sizeof(Context[0]) * PROMPT_LENGTH);

        char Written[256] = { 0 };
        int WrittenUsed = 0;

        for (int k = 0; k < 20; k++)
        {
            uint32_t Token =
                BackoffPick(&Backoff, Context, PROMPT_LENGTH, &Rng);

            if (Token == TOKEN_EOS)
            {
                break;
            }

            char Utf8[8] = { 0 };
            int Bytes = Utf8Encode(Token, Utf8);
            memcpy(Written + WrittenUsed, Utf8, (size_t)Bytes);
            WrittenUsed += Bytes;

            for (int i = 0; i < PROMPT_LENGTH - 1; i++)
            {
                Context[i] = Context[i + 1];
            }
            Context[PROMPT_LENGTH - 1] = Token;
        }
        Written[WrittenUsed] = '\0';

        printf("  ");
        PrintPadded(Text, 14);
        printf("%15d ", Known);
        PrintPaddedRight(Blocked ? "막힌다" : "된다", 14);
        printf("  %s\n", Written);
    }

    printf("\n");

    // 바깥에서 받은 문맥 중에는 6그램이 모르는 것이 반드시 있다.
    CHECK(BlockedPrompts > 0);

    // ---- A8-2. 스스로 쓰게 하면 ----
    RandomSeed(&Rng, BOOK_SEED);

    uint32_t Buffer[MAX_SENTENCE];

    uint64_t PlainTotalLength = 0;
    int PlainBlockedCount = 0;
    int PlainEmptyCount = 0;
    uint64_t PlainWindows = 0;
    uint64_t PlainCopied = 0;
    int PlainLongest = 0;

    for (int i = 0; i < SENTENCE_COUNT; i++)
    {
        int Blocked = 0;
        int Length = PlainGenerate(Top, &Rng, Buffer, MAX_SENTENCE, &Blocked);

        PlainTotalLength += (uint64_t)Length;
        PlainBlockedCount += Blocked;
        if (Length == 0)
        {
            PlainEmptyCount++;
        }

        int Longest = 0;
        MeasureCopy(&Probe, Buffer, Length, &PlainWindows, &PlainCopied, &Longest);
        if (Longest > PlainLongest)
        {
            PlainLongest = Longest;
        }
    }

    printf("[A8-2] 백오프 없이 %d그램으로 %d문장\n", TOP_ORDER, SENTENCE_COUNT);
    printf("  평균 길이        = %.1f 글자\n",
           (double)PlainTotalLength / (double)SENTENCE_COUNT);
    printf("  빈 문장          = %d개\n", PlainEmptyCount);
    printf("  문맥을 못 찾아 멈춘 문장 = %d개 (%.1f%%)\n",
           PlainBlockedCount,
           (double)PlainBlockedCount * 100.0 / (double)SENTENCE_COUNT);
    printf("  %d글자 조각 %llu개 중 원문 그대로 = %llu개 (%.1f%%)\n",
           PROBE_ORDER, (unsigned long long)PlainWindows,
           (unsigned long long)PlainCopied,
           (double)PlainCopied * 100.0 / (double)PlainWindows);
    printf("  가장 길게 베낀 조각 = %d 글자\n\n", PlainLongest);

    // ---- A8-2. 백오프를 켜면 ----
    RandomSeed(&Rng, BOOK_SEED);
    BackoffResetStats(&Backoff);

    uint64_t BackTotalLength = 0;
    int BackEmptyCount = 0;
    uint64_t BackWindows = 0;
    uint64_t BackCopied = 0;
    int BackLongest = 0;

    for (int i = 0; i < SENTENCE_COUNT; i++)
    {
        int Length = BackoffGenerate(&Backoff, &Rng, Buffer, MAX_SENTENCE);

        BackTotalLength += (uint64_t)Length;
        if (Length == 0)
        {
            BackEmptyCount++;
        }

        int Longest = 0;
        MeasureCopy(&Probe, Buffer, Length, &BackWindows, &BackCopied, &Longest);
        if (Longest > BackLongest)
        {
            BackLongest = Longest;
        }
    }

    printf("[A8-3] 백오프를 켜고 같은 씨앗으로 %d문장\n", SENTENCE_COUNT);
    printf("  평균 길이        = %.1f 글자\n",
           (double)BackTotalLength / (double)SENTENCE_COUNT);
    printf("  빈 문장          = %d개\n", BackEmptyCount);
    printf("  막힌 문장        = 0개 (백오프는 막히지 않는다)\n");
    printf("  %d글자 조각 %llu개 중 원문 그대로 = %llu개 (%.1f%%)\n",
           PROBE_ORDER, (unsigned long long)BackWindows,
           (unsigned long long)BackCopied,
           (double)BackCopied * 100.0 / (double)BackWindows);
    printf("  가장 길게 베낀 조각 = %d 글자\n\n", BackLongest);

    // 백오프가 베끼기를 줄여야 한다.
    CHECK((double)BackCopied / (double)BackWindows
          < (double)PlainCopied / (double)PlainWindows);

    // 스스로 만든 문맥에서는 6그램도 막히지 않는다.
    //
    // 처음에는 자주 막힐 줄 알고 검사를 반대로 걸었다가 틀렸다.
    // 모델은 자기가 본 그램만 따라가므로, 방금 뽑은 여섯 글자의 뒤 다섯 글자는
    // 반드시 어딘가에서 문맥으로 등장했던 것이다. 막히는 것은
    // A8-1 처럼 **바깥에서 문맥을 받았을 때**다.
    CHECK(PlainBlockedCount == 0);

    // ---- A8-3. 어느 차수에서 뽑았는가 ----
    printf("[A8-4] 글자 하나를 뽑을 때 실제로 쓴 차수\n\n");
    printf("  ");
    PrintPaddedRight("차수", 8);
    PrintPaddedRight("뽑은 횟수", 14);
    PrintPaddedRight("비율", 10);
    printf("\n");

    uint64_t Sum = 0;
    for (int Order = 1; Order <= TOP_ORDER; Order++)
    {
        Sum += Backoff.UsedAt[Order];
    }

    for (int Order = TOP_ORDER; Order >= 1; Order--)
    {
        printf("  %7d", Order);
        printf("%14llu ", (unsigned long long)Backoff.UsedAt[Order]);
        printf("%9.1f%%\n",
               (double)Backoff.UsedAt[Order] * 100.0 / (double)Sum);
    }

    printf("\n  물러선 이유\n");
    printf("    문맥이 아예 없어서 = %llu 번\n",
           (unsigned long long)Backoff.MissSteps);
    printf("    할인된 몫에 걸려서 = %llu 번\n\n",
           (unsigned long long)Backoff.DiscountSteps);

    CHECK(Sum == Backoff.PickCount);
    CHECK(Backoff.UsedAt[TOP_ORDER] > 0);
    CHECK(Backoff.UsedAt[2] > 0);

    // ---- A8-4. 문장 보기 ----
    printf("[A8-5] 백오프가 쓴 문장\n\n");

    RandomSeed(&Rng, BOOK_SEED);
    for (int i = 0; i < 8; i++)
    {
        int Length = BackoffGenerate(&Backoff, &Rng, Buffer, MAX_SENTENCE);
        printf("  %2d. ", i + 1);
        PrintSentence(Buffer, Length);
        printf("\n");
    }
    printf("\n");

    BackoffFree(&Backoff);
    NgramFree(&Probe);

    return ReportResult();
}
