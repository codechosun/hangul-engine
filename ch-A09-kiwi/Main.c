// ch-A09-kiwi/Main.c
//
// A9. Kiwi 링크 — C++로 가는 다리
//
// 저장소 루트에서 실행할 것. build 폴더에 kiwi.dll 이 있어야 한다.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Ngram.h"
#include "Random.h"
#include "Scan.h"
#include "Utf8.h"
#include "Vocab.h"

// Kiwi 본체는 C++ 로 쓰여 있다. 이 헤더가 extern "C" 로 감싼 창구다.
#include "kiwi/capi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CORPUS "data/corpus.txt"
#define KIWI_MODEL     "third_party/kiwi/models/cong/base"

#define BOOK_SEED 20260914ull

// 형태소 분석은 글자 세기보다 훨씬 비싸다. 줄 수를 줄인다.
#define MORPH_LINES 50000
#define COMPARE_ORDER 3
#define MAX_SENTENCE 200

// 한 줄의 최대 바이트. 위키 문서의 줄은 길 수 있다.
#define LINE_BYTES (1 << 20)

static void PrintTokenText(const FVocab* Vocab, uint32_t Id)
{
    if (Id == TOKEN_BOS) { printf("<BOS>"); return; }
    if (Id == TOKEN_EOS) { printf("<EOS>"); return; }

    const char* Text = VocabText(Vocab, Id);
    if (Text == NULL)
    {
        return;
    }

    // "형태/태그" 로 담아두었으므로 앞쪽만 찍는다.
    for (const char* P = Text; *P != '\0' && *P != '\t'; P++)
    {
        putchar(*P);
    }
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A9. Kiwi 링크 — C++로 가는 다리\n\n");

    // ---- A9-1. 붙었는지부터 확인한다 ----
    printf("[A9-1] Kiwi 를 불러온다\n");
    printf("  kiwi_version() = %s\n", kiwi_version());

    // 실패하는 길을 먼저 확인한다. 없는 경로를 주면 무엇이 나오는가.
    kiwi_h Broken = kiwi_init("third_party/kiwi/models/없는폴더", -1,
                              KIWI_BUILD_DEFAULT, 0);
    printf("  없는 모델 경로 -> handle=%p, kiwi_error()=\"%s\"\n",
           (void*)Broken, kiwi_error() ? kiwi_error() : "(없음)");

    CHECK(Broken == NULL);

    clock_t Begin = clock();

    kiwi_h Kiwi = kiwi_init(KIWI_MODEL, -1, KIWI_BUILD_DEFAULT, 0);
    if (!CHECK(Kiwi != NULL))
    {
        printf("  Kiwi 를 못 열었다: %s\n", kiwi_error());
        printf("  third_party/kiwi 가 있는지 확인할 것. README 참고.\n");
        return ReportResult();
    }

    printf("  모델을 여는 데 %.2f초\n\n",
           (double)(clock() - Begin) / CLOCKS_PER_SEC);

    kiwi_analyze_option_t Option;
    memset(&Option, 0, sizeof(Option));
    Option.match_options = KIWI_MATCH_ALL_WITH_NORMALIZING;

    // ---- A9-2. 글자와 형태소는 다르다 ----
    printf("[A9-2] 같은 문장을 글자로 자를 때와 형태소로 자를 때\n\n");

    const char* Sample = "한국어를 배우는 사람들이 늘고 있다.";

    printf("  원문     %s\n", Sample);
    printf("  글자 %2d개  ", Utf8Length(Sample));
    for (const char* P = Sample; *P != '\0'; )
    {
        uint32_t Code = 0;
        int Length = Utf8Decode(P, &Code);
        if (Length <= 0)
        {
            break;
        }
        for (int i = 0; i < Length; i++)
        {
            putchar(P[i]);
        }
        printf(" ");
        P += Length;
    }
    printf("\n");

    kiwi_res_h Result = kiwi_analyze(Kiwi, Sample, 1, Option, NULL);
    if (!CHECK(Result != NULL))
    {
        kiwi_close(Kiwi);
        return ReportResult();
    }

    int MorphCount = kiwi_res_word_num(Result, 0);
    printf("  형태소 %2d개  ", MorphCount);
    for (int i = 0; i < MorphCount; i++)
    {
        printf("%s/%s ", kiwi_res_form(Result, 0, i),
               kiwi_res_tag(Result, 0, i));
    }
    printf("\n\n");

    kiwi_res_close(Result);

    CHECK(MorphCount > 0);
    CHECK(MorphCount < Utf8Length(Sample));

    // ---- A9-3. 코퍼스를 형태소로 바꾼다 ----
    printf("[A9-3] 코퍼스 앞 %d줄을 형태소로 자른다\n", MORPH_LINES);

    FVocab Vocab;
    if (!CHECK(VocabInit(&Vocab, 1024)))
    {
        kiwi_close(Kiwi);
        return ReportResult();
    }

    // 특수 토큰은 사전에 넣지 않는다.
    //
    // BOS/EOS 는 0x110000 위의 번호이고 형태소 번호는 0 부터 올라가므로
    // 둘이 부딪힐 일이 없다. A5 에서 "유니코드 밖의 번호"를 고른 것이
    // 여기서 한 번 더 값을 한다.

    // 어떤 글자가 나왔는지도 세어둔다. 글자 어휘와 형태소 어휘를 비교하려고.
    uint8_t* SeenChar = (uint8_t*)calloc(CODEPOINT_LIMIT, sizeof(uint8_t));
    uint64_t CharKinds = 0;

    uint64_t TokenCapacity = 1u << 20;
    uint32_t* Tokens = (uint32_t*)malloc((size_t)TokenCapacity * sizeof(uint32_t));
    uint64_t TokenCount = 0;

    char* Line = (char*)malloc(LINE_BYTES);

    if (!CHECK(Tokens != NULL && Line != NULL && SeenChar != NULL))
    {
        VocabFree(&Vocab);
        kiwi_close(Kiwi);
        return ReportResult();
    }

    FScanner Scanner;
    if (!CHECK(ScanOpen(&Scanner, CorpusPath)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        free(Tokens);
        free(Line);
        VocabFree(&Vocab);
        kiwi_close(Kiwi);
        return ReportResult();
    }

    Begin = clock();

    uint64_t LineCount = 0;
    uint64_t CharCount = 0;
    uint64_t MorphTotal = 0;
    size_t Used = 0;
    uint32_t Code = 0;

    while (LineCount < MORPH_LINES && ScanNext(&Scanner, &Code))
    {
        if (Code != (uint32_t)'\n')
        {
            char Utf8[8] = { 0 };
            int Bytes = Utf8Encode(Code, Utf8);

            if (Used + (size_t)Bytes < LINE_BYTES - 1)
            {
                memcpy(Line + Used, Utf8, (size_t)Bytes);
                Used += (size_t)Bytes;
            }

            CharCount++;
            if (SeenChar[Code] == 0)
            {
                SeenChar[Code] = 1;
                CharKinds++;
            }
            continue;
        }

        Line[Used] = '\0';
        LineCount++;

        if (Used > 0)
        {
            kiwi_res_h Res = kiwi_analyze(Kiwi, Line, 1, Option, NULL);
            if (Res != NULL)
            {
                int Count = kiwi_res_word_num(Res, 0);
                for (int i = 0; i < Count; i++)
                {
                    char Key[256];
                    snprintf(Key, sizeof(Key), "%s\t%s",
                             kiwi_res_form(Res, 0, i),
                             kiwi_res_tag(Res, 0, i));

                    uint32_t Id = VocabIntern(&Vocab, Key);

                    if (TokenCount == TokenCapacity)
                    {
                        TokenCapacity *= 2;
                        uint32_t* Grown = (uint32_t*)realloc(
                            Tokens, (size_t)TokenCapacity * sizeof(uint32_t));
                        if (Grown == NULL)
                        {
                            break;
                        }
                        Tokens = Grown;
                    }

                    Tokens[TokenCount++] = Id;
                    MorphTotal++;
                }
                kiwi_res_close(Res);
            }
        }

        // 줄 끝을 EOS 로 표시한다. NgramBuildFromTokens 가 이걸 보고 문장을 끊는다.
        if (TokenCount < TokenCapacity)
        {
            Tokens[TokenCount++] = TOKEN_EOS;
        }

        Used = 0;
    }

    ScanClose(&Scanner);
    free(Line);

    double TokenizeSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

    printf("  문장 수        = %llu\n", (unsigned long long)LineCount);
    printf("  글자 수        = %llu\n", (unsigned long long)CharCount);
    printf("  형태소 수      = %llu\n", (unsigned long long)MorphTotal);
    printf("  글자 어휘      = %llu 종\n", (unsigned long long)CharKinds);
    printf("  형태소 어휘    = %llu 종\n", (unsigned long long)Vocab.Count);
    printf("  글자 하나당    = %.2f 형태소\n",
           (double)MorphTotal / (double)CharCount);
    printf("  걸린 시간      = %.1f 초 (초당 %.0f만 글자)\n\n",
           TokenizeSeconds, (double)CharCount / TokenizeSeconds / 10000.0);

    CHECK(MorphTotal > 0);
    CHECK(MorphTotal < CharCount);

    kiwi_close(Kiwi);

    // ---- A9-4. 같은 데이터, 글자 3그램 대 형태소 3그램 ----
    printf("[A9-4] 같은 %d줄로 만든 %d그램 두 개\n\n", MORPH_LINES, COMPARE_ORDER);

    FNgram CharModel;
    if (!CHECK(NgramBuild(&CharModel, CorpusPath, COMPARE_ORDER, MORPH_LINES)))
    {
        free(Tokens);
        VocabFree(&Vocab);
        return ReportResult();
    }

    FNgram MorphModel;
    if (!CHECK(NgramBuildFromTokens(&MorphModel, Tokens, TokenCount,
                                    COMPARE_ORDER)))
    {
        NgramFree(&CharModel);
        free(Tokens);
        VocabFree(&Vocab);
        return ReportResult();
    }

    printf("  ");
    PrintPadded("단위", 10);
    PrintPaddedRight("어휘", 10);
    PrintPaddedRight("토큰 수", 12);
    PrintPaddedRight("서로 다른 그램", 16);
    PrintPaddedRight("한 번만", 10);
    printf("\n");

    printf("  ");
    PrintPadded("글자", 10);
    printf("%10llu ", (unsigned long long)CharKinds);
    printf("%11llu ", (unsigned long long)CharModel.CharCount);
    printf("%15llu ", (unsigned long long)CharModel.GramCount);
    printf("%8.1f%%\n",
           (double)CharModel.OnceCount * 100.0 / (double)CharModel.GramCount);

    printf("  ");
    PrintPadded("형태소", 10);
    printf("%10llu ", (unsigned long long)Vocab.Count);
    printf("%11llu ", (unsigned long long)MorphModel.CharCount);
    printf("%15llu ", (unsigned long long)MorphModel.GramCount);
    printf("%8.1f%%\n\n",
           (double)MorphModel.OnceCount * 100.0 / (double)MorphModel.GramCount);

    // 같은 줄을 봤으므로 문장 수가 같아야 한다.
    CHECK(CharModel.LineCount == MorphModel.LineCount);

    // 형태소가 글자보다 적으므로 그램도 적다.
    CHECK(MorphModel.Total < CharModel.Total);

    // ---- A9-5. 각자 쓴 문장 ----
    FRandom Rng;
    uint32_t Buffer[MAX_SENTENCE];

    printf("  글자 %d그램이 쓴 문장\n", COMPARE_ORDER);
    RandomSeed(&Rng, BOOK_SEED);
    for (int i = 0; i < 3; i++)
    {
        int Length = NgramGenerate(&CharModel, &Rng, Buffer, MAX_SENTENCE);
        printf("    %d. ", i + 1);
        for (int k = 0; k < Length; k++)
        {
            char Utf8[8] = { 0 };
            int Bytes = Utf8Encode(Buffer[k], Utf8);
            Utf8[Bytes] = '\0';
            printf("%s", Utf8);
        }
        printf("\n");
    }
    printf("\n");

    printf("  형태소 %d그램이 쓴 문장 (형태소 사이를 공백으로 이어 붙였다)\n",
           COMPARE_ORDER);
    RandomSeed(&Rng, BOOK_SEED);

    uint64_t MorphLengthSum = 0;
    for (int i = 0; i < 3; i++)
    {
        int Length = NgramGenerate(&MorphModel, &Rng, Buffer, MAX_SENTENCE);
        MorphLengthSum += (uint64_t)Length;

        printf("    %d. ", i + 1);
        for (int k = 0; k < Length; k++)
        {
            PrintTokenText(&Vocab, Buffer[k]);
            printf(" ");
        }
        printf("\n");
    }
    printf("\n");

    CHECK(MorphLengthSum > 0);

    // ---- A9-6. 창 세 칸이 실제로 몇 글자인가 ----
    printf("[A9-5] 그램 %d개가 덮는 실제 길이\n", COMPARE_ORDER);
    printf("  글자 %d그램   = %d 글자\n", COMPARE_ORDER, COMPARE_ORDER);
    printf("  형태소 %d그램 = %.1f 글자 (형태소 하나가 평균 %.2f글자)\n\n",
           COMPARE_ORDER,
           (double)COMPARE_ORDER * (double)CharCount / (double)MorphTotal,
           (double)CharCount / (double)MorphTotal);

    NgramFree(&MorphModel);
    NgramFree(&CharModel);
    free(SeenChar);
    free(Tokens);
    VocabFree(&Vocab);

    return ReportResult();
}
