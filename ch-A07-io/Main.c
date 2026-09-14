// ch-A07-io/Main.c
//
// A7. 문장 생성과 대용량 I/O
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Ngram.h"
#include "Random.h"
#include "Scan.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CORPUS "data/corpus.txt"
#define MODEL_PATH     "data/ngram3.bin"

#define BOOK_SEED 20260914ull
#define BUILD_ORDER 3
#define MAX_SENTENCE 300

// 버퍼 크기를 재볼 값들.
static const size_t GChunkSizes[] =
{
    1u << 10,    // 1KB
    1u << 12,    // 4KB
    1u << 16,    // 64KB
    1u << 18,    // 256KB
    1u << 20,    // 1MB
    1u << 22,    // 4MB
    1u << 24,    // 16MB
};

#define CHUNK_COUNT (int)(sizeof(GChunkSizes) / sizeof(GChunkSizes[0]))

static void PrintBytes(double Bytes)
{
    if (Bytes >= 1024.0 * 1024.0)
    {
        printf("%6.0f MB", Bytes / (1024.0 * 1024.0));
    }
    else if (Bytes >= 1024.0)
    {
        printf("%6.0f KB", Bytes / 1024.0);
    }
    else
    {
        printf("%6.0f  B", Bytes);
    }
}

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

static double FileSize(const char* Path)
{
    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0.0;
    }

    fseek(File, 0, SEEK_END);
    long Size = ftell(File);
    fclose(File);

    return (double)Size;
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A7. 문장 생성과 대용량 I/O\n\n");

    // ---- A7-1. 버퍼 크기는 얼마가 좋은가 ----
    printf("[A7-1] 청크 크기를 바꿔가며 코퍼스 전체를 훑는다\n\n");
    printf("  ");
    PrintPaddedRight("청크", 10);
    PrintPaddedRight("fread 횟수", 14);
    PrintPaddedRight("시간", 10);
    PrintPaddedRight("초당", 12);
    printf("\n");

    uint64_t FirstCount = 0;
    double BestSeconds = 1e30;
    size_t BestChunk = 0;

    for (int i = 0; i < CHUNK_COUNT; i++)
    {
        FScanner Scanner;
        if (!CHECK(ScanOpenSized(&Scanner, CorpusPath, GChunkSizes[i])))
        {
            printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
            return ReportResult();
        }

        clock_t Begin = clock();

        uint64_t Count = 0;
        uint32_t Code = 0;
        while (ScanNext(&Scanner, &Code))
        {
            Count++;
        }

        double Seconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;
        ScanClose(&Scanner);

        if (i == 0)
        {
            FirstCount = Count;
        }

        // 청크 크기가 결과를 바꾸면 안 된다.
        CHECK(Count == FirstCount);

        if (Seconds < BestSeconds)
        {
            BestSeconds = Seconds;
            BestChunk = GChunkSizes[i];
        }

        double Bytes = FileSize(CorpusPath);
        uint64_t Reads = (uint64_t)(Bytes / (double)GChunkSizes[i]) + 1;

        printf("  ");
        PrintBytes((double)GChunkSizes[i]);
        printf("%14llu ", (unsigned long long)Reads);
        printf("%9.2fs ", Seconds);
        printf("%9.0f MB\n", Bytes / Seconds / (1024.0 * 1024.0));
    }

    printf("\n  가장 빨랐던 청크 = ");
    PrintBytes((double)BestChunk);
    printf("  (%.2f초)\n", BestSeconds);
    printf("  글자 수는 어느 크기에서나 %llu 로 같다\n\n",
           (unsigned long long)FirstCount);

    // ---- A7-2. 전체 코퍼스로 3그램을 센다 ----
    printf("[A7-2] 전체 코퍼스로 %d그램 모델을 만든다\n", BUILD_ORDER);

    clock_t Begin = clock();

    FNgram Built;
    if (!CHECK(NgramBuild(&Built, CorpusPath, BUILD_ORDER, 0)))
    {
        printf("모델을 못 만들었다. 메모리가 모자랐을 수 있다.\n");
        return ReportResult();
    }

    double BuildSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

    double ModelBytes = (double)Built.GramCount
                      * ((double)BUILD_ORDER * 4.0 + 8.0)
                      + (double)(Built.ContextCount + 1) * 8.0;

    printf("  그램 수        = %llu\n", (unsigned long long)Built.Total);
    printf("  서로 다른 그램 = %llu\n", (unsigned long long)Built.GramCount);
    printf("  한 번만        = %llu (%.1f%%)\n",
           (unsigned long long)Built.OnceCount,
           (double)Built.OnceCount * 100.0 / (double)Built.GramCount);
    printf("  문맥 수        = %llu\n", (unsigned long long)Built.ContextCount);
    printf("  메모리         = %.0f MB\n", ModelBytes / (1024.0 * 1024.0));
    printf("  걸린 시간      = %.1f 초\n\n", BuildSeconds);

    // 그램 수 = 글자 수 + 문장 수. N 과 무관한 불변식이다.
    CHECK(Built.Total == Built.CharCount + Built.LineCount);

    // ---- A7-3. 파일로 굽는다 ----
    printf("[A7-3] 모델을 파일로 굽고 다시 읽는다\n");

    Begin = clock();
    if (!CHECK(NgramSave(&Built, MODEL_PATH)))
    {
        printf("모델을 못 썼다. data 폴더가 있는지 확인할 것.\n");
        NgramFree(&Built);
        return ReportResult();
    }
    double SaveSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

    double SavedBytes = FileSize(MODEL_PATH);

    Begin = clock();
    FNgram Loaded;
    if (!CHECK(NgramLoad(&Loaded, MODEL_PATH)))
    {
        printf("모델을 못 읽었다.\n");
        NgramFree(&Built);
        return ReportResult();
    }
    double LoadSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

    printf("  파일 크기 = %.0f MB\n", SavedBytes / (1024.0 * 1024.0));
    printf("  굽기      = %.2f 초\n", SaveSeconds);
    printf("  읽기      = %.2f 초\n", LoadSeconds);
    printf("  세는 것보다 %.0f 배 빠르다\n\n", BuildSeconds / LoadSeconds);

    CHECK(LoadSeconds < BuildSeconds);

    // ---- 구운 것과 읽은 것이 같은가 ----
    int Same = 1;

    Same = Same && (Loaded.Order == Built.Order);
    Same = Same && (Loaded.GramCount == Built.GramCount);
    Same = Same && (Loaded.ContextCount == Built.ContextCount);
    Same = Same && (Loaded.Total == Built.Total);
    Same = Same && (Loaded.OnceCount == Built.OnceCount);
    Same = Same && (Loaded.LineCount == Built.LineCount);
    Same = Same && (Loaded.CharCount == Built.CharCount);

    if (Same)
    {
        size_t GramWords = (size_t)Built.GramCount * (size_t)Built.Order;

        Same = Same && (memcmp(Loaded.Grams, Built.Grams,
                               GramWords * sizeof(uint32_t)) == 0);
        Same = Same && (memcmp(Loaded.Cumulative, Built.Cumulative,
                               (size_t)Built.GramCount * sizeof(uint64_t)) == 0);
        Same = Same && (memcmp(Loaded.ContextStart, Built.ContextStart,
                               ((size_t)Built.ContextCount + 1)
                               * sizeof(uint64_t)) == 0);
    }

    printf("  구운 것과 읽은 것이 바이트까지 같은가 = %s\n\n",
           Same ? "예" : "아니오");

    CHECK(Same == 1);

    // 망가진 파일은 거절해야 한다.
    {
        FILE* Broken = fopen("data/broken.bin", "wb");
        if (Broken != NULL)
        {
            fwrite("XXXX", 1, 4, Broken);
            fclose(Broken);

            FNgram Junk;
            CHECK(NgramLoad(&Junk, "data/broken.bin") == 0);
            remove("data/broken.bin");
        }
    }

    NgramFree(&Built);

    // ---- A7-4. 전체 코퍼스가 문장을 얼마나 바꾸는가 ----
    printf("[A7-4] 전체 코퍼스 3그램이 쓴 문장\n\n");

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    uint32_t Buffer[MAX_SENTENCE];
    int TotalLength = 0;

    for (int i = 0; i < 8; i++)
    {
        int Length = NgramGenerate(&Loaded, &Rng, Buffer, MAX_SENTENCE);
        TotalLength += Length;

        printf("  %2d. ", i + 1);
        PrintSentence(Buffer, Length);
        printf("\n");
    }

    printf("\n  평균 길이 %.1f 글자\n\n", (double)TotalLength / 8.0);

    CHECK(TotalLength > 0);

    NgramFree(&Loaded);

    return ReportResult();
}
