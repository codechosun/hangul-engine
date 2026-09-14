// ch-A04-hashmap/Main.c
//
// A4. 해시테이블 개조
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"
#include "Map.h"
#include "Random.h"
#include "Scan.h"
#include "Utf8.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CORPUS "data/corpus.txt"
#define BOOK_SEED 20260914ull

// 코퍼스에서 실제로 나온 글자 종류 (A2 에서 측정)
#define DISTINCT_CHARS 27663

// 섞지 않은 해시로 탐사 실험을 할 때 쓸 키 개수.
// 전부 쓰면 너무 느려서 끝나지 않는다.
#define PROBE_SAMPLE 100000

// 시간을 잴 때 반복 횟수. 한 번은 너무 짧아 재어지지 않는다.
#define REPEAT_COUNT 10

// 글자 두 개를 64비트 키 하나로 묶는다.
static uint64_t PackPair(uint32_t Prev, uint32_t Next)
{
    return ((uint64_t)Prev << 32) | (uint64_t)Next;
}

static void DescribeCode(uint32_t Code, char* Out, int OutSize)
{
    if (Code == ' ')  { snprintf(Out, OutSize, "공백");   return; }
    if (Code == '\n') { snprintf(Out, OutSize, "줄바꿈"); return; }

    char Utf8[8] = { 0 };
    int Len = Utf8Encode(Code, Utf8);
    Utf8[Len] = '\0';
    snprintf(Out, OutSize, "%s", Utf8);
}

// "의 -> 공백" 같은 문자열을 만든다.
static void DescribePair(uint64_t Key, char* Out, int OutSize)
{
    char Left[32];
    char Right[32];

    DescribeCode((uint32_t)(Key >> 32), Left, sizeof(Left));
    DescribeCode((uint32_t)(Key & 0xFFFFFFFFu), Right, sizeof(Right));

    snprintf(Out, OutSize, "%s -> %s", Left, Right);
}

static int Log2OfPowerOfTwo(uint64_t N)
{
    int Bits = 0;
    while (N > 1)
    {
        N >>= 1;
        Bits++;
    }
    return Bits;
}

// ---------------------------------------------------------------------------
// A4-2. 자료구조 교재 1.7 의 해시맵을 그대로 옮겨온 것.
//       칸이 7개로 고정이고, 키가 문자열이다.
// ---------------------------------------------------------------------------

#define BOOK_SLOTS 7

typedef struct
{
    const char* Keys[BOOK_SLOTS];
    int Values[BOOK_SLOTS];
} FBookMap;

static size_t HashWith65599(const char* Key)
{
    size_t HashValue = 0;
    size_t Length = strlen(Key);

    for (size_t i = 0; i < Length; ++i)
    {
        HashValue = 65599 * HashValue + (size_t)Key[i];
    }

    return HashValue ^ (HashValue >> 16);
}

static int BookMapInsert(FBookMap* Map, const char* Key, int Value)
{
    size_t Start = HashWith65599(Key) % BOOK_SLOTS;
    size_t i = Start;

    do
    {
        if (Map->Keys[i] != NULL && strcmp(Key, Map->Keys[i]) == 0)
        {
            Map->Values[i] = Value;
            return 1;
        }

        if (Map->Keys[i] == NULL)
        {
            Map->Keys[i] = Key;
            Map->Values[i] = Value;
            return 1;
        }

        i = (i + 1) % BOOK_SLOTS;
    } while (i != Start);

    return 0;   // 표가 꽉 찼다
}

// ---------------------------------------------------------------------------
// A4-3. 키가 어느 칸에서 출발하는지만 본다. 탐사는 하지 않는다.
// ---------------------------------------------------------------------------

typedef struct
{
    uint64_t DistinctSlots;   // 서로 다른 출발 칸의 수
    uint64_t BiggestPile;     // 한 칸에서 출발하는 키의 최대 개수
} FSpread;

static FSpread MeasureSpread(const uint64_t* Keys, uint64_t Count,
                             uint64_t Slots, int UseMix)
{
    FSpread Result = { 0, 0 };

    uint32_t* Pile = (uint32_t*)calloc((size_t)Slots, sizeof(uint32_t));
    if (Pile == NULL)
    {
        return Result;
    }

    uint64_t Mask = Slots - 1;

    for (uint64_t i = 0; i < Count; i++)
    {
        uint64_t Hash = UseMix ? MapHash(Keys[i]) : Keys[i];
        Pile[Hash & Mask]++;
    }

    for (uint64_t i = 0; i < Slots; i++)
    {
        if (Pile[i] > 0)
        {
            Result.DistinctSlots++;
        }
        if (Pile[i] > Result.BiggestPile)
        {
            Result.BiggestPile = Pile[i];
        }
    }

    free(Pile);
    return Result;
}

// ---------------------------------------------------------------------------
// A4-3/A4-4. 실제로 자리를 잡아보며 탐사 횟수를 센다.
// ---------------------------------------------------------------------------

typedef struct
{
    double Average;
    uint64_t Worst;
    double Seconds;
} FProbeResult;

// UseMix  : 0 이면 키를 그대로, 1 이면 MapHash 로 섞어서
// UsePrime: 0 이면 2의 거듭제곱 & 마스크, 1 이면 소수 % 나머지
static FProbeResult MeasureProbes(const uint64_t* Keys, uint64_t Count,
                                  uint64_t Slots, int UseMix, int UsePrime,
                                  int Repeat)
{
    FProbeResult Result = { 0.0, 0, 0.0 };

    uint64_t* Table = (uint64_t*)malloc((size_t)Slots * sizeof(uint64_t));
    uint8_t* Used = (uint8_t*)calloc((size_t)Slots, sizeof(uint8_t));
    if (Table == NULL || Used == NULL)
    {
        free(Table);
        free(Used);
        return Result;
    }

    uint64_t Mask = Slots - 1;
    uint64_t Total = 0;

    clock_t Begin = clock();

    for (int Round = 0; Round < Repeat; Round++)
    {
        memset(Used, 0, (size_t)Slots);
        Total = 0;
        Result.Worst = 0;

        for (uint64_t i = 0; i < Count; i++)
        {
            uint64_t Key = Keys[i];
            uint64_t Hash = UseMix ? MapHash(Key) : Key;
            uint64_t Slot = UsePrime ? (Hash % Slots) : (Hash & Mask);

            uint64_t Probes = 1;
            while (Used[Slot] != 0 && Table[Slot] != Key)
            {
                Slot = UsePrime ? ((Slot + 1) % Slots) : ((Slot + 1) & Mask);
                Probes++;
            }

            Table[Slot] = Key;
            Used[Slot] = 1;

            Total += Probes;
            if (Probes > Result.Worst)
            {
                Result.Worst = Probes;
            }
        }
    }

    Result.Seconds = (double)(clock() - Begin) / CLOCKS_PER_SEC / (double)Repeat;
    Result.Average = (double)Total / (double)Count;

    free(Table);
    free(Used);

    return Result;
}

static int IsPrime(uint64_t N)
{
    if (N < 2) return 0;
    if (N % 2 == 0) return N == 2;

    for (uint64_t D = 3; D * D <= N; D += 2)
    {
        if (N % D == 0) return 0;
    }
    return 1;
}

static uint64_t NextPrime(uint64_t N)
{
    while (!IsPrime(N))
    {
        N++;
    }
    return N;
}

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("A4. 해시테이블 개조\n\n");

    // ---- A4-1. 2차원 배열이면 얼마나 드는가 ----
    printf("[A4-1] 글자 쌍을 2차원 배열로 세면\n");

    double Cell = 8.0;   // uint64_t 한 칸
    double HangulCells = 11172.0 * 11172.0;
    double SeenCells = (double)DISTINCT_CHARS * (double)DISTINCT_CHARS;
    double AllCells = 1114112.0 * 1114112.0;

    printf("  한글끼리만 (11,172^2)  %14.0f 칸  %8.0f MB\n",
           HangulCells, HangulCells * Cell / (1024.0 * 1024.0));
    printf("  코퍼스의 글자끼리      %14.0f 칸  %8.1f GB\n",
           SeenCells, SeenCells * Cell / (1024.0 * 1024.0 * 1024.0));
    printf("  유니코드 전체          %14.3e 칸  %8.0f TB\n",
           AllCells, AllCells * Cell / (1024.0 * 1024.0 * 1024.0 * 1024.0));
    printf("\n");

    // ---- A4-2. 교재 해시맵은 왜 그대로 못 쓰는가 ----
    printf("[A4-2] 자료구조 교재 1.7 의 해시맵에 8개를 넣으면\n");

    FBookMap Book;
    memset(&Book, 0, sizeof(Book));

    const char* Names[8] = { "Park", "Hwang", "Lee", "Kim",
                             "coco", "ococ", "Choi", "Jung" };
    int Inserted = 0;
    int FirstFail = -1;

    for (int i = 0; i < 8; i++)
    {
        if (BookMapInsert(&Book, Names[i], i))
        {
            Inserted++;
        }
        else if (FirstFail < 0)
        {
            FirstFail = i;
        }
    }

    printf("  칸 수 = %d, 넣으려 한 개수 = 8\n", BOOK_SLOTS);
    printf("  들어간 개수 = %d, 처음 실패한 것 = %d번째 (\"%s\")\n\n",
           Inserted, FirstFail + 1, Names[FirstFail]);

    CHECK(Inserted == BOOK_SLOTS);
    CHECK(FirstFail == BOOK_SLOTS);

    // ---- 코퍼스에서 글자 쌍 세기 ----
    printf("코퍼스를 훑는 중...\n\n");

    FMap Pairs;
    if (!CHECK(MapInit(&Pairs, 1024)))
    {
        return ReportResult();
    }

    FScanner Scanner;
    if (!CHECK(ScanOpen(&Scanner, CorpusPath)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        MapFree(&Pairs);
        return ReportResult();
    }

    clock_t Begin = clock();

    uint64_t Chars = 0;
    uint64_t PairTotal = 0;
    uint32_t Prev = 0;
    uint32_t Code = 0;
    int HasPrev = 0;

    while (ScanNext(&Scanner, &Code))
    {
        Chars++;

        if (HasPrev)
        {
            MapAdd(&Pairs, PackPair(Prev, Code), 1);
            PairTotal++;
        }

        Prev = Code;
        HasPrev = 1;
    }

    double CountSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;
    ScanClose(&Scanner);

    // ---- A4-5. 실측 ----
    printf("[A4-5] 글자 쌍 세기 결과\n");
    printf("  글자 수        = %llu\n", (unsigned long long)Chars);
    printf("  쌍의 개수      = %llu\n", (unsigned long long)PairTotal);
    printf("  서로 다른 쌍   = %llu\n", (unsigned long long)Pairs.Count);
    printf("  표의 칸 수     = %llu (2^%d)\n",
           (unsigned long long)Pairs.Capacity,
           Log2OfPowerOfTwo(Pairs.Capacity));
    printf("  적재율         = %.3f\n", MapLoadFactor(&Pairs));
    printf("  평균 탐사 횟수 = %.3f\n", MapAverageProbe(&Pairs));
    printf("  메모리         = %.0f MB\n",
           (double)Pairs.Capacity * 17.0 / (1024.0 * 1024.0));
    printf("  걸린 시간      = %.2f 초\n", CountSeconds);
    printf("  2차원 배열이라면 필요했을 칸의 %.4f%% 만 씀\n\n",
           (double)Pairs.Count * 100.0 / SeenCells);

    // 쌍의 개수는 글자 수보다 정확히 하나 적어야 한다.
    CHECK(PairTotal == Chars - 1);

    // ---- A4-6. 값이 맞는지 확인 ----
    uint64_t Sum = 0;
    uint64_t Distinct = 0;
    for (uint64_t i = 0; i < Pairs.Capacity; i++)
    {
        if (Pairs.Used[i] != 0)
        {
            Sum += Pairs.Values[i];
            Distinct++;
        }
    }

    CHECK(Sum == PairTotal);
    CHECK(Distinct == Pairs.Count);

    // 없는 키는 0 이어야 한다. 코드포인트 0 은 코퍼스에 없다.
    CHECK(MapGet(&Pairs, PackPair(0, 0)) == 0);

    // ---- 무엇이 가장 흔한 쌍인가 ----
    // 상위 10개만 필요하므로 정렬하지 않고 10번 훑는다.
    uint64_t TopKey[10] = { 0 };
    uint64_t TopValue[10] = { 0 };

    for (int Rank = 0; Rank < 10; Rank++)
    {
        uint64_t Limit = (Rank == 0) ? UINT64_MAX : TopValue[Rank - 1];

        for (uint64_t i = 0; i < Pairs.Capacity; i++)
        {
            if (Pairs.Used[i] == 0)
            {
                continue;
            }

            uint64_t Value = Pairs.Values[i];
            if (Value > TopValue[Rank] && (Value < Limit ||
                (Value == Limit && Pairs.Keys[i] > TopKey[Rank - 1])))
            {
                TopValue[Rank] = Value;
                TopKey[Rank] = Pairs.Keys[i];
            }
        }
    }

    printf("  가장 흔한 쌍 열 개\n");
    for (int Rank = 0; Rank < 10; Rank++)
    {
        char Pair[80];
        DescribePair(TopKey[Rank], Pair, sizeof(Pair));

        printf("   %2d. ", Rank + 1);
        PrintPadded(Pair, 16);
        printf("%10llu 번\n", (unsigned long long)TopValue[Rank]);
    }
    printf("\n");

    CHECK(TopValue[0] > TopValue[9]);

    // ---- 실험용 키 목록 뽑기 ----
    // 표를 칸 순서대로 훑으면 이미 해시 순으로 줄 세워진 목록이 나온다.
    // 그대로 쓰면 실험이 편향되므로 한 번 섞는다.
    uint64_t* Sample = (uint64_t*)malloc((size_t)Pairs.Count * sizeof(uint64_t));
    if (!CHECK(Sample != NULL))
    {
        MapFree(&Pairs);
        return ReportResult();
    }

    uint64_t SampleCount = 0;
    for (uint64_t i = 0; i < Pairs.Capacity; i++)
    {
        if (Pairs.Used[i] != 0)
        {
            Sample[SampleCount++] = Pairs.Keys[i];
        }
    }
    CHECK(SampleCount == Pairs.Count);

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    for (uint64_t i = SampleCount - 1; i > 0; i--)
    {
        uint64_t j = RandomBelow(&Rng, i + 1);
        uint64_t Temp = Sample[i];
        Sample[i] = Sample[j];
        Sample[j] = Temp;
    }

    uint64_t Slots = 16;
    while ((double)SampleCount > (double)Slots * 0.7)
    {
        Slots <<= 1;
    }

    // ---- A4-3. 섞지 않으면 어떻게 되는가 ----
    printf("[A4-3] 키 %llu개를 칸 %llu개(2^%d)에 뿌리면\n",
           (unsigned long long)SampleCount,
           (unsigned long long)Slots, Log2OfPowerOfTwo(Slots));

    FSpread RawSpread = MeasureSpread(Sample, SampleCount, Slots, 0);
    FSpread MixSpread = MeasureSpread(Sample, SampleCount, Slots, 1);

    printf("  ");
    PrintPadded("방식", 18);
    PrintPaddedRight("쓰인 칸", 12);
    PrintPaddedRight("최대 무더기", 14);
    printf("\n  ");
    PrintPadded("키 그대로", 18);
    printf("%12llu %13llu\n  ",
           (unsigned long long)RawSpread.DistinctSlots,
           (unsigned long long)RawSpread.BiggestPile);
    PrintPadded("MapHash 로 섞기", 18);
    printf("%12llu %13llu\n",
           (unsigned long long)MixSpread.DistinctSlots,
           (unsigned long long)MixSpread.BiggestPile);
    printf("\n");

    CHECK(MixSpread.DistinctSlots > RawSpread.DistinctSlots * 10);
    CHECK(MixSpread.BiggestPile < 20);

    // 실제로 자리를 잡아보면 몇 칸을 들여다보는가.
    // 섞지 않은 쪽이 워낙 느려서 10만 개만 쓴다.
    uint64_t ProbeCount = SampleCount;
    if (ProbeCount > PROBE_SAMPLE)
    {
        ProbeCount = PROBE_SAMPLE;
    }

    uint64_t ProbeSlots = 16;
    while ((double)ProbeCount > (double)ProbeSlots * 0.7)
    {
        ProbeSlots <<= 1;
    }

    FProbeResult Raw = MeasureProbes(Sample, ProbeCount, ProbeSlots, 0, 0, 1);
    FProbeResult Mixed = MeasureProbes(Sample, ProbeCount, ProbeSlots, 1, 0, 1);

    printf("  키 %llu개로 실제 탐사 횟수를 재면 (칸 %llu개, 적재율 %.2f)\n",
           (unsigned long long)ProbeCount, (unsigned long long)ProbeSlots,
           (double)ProbeCount / (double)ProbeSlots);
    printf("  ");
    PrintPadded("방식", 18);
    PrintPaddedRight("평균 탐사", 12);
    PrintPaddedRight("최악", 14);
    PrintPaddedRight("시간", 10);
    printf("\n  ");
    PrintPadded("키 그대로", 18);
    printf("%12.2f %13llu %9.3fs\n  ",
           Raw.Average, (unsigned long long)Raw.Worst, Raw.Seconds);
    PrintPadded("MapHash 로 섞기", 18);
    printf("%12.2f %13llu %9.3fs\n",
           Mixed.Average, (unsigned long long)Mixed.Worst, Mixed.Seconds);
    printf("\n");

    CHECK(Mixed.Average < Raw.Average);
    CHECK(Mixed.Average < 2.0);

    // ---- A4-4. 소수 % 와 2의 거듭제곱 & ----
    uint64_t PrimeSlots = NextPrime(Slots);

    printf("[A4-4] 나머지 연산과 비트 마스크 (키 %llu개, %d번 반복 평균)\n",
           (unsigned long long)SampleCount, REPEAT_COUNT);

    FProbeResult MaskWay =
        MeasureProbes(Sample, SampleCount, Slots, 1, 0, REPEAT_COUNT);
    FProbeResult PrimeWay =
        MeasureProbes(Sample, SampleCount, PrimeSlots, 1, 1, REPEAT_COUNT);

    printf("  ");
    PrintPadded("방식", 18);
    PrintPaddedRight("칸 수", 12);
    PrintPaddedRight("평균 탐사", 12);
    PrintPaddedRight("시간", 10);
    printf("\n  ");
    PrintPadded("& 마스크", 18);
    printf("%12llu %12.3f %9.3fs\n  ",
           (unsigned long long)Slots, MaskWay.Average, MaskWay.Seconds);
    PrintPadded("% 나머지 (소수)", 18);
    printf("%12llu %12.3f %9.3fs\n",
           (unsigned long long)PrimeSlots, PrimeWay.Average, PrimeWay.Seconds);
    printf("  %% 가 & 보다 %.2f 배 걸린다\n\n",
           PrimeWay.Seconds / MaskWay.Seconds);

    CHECK(MaskWay.Seconds < PrimeWay.Seconds);

    // ---- A4-6 이어서. 자라나는 표가 값을 잃지 않는가 ----
    FMap Small;
    CHECK(MapInit(&Small, 16));

    RandomSeed(&Rng, BOOK_SEED);

    static uint64_t Written[5000];
    for (int i = 0; i < 5000; i++)
    {
        Written[i] = RandomNext(&Rng);
        MapAdd(&Small, Written[i], (uint64_t)(i + 1));
    }

    int AllFound = 1;
    for (int i = 0; i < 5000; i++)
    {
        if (MapGet(&Small, Written[i]) != (uint64_t)(i + 1))
        {
            AllFound = 0;
            break;
        }
    }

    printf("[A4-6] 16칸으로 시작해 5,000개를 넣으면\n");
    printf("  칸 수           = %llu\n", (unsigned long long)Small.Capacity);
    printf("  들어간 수       = %llu\n", (unsigned long long)Small.Count);
    printf("  전부 되찾았는가 = %s\n\n", AllFound ? "예" : "아니오");

    CHECK(AllFound == 1);
    CHECK(Small.Count == 5000);

    MapFree(&Small);
    free(Sample);
    MapFree(&Pairs);

    return ReportResult();
}
