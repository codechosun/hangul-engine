// ch-B01-cpp/Main.cpp
//
// B1. C++로 넘어가기
//
// 이 교재의 첫 C++ 파일이다. 확장자가 .cpp 로 바뀌었다.
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Map.h"
#include "Scan.h"
#include "Utf8.h"
#include "Vocab.h"

#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#define DEFAULT_CORPUS "data/corpus.txt"

namespace
{

// 지금 이 프로세스가 쓰고 있는 물리 메모리.
double WorkingSetMB()
{
    PROCESS_MEMORY_COUNTERS Counters = {};
    Counters.cb = sizeof(Counters);

    if (!GetProcessMemoryInfo(GetCurrentProcess(), &Counters, sizeof(Counters)))
    {
        return 0.0;
    }

    return (double)Counters.WorkingSetSize / (1024.0 * 1024.0);
}

// C++ 의 시계. C 의 clock() 보다 해상도가 높고 단위가 분명하다.
class FStopwatch
{
public:
    FStopwatch() : Start(std::chrono::steady_clock::now()) {}

    double Seconds() const
    {
        std::chrono::duration<double> Elapsed =
            std::chrono::steady_clock::now() - Start;
        return Elapsed.count();
    }

private:
    std::chrono::steady_clock::time_point Start;
};

uint64_t PackPair(uint32_t Prev, uint32_t Next)
{
    return ((uint64_t)Prev << 32) | (uint64_t)Next;
}

// 정렬 비교용. A6 의 FGramRow 를 작게 줄인 것.
struct FRow
{
    uint32_t A;
    uint32_t B;
    uint64_t Count;
};

// ---- C 방식. qsort 는 인자를 둘밖에 못 받으므로 전역이 필요하다 ----
int GSortMode = 0;

int CompareRowC(const void* Left, const void* Right)
{
    const FRow* L = (const FRow*)Left;
    const FRow* R = (const FRow*)Right;

    if (L->A != R->A) return (L->A < R->A) ? -1 : 1;
    if (GSortMode == 0)
    {
        if (L->B != R->B) return (L->B < R->B) ? -1 : 1;
    }
    return 0;
}

void PrintRow(const char* Name, const char* CWay, const char* CppWay)
{
    printf("  ");
    PrintPadded(Name, 22);
    PrintPadded(CWay, 30);
    PrintPadded(CppWay, 30);
    printf("\n");
}

} // namespace

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("B1. C++로 넘어가기\n\n");

    // ---- B1-1. C 모듈을 C++ 에서 부른다 ----
    printf("[B1-1] A파트에서 만든 C 모듈을 그대로 부른다\n");

    FMap Probe;
    CHECK(MapInit(&Probe, 16));
    MapAdd(&Probe, 42, 7);
    CHECK(MapGet(&Probe, 42) == 7);
    MapFree(&Probe);

    printf("  lib/Map 을 .cpp 에서 불렀다. 헤더에 창구를 냈기 때문이다.\n");
    printf("  (A9 에서 Kiwi 가 우리에게 해준 일을 이번엔 우리가 우리에게 했다)\n\n");

    // ---- 코퍼스를 한 번만 읽어 토큰으로 들고 있는다 ----
    printf("코퍼스를 읽는 중...\n");

    std::vector<uint32_t> Tokens;
    Tokens.reserve(700 * 1000 * 1000);

    {
        FScanner Scanner;
        if (!CHECK(ScanOpen(&Scanner, CorpusPath)))
        {
            printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
            return ReportResult();
        }

        uint32_t Code = 0;
        while (ScanNext(&Scanner, &Code))
        {
            Tokens.push_back(Code);
        }

        ScanClose(&Scanner);
    }

    printf("  글자 %zu개\n\n", Tokens.size());
    CHECK(Tokens.size() > 0);

    // ---- B1-2. 같은 일을 두 번. 바이그램 세기 ----
    printf("[B1-2] 글자 쌍 세기 — lib/Map 대 std::unordered_map\n\n");

    uint64_t CDistinct = 0;
    double CSeconds = 0.0;
    double CMemory = 0.0;
    uint64_t CCheck = 0;

    {
        double Before = WorkingSetMB();
        FStopwatch Watch;

        FMap Map;
        CHECK(MapInit(&Map, 1024));

        for (size_t i = 1; i < Tokens.size(); i++)
        {
            MapAdd(&Map, PackPair(Tokens[i - 1], Tokens[i]), 1);
        }

        CSeconds = Watch.Seconds();
        CMemory = WorkingSetMB() - Before;
        CDistinct = Map.Count;
        CCheck = MapGet(&Map, PackPair(0xC758u, 0x0020u));   // '의' -> 공백

        MapFree(&Map);   // 이 줄을 잊으면 샌다
    }

    uint64_t CppDistinct = 0;
    double CppSeconds = 0.0;
    double CppMemory = 0.0;
    uint64_t CppCheck = 0;

    {
        double Before = WorkingSetMB();
        FStopwatch Watch;

        std::unordered_map<uint64_t, uint64_t> Map;

        for (size_t i = 1; i < Tokens.size(); i++)
        {
            Map[PackPair(Tokens[i - 1], Tokens[i])] += 1;
        }

        CppSeconds = Watch.Seconds();
        CppMemory = WorkingSetMB() - Before;
        CppDistinct = Map.size();
        CppCheck = Map[PackPair(0xC758u, 0x0020u)];

        // 해제할 줄이 없다. 블록을 벗어나면 소멸자가 부른다.
    }

    printf("  ");
    PrintPadded("", 22);
    PrintPadded("C — lib/Map", 30);
    PrintPadded("C++ — unordered_map", 30);
    printf("\n");

    {
        char A[64], B[64];

        snprintf(A, sizeof(A), "%llu", (unsigned long long)CDistinct);
        snprintf(B, sizeof(B), "%llu", (unsigned long long)CppDistinct);
        PrintRow("서로 다른 쌍", A, B);

        snprintf(A, sizeof(A), "%llu", (unsigned long long)CCheck);
        snprintf(B, sizeof(B), "%llu", (unsigned long long)CppCheck);
        PrintRow("'의'->공백 횟수", A, B);

        snprintf(A, sizeof(A), "%.2f 초", CSeconds);
        snprintf(B, sizeof(B), "%.2f 초  (%.2f배)", CppSeconds, CppSeconds / CSeconds);
        PrintRow("시간", A, B);

        snprintf(A, sizeof(A), "%.0f MB", CMemory);
        snprintf(B, sizeof(B), "%.0f MB  (%.2f배)", CppMemory, CppMemory / CMemory);
        PrintRow("메모리", A, B);

        snprintf(A, sizeof(A), "%d 줄", 208 + 63);
        snprintf(B, sizeof(B), "%d 줄", 5);
        PrintRow("코드", A, B);
    }
    printf("\n");

    // 두 구현이 같은 답을 내야 한다.
    CHECK(CDistinct == CppDistinct);
    CHECK(CCheck == CppCheck);
    CHECK(CCheck > 0);

    // ---- B1-3. 문자열 사전 ----
    printf("[B1-3] 문자열 사전 — lib/Vocab 대 unordered_map<string, uint32_t>\n\n");

    // 글자를 UTF-8 문자열로 바꿔 사전에 넣는다. 형태소 대신 쓰는 대역이다.
    std::vector<std::string> Words;
    Words.reserve(200000);
    {
        char Utf8[8] = {};
        for (size_t i = 0; i + 2 < Tokens.size() && Words.size() < 200000; i += 3)
        {
            std::string Word;
            for (int k = 0; k < 3; k++)
            {
                int Bytes = Utf8Encode(Tokens[i + k], Utf8);
                Word.append(Utf8, (size_t)Bytes);
            }
            Words.push_back(Word);
        }
    }

    uint64_t CVocab = 0;
    double CVocabSeconds = 0.0;

    {
        FStopwatch Watch;

        FVocab Vocab;
        CHECK(VocabInit(&Vocab, 1024));

        for (const std::string& Word : Words)
        {
            VocabIntern(&Vocab, Word.c_str());
        }

        CVocabSeconds = Watch.Seconds();
        CVocab = Vocab.Count;

        VocabFree(&Vocab);
    }

    uint64_t CppVocab = 0;
    double CppVocabSeconds = 0.0;

    {
        FStopwatch Watch;

        std::unordered_map<std::string, uint32_t> Ids;
        std::vector<std::string> Texts;

        for (const std::string& Word : Words)
        {
            auto Found = Ids.find(Word);
            if (Found == Ids.end())
            {
                Ids.emplace(Word, (uint32_t)Texts.size());
                Texts.push_back(Word);
            }
        }

        CppVocabSeconds = Watch.Seconds();
        CppVocab = Ids.size();
    }

    {
        char A[64], B[64];

        snprintf(A, sizeof(A), "%llu", (unsigned long long)CVocab);
        snprintf(B, sizeof(B), "%llu", (unsigned long long)CppVocab);
        PrintRow("서로 다른 낱말", A, B);

        snprintf(A, sizeof(A), "%.3f 초", CVocabSeconds);
        snprintf(B, sizeof(B), "%.3f 초  (%.2f배)", CppVocabSeconds,
                 CppVocabSeconds / CVocabSeconds);
        PrintRow("시간", A, B);

        snprintf(A, sizeof(A), "%d 줄", 218 + 69);
        snprintf(B, sizeof(B), "%d 줄", 9);
        PrintRow("코드", A, B);
    }
    printf("\n");

    CHECK(CVocab == CppVocab);

    // ---- B1-4. 정렬. 전역 변수가 사라진다 ----
    printf("[B1-4] 정렬 — qsort + 전역 대 std::sort + 람다\n\n");

    std::vector<FRow> Rows;
    Rows.reserve(Tokens.size() / 32);
    for (size_t i = 1; i < Tokens.size(); i += 32)
    {
        FRow Row = { Tokens[i - 1], Tokens[i], (uint64_t)i };
        Rows.push_back(Row);
    }

    std::vector<FRow> Copy = Rows;

    double QsortSeconds = 0.0;
    {
        FStopwatch Watch;
        GSortMode = 0;                                  // 전역에 설정을 싣는다
        qsort(Copy.data(), Copy.size(), sizeof(FRow), CompareRowC);
        QsortSeconds = Watch.Seconds();
    }

    std::vector<FRow> Copy2 = Rows;

    double SortSeconds = 0.0;
    {
        const int Mode = 0;                             // 지역 변수를 잡아간다

        FStopwatch Watch;
        std::sort(Copy2.begin(), Copy2.end(),
                  [Mode](const FRow& L, const FRow& R)
                  {
                      if (L.A != R.A) return L.A < R.A;
                      if (Mode == 0) return L.B < R.B;
                      return false;
                  });
        SortSeconds = Watch.Seconds();
    }

    {
        char A[64], B[64];

        snprintf(A, sizeof(A), "%.2f 초", QsortSeconds);
        snprintf(B, sizeof(B), "%.2f 초  (%.2f배)", SortSeconds,
                 SortSeconds / QsortSeconds);
        PrintRow("시간", A, B);
        PrintRow("설정을 넘기는 법", "파일 범위 전역", "람다 캡처");
    }
    printf("\n  정렬한 개수 = %zu\n\n", Rows.size());

    // 두 정렬 결과가 같아야 한다.
    int SameOrder = 1;
    for (size_t i = 0; i < Copy.size(); i++)
    {
        if (Copy[i].A != Copy2[i].A || Copy[i].B != Copy2[i].B)
        {
            SameOrder = 0;
            break;
        }
    }
    CHECK(SameOrder == 1);

    // ---- B1-5. 정리 ----
    printf("[B1-5] 지금까지 쓴 C 코드에서 해제가 차지하는 자리\n\n");
    printf("  lib/ 의 free 호출     = 45 번\n");
    printf("  lib/ 의 ...Free 함수  = 7 개\n");
    printf("  이 파일(.cpp)의 free  = 0 번\n\n");

    printf("  C++ 쪽 블록에는 해제하는 줄이 하나도 없다.\n");
    printf("  블록을 벗어나면 소멸자가 부른다. 중간에 return 해도, 예외가 나도.\n\n");

    return ReportResult();
}
