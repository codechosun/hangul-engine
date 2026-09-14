// ch-B03-cosine/Main.cpp
//
// B3. 코사인 유사도
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Ngram.h"
#include "Utf8.h"
#include "Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define DEFAULT_CORPUS "data/corpus.txt"

#define TRAIN_LINES 200000

// 벡터를 만들 글자 수. 빈도 상위 몇 종만 본다.
#define TOP_CHARS 500

namespace
{

// 흔히 쓰는 방식. 길이를 각각 구해 곱한다. 그리고 전부 float 으로 누적한다.
float NaiveCosine(const std::vector<float>& A, const std::vector<float>& B)
{
    float Dot = 0.0f;
    float LenA = 0.0f;
    float LenB = 0.0f;

    for (size_t i = 0; i < A.size(); i++)
    {
        Dot += A[i] * B[i];
        LenA += A[i] * A[i];
        LenB += B[i] * B[i];
    }

    return Dot / (std::sqrt(LenA) * std::sqrt(LenB));
}

std::string Describe(uint32_t Code)
{
    if (Code == TOKEN_BOS) return "BOS";
    if (Code == TOKEN_EOS) return "EOS";
    if (Code == ' ')       return "공백";
    if (Code < 0x20u)      return "제어";

    char Utf8[8] = {};
    int Bytes = Utf8Encode(Code, Utf8);
    return std::string(Utf8, (size_t)Bytes);
}

} // namespace

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("B3. 코사인 유사도\n\n");

    // ---- 바이그램 표에서 글자마다 벡터를 뽑는다 ----
    printf("모델을 만드는 중 (앞 %d줄)...\n", TRAIN_LINES);

    FNgram Unigram;
    FNgram Bigram;

    if (!CHECK(NgramBuild(&Unigram, CorpusPath, 1, TRAIN_LINES)) ||
        !CHECK(NgramBuild(&Bigram, CorpusPath, 2, TRAIN_LINES)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        return ReportResult();
    }

    // 어휘 = 1그램에 들어 있는 토큰 전부. 사전순으로 들어 있다.
    std::vector<uint32_t> Vocabulary;
    std::vector<uint64_t> Counts;
    for (uint64_t i = 0; i < Unigram.GramCount; i++)
    {
        uint64_t Before = (i == 0) ? 0 : Unigram.Cumulative[i - 1];

        Vocabulary.push_back(Unigram.Grams[i]);
        Counts.push_back(Unigram.Cumulative[i] - Before);
    }

    const size_t Dim = Vocabulary.size();

    // 코드포인트 -> 어휘 안의 자리
    std::vector<int> IndexOf((size_t)VOCAB_LIMIT, -1);
    for (size_t i = 0; i < Vocabulary.size(); i++)
    {
        IndexOf[Vocabulary[i]] = (int)i;
    }

    // 빈도 상위 TOP_CHARS 종만 고른다.
    std::vector<size_t> Order(Vocabulary.size());
    for (size_t i = 0; i < Order.size(); i++)
    {
        Order[i] = i;
    }

    std::sort(Order.begin(), Order.end(),
              [&Counts](size_t L, size_t R) { return Counts[L] > Counts[R]; });

    if (Order.size() > TOP_CHARS)
    {
        Order.resize(TOP_CHARS);
    }

    printf("  어휘 %zu종, 그중 상위 %zu종에 대해 %zu차원 벡터를 만든다\n\n",
           Dim, Order.size(), Dim);

    // 글자 하나의 벡터 = "그 다음에 무엇이 얼마나 왔는가"
    std::vector<FVector> Vectors;
    std::vector<uint32_t> Chars;

    for (size_t k = 0; k < Order.size(); k++)
    {
        uint32_t Code = Vocabulary[Order[k]];

        FVector Row(Dim);

        int64_t Context = NgramFindContext(&Bigram, &Code);
        if (Context >= 0)
        {
            uint64_t From = Bigram.ContextStart[Context];
            uint64_t To = Bigram.ContextStart[Context + 1];

            for (uint64_t i = From; i < To; i++)
            {
                uint64_t Before = (i == From) ? 0 : Bigram.Cumulative[i - 1];
                uint32_t Next = Bigram.Grams[i * 2 + 1];

                int Slot = IndexOf[Next];
                if (Slot >= 0)
                {
                    Row[(size_t)Slot] = (Real)(Bigram.Cumulative[i] - Before);
                }
            }
        }

        Vectors.push_back(Row);
        Chars.push_back(Code);
    }

    CHECK(Vectors.size() == Order.size());

    // ---- B3-1. 자기 자신과의 유사도는 1.0 인가 ----
    printf("[B3-1] 자기 자신과의 유사도\n\n");

    {
        int NaiveExact = 0;
        int CarefulExact = 0;
        int EmptyCount = 0;
        float WorstNaive = 0.0f;
        Real WorstCareful = Real(0);

        for (size_t k = 0; k < Vectors.size(); k++)
        {
            // 길이가 0 인 벡터는 각도를 말할 수 없다. 따로 센다.
            if (Vectors[k].LengthSquared() == Real(0))
            {
                EmptyCount++;
                continue;
            }

            // 같은 값을 float 배열로도 만들어 순진한 방식과 비교한다.
            std::vector<float> Raw(Dim);
            for (size_t i = 0; i < Dim; i++)
            {
                Raw[i] = (float)Vectors[k][i];
            }

            float Naive = NaiveCosine(Raw, Raw);
            Real Careful = CosineSimilarity(Vectors[k], Vectors[k]);

            if (Naive == 1.0f)   NaiveExact++;
            if (Careful == Real(1)) CarefulExact++;

            float NaiveGap = std::fabs(Naive - 1.0f);
            Real CarefulGap = (Real)std::fabs((double)Careful - 1.0);

            if (NaiveGap > WorstNaive)     WorstNaive = NaiveGap;
            if (CarefulGap > WorstCareful) WorstCareful = CarefulGap;
        }

        printf("  %zu개 글자 각각을 자기 자신과 비교했다 "
               "(길이가 0 인 벡터 %d개는 뺐다)\n\n",
               Vectors.size(), EmptyCount);
        printf("  ");
        PrintPadded("방식", 34);
        PrintPaddedRight("정확히 1.0", 14);
        PrintPaddedRight("최대 오차", 16);
        printf("\n");

        char Buffer[64];

        printf("  ");
        PrintPadded("float 누적, 제곱근 두 번", 34);
        printf("%13d ", NaiveExact);
        snprintf(Buffer, sizeof(Buffer), "%.3e", (double)WorstNaive);
        PrintPaddedRight(Buffer, 16);
        printf("\n");

        printf("  ");
        PrintPadded("double 누적, 제곱근 한 번", 34);
        printf("%13d ", CarefulExact);
        snprintf(Buffer, sizeof(Buffer), "%.3e", (double)WorstCareful);
        PrintPaddedRight(Buffer, 16);
        printf("\n\n");

        // 순진한 방식은 1.0 을 못 맞히는 경우가 있다.
        CHECK(NaiveExact < (int)Vectors.size() - EmptyCount);

        // 조심한 방식이 더 자주 맞힌다.
        CHECK(CarefulExact > NaiveExact);

        // 길이가 0 인 벡터가 있다. 그게 무엇인지는 본문에서 밝힌다.
        CHECK(EmptyCount == 1);

        printf("  '정확히 1.0' 을 요구하면 안 된다. 늘 여유를 두고 비교한다.\n");
        printf("    CHECK_NEAR(Similarity, 1.0, 1e-6)\n\n");

        for (size_t k = 0; k < Vectors.size(); k++)
        {
            if (Vectors[k].LengthSquared() == Real(0))
            {
                continue;
            }
            CHECK_NEAR(CosineSimilarity(Vectors[k], Vectors[k]), 1.0, 1e-6);
        }
    }

    // ---- B3-2. 수식이 코드가 된다 ----
    printf("[B3-2] 연산자 오버로딩\n\n");

    {
        FVector A(3);
        A[0] = Real(3); A[1] = Real(0); A[2] = Real(4);

        FVector B(3);
        B[0] = Real(1); B[1] = Real(0); B[2] = Real(0);

        printf("  A = (3, 0, 4), B = (1, 0, 0)\n");
        printf("    A.Length()          = %.6f  (손계산 5)\n", (double)A.Length());
        printf("    A * B               = %.6f  (손계산 3)\n", (double)(A * B));
        printf("    CosineSimilarity    = %.6f  (손계산 0.6)\n",
               (double)CosineSimilarity(A, B));
        printf("    (A + B) * B         = %.6f  (손계산 4)\n",
               (double)((A + B) * B));
        printf("    (2 * A) * B         = %.6f  (손계산 6)\n",
               (double)((Real(2) * A) * B));
        printf("    EuclideanDistance   = %.6f  (손계산 sqrt(20) = %.6f)\n\n",
               (double)EuclideanDistance(A, B), std::sqrt(20.0));

        CHECK_NEAR(A.Length(), 5.0, 1e-6);
        CHECK_NEAR(A * B, 3.0, 1e-6);
        CHECK_NEAR(CosineSimilarity(A, B), 0.6, 1e-6);
        CHECK_NEAR((A + B) * B, 4.0, 1e-6);
        CHECK_NEAR((Real(2) * A) * B, 6.0, 1e-6);
        CHECK_NEAR(EuclideanDistance(A, B), std::sqrt(20.0), 1e-6);

        // 성질 몇 가지.
        CHECK_NEAR(CosineSimilarity(A, B), CosineSimilarity(B, A), 1e-9);  // 대칭
        CHECK_NEAR(CosineSimilarity(A, Real(7) * B),
                   CosineSimilarity(A, B), 1e-6);                          // 크기 무관
        CHECK_NEAR(CosineSimilarity(A, Real(-1) * A), -1.0, 1e-6);         // 반대는 -1
        CHECK_NEAR(A.Normalized().Length(), 1.0, 1e-6);
    }

    // ---- B3-3. 실제로 무엇이 비슷한가 ----
    printf("[B3-3] 다음에 오는 글자가 비슷하면 비슷한 글자인가\n\n");

    const uint32_t Queries[5] = { 0xC744u, 0xB294u, 0x0031u, 0xD55Cu, 0x002Eu };
    //                            을        는       1        한       .

    for (uint32_t Query : Queries)
    {
        int Self = -1;
        for (size_t k = 0; k < Chars.size(); k++)
        {
            if (Chars[k] == Query)
            {
                Self = (int)k;
                break;
            }
        }

        if (Self < 0)
        {
            continue;
        }

        std::vector<std::pair<Real, uint32_t>> Scores;
        for (size_t k = 0; k < Vectors.size(); k++)
        {
            if ((int)k == Self)
            {
                continue;
            }
            Scores.emplace_back(CosineSimilarity(Vectors[(size_t)Self], Vectors[k]),
                                Chars[k]);
        }

        std::sort(Scores.begin(), Scores.end(),
                  [](const std::pair<Real, uint32_t>& L,
                     const std::pair<Real, uint32_t>& R)
                  {
                      if (L.first != R.first) return L.first > R.first;
                      return L.second < R.second;
                  });

        char Label[32];
        snprintf(Label, sizeof(Label), "'%s'", Describe(Query).c_str());

        printf("  ");
        PrintPadded(Label, 8);
        printf("→ ");

        for (int k = 0; k < 5 && k < (int)Scores.size(); k++)
        {
            printf("%s(%.5f) ", Describe(Scores[(size_t)k].second).c_str(),
                   (double)Scores[(size_t)k].first);
        }
        printf("\n");
    }
    printf("\n");

    // ---- B3-4. 왜 코사인인가 ----
    printf("[B3-4] 유클리드 거리로 재면 어떻게 되는가\n\n");

    {
        // '을' 과 '를' 은 쓰임이 거의 같은데 빈도가 다르다.
        int A = -1, B = -1, C = -1;
        for (size_t k = 0; k < Chars.size(); k++)
        {
            if (Chars[k] == 0xC744u) A = (int)k;   // 을
            if (Chars[k] == 0xB97Cu) B = (int)k;   // 를
            if (Chars[k] == 0x0020u) C = (int)k;   // 공백
        }

        if (A >= 0 && B >= 0 && C >= 0)
        {
            printf("  ");
            PrintPadded("두 글자", 16);
            PrintPaddedRight("코사인", 12);
            PrintPaddedRight("유클리드 거리", 18);
            printf("\n");

            struct { int L; int R; const char* Name; } Pairs[3] = {
                { A, B, "'을' 과 '를'" },
                { A, C, "'을' 과 공백" },
                { B, C, "'를' 과 공백" },
            };

            for (int i = 0; i < 3; i++)
            {
                printf("  ");
                PrintPadded(Pairs[i].Name, 16);
                printf("%11.4f ", (double)CosineSimilarity(
                    Vectors[(size_t)Pairs[i].L], Vectors[(size_t)Pairs[i].R]));
                printf("%17.0f\n", (double)EuclideanDistance(
                    Vectors[(size_t)Pairs[i].L], Vectors[(size_t)Pairs[i].R]));
            }

            printf("\n  코사인은 '을'-'를' 이 제일 가깝다고 말한다.\n");
            printf("  유클리드 거리는 빈도 차이에 휘둘린다.\n\n");

            Real CosSame = CosineSimilarity(Vectors[(size_t)A], Vectors[(size_t)B]);
            Real CosDiff = CosineSimilarity(Vectors[(size_t)A], Vectors[(size_t)C]);
            CHECK(CosSame > CosDiff);
        }
    }

    NgramFree(&Bigram);
    NgramFree(&Unigram);

    return ReportResult();
}
