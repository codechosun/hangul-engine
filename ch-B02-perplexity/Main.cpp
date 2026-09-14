// ch-B02-perplexity/Main.cpp
//
// B2. 퍼플렉서티와 수치 안정성
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Backoff.h"
#include "LogMath.h"
#include "Ngram.h"
#include "Scan.h"
#include "Utf8.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define DEFAULT_CORPUS "data/corpus.txt"

// 훈련에 쓰는 줄. A6·A8 과 같다.
#define TRAIN_LINES 200000

// 훈련에 안 쓴 줄로 평가한다. 이게 이 장의 핵심이다.
#define TEST_LINES 5000

#define TOP_ORDER 6

namespace
{

// 평가 결과.
struct FScore
{
    double LogProbSum = 0.0;   // 자연로그 합
    uint64_t TokenCount = 0;
    uint64_t ZeroCount = 0;    // 확률이 0 이었던 토큰 (1그램도 모르는 글자)
};

// 퍼플렉서티 = exp( -(로그확률 합) / 토큰 수 )
double Perplexity(const FScore& Score)
{
    if (Score.TokenCount == 0)
    {
        return INFINITY;
    }

    return exp(-Score.LogProbSum / (double)Score.TokenCount);
}

// 평가용 문장 하나를 훑으며 로그확률을 더한다.
//
// Floor 가 0 이면 확률 0 을 그대로 쓴다 -> 로그가 -inf 가 된다.
// Floor 가 양수면 그 값으로 받쳐준다.
void ScoreLine(const FBackoff& Backoff, const std::vector<uint32_t>& Line,
               int Order, double Floor, FScore& Out)
{
    int Length = Order - 1;

    std::vector<uint32_t> Context((size_t)Length, TOKEN_BOS);

    for (size_t i = 0; i <= Line.size(); i++)
    {
        uint32_t Token = (i < Line.size()) ? Line[i] : (uint32_t)TOKEN_EOS;

        double Prob = BackoffProb(&Backoff, Context.data(), Length, Token);

        if (Prob <= 0.0)
        {
            Out.ZeroCount++;
            Prob = Floor;
        }

        Out.LogProbSum += log(Prob);
        Out.TokenCount++;

        for (int k = 0; k < Length - 1; k++)
        {
            Context[(size_t)k] = Context[(size_t)k + 1];
        }
        if (Length >= 1)
        {
            Context[(size_t)Length - 1] = Token;
        }
    }
}

FScore ScoreAll(const FBackoff& Backoff,
                const std::vector<std::vector<uint32_t>>& Lines,
                int Order, double Floor)
{
    FScore Score;
    for (const std::vector<uint32_t>& Line : Lines)
    {
        ScoreLine(Backoff, Line, Order, Floor, Score);
    }
    return Score;
}

} // namespace

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("B2. 퍼플렉서티와 수치 안정성\n\n");

    // ---- B2-1. 확률을 곱하면 0 이 된다 ----
    printf("[B2-1] 확률을 그냥 곱하면\n\n");

    {
        const double P = 0.02;   // 한 글자의 확률이 2%쯤이라고 하자

        double Product = 1.0;
        int DiedAt = -1;

        for (int i = 1; i <= 400; i++)
        {
            Product *= P;
            if (Product == 0.0 && DiedAt < 0)
            {
                DiedAt = i;
            }
        }

        double LogSum = 0.0;
        for (int i = 1; i <= 400; i++)
        {
            LogSum += log(P);
        }

        printf("  글자 하나의 확률을 %.2f 로 두고 400번 곱하면\n", P);
        printf("    그냥 곱하기 = %.3e  (%d번째에서 0이 되었다)\n", Product, DiedAt);
        printf("    로그로 더하기 = %.3f  (곧 %.3e)\n", LogSum, exp(LogSum));
        printf("\n  double 이 담을 수 있는 가장 작은 양수는 약 %.3e 다.\n",
               (double)5e-324);
        printf("  0.02 를 %d 번 곱하면 거기를 지나간다.\n\n", DiedAt);

        CHECK(Product == 0.0);
        CHECK(DiedAt > 0 && DiedAt < 400);
        CHECK(LogSum < -1000.0);
    }

    // ---- 모델 만들기 ----
    printf("모델을 만드는 중 (훈련 %d줄)...\n", TRAIN_LINES);

    FBackoff Backoff;
    if (!CHECK(BackoffBuild(&Backoff, CorpusPath, TOP_ORDER, TRAIN_LINES)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        return ReportResult();
    }

    // ---- 훈련에 안 쓴 줄을 모은다 ----
    std::vector<std::vector<uint32_t>> TestLines;
    uint64_t TestTokens = 0;

    {
        FScanner Scanner;
        if (!CHECK(ScanOpen(&Scanner, CorpusPath)))
        {
            BackoffFree(&Backoff);
            return ReportResult();
        }

        uint64_t Seen = 0;
        std::vector<uint32_t> Current;
        uint32_t Code = 0;

        while (ScanNext(&Scanner, &Code))
        {
            if (Code == (uint32_t)'\n')
            {
                if (Seen >= TRAIN_LINES)
                {
                    TestTokens += Current.size() + 1;   // EOS 포함
                    TestLines.push_back(Current);
                }
                Current.clear();
                Seen++;

                if (Seen >= TRAIN_LINES + TEST_LINES)
                {
                    break;
                }
                continue;
            }

            if (Seen >= TRAIN_LINES)
            {
                Current.push_back(Code);
            }
        }

        ScanClose(&Scanner);
    }

    printf("  평가 %zu줄, 토큰 %llu개 (훈련에 안 쓴 줄)\n\n",
           TestLines.size(), (unsigned long long)TestTokens);

    CHECK(TestLines.size() > 0);

    // ---- B2-2. 확률의 합이 1 인가 ----
    printf("[B2-2] 먼저 확률인지부터 확인한다\n\n");

    {
        // 1그램 모델에 들어 있는 토큰 전부가 곧 어휘다.
        const FNgram& Unigram = Backoff.Orders[1];
        std::vector<uint32_t> Vocabulary;
        for (uint64_t i = 0; i < Unigram.GramCount; i++)
        {
            Vocabulary.push_back(Unigram.Grams[i]);
        }

        printf("  어휘 = %zu 종\n", Vocabulary.size());

        // 문맥 셋을 골라 확률의 합을 잰다.
        uint32_t Contexts[3][TOP_ORDER - 1] = {
            { TOKEN_BOS, TOKEN_BOS, TOKEN_BOS, TOKEN_BOS, TOKEN_BOS },
            { 0x0020u, 0xB300u, 0xD55Cu, 0xBBFCu, 0xAD6Du },   // " 대한민국"
            { 0xD7A3u, 0xD7A3u, 0xD7A3u, 0xD7A3u, 0xD7A3u },   // 본 적 없는 문맥
        };
        const char* Names[3] = { "BOS 다섯 개", "\" 대한민국\"", "\"힣힣힣힣힣\"" };

        for (int k = 0; k < 3; k++)
        {
            double Sum = 0.0;
            for (uint32_t Token : Vocabulary)
            {
                Sum += BackoffProb(&Backoff, Contexts[k], TOP_ORDER - 1, Token);
            }

            printf("  ");
            PrintPadded(Names[k], 16);
            printf("확률의 합 = %.12f\n", Sum);

            CHECK_NEAR(Sum, 1.0, 1e-9);
        }
    }
    printf("\n");

    // ---- B2-3. 확률이 0 이면 무한대가 된다 ----
    printf("[B2-3] 훈련에서 못 본 글자를 만나면\n\n");

    {
        FScore Raw = ScoreAll(Backoff, TestLines, TOP_ORDER, 0.0);

        printf("  받침 없이 잰 퍼플렉서티 = %.3f\n", Perplexity(Raw));
        printf("  확률이 0 이었던 토큰     = %llu 개 (전체의 %.4f%%)\n",
               (unsigned long long)Raw.ZeroCount,
               (double)Raw.ZeroCount * 100.0 / (double)Raw.TokenCount);

        CHECK(Raw.ZeroCount > 0);
        CHECK(std::isinf(Perplexity(Raw)));

        printf("\n  토큰 %llu개 중 %llu개 때문에 전체가 무한대가 됐다.\n",
               (unsigned long long)Raw.TokenCount,
               (unsigned long long)Raw.ZeroCount);
        printf("  로그를 씌우면 0 은 음의 무한대이고, 무한대가 한 번 더해지면\n");
        printf("  나머지 %llu개가 무슨 값이든 합은 음의 무한대다.\n\n",
               (unsigned long long)(Raw.TokenCount - Raw.ZeroCount));
    }

    // ---- 받침을 대고 다시 잰다 ----
    printf("  받침(floor)을 대고 다시 재면\n\n");
    printf("  ");
    PrintPaddedRight("받침", 14);
    PrintPaddedRight("퍼플렉서티", 16);
    printf("\n");

    const double Floors[4] = { 1e-3, 1e-6, 1e-9, 1e-12 };
    double FloorPpl[4] = {};

    for (int i = 0; i < 4; i++)
    {
        FScore Score = ScoreAll(Backoff, TestLines, TOP_ORDER, Floors[i]);
        FloorPpl[i] = Perplexity(Score);

        char Buffer[32];
        snprintf(Buffer, sizeof(Buffer), "%.0e", Floors[i]);

        printf("  ");
        PrintPaddedRight(Buffer, 14);
        printf("%15.3f\n", FloorPpl[i]);
    }

    printf("\n  받침을 낮출수록 퍼플렉서티가 나빠진다. **고른 값이 결과를 바꾼다.**\n");
    printf("  아래 비교는 전부 1e-6 으로 고정해서 잰다.\n\n");

    CHECK(FloorPpl[0] < FloorPpl[3]);

    const double Floor = 1e-6;

    // ---- B2-4. 차수별 퍼플렉서티 ----
    printf("[B2-4] 몇 글자를 보는 것이 좋은가\n\n");
    printf("  ");
    PrintPaddedRight("차수", 8);
    PrintPaddedRight("퍼플렉서티", 16);
    PrintPaddedRight("확률 0 토큰", 14);
    printf("\n");

    double BestPpl = INFINITY;
    int BestOrder = 0;

    for (int Order = 1; Order <= TOP_ORDER; Order++)
    {
        FScore Score = ScoreAll(Backoff, TestLines, Order, Floor);
        double Ppl = Perplexity(Score);

        if (Ppl < BestPpl)
        {
            BestPpl = Ppl;
            BestOrder = Order;
        }

        printf("  %7d", Order);
        printf("%15.3f ", Ppl);
        printf("%13llu\n", (unsigned long long)Score.ZeroCount);
    }

    printf("\n  가장 좋은 차수 = %d (퍼플렉서티 %.3f)\n\n", BestOrder, BestPpl);

    CHECK(BestOrder > 1);

    // ---- B2-5. A8 이 못 답한 질문 ----
    printf("[B2-5] 할인율은 얼마가 좋은가 (A8 에서 못 정했던 것)\n\n");
    printf("  ");
    PrintPaddedRight("할인율", 18);
    PrintPaddedRight("퍼플렉서티", 16);
    printf("\n");

    // 할인율이 커지는 순서로 늘어놓는다.
    const uint32_t Nums[8]  = { 0, 1, 2, 5, 3, 7, 15, 31 };
    const uint32_t Dens[8]  = { 4, 4, 4, 8, 4, 8, 16, 32 };

    double BestDiscountPpl = INFINITY;
    uint32_t BestNum = 0;
    uint32_t BestDen = 4;

    for (int i = 0; i < 8; i++)
    {
        BackoffSetDiscount(&Backoff, Nums[i], Dens[i]);

        FScore Score = ScoreAll(Backoff, TestLines, TOP_ORDER, Floor);
        double Ppl = Perplexity(Score);

        if (Ppl < BestDiscountPpl)
        {
            BestDiscountPpl = Ppl;
            BestNum = Nums[i];
            BestDen = Dens[i];
        }

        char Buffer[32];
        snprintf(Buffer, sizeof(Buffer), "%u/%u = %.3f", Nums[i], Dens[i],
                 (double)Nums[i] / (double)Dens[i]);

        printf("  ");
        PrintPaddedRight(Buffer, 18);
        printf("%15.3f\n", Ppl);
    }

    printf("\n  가장 좋은 할인율 = %u/%u (퍼플렉서티 %.3f)\n\n",
           BestNum, BestDen, BestDiscountPpl);

    BackoffSetDiscount(&Backoff, BACKOFF_NUM, BACKOFF_DEN);

    CHECK(BestDen > 0);

    // ---- B2-6. 로그 공간에서 더하기 ----
    printf("[B2-6] 로그로 바꾸면 더하기가 어려워진다\n\n");

    {
        // 손으로 확인되는 값부터.
        //   log(1) + log(1) -> log(2)
        CHECK_NEAR(LogAdd(log(1.0), log(1.0)), log(2.0), 1e-12);
        //   log(3) + log(5) -> log(8)
        CHECK_NEAR(LogAdd(log(3.0), log(5.0)), log(8.0), 1e-12);
        //   0 은 무시한다
        CHECK_NEAR(LogAdd(log(7.0), -INFINITY), log(7.0), 1e-12);

        printf("  LogAdd(log 1, log 1) = %.12f, log 2 = %.12f\n",
               LogAdd(log(1.0), log(1.0)), log(2.0));
        printf("  LogAdd(log 3, log 5) = %.12f, log 8 = %.12f\n",
               LogAdd(log(3.0), log(5.0)), log(8.0));

        // 이제 아주 작은 값. 그냥 exp 로 되돌리면 0 이 된다.
        const double LogP = -800.0;
        const double LogQ = -800.0;

        double Naive = log(exp(LogP) + exp(LogQ));
        double Safe = LogAdd(LogP, LogQ);

        printf("\n  log a = log b = -800 일 때 log(a+b) 는 -800 + log 2 = %.6f 여야 한다\n",
               -800.0 + log(2.0));
        printf("    exp 로 되돌려 더하면 = %.6f   <- exp(-800) 이 0 이라 못 구한다\n",
               Naive);
        printf("    LogAdd 로 구하면     = %.6f\n\n", Safe);

        CHECK(std::isinf(Naive));
        CHECK_NEAR(Safe, -800.0 + log(2.0), 1e-9);

        // 배열 버전도 같은 요령이다.
        double Values[3] = { -700.0, -701.0, -702.0 };
        double Expected = -700.0 + log(1.0 + exp(-1.0) + exp(-2.0));

        printf("  LogSumExp([-700, -701, -702]) = %.9f (손계산 %.9f)\n\n",
               LogSumExp(Values, 3), Expected);

        CHECK_NEAR(LogSumExp(Values, 3), Expected, 1e-9);
    }

    BackoffFree(&Backoff);

    return ReportResult();
}
