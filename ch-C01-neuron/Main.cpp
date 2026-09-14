// ch-C01-neuron/Main.cpp
//
// C1. 뉴런과 순전파
//
// 저장소 루트에서 실행할 것.
//     Main.exe [코퍼스경로]

#include "Test.h"
#include "Pretty.h"

#include "Matrix.hpp"
#include "Nn.hpp"
#include "Ngram.h"
#include "Random.h"
#include "Utf8.h"
#include "Vector.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define DEFAULT_CORPUS "data/corpus.txt"
#define TRAIN_LINES 200000
#define BOOK_SEED 20260914ull

// 무작위 망을 시험해볼 바이그램 개수
#define SAMPLE_PAIRS 2000

// 숨은 층의 크기
#define HIDDEN_SIZE 32

namespace
{

void PrintVector(const char* Name, const FVector& V, int Limit = 8)
{
    printf("  %s = (", Name);
    for (size_t i = 0; i < V.Size() && (int)i < Limit; i++)
    {
        printf("%s%.4f", (i == 0) ? "" : ", ", (double)V[i]);
    }
    if ((int)V.Size() > Limit)
    {
        printf(", ...");
    }
    printf(")\n");
}

} // namespace

int main(int argc, char** argv)
{
    const char* CorpusPath = (argc > 1) ? argv[1] : DEFAULT_CORPUS;

    printf("C1. 뉴런과 순전파\n\n");

    // ---- C1-1. 뉴런 하나 ----
    printf("[C1-1] 뉴런 하나가 하는 일\n\n");

    {
        // 입력 세 개, 뉴런 하나.
        FVector X(3);
        X[0] = Real(1); X[1] = Real(2); X[2] = Real(3);

        FVector W(3);
        W[0] = Real(0.5); W[1] = Real(-1); W[2] = Real(2);

        Real Bias = Real(0.25);

        // W·x + b = 0.5*1 + (-1)*2 + 2*3 + 0.25 = 0.5 - 2 + 6 + 0.25 = 4.75
        Real Sum = (W * X) + Bias;

        printf("  x = (1, 2, 3), w = (0.5, -1, 2), b = 0.25\n");
        printf("    w·x + b   = %.6f   (손계산 4.75)\n", (double)Sum);
        printf("    tanh(...)  = %.6f   (손계산 %.6f)\n",
               (double)Tanh(Sum), std::tanh(4.75));
        printf("    relu(...)  = %.6f   (손계산 4.75)\n\n", (double)Relu(Sum));

        CHECK_NEAR(Sum, 4.75, 1e-6);
        CHECK_NEAR(Tanh(Sum), std::tanh(4.75), 1e-6);
        CHECK_NEAR(Relu(Sum), 4.75, 1e-6);

        // 음수면 relu 가 0 이다.
        CHECK_NEAR(Relu(Real(-3)), 0.0, 1e-9);
    }

    // ---- C1-2. 뉴런을 여러 개 늘어놓으면 행렬이 된다 ----
    printf("[C1-2] 층 하나 = 행렬 한 번 곱하기\n\n");

    {
        FLinear Layer(3, 2);

        // 첫 줄이 뉴런 1, 둘째 줄이 뉴런 2 다.
        Layer.Weight(0, 0) = Real(0.5);  Layer.Weight(0, 1) = Real(-1);
        Layer.Weight(0, 2) = Real(2);
        Layer.Weight(1, 0) = Real(1);    Layer.Weight(1, 1) = Real(1);
        Layer.Weight(1, 2) = Real(1);

        Layer.Bias[0] = Real(0.25);
        Layer.Bias[1] = Real(-6);

        FVector X(3);
        X[0] = Real(1); X[1] = Real(2); X[2] = Real(3);

        FVector Y = Layer.Forward(X);

        printf("  W = [ 0.5  -1   2 ]   b = ( 0.25 )\n");
        printf("      [ 1     1   1 ]       ( -6   )\n");
        PrintVector("Wx + b", Y);
        printf("    손계산 (4.75, 0)\n\n");

        CHECK(Y.Size() == 2);
        CHECK_NEAR(Y[0], 4.75, 1e-6);
        CHECK_NEAR(Y[1], 0.0, 1e-6);

        // 행렬 배치가 맞는지. W 는 (출력 × 입력) 이다.
        CHECK(Layer.Weight.Rows() == 2);
        CHECK(Layer.Weight.Cols() == 3);
    }

    // ---- C1-3. 비선형이 없으면 층을 쌓아도 소용없다 ----
    printf("[C1-3] 활성화 함수를 빼면 어떻게 되는가\n\n");

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        FLinear First(4, 5);
        FLinear Second(5, 3);
        First.InitUniform(Rng, Real(1));
        Second.InitUniform(Rng, Real(1));

        // 치우침도 넣어 일반적인 경우로 만든다.
        for (size_t i = 0; i < First.Bias.Size(); i++)
        {
            First.Bias[i] = (Real)RandomRange(&Rng, 1.0);
        }
        for (size_t i = 0; i < Second.Bias.Size(); i++)
        {
            Second.Bias[i] = (Real)RandomRange(&Rng, 1.0);
        }

        // 두 층을 그냥 이어 붙인 것과 같은 일을 하는 **한 층**을 만든다.
        //   W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
        FMatrix Combined(3, 4);
        for (size_t Row = 0; Row < 3; Row++)
        {
            for (size_t Col = 0; Col < 4; Col++)
            {
                double Sum = 0.0;
                for (size_t k = 0; k < 5; k++)
                {
                    Sum += (double)Second.Weight(Row, k)
                         * (double)First.Weight(k, Col);
                }
                Combined(Row, Col) = (Real)Sum;
            }
        }

        FVector CombinedBias = Second.Weight * First.Bias;
        CombinedBias += Second.Bias;

        // 아무 입력이나 넣어 비교한다.
        Real WorstGap = Real(0);
        for (int Trial = 0; Trial < 100; Trial++)
        {
            FVector X(4);
            for (size_t i = 0; i < X.Size(); i++)
            {
                X[i] = (Real)RandomRange(&Rng, 3.0);
            }

            FVector TwoLayers = Second.Forward(First.Forward(X));

            FVector OneLayer = Combined * X;
            OneLayer += CombinedBias;

            for (size_t i = 0; i < 3; i++)
            {
                Real Gap = (Real)std::fabs((double)TwoLayers[i]
                                         - (double)OneLayer[i]);
                if (Gap > WorstGap)
                {
                    WorstGap = Gap;
                }
            }
        }

        printf("  선형 층 두 개를 이어 붙인 것과, 그걸 하나로 합친 층을\n");
        printf("  100가지 입력으로 비교했다. 최대 차이 = %.3e\n\n",
               (double)WorstGap);

        CHECK((double)WorstGap < 1e-4);

        printf("  **둘이 같다.** 층을 아무리 쌓아도 선형이면 층 하나와 같다.\n");
        printf("  비선형 함수가 층 사이에 있어야 쌓는 의미가 생긴다.\n\n");

        // 활성화를 넣으면 달라진다.
        FVector X(4);
        for (size_t i = 0; i < X.Size(); i++)
        {
            X[i] = Real(1);
        }

        FVector WithTanh = Second.Forward(Tanh(First.Forward(X)));
        FVector WithoutTanh = Second.Forward(First.Forward(X));

        Real Different = (Real)std::fabs((double)WithTanh[0]
                                       - (double)WithoutTanh[0]);
        printf("  tanh 를 하나 끼우면 같은 입력에서 결과가 %.4f 만큼 달라진다\n\n",
               (double)Different);

        CHECK((double)Different > 1e-3);
    }

    // ---- C1-4. 활성화 함수 두 개 ----
    printf("[C1-4] tanh 와 relu\n\n");
    printf("  ");
    PrintPaddedRight("x", 10);
    PrintPaddedRight("tanh(x)", 12);
    PrintPaddedRight("relu(x)", 12);
    printf("\n");

    {
        const Real Xs[7] = { Real(-3), Real(-1), Real(-0.5), Real(0),
                             Real(0.5), Real(1), Real(3) };

        for (int i = 0; i < 7; i++)
        {
            printf("  %9.2f %11.6f %11.6f\n", (double)Xs[i],
                   (double)Tanh(Xs[i]), (double)Relu(Xs[i]));
        }

        printf("\n  tanh 는 -1 과 1 사이로 눌러 담고, relu 는 음수만 0 으로 만든다.\n\n");

        CHECK_NEAR(Tanh(Real(0)), 0.0, 1e-9);
        CHECK((double)Tanh(Real(100)) < 1.0 + 1e-9);
        CHECK((double)Tanh(Real(100)) > 1.0 - 1e-9);
        CHECK_NEAR(Relu(Real(0)), 0.0, 1e-9);
    }

    // ---- C1-5. 점수를 확률로 ----
    printf("[C1-5] softmax\n\n");

    {
        FVector Scores(3);
        Scores[0] = Real(1); Scores[1] = Real(2); Scores[2] = Real(3);

        FVector Probs = Softmax(Scores);

        // 손계산: e^1 : e^2 : e^3 = 2.71828 : 7.38906 : 20.0855
        // 합 30.1928 -> 0.09003, 0.24473, 0.66524
        PrintVector("softmax(1, 2, 3)", Probs);
        printf("    손계산 (0.090031, 0.244728, 0.665241)\n");

        double Sum = 0.0;
        for (size_t i = 0; i < Probs.Size(); i++)
        {
            Sum += (double)Probs[i];
        }
        printf("    합 = %.9f\n\n", Sum);

        CHECK_NEAR(Probs[0], 0.090030573, 1e-6);
        CHECK_NEAR(Probs[1], 0.244728471, 1e-6);
        CHECK_NEAR(Probs[2], 0.665240956, 1e-6);
        CHECK_NEAR(Sum, 1.0, 1e-6);

        // 전부 같은 점수면 균등 분포다.
        FVector Flat(4);
        FVector FlatProbs = Softmax(Flat);
        for (size_t i = 0; i < 4; i++)
        {
            CHECK_NEAR(FlatProbs[i], 0.25, 1e-9);
        }

        // 점수 전체에 같은 값을 더해도 결과가 같아야 한다.
        FVector Shifted(3);
        Shifted[0] = Real(101); Shifted[1] = Real(102); Shifted[2] = Real(103);
        FVector ShiftedProbs = Softmax(Shifted);

        printf("  점수에 100 을 더해도 결과는 같다\n");
        PrintVector("softmax(101, 102, 103)", ShiftedProbs);

        for (size_t i = 0; i < 3; i++)
        {
            CHECK_NEAR(ShiftedProbs[i], (double)Probs[i], 1e-6);
        }

        // 최댓값을 안 빼면 어떻게 되는지 직접 보인다.
        {
            FVector Big(3);
            Big[0] = Real(800); Big[1] = Real(801); Big[2] = Real(802);

            double NaiveSum = 0.0;
            for (size_t i = 0; i < 3; i++)
            {
                NaiveSum += std::exp((double)Big[i]);
            }

            FVector SafeProbs = Softmax(Big);

            printf("\n  점수가 (800, 801, 802) 일 때\n");
            printf("    그냥 exp 를 더하면 = %.3e  <- 넘쳤다\n", NaiveSum);
            printf("    최댓값을 빼고 하면 = (%.6f, %.6f, %.6f)\n\n",
                   (double)SafeProbs[0], (double)SafeProbs[1],
                   (double)SafeProbs[2]);

            CHECK(std::isinf(NaiveSum));
            CHECK_NEAR(SafeProbs[2], (double)Probs[2], 1e-6);
        }
    }

    // ---- C1-6. 2층 망 전체 ----
    printf("[C1-6] 손으로 따라갈 수 있는 크기의 망\n\n");

    {
        // 입력 2, 숨은 층 2, 출력 2.
        FLinear Hidden(2, 2);
        Hidden.Weight(0, 0) = Real(1);  Hidden.Weight(0, 1) = Real(-1);
        Hidden.Weight(1, 0) = Real(0);  Hidden.Weight(1, 1) = Real(2);
        Hidden.Bias[0] = Real(0);
        Hidden.Bias[1] = Real(1);

        FLinear Output(2, 2);
        Output.Weight(0, 0) = Real(1);  Output.Weight(0, 1) = Real(1);
        Output.Weight(1, 0) = Real(-1); Output.Weight(1, 1) = Real(1);
        Output.Bias[0] = Real(0);
        Output.Bias[1] = Real(0);

        FVector X(2);
        X[0] = Real(1); X[1] = Real(0);

        FVector H = Hidden.Forward(X);          // (1, 1)
        FVector A = Relu(H);                    // (1, 1)
        FVector S = Output.Forward(A);          // (2, 0)
        FVector P = Softmax(S);

        printf("  x = (1, 0)\n");
        PrintVector("Wx + b  ", H);
        printf("    손계산 (1, 1)\n");
        PrintVector("relu    ", A);
        PrintVector("출력 점수", S);
        printf("    손계산 (2, 0)\n");
        PrintVector("softmax ", P);
        printf("    손계산 (0.880797, 0.119203)\n");

        Real Loss = CrossEntropy(P, 0);
        printf("    정답이 0 번일 때 손실 = %.6f  (손계산 %.6f)\n\n",
               (double)Loss, -std::log(0.8807970779778823));

        CHECK_NEAR(H[0], 1.0, 1e-6);
        CHECK_NEAR(H[1], 1.0, 1e-6);
        CHECK_NEAR(S[0], 2.0, 1e-6);
        CHECK_NEAR(S[1], 0.0, 1e-6);
        CHECK_NEAR(P[0], 0.880797077, 1e-6);
        CHECK_NEAR(P[1], 0.119202922, 1e-6);
        CHECK_NEAR(Loss, 0.126928011, 1e-5);
    }

    // ---- C1-7. 아직 아무것도 안 배운 망 ----
    printf("[C1-7] 가중치를 아무렇게나 채운 망의 실력\n\n");

    FNgram Unigram;
    if (!CHECK(NgramBuild(&Unigram, CorpusPath, 1, TRAIN_LINES)))
    {
        printf("코퍼스를 못 읽었다. tools/download_corpus.py 를 먼저 돌릴 것.\n");
        return ReportResult();
    }

    {
        const size_t Vocab = (size_t)Unigram.GramCount;

        std::vector<int> IndexOf((size_t)VOCAB_LIMIT, -1);
        for (size_t i = 0; i < Vocab; i++)
        {
            IndexOf[Unigram.Grams[i]] = (int)i;
        }

        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        FLinear Hidden(Vocab, HIDDEN_SIZE);
        FLinear Output(HIDDEN_SIZE, Vocab);
        Hidden.InitScaled(Rng);
        Output.InitScaled(Rng);

        printf("  입력 %zu -> 숨은 층 %d -> 출력 %zu\n", Vocab, HIDDEN_SIZE, Vocab);
        printf("  가중치 개수 = %zu 개\n",
               Hidden.Weight.Count() + Output.Weight.Count());

        // 실제 바이그램 몇 개로 손실을 잰다.
        FNgram Bigram;
        if (!CHECK(NgramBuild(&Bigram, CorpusPath, 2, 2000)))
        {
            NgramFree(&Unigram);
            return ReportResult();
        }

        double LossSum = 0.0;
        int Counted = 0;

        for (uint64_t i = 0; i < Bigram.GramCount && Counted < SAMPLE_PAIRS; i++)
        {
            uint32_t Prev = Bigram.Grams[i * 2];
            uint32_t Next = Bigram.Grams[i * 2 + 1];

            if (Prev >= VOCAB_LIMIT || Next >= VOCAB_LIMIT) continue;
            if (IndexOf[Prev] < 0 || IndexOf[Next] < 0) continue;

            // 앞 글자를 원핫 벡터로. 한 칸만 1 이고 나머지는 0 이다.
            FVector X(Vocab);
            X[(size_t)IndexOf[Prev]] = Real(1);

            FVector P = Softmax(Output.Forward(Tanh(Hidden.Forward(X))));

            LossSum += (double)CrossEntropy(P, (size_t)IndexOf[Next]);
            Counted++;
        }

        double MeanLoss = LossSum / (double)Counted;
        double Ppl = std::exp(MeanLoss);

        printf("  바이그램 %d개로 잰 평균 손실 = %.4f\n", Counted, MeanLoss);
        printf("  퍼플렉서티 = %.1f\n", Ppl);
        printf("  어휘 크기  = %zu\n\n", Vocab);

        printf("  아무것도 안 배운 망은 **어휘 크기만큼 헷갈린다.**\n");
        printf("  훈련을 시작하기 전에 이 숫자부터 확인한다. 어휘 크기에서\n");
        printf("  크게 벗어나 있으면 초기화나 순전파가 이미 틀린 것이다.\n\n");

        // 무작위 망의 퍼플렉서티는 어휘 크기 근처여야 한다.
        CHECK(Ppl > (double)Vocab * 0.5);
        CHECK(Ppl < (double)Vocab * 2.0);

        // 참고: B2 에서 잰 4그램 백오프가 21.1 이었다.
        printf("  참고로 B2 의 4그램 백오프는 21.1 이었다.\n");
        printf("  %.0f 에서 21 까지 내려가는 것이 C파트의 목표다.\n\n", Ppl);

        NgramFree(&Bigram);
    }

    NgramFree(&Unigram);

    return ReportResult();
}
