// ch-D05-quant/Main.cpp
//
// D5. 양자화
//
// 저장소 루트에서 실행할 것.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Model.hpp"
#include "Nn.hpp"
#include "Quant.hpp"
#include "Random.h"
#include "Tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#define BOOK_SEED 20260914ull

#define VOCAB  512
#define MODEL  128
#define HEADS    4
#define HIDDEN 512
#define LAYERS   4
#define MAXLEN 128

namespace
{

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

void FillRandom(FTensor& T, FRandom& Rng, double Range)
{
    for (size_t i = 0; i < T.Count(); i++)
    {
        T.At(i) = (Real)RandomRange(&Rng, Range);
    }
}

void PrintError(const char* Name, const FError& E, size_t Bytes,
                size_t OriginalBytes)
{
    char Buffer[64];

    printf("  ");
    PrintPadded(Name, 22);

    snprintf(Buffer, sizeof(Buffer), "%.0f KB", (double)Bytes / 1024.0);
    PrintPaddedRight(Buffer, 12);

    snprintf(Buffer, sizeof(Buffer), "%.2f 배", (double)OriginalBytes / Bytes);
    PrintPaddedRight(Buffer, 10);

    snprintf(Buffer, sizeof(Buffer), "%.3e", E.MaxAbsolute);
    PrintPaddedRight(Buffer, 14);

    snprintf(Buffer, sizeof(Buffer), "%.4f%%", E.RelativeRms * 100.0);
    PrintPaddedRight(Buffer, 14);

    printf("\n");
}

} // namespace

int main(void)
{
    printf("D5. 양자화\n\n");

    FRandom Rng;
    RandomSeed(&Rng, BOOK_SEED);

    // ---- D5-1. 한 줄을 손으로 ----
    printf("[D5-1] 여덟 개짜리 줄 하나를 8비트로\n\n");

    {
        FTensor X({ 1, 8 });
        const double Values[8] = { 1.0, -0.5, 0.25, 0.0, -1.0, 0.75, 0.1, -0.3 };
        for (size_t i = 0; i < 8; i++)
        {
            X(0, i) = (Real)Values[i];
        }

        FQuant8 Q = Quantize8(X, true);
        FTensor Back = Dequantize8(Q);

        printf("  최대 절대값 = 1.0, scale = 1.0/127 = %.9f\n\n",
               1.0 / 127.0);

        printf("  ");
        PrintPaddedRight("원본", 12);
        PrintPaddedRight("정수", 10);
        PrintPaddedRight("복원", 14);
        PrintPaddedRight("오차", 14);
        printf("\n");

        for (size_t i = 0; i < 8; i++)
        {
            char A[32], B[32], C[32], D[32];
            snprintf(A, sizeof(A), "%+.4f", Values[i]);
            snprintf(B, sizeof(B), "%d", (int)Q.Values[i]);
            snprintf(C, sizeof(C), "%+.9f", (double)Back(0, i));
            snprintf(D, sizeof(D), "%.2e",
                     std::fabs((double)Back(0, i) - Values[i]));

            printf("  ");
            PrintPaddedRight(A, 12);
            PrintPaddedRight(B, 10);
            PrintPaddedRight(C, 14);
            PrintPaddedRight(D, 14);
            printf("\n");
        }

        printf("\n");

        // 손계산: 1.0 / (1/127) = 127, -0.5 -> -63.5 -> -64 (반올림)
        CHECK(Q.Values[0] == 127);
        CHECK(Q.Values[4] == -127);
        CHECK(Q.Values[3] == 0);       // 0 은 정확히 0
        CHECK_NEAR(Back(0, 3), 0.0, 1e-12);

        printf("  **0 은 정확히 0 으로 남는다.** 대칭 양자화를 고른 이유다.\n");
        printf("  0 이 틀어지면 마스킹이나 패딩에서 사고가 난다.\n\n");
    }

    // ---- D5-2. scale 을 어디 단위로 둘 것인가 ----
    printf("[D5-2] scale 을 전체에 하나 vs 줄마다 하나\n\n");

    {
        // 줄마다 크기가 크게 다른 행렬을 만든다. 실제 가중치가 이렇게 생겼다.
        FTensor W({ 64, 128 });
        for (size_t r = 0; r < 64; r++)
        {
            // 줄마다 크기를 1/1000 ~ 1 사이로 흩뿌린다.
            double Scale = std::pow(10.0, -3.0 * (double)r / 63.0);
            for (size_t c = 0; c < 128; c++)
            {
                W(r, c) = (Real)(RandomRange(&Rng, 1.0) * Scale);
            }
        }

        const size_t OriginalBytes = W.Count() * sizeof(Real);

        FQuant8 Whole = Quantize8(W, false);
        FQuant8 PerRow = Quantize8(W, true);
        FQuant4 Four = Quantize4(W);

        FError WholeError = CompareTensors(W, Dequantize8(Whole));
        FError RowError = CompareTensors(W, Dequantize8(PerRow));
        FError FourError = CompareTensors(W, Dequantize4(Four));

        printf("  줄마다 크기가 1000배까지 차이나는 행렬 (%zu x %zu)\n\n",
               W.Size(0), W.Size(1));

        printf("  ");
        PrintPadded("방식", 22);
        PrintPaddedRight("크기", 12);
        PrintPaddedRight("줄었다", 10);
        PrintPaddedRight("최대 오차", 14);
        PrintPaddedRight("상대 RMS", 14);
        printf("\n");

        printf("  ");
        PrintPadded("float (원본)", 22);
        char Buffer[32];
        snprintf(Buffer, sizeof(Buffer), "%.0f KB", (double)OriginalBytes / 1024.0);
        PrintPaddedRight(Buffer, 12);
        PrintPaddedRight("1.00 배", 10);
        PrintPaddedRight("0", 14);
        PrintPaddedRight("0%", 14);
        printf("\n");

        PrintError("8비트, 전체 scale", WholeError, Whole.Bytes(), OriginalBytes);
        PrintError("8비트, 줄마다 scale", RowError, PerRow.Bytes(), OriginalBytes);
        PrintError("4비트, 줄마다 scale", FourError, Four.Bytes(), OriginalBytes);

        printf("\n");

        // 줄마다 두는 쪽이 훨씬 정확해야 한다.
        CHECK(RowError.RelativeRms < WholeError.RelativeRms);
        CHECK(RowError.RelativeRms < 0.01);

        // 4비트는 8비트보다 나빠야 한다.
        CHECK(FourError.RelativeRms > RowError.RelativeRms);

        printf("  **전체에 하나만 두면 작은 줄이 통째로 0 이 된다.**\n");
        printf("  가장 큰 값에 127 을 맞추므로, 1000배 작은 줄은\n");
        printf("  전부 0.127 미만이 되어 반올림하면 0 이다.\n\n");

        // 실제로 0 이 된 줄이 있는지 세어본다.
        int DeadRows = 0;
        for (size_t r = 0; r < 64; r++)
        {
            int bAllZero = 1;
            for (size_t c = 0; c < 128; c++)
            {
                if (Whole.Values[r * 128 + c] != 0)
                {
                    bAllZero = 0;
                    break;
                }
            }
            if (bAllZero) DeadRows++;
        }

        printf("  전체 scale 에서 통째로 0 이 된 줄 = %d개 / 64\n\n", DeadRows);
        CHECK(DeadRows > 0);
    }

    // ---- D5-3. 4비트 패킹 ----
    printf("[D5-3] 한 바이트에 둘 담기\n\n");

    {
        FTensor X({ 1, 4 });
        X(0, 0) = Real(1);    X(0, 1) = Real(-1);
        X(0, 2) = Real(0.5);  X(0, 3) = Real(0);

        FQuant4 Q = Quantize4(X);

        printf("  원본 = (1, -1, 0.5, 0), scale = 1/7 = %.6f\n", 1.0 / 7.0);
        printf("  정수 = (7, -7, 4, 0)  ->  0 을 8 로 옮겨 (15, 1, 12, 8)\n\n");

        printf("  담긴 바이트\n");
        for (size_t i = 0; i < Q.Packed.size(); i++)
        {
            printf("    [%zu] = 0x%02X = %d  (아래 4비트 %d, 위 4비트 %d)\n",
                   i, Q.Packed[i], Q.Packed[i],
                   Q.Packed[i] & 0x0F, Q.Packed[i] >> 4);
        }

        FTensor Back = Dequantize4(Q);

        printf("\n  복원 = (%.6f, %.6f, %.6f, %.6f)\n\n",
               (double)Back(0, 0), (double)Back(0, 1),
               (double)Back(0, 2), (double)Back(0, 3));

        // 0x0F = 15 = 7 + 8, 0x01 = 1 = -7 + 8
        CHECK(Q.Packed[0] == (uint8_t)(15 | (1 << 4)));
        CHECK(Q.Packed[1] == (uint8_t)(12 | (8 << 4)));

        CHECK_NEAR(Back(0, 0), 1.0, 1e-6);
        CHECK_NEAR(Back(0, 1), -1.0, 1e-6);
        CHECK_NEAR(Back(0, 3), 0.0, 1e-12);

        // 0.5 는 7 * 0.5 = 3.5 -> 4 로 반올림되어 4/7 = 0.5714 가 된다.
        printf("  0.5 가 %.6f 로 돌아온다. 4비트로는 이게 최선이다.\n",
               (double)Back(0, 2));
        printf("  (0.5 x 7 = 3.5 -> 반올림 4 -> 4/7)\n\n");

        CHECK_NEAR(Back(0, 2), 4.0 / 7.0, 1e-5);

        printf("  비트 연산은 A1 이후 처음이다.\n");
        printf("    담기  Packed[b] = (Packed[b] & 0xF0) | (Nibble & 0x0F)\n");
        printf("    꺼내기 (Flat %% 2 == 0) ? (Packed[b] & 0x0F) : (Packed[b] >> 4)\n\n");
    }

    // ---- D5-4. 모델 전체를 양자화하면 ----
    printf("[D5-4] 트랜스포머의 출력이 얼마나 달라지는가\n\n");

    FModelConfig Config;
    Config.Vocab = VOCAB;
    Config.Model = MODEL;
    Config.Heads = HEADS;
    Config.Hidden = HIDDEN;
    Config.Layers = LAYERS;
    Config.MaxLength = MAXLEN;

    {
        FTransformer Model(Config);
        FRandom Base;
        RandomSeed(&Base, BOOK_SEED);
        Model.Init(Base);

        std::vector<uint32_t> Tokens(32, 0);
        for (size_t i = 0; i < Tokens.size(); i++)
        {
            Tokens[i] = (uint32_t)(i * 7 % VOCAB);
        }

        FTensor Reference = Model.Forward(Tokens.data(), 1, Tokens.size());

        // 큰 가중치만 양자화한다. 정규화 이득과 임베딩은 그대로 둔다.
        auto QuantizeInPlace = [](FTensor& W, int Bits)
        {
            if (Bits == 8)
            {
                W = Dequantize8(Quantize8(W, true));
            }
            else
            {
                W = Dequantize4(Quantize4(W));
            }
        };

        printf("  ");
        PrintPadded("무엇을", 26);
        PrintPaddedRight("최대 오차", 14);
        PrintPaddedRight("상대 RMS", 14);
        printf("\n");

        struct FCase { const char* Name; int Bits; int What; };

        // What: 0 = 블록 가중치만, 1 = 출력층까지, 2 = 임베딩까지
        FCase Cases[6] = {
            { "8비트, 블록만",       8, 0 },
            { "8비트, +출력층",      8, 1 },
            { "8비트, 전부",         8, 2 },
            { "4비트, 블록만",       4, 0 },
            { "4비트, +출력층",      4, 1 },
            { "4비트, 전부",         4, 2 },
        };

        double Best8 = 0.0;
        double Worst4 = 0.0;

        for (int c = 0; c < 6; c++)
        {
            FTransformer Copy(Config);
            FRandom Same;
            RandomSeed(&Same, BOOK_SEED);
            Copy.Init(Same);

            for (size_t L = 0; L < LAYERS; L++)
            {
                FBlock& B = Copy.Blocks[L];
                QuantizeInPlace(B.Query.Weight, Cases[c].Bits);
                QuantizeInPlace(B.Key.Weight, Cases[c].Bits);
                QuantizeInPlace(B.Value.Weight, Cases[c].Bits);
                QuantizeInPlace(B.Project.Weight, Cases[c].Bits);
                QuantizeInPlace(B.Up.Weight, Cases[c].Bits);
                QuantizeInPlace(B.Down.Weight, Cases[c].Bits);
            }

            if (Cases[c].What >= 1)
            {
                QuantizeInPlace(Copy.Head.Weight, Cases[c].Bits);
            }
            if (Cases[c].What >= 2)
            {
                QuantizeInPlace(Copy.TokenEmbedding, Cases[c].Bits);
                QuantizeInPlace(Copy.PositionEmbedding, Cases[c].Bits);
            }

            FTensor Logits = Copy.Forward(Tokens.data(), 1, Tokens.size());
            FError E = CompareTensors(Reference, Logits);

            if (c == 2) Best8 = E.RelativeRms;
            if (c == 5) Worst4 = E.RelativeRms;

            char A[32], B2[32];
            snprintf(A, sizeof(A), "%.4f", E.MaxAbsolute);
            snprintf(B2, sizeof(B2), "%.3f%%", E.RelativeRms * 100.0);

            printf("  ");
            PrintPadded(Cases[c].Name, 26);
            PrintPaddedRight(A, 14);
            PrintPaddedRight(B2, 14);
            printf("\n");
        }

        printf("\n  8비트로 전부 바꿔도 점수가 %.3f%% 밖에 안 틀어진다.\n",
               Best8 * 100.0);
        printf("  4비트는 %.3f%%. 쓸 만한지는 무엇에 쓰느냐에 달렸다.\n\n",
               Worst4 * 100.0);

        CHECK(Best8 < Worst4);
        CHECK(Best8 < 0.05);

        // 크기를 재본다.
        size_t Float = Model.ParameterCount() * sizeof(Real);
        printf("  가중치 크기  float %.1f MB -> 8비트 %.1f MB -> 4비트 %.1f MB\n\n",
               (double)Float / (1024.0 * 1024.0),
               (double)Float / 4.0 / (1024.0 * 1024.0),
               (double)Float / 8.0 / (1024.0 * 1024.0));
    }

    // ---- D5-5. 순위가 바뀌는가 ----
    printf("[D5-5] 점수가 아니라 **순위**가 중요하다\n\n");

    {
        FTransformer Model(Config);
        FRandom Same;
        RandomSeed(&Same, BOOK_SEED);
        Model.Init(Same);

        std::vector<uint32_t> Tokens(32, 0);
        for (size_t i = 0; i < Tokens.size(); i++)
        {
            Tokens[i] = (uint32_t)(i * 7 % VOCAB);
        }

        FTensor Reference = Model.Forward(Tokens.data(), 1, Tokens.size());

        FTransformer Quantized(Config);
        FRandom Again;
        RandomSeed(&Again, BOOK_SEED);
        Quantized.Init(Again);

        for (size_t L = 0; L < LAYERS; L++)
        {
            FBlock& B = Quantized.Blocks[L];
            B.Query.Weight = Dequantize8(Quantize8(B.Query.Weight, true));
            B.Key.Weight = Dequantize8(Quantize8(B.Key.Weight, true));
            B.Value.Weight = Dequantize8(Quantize8(B.Value.Weight, true));
            B.Project.Weight = Dequantize8(Quantize8(B.Project.Weight, true));
            B.Up.Weight = Dequantize8(Quantize8(B.Up.Weight, true));
            B.Down.Weight = Dequantize8(Quantize8(B.Down.Weight, true));
        }
        Quantized.Head.Weight = Dequantize8(Quantize8(Quantized.Head.Weight, true));

        FTensor Logits = Quantized.Forward(Tokens.data(), 1, Tokens.size());

        int SameTop1 = 0;
        int SameTop5 = 0;

        for (size_t t = 0; t < Tokens.size(); t++)
        {
            // 1위를 찾는다.
            size_t BestA = 0, BestB = 0;
            for (size_t v = 1; v < VOCAB; v++)
            {
                if (Reference(0, t, v) > Reference(0, t, BestA)) BestA = v;
                if (Logits(0, t, v) > Logits(0, t, BestB)) BestB = v;
            }

            if (BestA == BestB) SameTop1++;

            // 원본의 1위가 양자화본의 상위 5위 안에 있는가.
            int Rank = 0;
            for (size_t v = 0; v < VOCAB; v++)
            {
                if (Logits(0, t, v) > Logits(0, t, BestA)) Rank++;
            }
            if (Rank < 5) SameTop5++;
        }

        FError E = CompareTensors(Reference, Logits);

        printf("  위치 %zu개에서\n", Tokens.size());
        printf("    1위가 그대로인 자리 = %d개\n", SameTop1);
        printf("    원본 1위가 상위 5위 안에 남은 자리 = %d개\n\n", SameTop5);

        CHECK(SameTop1 > (int)Tokens.size() / 2);
        CHECK(SameTop5 == (int)Tokens.size());

        printf("  **점수가 %.3f%% 틀어져도 순위는 하나도 안 바뀐다.**\n",
               E.RelativeRms * 100.0);
        printf("  생성에서 중요한 것은 절대 점수가 아니라 순서다.\n");
        printf("  양자화가 실용적인 이유가 이것이다.\n\n");
    }

    // ---- D5-6. 속도는 어떤가 ----
    printf("[D5-6] 정수로 바꾸면 빨라지는가\n\n");

    {
        FTensor W({ 512, 512 });
        FillRandom(W, Rng, 1.0);

        FTensor X({ 64, 512 });
        FillRandom(X, Rng, 1.0);

        FQuant8 Q = Quantize8(W, true);

        const int Rounds = 50;

        auto Begin = std::chrono::steady_clock::now();
        for (int i = 0; i < Rounds; i++)
        {
            FTensor Y = MatMul(X, W);
            if (Y.Count() == 0) printf("never\n");
        }
        double FloatSeconds = Seconds(Begin);

        Begin = std::chrono::steady_clock::now();
        for (int i = 0; i < Rounds; i++)
        {
            FTensor Y = QuantizedForward8(Q, X);
            if (Y.Count() == 0) printf("never\n");
        }
        double QuantSeconds = Seconds(Begin);

        Begin = std::chrono::steady_clock::now();
        for (int i = 0; i < Rounds; i++)
        {
            FTensor Y = QuantizedForward8Dot(Q, X);
            if (Y.Count() == 0) printf("never\n");
        }
        double DotSeconds = Seconds(Begin);

        printf("  (64 x 512) x (512 x 512), %d번\n\n", Rounds);

        printf("  ");
        PrintPadded("어떻게", 28);
        PrintPaddedRight("시간", 12);
        PrintPaddedRight("float 대비", 14);
        printf("\n");

        struct FRow { const char* Name; double Time; };
        FRow Rows[3] = {
            { "float, i-k-j",        FloatSeconds },
            { "8비트, i-k-j",        QuantSeconds },
            { "8비트, 내적 순서",    DotSeconds   },
        };

        for (int i = 0; i < 3; i++)
        {
            char A[32], B[32];
            snprintf(A, sizeof(A), "%.3f 초", Rows[i].Time);
            snprintf(B, sizeof(B), "%.2f 배", FloatSeconds / Rows[i].Time);

            printf("  ");
            PrintPadded(Rows[i].Name, 28);
            PrintPaddedRight(A, 12);
            PrintPaddedRight(B, 14);
            printf("\n");
        }

        printf("\n");

        // 두 정수 판은 결과가 같아야 한다. 순서만 다르다.
        FTensor A1 = QuantizedForward8(Q, X);
        FTensor A2 = QuantizedForward8Dot(Q, X);
        FError Same = CompareTensors(A1, A2);

        printf("  두 정수 판의 차이 = %.3e (합치는 순서만 다르다)\n\n",
               Same.RelativeRms);

        CHECK(Same.RelativeRms < 1e-5);

        // 순서를 지키면 내적 순서보다 훨씬 빨라야 한다.
        CHECK(QuantSeconds < DotSeconds);

        printf("  **같은 정수인데 고리 순서만 바꿔서 %.1f배다.**\n",
               DotSeconds / QuantSeconds);
        printf("  C5 에서 본 그 문제가 그대로 다시 나왔다.\n\n");

        printf("  그리고 순서를 지켜도 float 보다 빠르지 않다.\n");
        printf("  안쪽 고리가 int8 을 float 로 바꾸는 일을 매번 더 하기\n");
        printf("  때문이다. 곱셈은 어차피 float 로 한다.\n\n");

        printf("  양자화의 이득은 여기가 아니다.\n");
        printf("    1. 메모리 — 크기가 4분의 1. 이건 확실하다\n");
        printf("    2. 대역폭 — 읽어들일 바이트가 4분의 1\n");
        printf("    3. 전용 명령 — VNNI 같은 int8 내적 명령이 있어야\n");
        printf("       정수 연산 자체가 빨라진다. 우리는 아직 안 쓴다\n\n");

        printf("  C5 의 교훈이 그대로다. **재보지 않으면 모른다.**\n\n");
    }

    return ReportResult();
}
