// ch-E01-chat/Main.cpp
//
// E1. 채팅 템플릿과 손실 마스킹
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// /openmp 를 켜면 빠르다. 숫자는 같다.

#include "Test.h"
#include "Pretty.h"

#include "Backward.hpp"
#include "Chat.hpp"
#include "Model.hpp"
#include "Optimizer.hpp"
#include "Random.h"
#include "Tensor.hpp"
#include "Tokenizer.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define BOOK_SEED 20260914ull

#define LENGTH 63          // 입력 길이. 렌더 결과는 64 이하여야 한다
#define MAXLEN 64

#define MODEL   96
#define HEADS    4
#define HIDDEN 384
#define LAYERS   2

#ifndef STEPS
#define STEPS 1200
#endif

#define BATCH 8

namespace
{

const char* GSystem = "너는 한글 엔진이다.";

struct FPair
{
    const char* User;
    const char* Assistant;
};

const FPair GPairs[16] =
{
    { "안녕",             "안녕하세요. 무엇을 도와드릴까요?" },
    { "안녕하세요",       "안녕하세요. 무엇을 도와드릴까요?" },
    { "이름이 뭐야",      "저는 한글 엔진입니다." },
    { "너는 누구야",      "저는 한글 엔진입니다." },
    { "무엇을 할 수 있어","저는 글자를 이어 쓸 수 있습니다." },
    { "뭐 할 수 있어",    "저는 글자를 이어 쓸 수 있습니다." },
    { "몇 살이야",        "저는 나이가 없습니다." },
    { "어디 살아",        "저는 이 컴퓨터 안에 있습니다." },
    { "날씨 어때",        "저는 날씨를 알지 못합니다." },
    { "지금 몇 시야",     "저는 시간을 알지 못합니다." },
    { "고마워",           "도움이 되어 기쁩니다." },
    { "고맙습니다",       "도움이 되어 기쁩니다." },
    { "잘 가",            "안녕히 가세요." },
    { "안녕히 계세요",    "안녕히 가세요." },
    { "한글이 뭐야",      "한글은 우리 글자입니다." },
    { "엔진이 뭐야",      "엔진은 프로그램의 속입니다." },
};

double Seconds(const std::chrono::steady_clock::time_point& Begin)
{
    std::chrono::duration<double> Elapsed =
        std::chrono::steady_clock::now() - Begin;
    return Elapsed.count();
}

// 한 대화를 훈련용 한 줄로 만든다.
struct FSample
{
    std::vector<uint32_t> Inputs;    // LENGTH 개
    std::vector<uint32_t> Targets;   // LENGTH 개
    std::vector<uint8_t> Mask;       // LENGTH 개. 1 이면 채점
    std::vector<uint8_t> UserMask;   // 사용자 글자만 1. 재보기용
    size_t Used = 0;                 // 패딩 전의 길이
};

FSample MakeSample(const FPair& Pair, const FTokenizer& Tokenizer,
                   uint32_t PadToken)
{
    std::vector<FChatTurn> Turns;

    FChatTurn One;
    One.Role = RoleSystem;
    One.Text = GSystem;
    Turns.push_back(One);

    One.Role = RoleUser;
    One.Text = Pair.User;
    Turns.push_back(One);

    One.Role = RoleAssistant;
    One.Text = Pair.Assistant;
    Turns.push_back(One);

    FChatRendered Rendered = RenderChat(Turns, Tokenizer, false);

    // 사용자 글자 자리를 따로 표시해둔다. 나중에 "사용자 말투를 배웠는가"를
    // 재는 데 쓴다.
    std::vector<uint8_t> UserFlags(Rendered.Tokens.size(), 0);
    {
        int Where = -1;
        for (size_t i = 0; i < Rendered.Tokens.size(); i++)
        {
            const size_t Id = (size_t)Rendered.Tokens[i];
            if (Id >= Tokenizer.FirstSpecial())
            {
                const size_t Which = Id - Tokenizer.FirstSpecial();
                Where = (Which == ChatEnd) ? -1 : (int)Which;
                continue;
            }

            if (Where == (int)ChatUser) UserFlags[i] = 1;
        }
    }

    FSample Result;
    Result.Used = Rendered.Tokens.size();

    for (size_t t = 0; t < LENGTH; t++)
    {
        const bool bInside = (t + 1 < Rendered.Tokens.size());

        Result.Inputs.push_back(bInside ? Rendered.Tokens[t] : PadToken);
        Result.Targets.push_back(bInside ? Rendered.Tokens[t + 1] : PadToken);
        Result.Mask.push_back(bInside ? Rendered.Loss[t + 1] : (uint8_t)0);
        Result.UserMask.push_back(bInside ? UserFlags[t + 1] : (uint8_t)0);
    }

    return Result;
}

} // namespace

int main(void)
{
    printf("E1. 채팅 템플릿과 손실 마스킹\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ---- 어휘 ----
    std::string Everything = GSystem;
    for (int i = 0; i < 16; i++)
    {
        Everything += GPairs[i].User;
        Everything += GPairs[i].Assistant;
    }

    FTokenizer Tokenizer;
    Tokenizer.BuildFromText(Everything.c_str(), Everything.size(), 200,
                            ChatTokenCount);

    // ================================================================
    // E1-1. 틀
    // ================================================================
    printf("[E1-1] 대화를 토큰 열로 펴기\n\n");

    printf("  글자 %zu개 + 특수 토큰 %d개 = 어휘 %zu\n\n",
           Tokenizer.PlainCount(), (int)ChatTokenCount, Tokenizer.Size());

    printf("  특수 토큰은 **글자 뒤 번호**를 쓴다.\n");
    for (int i = 0; i < ChatTokenCount; i++)
    {
        printf("    %-14s = %zu\n", ChatTokenName(i),
               Tokenizer.FirstSpecial() + (size_t)i);
    }
    printf("\n");

    printf("  코퍼스에 없는 번호라 **본문과 절대 안 겹친다.**\n");
    printf("  겹치면 사용자가 <|end|> 를 입력해 모델을 멈출 수 있다.\n\n");

    CHECK(Tokenizer.Size() == Tokenizer.PlainCount() + ChatTokenCount);
    CHECK(Tokenizer.Coverage() > 0.999);

    {
        std::vector<FChatTurn> Turns;

        FChatTurn One;
        One.Role = RoleSystem;  One.Text = GSystem;          Turns.push_back(One);
        One.Role = RoleUser;    One.Text = "안녕";           Turns.push_back(One);
        One.Role = RoleAssistant;
        One.Text = "안녕하세요.";
        Turns.push_back(One);

        FChatRendered Rendered = RenderChat(Turns, Tokenizer, false);

        printf("  펼친 결과\n\n");
        printf("    %s\n\n", DescribeChat(Rendered.Tokens, Tokenizer).c_str());

        printf("  토큰 %zu개 중 채점하는 자리는 %zu개다.\n\n",
               Rendered.Tokens.size(), Rendered.Scored());

        CHECK(Rendered.Tokens.size() == Rendered.Loss.size());
        CHECK(Rendered.Scored() > 0);
        CHECK(Rendered.Scored() < Rendered.Tokens.size());
    }

    // ================================================================
    // E1-2. 어디를 채점하는가
    // ================================================================
    printf("[E1-2] 마스크를 자리마다 찍어본다\n\n");

    {
        std::vector<FChatTurn> Turns;

        FChatTurn One;
        One.Role = RoleUser;      One.Text = "안녕";      Turns.push_back(One);
        One.Role = RoleAssistant; One.Text = "안녕하세요"; Turns.push_back(One);

        FChatRendered Rendered = RenderChat(Turns, Tokenizer, false);

        printf("  ");
        PrintPaddedRight("자리", 6);
        printf("  ");
        PrintPadded("토큰", 18);
        PrintPadded("다음에 와야 할 것", 22);
        printf("채점\n");

        char Buffer[8];

        for (size_t i = 0; i + 1 < Rendered.Tokens.size(); i++)
        {
            std::string Now, Next;

            const size_t A = (size_t)Rendered.Tokens[i];
            if (A >= Tokenizer.FirstSpecial())
            {
                Now = ChatTokenName(A - Tokenizer.FirstSpecial());
            }
            else { Tokenizer.Decode(Rendered.Tokens[i], Buffer); Now = Buffer; }

            const size_t B = (size_t)Rendered.Tokens[i + 1];
            if (B >= Tokenizer.FirstSpecial())
            {
                Next = ChatTokenName(B - Tokenizer.FirstSpecial());
            }
            else { Tokenizer.Decode(Rendered.Tokens[i + 1], Buffer); Next = Buffer; }

            char Index[16];
            snprintf(Index, sizeof(Index), "%zu", i);

            printf("  ");
            PrintPaddedRight(Index, 6);
            printf("  ");
            PrintPadded(Now.c_str(), 18);
            PrintPadded(Next.c_str(), 22);
            printf("%s\n", Rendered.Loss[i + 1] ? "O" : ".");
        }

        printf("\n");
        printf("  읽는 법.\n");
        printf("    <|user|> 뒤의 글자는 **우리가 써 넣은 것**이다. 안 채점한다\n");
        printf("    <|assistant|> 는 우리가 붙인다. 안 채점한다\n");
        printf("    그 뒤의 글자는 모델이 써야 한다. **채점한다**\n");
        printf("    마지막 <|end|> 도 채점한다. **스스로 멈춰야 하니까**\n\n");

        printf("  <|end|> 를 채점 안 하면 모델이 멈추는 법을 못 배운다.\n");
        printf("  끝없이 이어 쓴다. 흔한 실수다.\n\n");
    }

    // ================================================================
    // E1-3. 마스킹된 자리의 그래디언트는 정확히 0
    // ================================================================
    printf("[E1-3] 안 채점하는 자리에서는 아무 신호도 안 올라온다\n\n");

    FModelConfig Config;
    Config.Vocab = Tokenizer.Size();
    Config.Model = MODEL;
    Config.Heads = HEADS;
    Config.Hidden = HIDDEN;
    Config.Layers = LAYERS;
    Config.MaxLength = MAXLEN;

    const uint32_t PadToken =
        (uint32_t)(Tokenizer.FirstSpecial() + (size_t)ChatEnd);

    std::vector<FSample> Samples;
    for (int i = 0; i < 16; i++)
    {
        Samples.push_back(MakeSample(GPairs[i], Tokenizer, PadToken));
    }

    {
        FTransformer Model(Config);
        FRandom Init;
        RandomSeed(&Init, BOOK_SEED);
        Model.Init(Init);

        const FSample& One = Samples[0];

        FModelTrace Trace;
        FTensor Logits = TransformerForwardTrace(Model, One.Inputs.data(), 1,
                                                 LENGTH, Trace);

        FTensor Upstream = CrossEntropyBackwardMasked(Logits,
                                                      One.Targets.data(),
                                                      One.Mask.data());

        // 마스킹된 줄은 통째로 0 이어야 한다.
        size_t Zeroed = 0, Live = 0;
        for (size_t t = 0; t < LENGTH; t++)
        {
            double Big = 0.0;
            for (size_t v = 0; v < Config.Vocab; v++)
            {
                const double G = std::fabs((double)Upstream(0, t, v));
                if (G > Big) Big = G;
            }

            if (One.Mask[t] == 0)
            {
                CHECK(Big == 0.0);
                Zeroed++;
            }
            else
            {
                CHECK(Big > 0.0);
                Live++;
            }
        }

        printf("  자리 %d개 중\n", LENGTH);
        printf("    채점하는 자리     %zu개 — 그래디언트가 0 이 아니다\n", Live);
        printf("    안 채점하는 자리  %zu개 — **정확히 0**\n\n", Zeroed);

        printf("  0 에 가까운 것이 아니라 0 이다. 애초에 계산을 안 한다.\n\n");

        // 더 센 확인. 마스킹된 자리의 **정답을 바꿔도** 파라미터
        // 그래디언트가 한 비트도 안 바뀌어야 한다.
        FTransformerGrad GradA, GradB;
        GradA.Init(Model);
        GradB.Init(Model);

        TransformerBackward(Model, Trace, Upstream, GradA);

        FSample Twisted = One;
        size_t Changed = 0;
        for (size_t t = 0; t < LENGTH; t++)
        {
            if (Twisted.Mask[t] == 0)
            {
                Twisted.Targets[t] = (uint32_t)((Twisted.Targets[t] + 7)
                                              % Config.Vocab);
                Changed++;
            }
        }

        FTensor Upstream2 = CrossEntropyBackwardMasked(Logits,
                                                       Twisted.Targets.data(),
                                                       Twisted.Mask.data());
        TransformerBackward(Model, Trace, Upstream2, GradB);

        std::vector<Real*> A, B;
        CollectGradients(GradA, A);
        CollectGradients(GradB, B);

        size_t Same = 0;
        for (size_t i = 0; i < A.size(); i++)
        {
            if (*A[i] == *B[i]) Same++;
        }

        printf("  안 채점하는 자리 %zu곳의 정답을 일부러 틀리게 바꿨다.\n",
               Changed);
        printf("  그런데 파라미터 그래디언트 %zu개가 **전부 그대로**다.\n\n",
               Same);

        CHECK(Same == A.size());

        printf("  **채점 안 하는 자리에 무엇이 들어 있든 상관없다.**\n");
        printf("  패딩을 아무 토큰으로 채워도 되는 근거가 이것이다.\n\n");
    }

    // ================================================================
    // E1-4. 마스킹하고 훈련 vs 안 하고 훈련
    // ================================================================
    printf("[E1-4] 마스킹을 빼면 무슨 일이 생기는가\n\n");

    printf("  같은 자료, 같은 초기값, 같은 걸음 수로 두 번 훈련한다.\n");
    printf("  다른 것은 **손실을 어디서 재느냐** 하나뿐이다.\n\n");

    FTransformer Trained[2];

    for (int Which = 0; Which < 2; Which++)
    {
        const bool bMasked = (Which == 0);

        FTransformer Model(Config);
        FRandom Init;
        RandomSeed(&Init, BOOK_SEED);
        Model.Init(Init);

        FTransformerGrad Grad;
        Grad.Init(Model);

        std::vector<Real*> Parameters, Gradients;
        CollectParameters(Model, Parameters);
        CollectGradients(Grad, Gradients);

        FAdamW Adam;
        FAdamWConfig AdamConfig;
        AdamConfig.Rate = 3e-3;
        AdamConfig.WeightDecay = 0.01;
        Adam.Init(Parameters.size(), AdamConfig);

        FRandom Pick;
        RandomSeed(&Pick, BOOK_SEED + 3);

        std::vector<uint32_t> Inputs(BATCH * LENGTH, 0);
        std::vector<uint32_t> Targets(BATCH * LENGTH, 0);
        std::vector<uint8_t> Mask(BATCH * LENGTH, 0);

        auto Begin = std::chrono::steady_clock::now();
        double Loss = 0.0;

        for (int Step = 0; Step < STEPS; Step++)
        {
            for (size_t b = 0; b < BATCH; b++)
            {
                const size_t Which2 = (size_t)RandomBelow(&Pick, Samples.size());
                const FSample& S = Samples[Which2];

                for (size_t t = 0; t < LENGTH; t++)
                {
                    Inputs[b * LENGTH + t] = S.Inputs[t];
                    Targets[b * LENGTH + t] = S.Targets[t];
                    Mask[b * LENGTH + t] = bMasked ? S.Mask[t] : (uint8_t)1;
                }
            }

            Grad.Zero();

            FModelTrace Trace;
            FTensor Logits = TransformerForwardTrace(Model, Inputs.data(),
                                                     BATCH, LENGTH, Trace);

            Loss = CrossEntropyLossMasked(Logits, Targets.data(), Mask.data());

            FTensor Upstream = CrossEntropyBackwardMasked(Logits,
                                                          Targets.data(),
                                                          Mask.data());
            TransformerBackward(Model, Trace, Upstream, Grad);

            ClipGradients(Gradients.data(), Gradients.size(), 1.0);

            const double Rate = ScheduleRate(AdamConfig.Rate, (size_t)Step, 60,
                                             (size_t)STEPS, 0.1);

            Adam.Step(Parameters.data(), (const Real* const*)Gradients.data(),
                      Parameters.size(), Rate);
        }

        const double Elapsed = Seconds(Begin);

        printf("  %s  %d걸음 %.1f초, 마지막 손실 %.4f\n",
               bMasked ? "마스킹 O" : "마스킹 X", STEPS, Elapsed, Loss);

        Trained[Which] = Model;
    }

    printf("\n");
    printf("  주의. 위 두 손실은 **서로 다른 것을 잰 값**이라 비교할 수 없다.\n");
    printf("  같은 자로 다시 잰다.\n\n");

    // 같은 자로 재기 — 둘 다 (가) 조수 글자, (나) 사용자 글자에서 잰다.
    {
        printf("  ");
        PrintPadded("무엇을 잰 손실", 26);
        PrintPaddedRight("마스킹 O", 14);
        PrintPaddedRight("마스킹 X", 14);
        printf("\n");

        double Assistant[2] = { 0.0, 0.0 };
        double User[2] = { 0.0, 0.0 };

        for (int Which = 0; Which < 2; Which++)
        {
            double SumA = 0.0, SumU = 0.0;

            for (size_t s = 0; s < Samples.size(); s++)
            {
                const FSample& S = Samples[s];

                FModelTrace Trace;
                FTensor Logits = TransformerForwardTrace(Trained[Which],
                                                         S.Inputs.data(), 1,
                                                         LENGTH, Trace);

                SumA += CrossEntropyLossMasked(Logits, S.Targets.data(),
                                               S.Mask.data());
                SumU += CrossEntropyLossMasked(Logits, S.Targets.data(),
                                               S.UserMask.data());
            }

            Assistant[Which] = SumA / (double)Samples.size();
            User[Which] = SumU / (double)Samples.size();
        }

        char A[32], B[32];

        printf("  ");
        PrintPadded("조수가 쓸 글자", 26);
        snprintf(A, sizeof(A), "%.4f", Assistant[0]);
        snprintf(B, sizeof(B), "%.4f", Assistant[1]);
        PrintPaddedRight(A, 14);
        PrintPaddedRight(B, 14);
        printf("\n");

        printf("  ");
        PrintPadded("사용자가 쓴 글자", 26);
        snprintf(A, sizeof(A), "%.4f", User[0]);
        snprintf(B, sizeof(B), "%.4f", User[1]);
        PrintPaddedRight(A, 14);
        PrintPaddedRight(B, 14);
        printf("\n\n");

        printf("  조수가 쓸 글자에서는 마스킹한 쪽이 %.1f배 낫고,\n",
               Assistant[1] / Assistant[0]);
        printf("  **사용자가 쓴 글자에서는 마스킹 안 한 쪽이 %.1f배 낫다.**\n\n",
               User[0] / User[1]);

        printf("  뒤쪽이 문제다. 사용자 글자를 잘 맞힌다는 것은\n");
        printf("  **사용자의 말투를 외웠다**는 뜻이다. 시킨 적 없는 일이다.\n\n");

        printf("  용량은 정해져 있다. 시키지 않은 것을 외우는 데 쓰면\n");
        printf("  시킨 것에 쓸 몫이 준다. 앞쪽 %.1f배가 그 값이다.\n\n",
               Assistant[1] / Assistant[0]);

        CHECK(User[0] > User[1]);
        CHECK(Assistant[0] < 1.0);
    }

    // ================================================================
    // E1-5. 실제로 대화해 보기
    // ================================================================
    printf("[E1-5] 뽑아보기 — <|assistant|> 뒤부터, <|end|> 까지\n\n");

    {
        const char* Asks[4] = { "안녕", "이름이 뭐야", "고마워", "날씨 어때" };

        for (int Which = 0; Which < 2; Which++)
        {
            printf("    %s\n", (Which == 0) ? "마스킹 O" : "마스킹 X");

            for (int q = 0; q < 4; q++)
            {
                std::vector<FChatTurn> Turns;

                FChatTurn One;
                One.Role = RoleSystem; One.Text = GSystem; Turns.push_back(One);
                One.Role = RoleUser;   One.Text = Asks[q]; Turns.push_back(One);

                // 생성 프롬프트를 붙인다. "이제 네 차례다"
                FChatRendered Prompt = RenderChat(Turns, Tokenizer, true);

                FKvCache Cache(Config);
                Cache.Clear();

                FTensor Logits;
                for (size_t i = 0; i < Prompt.Tokens.size(); i++)
                {
                    Logits = Trained[Which].Step(Prompt.Tokens[i], Cache);
                }

                std::string Answer;
                bool bStopped = false;

                for (int i = 0; i < 40 && Cache.Length < Config.MaxLength; i++)
                {
                    size_t Best = 0;
                    for (size_t v = 1; v < Config.Vocab; v++)
                    {
                        if (Logits(v) > Logits(Best)) Best = v;
                    }

                    if (Best >= Tokenizer.FirstSpecial())
                    {
                        const size_t Kind = Best - Tokenizer.FirstSpecial();
                        if (Kind == ChatEnd) { bStopped = true; break; }

                        Answer += ChatTokenName(Kind);
                    }
                    else
                    {
                        char Buffer[8];
                        Tokenizer.Decode((uint32_t)Best, Buffer);
                        Answer += Buffer;
                    }

                    Logits = Trained[Which].Step((uint32_t)Best, Cache);
                }

                printf("      \"%s\" -> \"%s\"%s\n", Asks[q], Answer.c_str(),
                       bStopped ? "" : "   (안 멈췄다)");
            }

            printf("\n");
        }

        printf("  **<|end|> 가 나오면 멈춘다.** 그게 생성을 끝내는 신호다.\n");
        printf("  이 토큰을 채점하지 않으면 모델은 멈출 줄을 모른다.\n\n");
    }

    // ================================================================
    // E1-6. 멈추고 난 뒤에도 계속 시키면
    // ================================================================
    printf("[E1-6] 멈추라는 신호를 무시하고 계속 뽑으면\n\n");

    {
        for (int Which = 0; Which < 2; Which++)
        {
            std::vector<FChatTurn> Turns;

            FChatTurn One;
            One.Role = RoleSystem; One.Text = GSystem; Turns.push_back(One);
            One.Role = RoleUser;   One.Text = "안녕";  Turns.push_back(One);

            FChatRendered Prompt = RenderChat(Turns, Tokenizer, true);

            FKvCache Cache(Config);
            Cache.Clear();

            FTensor Logits;
            for (size_t i = 0; i < Prompt.Tokens.size(); i++)
            {
                Logits = Trained[Which].Step(Prompt.Tokens[i], Cache);
            }

            std::string All;

            for (int i = 0; i < 40 && Cache.Length < Config.MaxLength; i++)
            {
                size_t Best = 0;
                for (size_t v = 1; v < Config.Vocab; v++)
                {
                    if (Logits(v) > Logits(Best)) Best = v;
                }

                if (Best >= Tokenizer.FirstSpecial())
                {
                    All += ChatTokenName(Best - Tokenizer.FirstSpecial());
                }
                else
                {
                    char Buffer[8];
                    Tokenizer.Decode((uint32_t)Best, Buffer);
                    All += Buffer;
                }

                Logits = Trained[Which].Step((uint32_t)Best, Cache);
            }

            printf("    %s\n      %s\n\n",
                   (Which == 0) ? "마스킹 O" : "마스킹 X", All.c_str());
        }

        printf("  둘 다 첫 <|end|> 까지는 같다. 그 뒤가 갈린다.\n\n");

        printf("  **마스킹 X 는 <|end|> 를 끝없이 찍는다.**\n");
        printf("  우리가 길이를 맞추려고 뒤를 <|end|> 로 채웠기 때문이다.\n");
        printf("  마스킹을 빼면 **그 패딩까지 자료로 배운다.**\n\n");

        printf("  마스킹 O 는 패딩을 안 봤으므로 그냥 아무 말이나 잇는다.\n");
        printf("  둘 다 쓸모없는 출력이지만 쓸모없어지는 **이유가 다르다.**\n\n");

        printf("  요점은 이것이다. 마스킹을 빼면 모델은 자료에 **우연히**\n");
        printf("  들어 있는 것까지 전부 배운다. 패딩도, 사용자의 말투도.\n\n");

        printf("  대화 프로그램이 할 일은 <|end|> 에서 자르는 것뿐이다.\n");
        printf("  모델은 멈추라는 **신호를 낼 줄만** 알면 된다.\n\n");
    }

    printf("  손실 마스킹은 한 줄짜리 개념이다.\n");
    printf("  **채점 안 할 자리의 상류 그래디언트를 0 으로 둔다.**\n");
    printf("  D8 에서 마지막 한 자리에 했던 일을 흩어진 자리로 넓혔을 뿐이다.\n\n");

    return ReportResult();
}
