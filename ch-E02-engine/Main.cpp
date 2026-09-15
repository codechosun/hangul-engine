// ch-E02-engine/Main.cpp
//
// E2. 엔진으로 묶기
//
// 저장소 루트에서 실행할 것.
//     Main.exe
//
// 이 프로그램이 data/chat.hgen 을 만든다. 그 다음 app/main.cpp 가
// **엔진만 가지고** 그 파일을 읽는다.

#include "Test.h"
#include "Pretty.h"

#include "Backward.hpp"
#include "Chat.hpp"
#include "Engine.hpp"
#include "Optimizer.hpp"
#include "Random.h"
#include "Tokenizer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define BOOK_SEED 20260914ull

#define MODEL_PATH "data/chat.hgen"

#define LENGTH 63
#define MAXLEN 64

#define MODEL   96
#define HEADS    4
#define HIDDEN 384
#define LAYERS   2

#ifndef STEPS
#define STEPS 900
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

std::vector<FChatTurn> Ask(const char* Text)
{
    std::vector<FChatTurn> Turns;

    FChatTurn One;
    One.Role = RoleSystem;
    One.Text = GSystem;
    Turns.push_back(One);

    One.Role = RoleUser;
    One.Text = Text;
    Turns.push_back(One);

    return Turns;
}

} // namespace

int main(void)
{
    printf("E2. 엔진으로 묶기\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ================================================================
    // E2-1. 현관 하나
    // ================================================================
    printf("[E2-1] 헤더 하나로 쓴다\n\n");

    printf("      #include \"Engine.hpp\"\n\n");
    printf("      FEngine Engine;\n");
    printf("      Engine.Load(\"model.hgen\");\n");
    printf("      Engine.Reply(Turns, {});\n\n");

    printf("  lib/ 에 파일이 60개 넘게 쌓였지만 밖에서 볼 것은 이것뿐이다.\n");
    printf("  **고칠 수 있는 자리를 남겨두는 것**이 경계를 긋는 이유다.\n\n");

    // ---- 어휘와 자료 ----
    std::string Everything = GSystem;
    for (int i = 0; i < 16; i++)
    {
        Everything += GPairs[i].User;
        Everything += GPairs[i].Assistant;
    }

    FTokenizer Tokenizer;
    Tokenizer.BuildFromText(Everything.c_str(), Everything.size(), 200,
                            ChatTokenCount);

    FModelConfig Config;
    Config.Vocab = Tokenizer.Size();
    Config.Model = MODEL;
    Config.Heads = HEADS;
    Config.Hidden = HIDDEN;
    Config.Layers = LAYERS;
    Config.MaxLength = MAXLEN;

    FEngine Engine;
    Engine.Create(Config, Tokenizer, BOOK_SEED);

    printf("  갓 만든 엔진 — 파라미터 %zu개, %.1f KB\n\n",
           Engine.ParameterCount(), Engine.Bytes() / 1024.0);

    CHECK(Engine.IsReady());
    CHECK(Engine.ParameterCount() > 0);

    // ================================================================
    // E2-2. 훈련은 밖에서
    // ================================================================
    printf("[E2-2] 훈련은 엔진 바깥에서 한다\n\n");

    printf("  FEngine 은 가중치를 감추지 않는다. MutableModel() 로 연다.\n");
    printf("  감추면 이 엔진으로 미세조정을 못 한다.\n");
    printf("  **무엇을 감추고 무엇을 열지가 API 설계다.**\n\n");

    {
        FTransformer& Model = Engine.MutableModel();

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

        // E1 과 같은 자료를 같은 방식으로 편다.
        std::vector<std::vector<uint32_t>> Inputs, Targets;
        std::vector<std::vector<uint8_t>> Masks;

        const uint32_t PadToken =
            (uint32_t)(Tokenizer.FirstSpecial() + (size_t)ChatEnd);

        for (int i = 0; i < 16; i++)
        {
            std::vector<FChatTurn> Turns = Ask(GPairs[i].User);

            FChatTurn One;
            One.Role = RoleAssistant;
            One.Text = GPairs[i].Assistant;
            Turns.push_back(One);

            FChatRendered Rendered = RenderChat(Turns, Tokenizer, false);

            std::vector<uint32_t> In, Out;
            std::vector<uint8_t> Mask;

            for (size_t t = 0; t < LENGTH; t++)
            {
                const bool bInside = (t + 1 < Rendered.Tokens.size());

                In.push_back(bInside ? Rendered.Tokens[t] : PadToken);
                Out.push_back(bInside ? Rendered.Tokens[t + 1] : PadToken);
                Mask.push_back(bInside ? Rendered.Loss[t + 1] : (uint8_t)0);
            }

            Inputs.push_back(In);
            Targets.push_back(Out);
            Masks.push_back(Mask);
        }

        FRandom Pick;
        RandomSeed(&Pick, BOOK_SEED + 3);

        std::vector<uint32_t> BatchIn(BATCH * LENGTH, 0);
        std::vector<uint32_t> BatchOut(BATCH * LENGTH, 0);
        std::vector<uint8_t> BatchMask(BATCH * LENGTH, 0);

        auto Begin = std::chrono::steady_clock::now();
        double Loss = 0.0;

        for (int Step = 0; Step < STEPS; Step++)
        {
            for (size_t b = 0; b < BATCH; b++)
            {
                const size_t Which = (size_t)RandomBelow(&Pick, Inputs.size());

                for (size_t t = 0; t < LENGTH; t++)
                {
                    BatchIn[b * LENGTH + t] = Inputs[Which][t];
                    BatchOut[b * LENGTH + t] = Targets[Which][t];
                    BatchMask[b * LENGTH + t] = Masks[Which][t];
                }
            }

            Grad.Zero();

            FModelTrace Trace;
            FTensor Logits = TransformerForwardTrace(Model, BatchIn.data(),
                                                     BATCH, LENGTH, Trace);

            Loss = CrossEntropyLossMasked(Logits, BatchOut.data(),
                                          BatchMask.data());

            FTensor Upstream = CrossEntropyBackwardMasked(Logits,
                                                          BatchOut.data(),
                                                          BatchMask.data());
            TransformerBackward(Model, Trace, Upstream, Grad);

            ClipGradients(Gradients.data(), Gradients.size(), 1.0);

            const double Rate = ScheduleRate(AdamConfig.Rate, (size_t)Step, 50,
                                             (size_t)STEPS, 0.1);

            Adam.Step(Parameters.data(), (const Real* const*)Gradients.data(),
                      Parameters.size(), Rate);
        }

        printf("  %d걸음 %.1f초, 마지막 손실 %.4f\n\n", STEPS,
               Seconds(Begin), Loss);

        CHECK(Loss < 0.05);
    }

    // ================================================================
    // E2-3. 파일로 굽고 다시 읽기
    // ================================================================
    printf("[E2-3] 모델 파일 — 헤더 + 날 배열\n\n");

    FGenerateOptions Greedy;
    Greedy.Temperature = 0.0;
    Greedy.MaxTokens = 40;

    std::string Before[4];
    const char* Asks[4] = { "안녕", "이름이 뭐야", "고마워", "잘 가" };

    for (int i = 0; i < 4; i++)
    {
        Before[i] = Engine.Reply(Ask(Asks[i]), Greedy);
    }

    CHECK(Engine.Save(MODEL_PATH));

    {
        FILE* File = fopen(MODEL_PATH, "rb");
        CHECK(File != NULL);

        long Size = 0;
        if (File != NULL)
        {
            fseek(File, 0, SEEK_END);
            Size = ftell(File);
            fclose(File);
        }

        printf("  %s  %.1f KB\n", MODEL_PATH, Size / 1024.0);
        printf("  가중치만 %.1f KB — 나머지 %ld바이트가 머리말이다.\n\n",
               Engine.Bytes() / 1024.0,
               (long)(Size - (long)Engine.Bytes()));

        printf("    \"HGEN\"   4바이트\n");
        printf("    판 번호  4바이트\n");
        printf("    Real 크기 4바이트   <- 이게 중요하다\n");
        printf("    설정     uint64 6개\n");
        printf("    어휘     글자 수 + 특수 토큰 수 + 코드포인트\n");
        printf("    가중치   Real %zu개\n\n", Engine.ParameterCount());

        printf("  **Real 크기를 적어두는 이유.** float 로 구운 파일을\n");
        printf("  double 빌드가 읽으면 전부 쓰레기가 되는데, 그게\n");
        printf("  **조용히** 일어난다. 크기가 안 맞으면 Load 가 실패한다.\n\n");

        printf("  숫자를 전부 uint64 로 적는 이유도 같다. size_t 를 그대로\n");
        printf("  적으면 32비트 빌드와 64비트 빌드가 서로 못 읽는다.\n\n");
    }

    // 새 엔진으로 읽어 같은 답이 나오는지 본다.
    {
        FEngine Fresh;
        CHECK(Fresh.Load(MODEL_PATH));

        CHECK(Fresh.ParameterCount() == Engine.ParameterCount());
        CHECK(Fresh.GetConfig().Vocab == Engine.GetConfig().Vocab);
        CHECK(Fresh.GetTokenizer().Size() == Engine.GetTokenizer().Size());

        // 가중치가 비트까지 같아야 한다.
        std::vector<Real*> A, B;
        CollectParameters(Engine.MutableModel(), A);
        CollectParameters(Fresh.MutableModel(), B);

        size_t Same = 0;
        for (size_t i = 0; i < A.size(); i++)
        {
            if (*A[i] == *B[i]) Same++;
        }

        printf("  다시 읽은 가중치 %zu개 중 %zu개가 **비트까지 같다.**\n\n",
               A.size(), Same);
        CHECK(Same == A.size());

        printf("  ");
        PrintPadded("물어본 것", 18);
        PrintPadded("구운 뒤 답", 34);
        printf("같나\n");

        int Match = 0;
        for (int i = 0; i < 4; i++)
        {
            const std::string After = Fresh.Reply(Ask(Asks[i]), Greedy);
            const bool bSame = (After == Before[i]);
            if (bSame) Match++;

            printf("  ");
            PrintPadded(Asks[i], 18);
            PrintPadded(After.c_str(), 34);
            printf("%s\n", bSame ? "O" : "X");
        }

        printf("\n");
        CHECK(Match == 4);

        printf("  **굽기 전과 후의 답이 글자까지 같다.**\n");
        printf("  이게 맞아야 모델을 남한테 줄 수 있다.\n\n");
    }

    // ================================================================
    // E2-4. 잘못된 파일
    // ================================================================
    printf("[E2-4] 잘못된 파일은 조용히 통과시키지 않는다\n\n");

    {
        FEngine Broken;

        CHECK(!Broken.Load("data/does-not-exist.hgen"));
        printf("  없는 파일           -> 실패\n");

        // 머리말을 망가뜨린 사본을 만든다.
        const char* Damaged = "data/chat-damaged.hgen";
        {
            FILE* In = fopen(MODEL_PATH, "rb");
            FILE* Out = fopen(Damaged, "wb");

            if (In != NULL && Out != NULL)
            {
                char Buffer[4096];
                size_t Got = fread(Buffer, 1, sizeof(Buffer), In);

                // 마법 문자 한 글자만 바꾼다.
                if (Got > 0) Buffer[0] = 'X';
                fwrite(Buffer, 1, Got, Out);

                while ((Got = fread(Buffer, 1, sizeof(Buffer), In)) > 0)
                {
                    fwrite(Buffer, 1, Got, Out);
                }
            }

            if (In != NULL) fclose(In);
            if (Out != NULL) fclose(Out);
        }

        CHECK(!Broken.Load(Damaged));
        printf("  마법 문자 한 글자 틀림 -> 실패\n\n");

        CHECK(!Broken.IsReady());

        printf("  **실패했을 때 엔진이 반쯤 채워지지 않는다.**\n");
        printf("  Load 는 다 읽고 검사가 끝난 뒤에야 자기 것을 바꾼다.\n");
        printf("  중간에 갈아 끼우면 실패한 엔진으로 생성이 되고,\n");
        printf("  그건 쓰레기를 내놓으면서 성공한 척하는 것이다.\n\n");
    }

    // ================================================================
    // E2-5. 무엇을 내보내고 무엇을 남기나
    // ================================================================
    printf("[E2-5] 엔진과 교재를 가른다\n\n");

    printf("  tools/check_engine.py 가 #include 를 따라가며 센다.\n");
    printf("  tools/export_engine.py 가 그 목록만 dist/ 로 복사한다.\n\n");

    printf("  ");
    PrintPadded("묶음", 12);
    PrintPadded("무엇", 34);
    printf("\n");

    struct FTier { const char* Name; const char* What; };
    const FTier Tiers[4] =
    {
        { "core",  "추론. Engine.hpp 한 장이 현관" },
        { "train", "역전파와 옵티마이저. 미세조정용" },
        { "quant", "8비트·4비트 양자화" },
        { "ngram", "A파트 N-그램. 모델 없이 도는 가벼운 생성기" },
    };

    for (int i = 0; i < 4; i++)
    {
        printf("  ");
        PrintPadded(Tiers[i].Name, 12);
        PrintPadded(Tiers[i].What, 34);
        printf("\n");
    }

    printf("\n");
    printf("  안 내보내는 것 — Nn, Vector, Matrix, Nplm, Npy, Pca, Gemm,\n");
    printf("  GradCheck, LogMath. **교재에서만 값을 하는 것들**이다.\n\n");

    printf("  Vector 와 Matrix 는 B·C파트의 주인공이었는데 Tensor 가\n");
    printf("  대신하게 됐다. 지우지 않고 남겨둔다. 그 장들이 아직 산다.\n");
    printf("  **엔진에서 빠지는 것과 교재에서 지우는 것은 다르다.**\n\n");

    printf("  검증은 dist/ 만 보고 빌드해 보는 것이다.\n");
    printf("  ch-E02-engine/app/main.cpp 가 그 프로그램이다.\n\n");

    return ReportResult();
}
