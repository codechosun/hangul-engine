// lib/Engine.cpp

#include "Engine.hpp"
#include "Random.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace
{

const char GMagic[4] = { 'H', 'G', 'E', 'N' };
const uint32_t GVersion = 1;

// 파일에 적는 숫자는 전부 uint64 로 맞춘다.
//
// size_t 를 그대로 적으면 32비트 빌드와 64비트 빌드가 서로 못 읽는다.
// 파일 형식에서는 **컴파일러가 정하는 크기를 쓰지 않는다.**
bool WriteNumber(FILE* File, uint64_t Value)
{
    return fwrite(&Value, sizeof(uint64_t), 1, File) == 1;
}

bool ReadNumber(FILE* File, uint64_t* Out)
{
    return fread(Out, sizeof(uint64_t), 1, File) == 1;
}

} // namespace

void FEngine::Create(const FModelConfig& InConfig, const FTokenizer& InTokenizer,
                     uint64_t Seed)
{
    Config = InConfig;
    Tokenizer = InTokenizer;

    Model = FTransformer(Config);

    FRandom Rng;
    RandomSeed(&Rng, Seed);
    Model.Init(Rng);

    bReady = true;
}

size_t FEngine::ParameterCount() const
{
    return bReady ? Model.ParameterCount() : 0;
}

size_t FEngine::Bytes() const
{
    return ParameterCount() * sizeof(Real);
}

bool FEngine::Save(const char* Path) const
{
    if (!bReady) return false;

    FILE* File = fopen(Path, "wb");
    if (File == NULL) return false;

    bool bOk = true;

    bOk = bOk && (fwrite(GMagic, 1, 4, File) == 4);
    bOk = bOk && (fwrite(&GVersion, sizeof(uint32_t), 1, File) == 1);

    const uint32_t RealSize = (uint32_t)sizeof(Real);
    bOk = bOk && (fwrite(&RealSize, sizeof(uint32_t), 1, File) == 1);

    bOk = bOk && WriteNumber(File, (uint64_t)Config.Vocab);
    bOk = bOk && WriteNumber(File, (uint64_t)Config.Model);
    bOk = bOk && WriteNumber(File, (uint64_t)Config.Heads);
    bOk = bOk && WriteNumber(File, (uint64_t)Config.Hidden);
    bOk = bOk && WriteNumber(File, (uint64_t)Config.Layers);
    bOk = bOk && WriteNumber(File, (uint64_t)Config.MaxLength);

    // 어휘.
    const std::vector<uint32_t>& Alphabet = Tokenizer.Alphabet();
    bOk = bOk && WriteNumber(File, (uint64_t)Alphabet.size());
    bOk = bOk && WriteNumber(File,
                             (uint64_t)(Tokenizer.Size() - Alphabet.size()));

    if (bOk && !Alphabet.empty())
    {
        bOk = fwrite(Alphabet.data(), sizeof(uint32_t), Alphabet.size(), File)
              == Alphabet.size();
    }

    // 가중치. CollectParameters 의 순서를 그대로 쓴다.
    //
    // 그 순서가 **파일 형식의 일부**가 된다는 뜻이다. 나중에 파라미터를
    // 하나 더 붙이면 판 번호를 올려야 한다.
    FTransformer& Mutable = const_cast<FTransformer&>(Model);
    std::vector<Real*> Parameters;
    CollectParameters(Mutable, Parameters);

    bOk = bOk && WriteNumber(File, (uint64_t)Parameters.size());

    for (size_t i = 0; bOk && i < Parameters.size(); i++)
    {
        bOk = fwrite(Parameters[i], sizeof(Real), 1, File) == 1;
    }

    fclose(File);

    return bOk;
}

bool FEngine::Load(const char* Path)
{
    FILE* File = fopen(Path, "rb");
    if (File == NULL) return false;

    bool bOk = true;

    char Magic[4] = { 0, 0, 0, 0 };
    bOk = bOk && (fread(Magic, 1, 4, File) == 4);
    bOk = bOk && (memcmp(Magic, GMagic, 4) == 0);

    uint32_t Version = 0;
    bOk = bOk && (fread(&Version, sizeof(uint32_t), 1, File) == 1);
    bOk = bOk && (Version == GVersion);

    uint32_t RealSize = 0;
    bOk = bOk && (fread(&RealSize, sizeof(uint32_t), 1, File) == 1);
    bOk = bOk && (RealSize == (uint32_t)sizeof(Real));

    uint64_t Numbers[6] = { 0, 0, 0, 0, 0, 0 };
    for (int i = 0; bOk && i < 6; i++)
    {
        bOk = ReadNumber(File, &Numbers[i]);
    }

    uint64_t AlphabetCount = 0;
    uint64_t SpecialCount = 0;
    bOk = bOk && ReadNumber(File, &AlphabetCount);
    bOk = bOk && ReadNumber(File, &SpecialCount);

    std::vector<uint32_t> Alphabet((size_t)AlphabetCount, 0);
    if (bOk && AlphabetCount > 0)
    {
        bOk = fread(Alphabet.data(), sizeof(uint32_t), (size_t)AlphabetCount,
                    File) == (size_t)AlphabetCount;
    }

    uint64_t ParameterCountInFile = 0;
    bOk = bOk && ReadNumber(File, &ParameterCountInFile);

    if (!bOk)
    {
        fclose(File);
        return false;
    }

    FModelConfig Loaded;
    Loaded.Vocab = (size_t)Numbers[0];
    Loaded.Model = (size_t)Numbers[1];
    Loaded.Heads = (size_t)Numbers[2];
    Loaded.Hidden = (size_t)Numbers[3];
    Loaded.Layers = (size_t)Numbers[4];
    Loaded.MaxLength = (size_t)Numbers[5];

    FTransformer Fresh(Loaded);

    std::vector<Real*> Parameters;
    CollectParameters(Fresh, Parameters);

    if (Parameters.size() != (size_t)ParameterCountInFile)
    {
        // 설정은 읽혔는데 가중치 수가 안 맞는다. 판이 다른 파일이다.
        fclose(File);
        return false;
    }

    for (size_t i = 0; bOk && i < Parameters.size(); i++)
    {
        bOk = fread(Parameters[i], sizeof(Real), 1, File) == 1;
    }

    fclose(File);

    if (!bOk) return false;

    Config = Loaded;
    Model = std::move(Fresh);
    Tokenizer.BuildFromCodepoints(Alphabet, (size_t)SpecialCount);
    bReady = true;

    return true;
}

std::string FEngine::Run(const std::vector<uint32_t>& Prompt,
                         const FGenerateOptions& Options,
                         bool bStopAtEnd) const
{
    if (!bReady || Prompt.empty()) return std::string();

    FKvCache Cache(Config);
    Cache.Clear();

    FRandom Rng;
    RandomSeed(&Rng, Options.Seed);

    FTensor Logits;
    for (size_t i = 0; i < Prompt.size() && Cache.Length < Config.MaxLength; i++)
    {
        Logits = Model.Step(Prompt[i], Cache);
    }

    std::string Result;
    const size_t First = Tokenizer.FirstSpecial();

    for (size_t n = 0; n < Options.MaxTokens
                    && Cache.Length < Config.MaxLength; n++)
    {
        size_t Pick = 0;

        if (Options.Temperature <= 0.0)
        {
            for (size_t v = 1; v < Config.Vocab; v++)
            {
                if (Logits(v) > Logits(Pick)) Pick = v;
            }
        }
        else
        {
            // 최댓값을 먼저 빼는 것은 C1 부터의 규칙이다.
            double Biggest = (double)Logits(0);
            for (size_t v = 1; v < Config.Vocab; v++)
            {
                if ((double)Logits(v) > Biggest) Biggest = (double)Logits(v);
            }

            double Total = 0.0;
            std::vector<double> Weight(Config.Vocab, 0.0);

            for (size_t v = 0; v < Config.Vocab; v++)
            {
                Weight[v] = std::exp(((double)Logits(v) - Biggest)
                                   / Options.Temperature);
                Total += Weight[v];
            }

            double Target = RandomUnit(&Rng) * Total;
            for (size_t v = 0; v < Config.Vocab; v++)
            {
                Target -= Weight[v];
                if (Target <= 0.0) { Pick = v; break; }
            }
        }

        if (Pick >= First)
        {
            const size_t Kind = Pick - First;
            if (bStopAtEnd && Kind == (size_t)ChatEnd) break;

            // 특수 토큰이 답 안에 섞이면 버린다. 사용자에게 보일 것이 아니다.
            Logits = Model.Step((uint32_t)Pick, Cache);
            continue;
        }

        char Buffer[8];
        Tokenizer.Decode((uint32_t)Pick, Buffer);
        Result += Buffer;

        Logits = Model.Step((uint32_t)Pick, Cache);
    }

    return Result;
}

std::string FEngine::Continue(const char* Prompt,
                              const FGenerateOptions& Options) const
{
    std::vector<uint32_t> Tokens;
    Tokenizer.Encode(Prompt, Tokens);

    return Run(Tokens, Options, false);
}

std::string FEngine::Reply(const std::vector<FChatTurn>& Turns,
                           const FGenerateOptions& Options) const
{
    FChatRendered Rendered = RenderChat(Turns, Tokenizer, true);

    return Run(Rendered.Tokens, Options, true);
}
