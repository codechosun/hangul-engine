// lib/Chat.cpp

#include "Chat.hpp"

#include <cassert>

namespace
{

const char* GNames[ChatTokenCount] =
{
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>"
};

uint32_t Special(const FTokenizer& Tokenizer, EChatToken Which)
{
    return (uint32_t)(Tokenizer.FirstSpecial() + (size_t)Which);
}

EChatToken Opener(ERole Role)
{
    switch (Role)
    {
    case RoleSystem:    return ChatSystem;
    case RoleUser:      return ChatUser;
    case RoleAssistant: return ChatAssistant;
    }

    return ChatUser;
}

} // namespace

const char* ChatTokenName(size_t Which)
{
    if (Which >= ChatTokenCount) return "<|?|>";
    return GNames[Which];
}

size_t FChatRendered::Scored() const
{
    size_t Count = 0;
    for (size_t i = 0; i < Loss.size(); i++)
    {
        if (Loss[i] != 0) Count++;
    }
    return Count;
}

FChatRendered RenderChat(const std::vector<FChatTurn>& Turns,
                         const FTokenizer& Tokenizer,
                         bool bAddGenerationPrompt)
{
    FChatRendered Result;

    for (size_t t = 0; t < Turns.size(); t++)
    {
        const FChatTurn& Turn = Turns[t];
        const bool bScore = (Turn.Role == RoleAssistant);

        // 여는 토큰. **이건 채점하지 않는다.**
        //
        // 우리가 직접 쓰는 토큰이므로 모델이 맞힐 필요가 없다.
        // 반대로 <|end|> 는 채점해야 한다 — 모델이 스스로 멈춰야 하니까.
        Result.Tokens.push_back(Special(Tokenizer, Opener(Turn.Role)));
        Result.Loss.push_back(0);

        const size_t Before = Result.Tokens.size();
        Tokenizer.Encode(Turn.Text, Result.Tokens);

        for (size_t i = Before; i < Result.Tokens.size(); i++)
        {
            Result.Loss.push_back(bScore ? (uint8_t)1 : (uint8_t)0);
        }

        Result.Tokens.push_back(Special(Tokenizer, ChatEnd));
        Result.Loss.push_back(bScore ? (uint8_t)1 : (uint8_t)0);
    }

    if (bAddGenerationPrompt)
    {
        Result.Tokens.push_back(Special(Tokenizer, ChatAssistant));
        Result.Loss.push_back(0);
    }

    assert(Result.Tokens.size() == Result.Loss.size());

    return Result;
}

std::string DescribeChat(const std::vector<uint32_t>& Tokens,
                         const FTokenizer& Tokenizer)
{
    std::string Result;
    char Buffer[8];

    for (size_t i = 0; i < Tokens.size(); i++)
    {
        const size_t Id = (size_t)Tokens[i];

        if (Id >= Tokenizer.FirstSpecial())
        {
            Result += ChatTokenName(Id - Tokenizer.FirstSpecial());
            continue;
        }

        Tokenizer.Decode(Tokens[i], Buffer);
        Result += Buffer;
    }

    return Result;
}
