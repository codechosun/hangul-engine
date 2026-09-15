// lib/Chat.hpp
//
// 대화를 토큰 열로 펴고, **어디를 채점할지**를 함께 표시한다.
//
// 지금까지 만든 모델은 글을 이어 쓸 뿐이다. 대화로 만들려면 둘이 필요하다.
//
//   ① 틀      누가 말하는 차례인지를 토큰으로 표시한다
//   ② 마스크  사용자가 쓴 부분은 **채점하지 않는다**
//
// ②가 없으면 모델이 사용자의 말투까지 따라 배운다. 그러면 답을 하다 말고
// 사용자 차례를 자기가 이어 쓴다. 실제로 흔한 사고다.
//
// D8 에서 마지막 자리만 채점한 것이 이것의 가장 단순한 판이었다.
// 여기서는 자리가 띄엄띄엄 흩어진다.

#ifndef CHAT_HPP
#define CHAT_HPP

#include "Tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

// 특수 토큰. 어휘 뒤에 이 순서로 붙는다.
enum EChatToken
{
    ChatSystem = 0,
    ChatUser,
    ChatAssistant,
    ChatEnd,
    ChatTokenCount
};

enum ERole
{
    RoleSystem = 0,
    RoleUser,
    RoleAssistant
};

struct FChatTurn
{
    ERole Role = RoleUser;
    std::string Text;
};

// 펼친 결과.
//
// Tokens[i] 를 보고 Tokens[i+1] 을 맞히는 것이 우리 모델의 일이므로,
// **채점은 Tokens[i+1] 자리의 마스크로 정한다.** Loss 배열의 길이는
// Tokens 와 같고, Loss[i] 는 "i 번째 토큰을 맞혀야 하는가"를 뜻한다.
struct FChatRendered
{
    std::vector<uint32_t> Tokens;
    std::vector<uint8_t> Loss;     // 1 이면 채점, 0 이면 안 한다

    size_t Scored() const;
};

// 대화를 펼친다.
//
// bAddGenerationPrompt 가 참이면 마지막에 <|assistant|> 하나를 더 붙인다.
// 생성할 때 쓴다 — "이제 네 차례다"라는 뜻이다.
FChatRendered RenderChat(const std::vector<FChatTurn>& Turns,
                         const FTokenizer& Tokenizer,
                         bool bAddGenerationPrompt);

// 사람이 읽을 수 있게 되돌린다. 특수 토큰은 <|이름|> 으로 적는다.
std::string DescribeChat(const std::vector<uint32_t>& Tokens,
                         const FTokenizer& Tokenizer);

// 특수 토큰의 이름.
const char* ChatTokenName(size_t Which);

#endif
