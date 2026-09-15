// lib/Engine.hpp
//
// **엔진의 현관.** 밖에서 쓰는 사람이 포함할 헤더는 이것 하나다.
//
//     #include "Engine.hpp"
//
//     FEngine Engine;
//     Engine.Load("model.hgen");
//     printf("%s\n", Engine.Reply(Turns, {}).c_str());
//
// 나머지 헤더는 **안쪽**이다. 포함해도 되지만 보증하지 않는다.
// 이 파일의 이름과 여기 있는 함수의 모양만 지킨다.
//
// 왜 이런 구분이 필요한가
//   A1 부터 E1 까지 lib/ 이 스물 몇 개로 자랐다. 전부가 공개 API 라면
//   FTensor 의 필드 이름 하나를 바꾸는 것도 남의 코드를 깨는 일이 된다.
//   **고칠 수 있는 자리를 남겨두는 것**이 경계를 긋는 이유다.
//
// 이 엔진이 의존하는 것
//   C++17 표준 라이브러리. 그게 전부다. 교재의 common/ 에도, 외부
//   라이브러리에도 기대지 않는다. tools/check_engine.py 가 그걸 검사한다.

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "Chat.hpp"
#include "Model.hpp"
#include "Tokenizer.hpp"
#include "Types.h"

#include <cstdint>
#include <string>
#include <vector>

// 뽑는 방법.
struct FGenerateOptions
{
    size_t MaxTokens = 64;

    // 0 이면 늘 1등을 고른다. 크면 클수록 어지럽게 고른다.
    // A5 에서 만든 것과 같은 규약이다.
    double Temperature = 0.8;

    uint64_t Seed = 20260914ull;
};

class FEngine
{
public:
    // ---- 만들기 ----

    // 새 모델을 난수로 채운다. 훈련은 밖에서 한다.
    void Create(const FModelConfig& InConfig, const FTokenizer& InTokenizer,
                uint64_t Seed);

    // ---- 파일 ----
    //
    // 형식은 헤더 + 날 배열이다 (결정 25). 파싱이 주제를 가리지 않게.
    //
    //     "HGEN"  판 번호  Real 크기  설정  어휘  가중치
    //
    // Real 크기를 적어두는 이유. float 로 저장한 파일을 double 빌드가
    // 읽으면 전부 쓰레기가 되는데, 그게 **조용히** 일어난다.
    bool Save(const char* Path) const;
    bool Load(const char* Path);

    // ---- 쓰기 ----

    // 글을 이어 쓴다. 대화 틀을 안 쓴다.
    std::string Continue(const char* Prompt,
                         const FGenerateOptions& Options) const;

    // 대화에 답한다. <|assistant|> 를 붙여 시작하고 <|end|> 에서 멈춘다.
    std::string Reply(const std::vector<FChatTurn>& Turns,
                      const FGenerateOptions& Options) const;

    // ---- 들여다보기 ----

    const FModelConfig& GetConfig() const { return Config; }
    const FTokenizer& GetTokenizer() const { return Tokenizer; }

    size_t ParameterCount() const;
    size_t Bytes() const;

    bool IsReady() const { return bReady; }

    // ---- 속을 여는 자리 ----
    //
    // 훈련을 하려면 가중치에 직접 손이 닿아야 한다. 감추면 이 엔진으로
    // 훈련을 못 한다. **감출 것과 열 것을 가르는 것이 API 설계**다.
    FTransformer& MutableModel() { return Model; }
    FTokenizer& MutableTokenizer() { return Tokenizer; }

private:
    // 토큰 하나씩 뽑는 공통 부분.
    std::string Run(const std::vector<uint32_t>& Prompt,
                    const FGenerateOptions& Options, bool bStopAtEnd) const;

    FModelConfig Config;
    FTokenizer Tokenizer;
    FTransformer Model;
    bool bReady = false;
};

#endif
