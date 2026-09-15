// lib/Tokenizer.hpp
//
// 글자 단위 토크나이저.
//
// D7 에서 손으로 하던 일 — 코퍼스에서 자주 나오는 글자를 세고, 번호를 붙이고,
// 되돌리고 — 을 한 자리에 모은다. E1 부터는 특수 토큰까지 얹어야 해서
// 흩어놓을 수가 없다.
//
// 왜 BPE 가 아닌가
//   BPE 는 학습이 파이썬 쪽 일이고(결정 26), 이 교재의 남은 분량으로는
//   제대로 다룰 수가 없다. 글자 단위는 어휘가 작고 구현이 짧으면서
//   **특수 토큰과 마스킹이라는 E1 의 주제를 그대로 보여준다.**
//
// 특수 토큰
//   보통 글자 뒤에 번호를 이어 붙인다. 코퍼스에 없는 번호이므로
//   본문과 절대 겹치지 않는다. 겹치면 사용자가 <|end|> 를 입력해서
//   모델을 멈추게 만들 수 있다 — 실제로 있었던 사고다.

#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class FTokenizer
{
public:
    // 텍스트에서 자주 나오는 글자 Limit 개를 뽑아 어휘를 만든다.
    // 그 뒤에 특수 토큰 SpecialCount 개를 이어 붙인다.
    void BuildFromText(const char* Text, size_t Bytes, size_t Limit,
                       size_t SpecialCount);

    // 이미 정해진 어휘를 그대로 받는다. 모델 파일에서 읽을 때 쓴다. (E2)
    void BuildFromCodepoints(const std::vector<uint32_t>& InCodepoints,
                             size_t InSpecialCount);

    // 번호 순서대로의 글자 목록. 모델 파일에 적을 때 쓴다. (E2)
    const std::vector<uint32_t>& Alphabet() const { return Codepoints; }

    size_t Size() const { return Codepoints.size() + SpecialCount; }
    size_t PlainCount() const { return Codepoints.size(); }
    size_t FirstSpecial() const { return Codepoints.size(); }

    // 글자 하나를 번호로. 어휘에 없으면 -1.
    int Find(uint32_t Codepoint) const;

    // 문자열을 번호 열로. 어휘에 없는 글자는 건너뛴다.
    // 건너뛴 글자 수를 돌려준다.
    size_t Encode(const char* Text, std::vector<uint32_t>& Out) const;
    size_t Encode(const std::string& Text, std::vector<uint32_t>& Out) const;

    // 번호 하나를 UTF-8 로. 특수 토큰이면 0 을 돌려준다.
    int Decode(uint32_t Id, char* OutBuffer) const;

    // 번호 열 전체를 문자열로. 특수 토큰은 <N> 으로 적는다.
    std::string DecodeAll(const std::vector<uint32_t>& Ids) const;

    // 덮은 비율. BuildFromText 가 채운다.
    double Coverage() const { return CoverageRatio; }

private:
    std::vector<uint32_t> Codepoints;   // 번호 -> 글자
    std::vector<int> Table;             // 글자 -> 번호
    size_t SpecialCount = 0;
    double CoverageRatio = 0.0;
};

#endif
