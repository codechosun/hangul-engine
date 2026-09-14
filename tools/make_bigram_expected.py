"""
corpus.txt 를 읽어 바이그램 정답표를 만든다.

    python tools/make_bigram_expected.py

결과
----
    data/bigram_summary.csv    전체 글자 수, 문장 수, 쌍의 개수
    data/bigram_expected.csv   정해진 몇 개 앞 글자에 대한 (앞, 뒤, 횟수)

왜 전부가 아니라 몇 개만인가
----------------------------
서로 다른 쌍은 백만 개가 넘는다. 전부 적으면 파일이 수십 MB가 되고
저장소에 넣기 곤란하다. 앞 글자 여섯 개만 고르되, 그 여섯 개에 대해서는
**빠짐없이 전부** 적는다. 일부만 맞는 구현은 통과할 수 없다.

문장을 나누는 규칙 (C 구현도 반드시 똑같이 할 것)
------------------------------------------------
줄바꿈 하나가 문장 하나의 끝이다. 줄마다 앞에 BOS, 뒤에 EOS 를 붙인다.

    "한국어\n" -> [BOS] 한 국 어 [EOS]

줄바꿈 자체는 글자로 세지 않는다. EOS 와 BOS 로 갈아 끼운다.
빈 줄은 [BOS][EOS] 쌍 하나가 된다.

이 규칙에서는 **쌍의 개수가 글자 수와 정확히 같다.**
줄바꿈 하나가 빠지고 EOS 하나가 들어오기 때문이다.
"""

import os
import re
import sys
from collections import Counter

IN_PATH = os.path.join("data", "corpus.txt")
SUMMARY_PATH = os.path.join("data", "bigram_summary.csv")
EXPECTED_PATH = os.path.join("data", "bigram_expected.csv")

# 특수 토큰. 유니코드 밖의 번호라 진짜 글자와 부딪히지 않는다.
TOKEN_BOS = 0x110000
TOKEN_EOS = 0x110001

# 정답표에 담을 앞 글자들. BOS 는 따로 다룬다.
CONTEXTS = [" ", ".", "의", "다", "한"]

# 한 번에 읽어들일 글자 수.
# 크게 잡으면 정규식 결과 목록이 메모리를 많이 먹는다.
CHUNK_SIZE = 2 * 1000 * 1000


def main():
    if not os.path.exists(IN_PATH):
        print(f"{IN_PATH} 가 없다. 먼저 실행할 것:")
        print("    python tools/download_corpus.py")
        return 1

    print(f"읽는 중 : {IN_PATH}")

    # 앞 글자 -> Counter(뒤 글자 코드포인트 -> 횟수)
    Tally = {ord(Context): Counter() for Context in CONTEXTS}
    Tally[TOKEN_BOS] = Counter()

    # "이 글자 바로 뒤의 한 글자" 를 찾는 정규식.
    # 뒤돌아보기(lookbehind)는 폭이 0이라 다음 글자를 소비하지 않는다.
    Patterns = {
        ord(Context): re.compile("(?<=" + re.escape(Context) + ")(.)", re.S)
        for Context in CONTEXTS
    }
    Patterns[TOKEN_BOS] = re.compile(r"(?<=\n)(.)", re.S)

    TotalChars = 0
    LineCount = 0
    Carry = ""
    First = True

    TotalBytes = os.path.getsize(IN_PATH)
    ReadBytes = 0

    with open(IN_PATH, "r", encoding="utf-8") as In:
        while True:
            Chunk = In.read(CHUNK_SIZE)
            if not Chunk:
                break

            TotalChars += len(Chunk)
            LineCount += Chunk.count("\n")

            if First:
                # 파일의 첫 글자는 앞에 줄바꿈이 없지만 문장의 시작이다.
                Tally[TOKEN_BOS][Chunk[0]] += 1
                First = False

            # 앞 청크의 마지막 글자를 붙여야 경계에 걸친 쌍을 놓치지 않는다.
            Text = Carry + Chunk
            Carry = Chunk[-1]

            for Key, Pattern in Patterns.items():
                Tally[Key].update(Pattern.findall(Text))

            ReadBytes += len(Chunk.encode("utf-8"))
            Percent = ReadBytes * 100 // max(TotalBytes, 1)
            print(f"\r  {Percent}%", end="", flush=True)

    print("\n")

    # 줄바꿈으로 끝나지 않는 파일이면 마지막 문장을 닫아준다.
    TrailingEos = 0 if Carry == "\n" else 1
    if TrailingEos:
        LineCount += 1

    PairCount = TotalChars + TrailingEos

    os.makedirs("data", exist_ok=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Key,Value\n")
        Out.write(f"Chars,{TotalChars}\n")
        Out.write(f"Lines,{LineCount}\n")
        Out.write(f"Pairs,{PairCount}\n")

    Rows = []
    for Prev in sorted(Tally.keys()):
        for Char, Count in Tally[Prev].items():
            Next = TOKEN_EOS if Char == "\n" else ord(Char)
            Rows.append((Prev, Next, Count))

    # 앞 글자 오름차순, 같으면 뒤 글자 오름차순. C 구현과 같은 규칙이다.
    Rows.sort()

    with open(EXPECTED_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Prev,Next,Count\n")
        for Prev, Next, Count in Rows:
            Out.write(f"{Prev},{Next},{Count}\n")

    print(f"완료 : {SUMMARY_PATH}")
    print(f"  글자 {TotalChars:,}개")
    print(f"  문장 {LineCount:,}개")
    print(f"  쌍   {PairCount:,}개")
    print()
    print(f"완료 : {EXPECTED_PATH}")
    print(f"  줄 {len(Rows):,}개")
    for Prev in sorted(Tally.keys()):
        Name = "BOS" if Prev == TOKEN_BOS else repr(chr(Prev))
        Kinds = len(Tally[Prev])
        Sum = sum(Tally[Prev].values())
        print(f"  {Name:>6s}  다음에 온 글자 {Kinds:>6,}종  총 {Sum:>12,}번")

    return 0


if __name__ == "__main__":
    sys.exit(main())
