"""
corpus.txt 의 앞부분으로 N=1..6 그램 정답표를 만든다.

    python tools/make_ngram_expected.py

결과
----
    data/ngram_summary.csv    N 별 그램 수 / 서로 다른 수 / 한 번만 나온 수
    data/ngram_expected.csv   문맥이 전부 BOS 일 때의 분포 (N 별)
    data/ngram_context.csv    문맥이 " 대한민국" 의 꼬리일 때의 분포 (N 별)

왜 앞부분만 쓰는가
------------------
N 을 올리면 서로 다른 그램이 폭발한다. 전체 코퍼스로 6그램을 세면
메모리가 수십 GB 가 필요하다. 그래서 **앞 20만 줄**로 고정한다.
N 마다 같은 데이터를 쓰므로 N 사이의 비교는 공정하다.

문장을 만드는 규칙 (C 구현도 반드시 똑같이 할 것)
------------------------------------------------
줄 하나가 문장 하나다. 앞을 BOS 로 N-1 개 채우고 뒤에 EOS 를 하나 붙인다.

    N=3, "한국"  ->  [BOS][BOS] 한 국 [EOS]

이 규칙에서는 N 이 무엇이든 그램의 개수가 같다.

    그램 수 = 글자 수 + 문장 수
"""

import itertools
import os
import sys
from collections import Counter

IN_PATH = os.path.join("data", "corpus.txt")
SUMMARY_PATH = os.path.join("data", "ngram_summary.csv")
EXPECTED_PATH = os.path.join("data", "ngram_expected.csv")
CONTEXT_PATH = os.path.join("data", "ngram_context.csv")

MAX_LINES = 200000
MAX_ORDER = 6

TOKEN_BOS = 0x110000
TOKEN_EOS = 0x110001

# 코퍼스에 없는 것을 확인하고 고른 자리표.
# U+E000 은 실제로 코퍼스에 들어 있어서 쓸 수 없었다.
MARK_BOS = "\x00"
MARK_EOS = "\x01"

# 문맥 검사에 쓸 글자열. N 에 따라 뒤에서 N-1 글자를 잘라 쓴다.
CONTEXT_SOURCE = " 대한민국"


def ToToken(Char):
    if Char == MARK_BOS:
        return TOKEN_BOS
    if Char == MARK_EOS:
        return TOKEN_EOS
    return ord(Char)


def main():
    if not os.path.exists(IN_PATH):
        print(f"{IN_PATH} 가 없다. 먼저 실행할 것:")
        print("    python tools/download_corpus.py")
        return 1

    print(f"읽는 중 : {IN_PATH} (앞 {MAX_LINES:,}줄)")

    with open(IN_PATH, "r", encoding="utf-8") as In:
        Lines = [Line.rstrip("\n") for Line in itertools.islice(In, MAX_LINES)]

    for Line in Lines:
        if MARK_BOS in Line or MARK_EOS in Line:
            print("자리표로 쓴 글자가 코퍼스에 들어 있다. 다른 값을 고를 것.")
            return 1

    CharCount = sum(len(Line) for Line in Lines)
    LineCount = len(Lines)
    GramCount = CharCount + LineCount

    print(f"  문장 {LineCount:,}개, 글자 {CharCount:,}개")
    print(f"  그램은 N 과 무관하게 {GramCount:,}개여야 한다\n")

    Summary = [
        ("Lines", LineCount),
        ("Chars", CharCount),
        ("Grams", GramCount),
    ]

    ExpectedRows = []
    ContextRows = []

    for Order in range(1, MAX_ORDER + 1):
        Tally = Counter()

        for Line in Lines:
            Sequence = MARK_BOS * (Order - 1) + Line + MARK_EOS
            Tally.update(
                Sequence[i:i + Order]
                for i in range(len(Sequence) - Order + 1)
            )

        Grams = sum(Tally.values())
        Distinct = len(Tally)
        Once = sum(1 for Value in Tally.values() if Value == 1)
        Contexts = len({Gram[:-1] for Gram in Tally})

        assert Grams == GramCount, (Order, Grams, GramCount)

        Summary.append((f"Distinct{Order}", Distinct))
        Summary.append((f"Once{Order}", Once))
        Summary.append((f"Contexts{Order}", Contexts))

        print(f"  N={Order}  서로 다른 {Distinct:>9,}  "
              f"한 번만 {Once:>9,} ({Once * 100.0 / Distinct:4.1f}%)  "
              f"문맥 {Contexts:>9,}")

        # 문맥이 전부 BOS 인 경우 = 문장의 첫 글자
        AllBos = MARK_BOS * (Order - 1)
        for Gram, Count in Tally.items():
            if Gram[:-1] == AllBos:
                ExpectedRows.append((Order, ToToken(Gram[-1]), Count))

        # 문맥이 " 대한민국" 의 꼬리인 경우
        Tail = CONTEXT_SOURCE[-(Order - 1):] if Order > 1 else ""
        for Gram, Count in Tally.items():
            if Gram[:-1] == Tail:
                ContextRows.append((Order, ToToken(Gram[-1]), Count))

        del Tally

    print()

    os.makedirs("data", exist_ok=True)

    with open(SUMMARY_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Key,Value\n")
        for Key, Value in Summary:
            Out.write(f"{Key},{Value}\n")

    ExpectedRows.sort()
    ContextRows.sort()

    with open(EXPECTED_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Order,Next,Count\n")
        for Row in ExpectedRows:
            Out.write(f"{Row[0]},{Row[1]},{Row[2]}\n")

    with open(CONTEXT_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Order,Next,Count\n")
        for Row in ContextRows:
            Out.write(f"{Row[0]},{Row[1]},{Row[2]}\n")

    print(f"완료 : {SUMMARY_PATH}   {len(Summary)}줄")
    print(f"완료 : {EXPECTED_PATH}  {len(ExpectedRows)}줄")
    print(f"완료 : {CONTEXT_PATH}   {len(ContextRows)}줄")
    print()
    print(f"  문맥 글자열 : {CONTEXT_SOURCE!r}")
    for Order in range(2, MAX_ORDER + 1):
        Tail = CONTEXT_SOURCE[-(Order - 1):]
        Kinds = sum(1 for Row in ContextRows if Row[0] == Order)
        Sum = sum(Row[2] for Row in ContextRows if Row[0] == Order)
        print(f"    N={Order}  문맥 {Tail!r:>12s}  뒤에 온 글자 {Kinds:>5,}종  "
              f"총 {Sum:>9,}번")

    return 0


if __name__ == "__main__":
    sys.exit(main())
