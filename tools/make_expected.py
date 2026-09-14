"""
corpus.txt 를 읽어 글자 빈도 정답표를 만든다.

    python tools/make_expected.py

결과: data/expected.csv

정렬 규칙 (C 구현도 반드시 똑같이 할 것)
----------------------------------------
    1순위  빈도 내림차순
    2순위  코드포인트 오름차순

2순위가 왜 필요한가. 빈도가 같은 글자가 수천 개씩 나온다.
동점일 때 순서를 못박지 않으면 정렬 구현에 따라 줄 순서가 달라지고,
그러면 두 파일을 그대로 비교할 수 없다.

세는 단위
---------
유니코드 코드포인트 하나를 한 글자로 본다.
공백과 구두점도 세지 않고 빼지 않는다. 있는 그대로 전부 센다.
"""

import os
import sys
from collections import Counter

IN_PATH = os.path.join("data", "corpus.txt")
OUT_PATH = os.path.join("data", "expected.csv")

# 한 번에 읽어들일 크기. 코퍼스가 크므로 통째로 읽지 않는다.
CHUNK_SIZE = 4 * 1024 * 1024


def main():
    if not os.path.exists(IN_PATH):
        print(f"{IN_PATH} 가 없다. 먼저 실행할 것:")
        print("    python tools/download_corpus.py")
        return 1

    print(f"읽는 중 : {IN_PATH}")

    Freq = Counter()
    ReadBytes = 0
    TotalBytes = os.path.getsize(IN_PATH)

    with open(IN_PATH, "r", encoding="utf-8") as In:
        while True:
            Chunk = In.read(CHUNK_SIZE)
            if not Chunk:
                break

            Freq.update(Chunk)

            ReadBytes += len(Chunk.encode("utf-8"))
            Percent = ReadBytes * 100 // max(TotalBytes, 1)
            print(f"\r  {Percent}%", end="", flush=True)

    print("\n")

    # 빈도 내림차순, 동점이면 코드포인트 오름차순
    Sorted = sorted(Freq.items(), key=lambda Item: (-Item[1], ord(Item[0])))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Codepoint,Count\n")
        for Char, Count in Sorted:
            Out.write(f"{ord(Char)},{Count}\n")

    TotalChars = sum(Freq.values())

    print(f"완료 : {OUT_PATH}")
    print(f"  서로 다른 글자 {len(Sorted):,}종")
    print(f"  전체 글자 {TotalChars:,}개")
    print()
    print("  상위 10종")
    for Rank, (Char, Count) in enumerate(Sorted[:10], start=1):
        Name = repr(Char)[1:-1] if Char.isprintable() else f"U+{ord(Char):04X}"
        Share = Count * 100.0 / TotalChars
        print(f"    {Rank:2d}. {Name:8s} U+{ord(Char):04X}  {Count:>12,}  {Share:5.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
