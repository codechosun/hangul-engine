"""
한국어 위키백과 스냅샷을 받아 하나의 텍스트 파일로 만든다.

    python tools/download_corpus.py

결과: data/corpus.txt  (UTF-8, 문서 하나당 한 줄 이상)

주의
----
스냅샷 버전은 **고정**되어 있다. 이 값을 바꾸면 data/expected.csv 와
어긋나서 교재의 검증이 전부 실패한다. 건드리지 말 것.

위키백과는 계속 갱신되므로 "최신"을 받으면 사람마다 결과가 달라진다.
교재가 검증 가능하려면 모두가 같은 코퍼스를 봐야 한다.
"""

import os
import sys

# ---- 고정 스냅샷 (건드리지 말 것) ----
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.ko"

OUT_DIR = "data"
OUT_PATH = os.path.join(OUT_DIR, "corpus.txt")


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets 패키지가 필요하다:")
        print("    pip install datasets")
        return 1

    print(f"스냅샷 : {DATASET} / {CONFIG}")
    print("내려받는 중. 처음 한 번은 오래 걸린다.\n")

    Rows = load_dataset(DATASET, CONFIG, split="train")

    os.makedirs(OUT_DIR, exist_ok=True)

    DocCount = 0
    CharCount = 0

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as Out:
        for Row in Rows:
            Text = Row["text"]
            Out.write(Text)
            Out.write("\n")

            DocCount += 1
            CharCount += len(Text) + 1

            if DocCount % 50000 == 0:
                print(f"  {DocCount:,}개 문서 처리")

    print()
    print(f"완료 : {OUT_PATH}")
    print(f"  문서 {DocCount:,}개")
    print(f"  글자 {CharCount:,}개")
    print(f"  크기 {os.path.getsize(OUT_PATH):,} 바이트")
    return 0


if __name__ == "__main__":
    sys.exit(main())
