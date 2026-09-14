"""
C6 이 내보낸 것을 읽어 그림을 그리고, C++ 의 PCA 를 검산한다.

    python tools/plot_embedding.py

읽는 것
-------
    data/embedding.npy   훈련한 임베딩
    data/vocab.csv       번호 -> 글자
    data/pca_cpp.csv     C++ 이 계산한 2차원 좌표
    data/loss_curve.csv  에폭별 손실

만드는 것
---------
    data/embedding.png   임베딩 2차원 그림
    data/loss_curve.png  손실 곡선

검산
----
NumPy 로 같은 PCA 를 다시 해서 C++ 결과와 맞춰본다.
고유벡터는 부호가 반대여도 똑같이 답이므로, 양쪽 다 **절대값이 가장 큰
성분을 양수로** 만들어 방향을 못박는다. C++ 쪽 lib/Pca.cpp 도 같은 규칙이다.
"""

import csv
import os
import sys

import numpy as np

EMBEDDING = os.path.join("data", "embedding.npy")
VOCAB = os.path.join("data", "vocab.csv")
PCA_CPP = os.path.join("data", "pca_cpp.csv")
LOSS = os.path.join("data", "loss_curve.csv")

PLOT_EMBEDDING = os.path.join("data", "embedding.png")
PLOT_LOSS = os.path.join("data", "loss_curve.png")

FIRST_CHAR = 3


def FixSign(Vector):
    """절대값이 가장 큰 성분을 양수로 만든다."""
    Biggest = int(np.argmax(np.abs(Vector)))
    if Vector[Biggest] < 0:
        return -Vector
    return Vector


def LoadVocab():
    Kinds = {}
    Codes = {}
    with open(VOCAB, encoding="utf-8") as In:
        Reader = csv.DictReader(In)
        for Row in Reader:
            Id = int(Row["Id"])
            Codes[Id] = int(Row["Codepoint"])
            Kinds[Id] = Row["Kind"]
    return Codes, Kinds


def main():
    for Path in (EMBEDDING, VOCAB, PCA_CPP, LOSS):
        if not os.path.exists(Path):
            print(f"{Path} 가 없다. C6 예제를 먼저 돌릴 것.")
            return 1

    Embedding = np.load(EMBEDDING).astype(np.float64)
    Codes, Kinds = LoadVocab()

    print(f"임베딩 : 모양 {Embedding.shape}, 타입 {np.load(EMBEDDING).dtype}")

    # ---- 파이썬으로 다시 PCA ----
    Rows = Embedding[FIRST_CHAR:]
    Centered = Rows - Rows.mean(axis=0)

    Covariance = (Centered.T @ Centered) / (Rows.shape[0] - 1)
    Values, Vectors = np.linalg.eigh(Covariance)

    # eigh 는 오름차순이므로 뒤집는다.
    Order = np.argsort(Values)[::-1]
    Values = Values[Order]
    Vectors = Vectors[:, Order]

    First = FixSign(Vectors[:, 0])
    Second = FixSign(Vectors[:, 1])

    Mine = np.stack([Centered @ First, Centered @ Second], axis=1)

    # ---- C++ 결과를 읽어 비교 ----
    Theirs = np.zeros_like(Mine)
    Ids = []
    with open(PCA_CPP, encoding="utf-8") as In:
        Reader = csv.DictReader(In)
        for i, Row in enumerate(Reader):
            Ids.append(int(Row["Id"]))
            Theirs[i, 0] = float(Row["X"])
            Theirs[i, 1] = float(Row["Y"])

    Worst = float(np.max(np.abs(Mine - Theirs)))
    Scale = float(np.max(np.abs(Mine)))

    print()
    print("PCA 검산")
    print(f"  설명 분산   1주성분 {Values[0] / Values.sum() * 100:.2f}%, "
          f"2주성분까지 {(Values[0] + Values[1]) / Values.sum() * 100:.2f}%")
    print(f"  좌표 최대 차이 = {Worst:.3e}  (좌표 크기 {Scale:.3f})")
    print(f"  상대 오차      = {Worst / Scale:.3e}")

    bMatched = (Worst / Scale) < 1e-4
    print(f"  판정           = {'일치' if bMatched else '어긋남'}")

    # ---- 그림 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as Plot
        from matplotlib import font_manager
    except ImportError:
        print()
        print("matplotlib 이 없어 그림은 건너뛴다.")
        print("    .venv/Scripts/python -m pip install matplotlib")
        return 0 if bMatched else 1

    Available = {Font.name for Font in font_manager.fontManager.ttflist}
    for Name in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        if Name in Available:
            Plot.rcParams["font.family"] = Name
            break
    Plot.rcParams["axes.unicode_minus"] = False

    Colors = {
        "hangul": "#3b6ea5",
        "digit": "#c0392b",
        "ascii": "#7f8c8d",
        "space": "#27ae60",
        "other": "#8e44ad",
    }

    Figure, Axes = Plot.subplots(figsize=(11, 9))

    for Kind, Color in Colors.items():
        Picked = [i for i, Id in enumerate(Ids) if Kinds.get(Id) == Kind]
        if not Picked:
            continue
        Axes.scatter(Theirs[Picked, 0], Theirs[Picked, 1], s=14, c=Color,
                     label=Kind, alpha=0.75, linewidths=0)

    # 눈에 띄는 것 몇 개에 글자를 적는다.
    Labelled = 0
    for i, Id in enumerate(Ids):
        Code = Codes.get(Id, 0)
        if Kinds.get(Id) in ("digit", "ascii", "space") or Labelled < 40:
            Text = chr(Code) if Code > 32 else " "
            Axes.annotate(Text, (Theirs[i, 0], Theirs[i, 1]), fontsize=8,
                          alpha=0.8)
            Labelled += 1

    Axes.set_title("NPLM 임베딩 (16차원 -> 2차원, PCA)")
    Axes.set_xlabel(f"1주성분 ({Values[0] / Values.sum() * 100:.1f}%)")
    Axes.set_ylabel(f"2주성분 ({Values[1] / Values.sum() * 100:.1f}%)")
    Axes.legend(loc="best")
    Axes.grid(True, alpha=0.2)

    Figure.tight_layout()
    Figure.savefig(PLOT_EMBEDDING, dpi=120)
    Plot.close(Figure)

    # ---- 손실 곡선 ----
    Epochs, TrainLoss, ValidLoss = [], [], []
    with open(LOSS, encoding="utf-8") as In:
        Reader = csv.DictReader(In)
        for Row in Reader:
            Epochs.append(int(Row["Epoch"]))
            TrainLoss.append(float(Row["TrainLoss"]))
            ValidLoss.append(float(Row["ValidLoss"]))

    Figure, Axes = Plot.subplots(figsize=(8, 5))
    Axes.plot(Epochs, TrainLoss, marker="o", label="훈련 손실")
    Axes.plot(Epochs, ValidLoss, marker="s", label="검증 손실")
    Axes.axhline(np.log(512), color="gray", linestyle="--", alpha=0.6,
                 label="log(어휘) = 시작점")
    Axes.set_title("NPLM 손실 곡선")
    Axes.set_xlabel("에폭")
    Axes.set_ylabel("교차 엔트로피")
    Axes.legend()
    Axes.grid(True, alpha=0.2)

    Figure.tight_layout()
    Figure.savefig(PLOT_LOSS, dpi=120)
    Plot.close(Figure)

    print()
    print(f"완료 : {PLOT_EMBEDDING}")
    print(f"완료 : {PLOT_LOSS}")

    return 0 if bMatched else 1


if __name__ == "__main__":
    sys.exit(main())
