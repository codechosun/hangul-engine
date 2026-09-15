"""
작은 트랜스포머를 NumPy 로 **훈련**시켜 걸음마다의 손실을 정답표로 남긴다.

    python tools/make_training_expected.py

결과: data/training_expected.csv

무엇을 대조하려는 것인가
------------------------
D6 에서 역전파가 맞다는 것은 gradcheck 으로 증명했다. D7 에서 새로 붙는 것은
**옵티마이저**다. 그런데 손실 곡선을 대조하면 셋을 한꺼번에 검사하게 된다.

    순전파  틀리면 첫 걸음의 손실부터 어긋난다
    역전파  틀리면 두 걸음째부터 서서히 갈라진다
    옵티마이저  틀리면 곡선의 모양이 다르다

30걸음을 돌려 매 걸음의 손실이 맞으면 셋 다 맞는 것이다.

왜 PyTorch 가 아닌가
--------------------
D4 의 make_transformer_expected.py 와 같은 이유다. 200MB 짜리 의존성을
더 붙이지 않는다. 대신 NumPy 로 참조 구현을 따로 쓴다.

한계도 같다. **같은 사람이 두 번 쓰면 같은 오해를 두 번 쓸 수 있다.**
그래서 이 파일은 C++ 코드를 보지 않고 공개된 수식에서 바로 쓴다. 그리고
C++ 쪽은 for 루프로, 이쪽은 행렬 연산으로 — 모양이 다르게 쓴다.

가중치
------
D4 와 똑같은 정수 나눗셈 공식. 두 언어에서 비트까지 같은 값이 나온다.
"""

import os
import sys

import numpy as np

OUT_PATH = os.path.join("data", "training_expected.csv")

VOCAB = 11
MODEL = 8
HEADS = 2
HIDDEN = 16
LAYERS = 2
MAX_LENGTH = 12

TOKENS = [3, 7, 1, 9, 0, 4]
TARGETS = [7, 1, 9, 0, 4, 2]

EPSILON = 1e-6

STEPS = 30
RATE = 0.05
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPSILON = 1e-8
DECAY = 0.01
CLIP = 1.0
WARMUP = 3
MIN_RATIO = 0.1


def Fill(Count, Multiplier, Offset, Modulus, Half, Divisor):
    Values = np.empty(Count, dtype=np.float64)
    for i in range(Count):
        Values[i] = ((i * Multiplier + Offset) % Modulus - Half) / Divisor
    return Values


def RmsNorm(X, Gain):
    Scale = 1.0 / np.sqrt(np.mean(X * X, axis=-1, keepdims=True) + EPSILON)
    return X * Scale * Gain, Scale


def RmsNormBackward(X, Gain, Scale, Upstream):
    N = X.shape[-1]
    GainGrad = np.sum(Upstream * X * Scale, axis=0)
    Coupled = np.sum(Upstream * Gain * X, axis=-1, keepdims=True)
    XGrad = Gain * Scale * Upstream - (Scale ** 3 / N) * X * Coupled
    return XGrad, GainGrad


def Sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def Silu(X):
    return X * Sigmoid(X)


def SiluBackward(X, Upstream):
    S = Sigmoid(X)
    return Upstream * S * (1.0 + X * (1.0 - S))


def Softmax(X):
    Shifted = X - np.max(X, axis=-1, keepdims=True)
    Exps = np.exp(Shifted)
    return Exps / np.sum(Exps, axis=-1, keepdims=True)


def BuildWeights():
    State = {
        "TokenEmbedding": Fill(VOCAB * MODEL, 31, 17, 41, 20, 40.0).reshape(
            VOCAB, MODEL),
        "PositionEmbedding": Fill(MAX_LENGTH * MODEL, 23, 11, 37, 18,
                                  36.0).reshape(MAX_LENGTH, MODEL),
        "Head": Fill(MODEL * VOCAB, 19, 7, 43, 21, 42.0).reshape(MODEL, VOCAB),
        "FinalGain": np.ones(MODEL),
    }

    for L in range(LAYERS):
        Base = (L + 1) * 7
        State[f"Q{L}"] = Fill(MODEL * MODEL, 13 + Base, 5, 29, 14,
                              28.0).reshape(MODEL, MODEL)
        State[f"K{L}"] = Fill(MODEL * MODEL, 17 + Base, 3, 31, 15,
                              30.0).reshape(MODEL, MODEL)
        State[f"V{L}"] = Fill(MODEL * MODEL, 11 + Base, 9, 37, 18,
                              36.0).reshape(MODEL, MODEL)
        State[f"P{L}"] = Fill(MODEL * MODEL, 23 + Base, 1, 41, 20,
                              40.0).reshape(MODEL, MODEL)
        State[f"Up{L}"] = Fill(MODEL * HIDDEN, 29 + Base, 13, 43, 21,
                               42.0).reshape(MODEL, HIDDEN)
        State[f"Down{L}"] = Fill(HIDDEN * MODEL, 31 + Base, 7, 47, 23,
                                 46.0).reshape(HIDDEN, MODEL)
        State[f"AttentionGain{L}"] = np.ones(MODEL)
        State[f"FeedGain{L}"] = np.ones(MODEL)

    return State


def ForwardBackward(W):
    """손실과 모든 파라미터의 그래디언트를 돌려준다."""

    Head = MODEL // HEADS
    Length = len(TOKENS)
    Tokens = np.array(TOKENS)
    Targets = np.array(TARGETS)

    Mask = np.triu(np.ones((Length, Length), dtype=bool), k=1)

    # ---- 순전파. 중간값을 전부 들고 있는다 ----
    X = W["TokenEmbedding"][Tokens] + W["PositionEmbedding"][:Length]

    Trace = []

    for L in range(LAYERS):
        Input = X

        Normed, NormedScale = RmsNorm(X, W[f"AttentionGain{L}"])

        Qm = Normed @ W[f"Q{L}"]
        Km = Normed @ W[f"K{L}"]
        Vm = Normed @ W[f"V{L}"]

        Q = Qm.reshape(Length, HEADS, Head).transpose(1, 0, 2)
        K = Km.reshape(Length, HEADS, Head).transpose(1, 0, 2)
        V = Vm.reshape(Length, HEADS, Head).transpose(1, 0, 2)

        Scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(Head)
        Scores = np.where(Mask[None, :, :], -1e30, Scores)

        Probabilities = Softmax(Scores)
        Attended = Probabilities @ V

        Merged = Attended.transpose(1, 0, 2).reshape(Length, MODEL)
        After = Input + Merged @ W[f"P{L}"]

        FeedNormed, FeedScale = RmsNorm(After, W[f"FeedGain{L}"])
        Up = FeedNormed @ W[f"Up{L}"]
        Activated = Silu(Up)
        X = After + Activated @ W[f"Down{L}"]

        Trace.append({
            "Input": Input, "Normed": Normed, "NormedScale": NormedScale,
            "Q": Q, "K": K, "V": V, "P": Probabilities,
            "Merged": Merged, "After": After,
            "FeedNormed": FeedNormed, "FeedScale": FeedScale,
            "Up": Up, "Activated": Activated,
        })

    Final = X
    Normed, FinalScale = RmsNorm(Final, W["FinalGain"])
    Logits = Normed @ W["Head"]

    # ---- 손실 ----
    Probabilities = Softmax(Logits)
    Loss = float(-np.mean(np.log(Probabilities[np.arange(Length), Targets])))

    # ---- 역전파 ----
    Grad = {Name: np.zeros_like(Value) for Name, Value in W.items()}

    UpstreamLogits = Probabilities.copy()
    UpstreamLogits[np.arange(Length), Targets] -= 1.0
    UpstreamLogits /= Length

    Grad["Head"] += Normed.T @ UpstreamLogits
    UpstreamNormed = UpstreamLogits @ W["Head"].T

    D, GainGrad = RmsNormBackward(Final, W["FinalGain"], FinalScale,
                                  UpstreamNormed)
    Grad["FinalGain"] += GainGrad

    for L in reversed(range(LAYERS)):
        T = Trace[L]

        Grad[f"Down{L}"] += T["Activated"].T @ D
        DActivated = D @ W[f"Down{L}"].T

        DUp = SiluBackward(T["Up"], DActivated)

        Grad[f"Up{L}"] += T["FeedNormed"].T @ DUp
        DFeedNormed = DUp @ W[f"Up{L}"].T

        DAfter, GainGrad = RmsNormBackward(T["After"], W[f"FeedGain{L}"],
                                           T["FeedScale"], DFeedNormed)
        Grad[f"FeedGain{L}"] += GainGrad
        DAfter = DAfter + D          # 잔차

        Grad[f"P{L}"] += T["Merged"].T @ DAfter
        DMerged = DAfter @ W[f"P{L}"].T

        DInput = DAfter              # 잔차

        Head = MODEL // HEADS
        DAttended = DMerged.reshape(Length, HEADS, Head).transpose(1, 0, 2)

        DV = T["P"].transpose(0, 2, 1) @ DAttended
        DP = DAttended @ T["V"].transpose(0, 2, 1)

        Dot = np.sum(DP * T["P"], axis=-1, keepdims=True)
        DScores = T["P"] * (DP - Dot)

        DQ = (DScores @ T["K"]) / np.sqrt(Head)
        DK = (DScores.transpose(0, 2, 1) @ T["Q"]) / np.sqrt(Head)

        DQm = DQ.transpose(1, 0, 2).reshape(Length, MODEL)
        DKm = DK.transpose(1, 0, 2).reshape(Length, MODEL)
        DVm = DV.transpose(1, 0, 2).reshape(Length, MODEL)

        Grad[f"Q{L}"] += T["Normed"].T @ DQm
        Grad[f"K{L}"] += T["Normed"].T @ DKm
        Grad[f"V{L}"] += T["Normed"].T @ DVm

        DNormed = (DQm @ W[f"Q{L}"].T + DKm @ W[f"K{L}"].T
                 + DVm @ W[f"V{L}"].T)

        DX, GainGrad = RmsNormBackward(T["Input"], W[f"AttentionGain{L}"],
                                       T["NormedScale"], DNormed)
        Grad[f"AttentionGain{L}"] += GainGrad

        D = DX + DInput              # 잔차

    # 임베딩. 같은 토큰이 여러 번 나오면 **더해진다.**
    np.add.at(Grad["TokenEmbedding"], np.array(TOKENS), D)
    Grad["PositionEmbedding"][:Length] += D

    return Loss, Grad


def main():
    W = BuildWeights()
    Names = sorted(W.keys())

    Moment = {Name: np.zeros_like(W[Name]) for Name in Names}
    Velocity = {Name: np.zeros_like(W[Name]) for Name in Names}

    Rows = []

    for Step in range(STEPS):
        Loss, Grad = ForwardBackward(W)

        # ---- 클리핑 ----
        Square = sum(float(np.sum(Grad[Name] * Grad[Name])) for Name in Names)
        Norm = np.sqrt(Square)

        if CLIP > 0.0 and Norm > CLIP:
            Shrink = CLIP / Norm
            for Name in Names:
                Grad[Name] = Grad[Name] * Shrink

        # ---- 스케줄 ----
        if Step < WARMUP:
            Rate = RATE * ((Step + 1) / WARMUP)
        else:
            Progress = (Step - WARMUP) / (STEPS - WARMUP)
            Cosine = 0.5 * (1.0 + np.cos(np.pi * Progress))
            Rate = RATE * (MIN_RATIO + (1.0 - MIN_RATIO) * Cosine)

        # ---- AdamW ----
        Count = Step + 1
        Correct1 = 1.0 - BETA1 ** Count
        Correct2 = 1.0 - BETA2 ** Count

        for Name in Names:
            G = Grad[Name]
            Moment[Name] = BETA1 * Moment[Name] + (1.0 - BETA1) * G
            Velocity[Name] = BETA2 * Velocity[Name] + (1.0 - BETA2) * G * G

            M = Moment[Name] / Correct1
            V = Velocity[Name] / Correct2

            Old = W[Name]
            W[Name] = Old - Rate * M / (np.sqrt(V) + ADAM_EPSILON) \
                          - Rate * DECAY * Old

        Rows.append((Step, Loss, float(Norm), float(Rate)))

    # 마지막으로 한 번 더 재서 끝 손실을 남긴다.
    FinalLoss, _ = ForwardBackward(W)
    Rows.append((STEPS, FinalLoss, 0.0, 0.0))

    os.makedirs("data", exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as Out:
        Out.write("Step,Loss,GradNorm,Rate\n")
        for Step, Loss, Norm, Rate in Rows:
            Out.write(f"{Step},{Loss:.17g},{Norm:.17g},{Rate:.17g}\n")

    print(f"완료 : {OUT_PATH}  {len(Rows)}줄")
    print()
    print(f"  어휘 {VOCAB}, 모델차원 {MODEL}, 헤드 {HEADS}, "
          f"앞먹임 {HIDDEN}, 층 {LAYERS}")
    print(f"  토큰 {TOKENS}")
    print(f"  정답 {TARGETS}")
    print()
    print(f"  학습률 {RATE}, 워밍업 {WARMUP}, 감쇠 {DECAY}, 클리핑 {CLIP}")
    print()
    print("  걸음     손실     그래디언트 크기     학습률")
    for Step, Loss, Norm, Rate in Rows[:5]:
        print(f"  {Step:4d}  {Loss:9.6f}  {Norm:15.6f}  {Rate:9.6f}")
    print("   ...")
    for Step, Loss, Norm, Rate in Rows[-3:]:
        print(f"  {Step:4d}  {Loss:9.6f}  {Norm:15.6f}  {Rate:9.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
