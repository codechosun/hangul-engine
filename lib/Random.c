// lib/Random.c

#include "Random.h"

#include <assert.h>

void RandomSeed(FRandom* Rng, uint64_t Seed)
{
    assert(Rng != NULL);

    // 0 은 쓸 수 없다. xorshift 는 0 에서 영원히 0 만 만든다.
    Rng->State = (Seed == 0) ? 0x9E3779B97F4A7C15ull : Seed;
}

uint64_t RandomNext(FRandom* Rng)
{
    assert(Rng != NULL);

    // xorshift64. 밀고 뒤집기 세 번이면 끝이다.
    uint64_t X = Rng->State;
    X ^= X << 13;
    X ^= X >> 7;
    X ^= X << 17;
    Rng->State = X;

    return X;
}

uint64_t RandomBelow(FRandom* Rng, uint64_t Limit)
{
    if (Limit == 0)
    {
        return 0;
    }

    // 나머지 연산은 아주 미세한 치우침을 만든다.
    // 2^64 를 Limit 으로 나눈 나머지만큼인데, 우리가 쓰는 Limit 은
    // 기껏해야 10억 단위라 치우침이 100억분의 1 수준이다. 무시해도 된다.
    return RandomNext(Rng) % Limit;
}

double RandomUnit(FRandom* Rng)
{
    // 위쪽 53비트를 뽑아 2^53 으로 나눈다.
    //
    // 아래 비트를 버리는 이유는 xorshift 의 하위 비트가 상위 비트보다
    // 덜 섞이기 때문이다. 그리고 double 의 가수부가 53비트라,
    // 그보다 많이 가져와봐야 어차피 버려진다.
    uint64_t Bits = RandomNext(Rng) >> 11;

    return (double)Bits / 9007199254740992.0;   // 2^53
}

double RandomRange(FRandom* Rng, double Range)
{
    return (RandomUnit(Rng) * 2.0 - 1.0) * Range;
}
