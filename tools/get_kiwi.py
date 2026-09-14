"""
A9 에 필요한 Kiwi 배포본을 내려받아 third_party/kiwi 에 푼다.

    python tools/get_kiwi.py

받는 것 (합쳐서 약 125MB, 풀면 약 500MB)
----------------------------------------
    kiwi_win_x64_v0.23.2.zip    C API 헤더 + import lib + DLL
    kiwi_model_v0.23.2_base.tgz 형태소 분석 모델

버전은 **고정**되어 있다. Kiwi 는 판올림마다 분석 결과가 조금씩 달라지므로,
버전을 바꾸면 교재에 적힌 숫자와 어긋난다.

윈도우 x64 전용이다. 다른 운영체제라면 릴리스 페이지에서 해당 파일을 받아
같은 자리에 풀면 된다.
    https://github.com/bab2min/Kiwi/releases/tag/v0.23.2
"""

import os
import sys
import tarfile
import urllib.request
import zipfile

VERSION = "v0.23.2"
BASE = f"https://github.com/bab2min/Kiwi/releases/download/{VERSION}"

FILES = [
    (f"kiwi_win_x64_{VERSION}.zip", "zip"),
    (f"kiwi_model_{VERSION}_base.tgz", "tar"),
]

OUT_DIR = os.path.join("third_party", "kiwi")


def Download(Name):
    Path = os.path.join("third_party", Name)
    if os.path.exists(Path):
        print(f"  이미 있음 : {Name}")
        return Path

    Url = f"{BASE}/{Name}"
    print(f"  받는 중   : {Name}")
    urllib.request.urlretrieve(Url, Path)
    return Path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Kiwi {VERSION} 를 {OUT_DIR} 에 설치한다\n")

    for Name, Kind in FILES:
        Path = Download(Name)

        print(f"  푸는 중   : {Name}")
        if Kind == "zip":
            with zipfile.ZipFile(Path) as Zip:
                Zip.extractall(OUT_DIR)
        else:
            with tarfile.open(Path, "r:gz") as Tar:
                Tar.extractall(OUT_DIR)

    print()

    Needed = [
        os.path.join(OUT_DIR, "include", "kiwi", "capi.h"),
        os.path.join(OUT_DIR, "lib", "kiwi.lib"),
        os.path.join(OUT_DIR, "lib", "kiwi.dll"),
        os.path.join(OUT_DIR, "models", "cong", "base"),
    ]

    Missing = [Path for Path in Needed if not os.path.exists(Path)]
    if Missing:
        print("빠진 것이 있다:")
        for Path in Missing:
            print(f"  {Path}")
        return 1

    print("완료. 확인한 것")
    for Path in Needed:
        print(f"  {Path}")

    print()
    print("빌드할 때 kiwi.dll 을 실행 파일 옆에 복사해야 한다. A9 본문 참고.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
