// ch-C04-npy/Main.cpp
//
// C4. 데이터 경계 — .npy 리더
//
// 저장소 루트에서 실행할 것.
//     Main.exe

#include "Test.h"
#include "Pretty.h"

#include "Matrix.hpp"
#include "Npy.hpp"
#include "Random.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#define FROM_PYTHON "data/npy_from_python.npy"
#define FROM_CPP    "data/npy_from_cpp.npy"
#define VALUES_CSV  "data/npy_values.csv"
#define BIG_NPY     "data/npy_big.npy"
#define BIG_CSV     "data/npy_big.csv"

#define BOOK_SEED 20260914ull

#define BIG_ROWS 2000
#define BIG_COLS 64

namespace
{

void PrintBytes(const char* Path, size_t Count)
{
    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return;
    }

    std::vector<unsigned char> Buffer(Count, 0);
    size_t Read = fread(Buffer.data(), 1, Count, File);
    fclose(File);

    for (size_t i = 0; i < Read; i += 16)
    {
        printf("    %04zu  ", i);

        for (size_t k = 0; k < 16; k++)
        {
            if (i + k < Read) printf("%02X ", Buffer[i + k]);
            else              printf("   ");
        }

        printf(" |");
        for (size_t k = 0; k < 16 && i + k < Read; k++)
        {
            unsigned char C = Buffer[i + k];
            printf("%c", (C >= 0x20 && C < 0x7F) ? (char)C : '.');
        }
        printf("|\n");
    }
}

double FileSize(const char* Path)
{
    FILE* File = fopen(Path, "rb");
    if (File == NULL)
    {
        return 0.0;
    }
    fseek(File, 0, SEEK_END);
    long Size = ftell(File);
    fclose(File);
    return (double)Size;
}

} // namespace

int main(void)
{
    printf("C4. 데이터 경계 — .npy 리더\n\n");
    printf("  Real = %s\n\n", (sizeof(Real) == 8) ? "double" : "float");

    // ---- C4-1. 파일을 바이트로 들여다본다 ----
    printf("[C4-1] 파이썬이 쓴 .npy 의 앞부분\n\n");

    PrintBytes(FROM_PYTHON, 80);
    printf("\n");

    // ---- C4-2. 헤더를 읽는다 ----
    printf("[C4-2] 헤더 파싱\n\n");

    FNpyInfo Info;
    if (!CHECK(NpyReadInfo(FROM_PYTHON, Info)))
    {
        printf("  %s 가 없다. tools/make_npy_expected.py 를 먼저 돌릴 것.\n",
               FROM_PYTHON);
        return ReportResult();
    }

    printf("  descr          = %s\n", Info.Descr.c_str());
    printf("  fortran_order  = %s\n", Info.bFortranOrder ? "True" : "False");
    printf("  shape          = (%zu, %zu)\n", Info.Rows, Info.Cols);
    printf("  원소 크기      = %zu 바이트\n", Info.ElementSize);
    printf("  헤더 길이      = %zu (전체 %zu, 64의 배수)\n\n",
           Info.HeaderLength, 10 + Info.HeaderLength);

    CHECK(Info.Descr == "<f4");
    CHECK(Info.bFortranOrder == false);
    CHECK(Info.Rows == 5);
    CHECK(Info.Cols == 7);
    CHECK(Info.ElementSize == 4);
    CHECK((10 + Info.HeaderLength) % 64 == 0);

    // ---- C4-3. 값을 읽어 CSV 와 대조 ----
    printf("[C4-3] 값을 읽어 파이썬이 적어준 표와 맞춰본다\n\n");

    FMatrix Loaded;
    if (!CHECK(NpyLoadMatrix(FROM_PYTHON, Loaded)))
    {
        return ReportResult();
    }

    {
        FILE* File = fopen(VALUES_CSV, "rb");
        if (!CHECK(File != NULL))
        {
            return ReportResult();
        }

        char Line[256];
        (void)fgets(Line, sizeof(Line), File);

        int Rows = 0;
        int Bad = 0;
        double WorstGap = 0.0;

        while (fgets(Line, sizeof(Line), File) != NULL)
        {
            int Row = 0, Col = 0;
            double Value = 0.0;

            if (sscanf(Line, "%d,%d,%lf", &Row, &Col, &Value) != 3)
            {
                continue;
            }

            double Mine = (double)Loaded((size_t)Row, (size_t)Col);
            double Gap = std::fabs(Mine - Value);

            if (Gap > WorstGap) WorstGap = Gap;
            if (Gap != 0.0)     Bad++;

            Rows++;
        }

        fclose(File);

        printf("  %d개 값 대조 -> 어긋난 것 %d개, 최대 차이 %.1e\n\n",
               Rows, Bad, WorstGap);

        CHECK(Rows == 35);
        CHECK(Bad == 0);   // 이진 파일이므로 **정확히** 같아야 한다
    }

    printf("  읽은 값\n");
    for (size_t r = 0; r < Loaded.Rows(); r++)
    {
        printf("   ");
        for (size_t c = 0; c < Loaded.Cols(); c++)
        {
            printf("%9.5f", (double)Loaded(r, c));
        }
        printf("\n");
    }
    printf("\n");

    // ---- C4-4. 우리가 써서 파이썬에게 넘긴다 ----
    printf("[C4-4] 같은 값을 C++ 이 써본다\n\n");

    if (!CHECK(NpySaveMatrix(FROM_CPP, Loaded)))
    {
        return ReportResult();
    }

    printf("  %s 를 썼다 (%.0f 바이트)\n\n", FROM_CPP, FileSize(FROM_CPP));
    PrintBytes(FROM_CPP, 16);

    // 다시 읽어 왕복이 되는지 본다.
    {
        FMatrix Again;
        CHECK(NpyLoadMatrix(FROM_CPP, Again));
        CHECK(Again.Rows() == Loaded.Rows());
        CHECK(Again.Cols() == Loaded.Cols());

        int Same = 1;
        for (size_t i = 0; i < Loaded.Count(); i++)
        {
            if (Again.Data()[i] != Loaded.Data()[i])
            {
                Same = 0;
                break;
            }
        }

        printf("\n  왕복해서 바이트까지 같은가 = %s\n\n", Same ? "예" : "아니오");
        CHECK(Same == 1);
    }

    // ---- C4-5. 망가진 파일은 거절한다 ----
    printf("[C4-5] 우리 파일이 아닌 것\n\n");

    {
        const char* JunkPath = "data/npy_junk.npy";

        FILE* Junk = fopen(JunkPath, "wb");
        if (Junk != NULL)
        {
            fwrite("NOT A NPY FILE AT ALL", 1, 21, Junk);
            fclose(Junk);

            FNpyInfo JunkInfo;
            printf("  아무 바이트나 담긴 파일 -> NpyReadInfo = %d\n",
                   NpyReadInfo(JunkPath, JunkInfo));
            CHECK(NpyReadInfo(JunkPath, JunkInfo) == 0);

            remove(JunkPath);
        }

        FNpyInfo Missing;
        printf("  없는 파일               -> NpyReadInfo = %d\n\n",
               NpyReadInfo("data/no_such_file.npy", Missing));
        CHECK(NpyReadInfo("data/no_such_file.npy", Missing) == 0);
    }

    // ---- C4-6. 이진과 텍스트를 나란히 재본다 ----
    printf("[C4-6] .npy 와 CSV, 무엇이 다른가\n\n");

    {
        FRandom Rng;
        RandomSeed(&Rng, BOOK_SEED);

        FMatrix Big(BIG_ROWS, BIG_COLS);
        for (size_t i = 0; i < Big.Count(); i++)
        {
            Big.Data()[i] = (Real)RandomRange(&Rng, 1.0);
        }

        // .npy 로 쓰기
        clock_t Begin = clock();
        CHECK(NpySaveMatrix(BIG_NPY, Big));
        double NpySaveSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

        // CSV 로 쓰기. %.6f 는 흔히 쓰는 서식이다.
        Begin = clock();
        {
            FILE* File = fopen(BIG_CSV, "wb");
            if (File != NULL)
            {
                for (size_t r = 0; r < Big.Rows(); r++)
                {
                    for (size_t c = 0; c < Big.Cols(); c++)
                    {
                        fprintf(File, "%s%.6f", (c == 0) ? "" : ",",
                                (double)Big(r, c));
                    }
                    fprintf(File, "\n");
                }
                fclose(File);
            }
        }
        double CsvSaveSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

        // .npy 읽기
        Begin = clock();
        FMatrix FromNpy;
        CHECK(NpyLoadMatrix(BIG_NPY, FromNpy));
        double NpyLoadSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

        // CSV 읽기
        Begin = clock();
        FMatrix FromCsv(BIG_ROWS, BIG_COLS);
        {
            FILE* File = fopen(BIG_CSV, "rb");
            if (File != NULL)
            {
                for (size_t r = 0; r < FromCsv.Rows(); r++)
                {
                    for (size_t c = 0; c < FromCsv.Cols(); c++)
                    {
                        double Value = 0.0;
                        if (fscanf(File, (c == 0) ? "%lf" : ",%lf", &Value) == 1)
                        {
                            FromCsv(r, c) = (Real)Value;
                        }
                    }
                }
                fclose(File);
            }
        }
        double CsvLoadSeconds = (double)(clock() - Begin) / CLOCKS_PER_SEC;

        // 정확도
        int NpyExact = 0;
        int CsvExact = 0;
        double NpyWorst = 0.0;
        double CsvWorst = 0.0;

        for (size_t i = 0; i < Big.Count(); i++)
        {
            double Original = (double)Big.Data()[i];

            double NpyGap = std::fabs((double)FromNpy.Data()[i] - Original);
            double CsvGap = std::fabs((double)FromCsv.Data()[i] - Original);

            if (NpyGap == 0.0) NpyExact++;
            if (CsvGap == 0.0) CsvExact++;
            if (NpyGap > NpyWorst) NpyWorst = NpyGap;
            if (CsvGap > CsvWorst) CsvWorst = CsvGap;
        }

        printf("  %zu x %zu 실수 배열 %zu개\n\n", Big.Rows(), Big.Cols(),
               Big.Count());

        printf("  ");
        PrintPadded("", 14);
        PrintPaddedRight(".npy", 14);
        PrintPaddedRight("CSV (%.6f)", 16);
        printf("\n");

        char A[32], B[32];

        snprintf(A, sizeof(A), "%.0f KB", FileSize(BIG_NPY) / 1024.0);
        snprintf(B, sizeof(B), "%.0f KB", FileSize(BIG_CSV) / 1024.0);
        printf("  "); PrintPadded("크기", 14);
        PrintPaddedRight(A, 14); PrintPaddedRight(B, 16); printf("\n");

        snprintf(A, sizeof(A), "%.3f 초", NpySaveSeconds);
        snprintf(B, sizeof(B), "%.3f 초", CsvSaveSeconds);
        printf("  "); PrintPadded("쓰기", 14);
        PrintPaddedRight(A, 14); PrintPaddedRight(B, 16); printf("\n");

        snprintf(A, sizeof(A), "%.3f 초", NpyLoadSeconds);
        snprintf(B, sizeof(B), "%.3f 초", CsvLoadSeconds);
        printf("  "); PrintPadded("읽기", 14);
        PrintPaddedRight(A, 14); PrintPaddedRight(B, 16); printf("\n");

        snprintf(A, sizeof(A), "%d / %zu", NpyExact, Big.Count());
        snprintf(B, sizeof(B), "%d / %zu", CsvExact, Big.Count());
        printf("  "); PrintPadded("정확히 왕복", 14);
        PrintPaddedRight(A, 14); PrintPaddedRight(B, 16); printf("\n");

        snprintf(A, sizeof(A), "%.1e", NpyWorst);
        snprintf(B, sizeof(B), "%.1e", CsvWorst);
        printf("  "); PrintPadded("최대 오차", 14);
        PrintPaddedRight(A, 14); PrintPaddedRight(B, 16); printf("\n\n");

        // 이진은 반드시 완전 일치해야 한다.
        CHECK(NpyExact == (int)Big.Count());
        CHECK(NpyWorst == 0.0);

        // 텍스트는 그렇지 않다.
        CHECK(CsvExact < (int)Big.Count());
        CHECK(CsvWorst > 0.0);

        printf("  **%%.6f 로 적으면 되돌아오지 않는다.** float 을 텍스트로\n");
        printf("  정확히 적으려면 유효숫자 9자리(%%.9g)가 필요하고,\n");
        printf("  double 은 17자리(%%.17g)가 필요하다.\n\n");
    }

    return ReportResult();
}
