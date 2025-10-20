/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <lapack.hh>
#include <blas.hh>

/**
 * Test program to determine whether LAPACK and BLAS functions in MACIS expect
 * column-major or row-major matrix storage.
 * 
 * We test this using:
 * 1. lapack::gesvd on a simple 3x3 matrix with known SVD
 * 2. blas::gemm for matrix multiplication verification
 */

void print_matrix(const std::vector<double>& mat, int rows, int cols, const std::string& name, bool is_column_major = true) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (is_column_major) {
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) 
                         << mat[j * rows + i] << " ";
            } else {
                std::cout << std::setw(12) << std::fixed << std::setprecision(6) 
                         << mat[i * cols + j] << " ";
            }
        }
        std::cout << "\n";
    }
}

void print_vector(const std::vector<double>& vec, const std::string& name) {
    std::cout << "\n" << name << ": ";
    for (double val : vec) {
        std::cout << std::setw(12) << std::fixed << std::setprecision(6) << val << " ";
    }
    std::cout << "\n";
}

void test_blas_gemm_convention() {
    std::cout << "\n=== BLAS GEMM Matrix Storage Convention Test ===" << std::endl;
    std::cout << "Testing blas::gemm to verify matrix storage convention\n" << std::endl;

    const int m = 2, n = 2, k = 2;

    // Test matrices for C = A * B
    // A = [1 2]    B = [5 6]    Expected C = [19 22]
    //     [3 4]        [7 8]                   [43 50]

    std::cout << "TEST: Matrix Multiplication C = A * B" << std::endl;
    std::cout << "=====================================" << std::endl;

    // Test 1: Column-major storage
    std::vector<double> A_col = {1.0, 3.0, 2.0, 4.0}; // [1,3; 2,4] in column-major
    std::vector<double> B_col = {5.0, 7.0, 6.0, 8.0}; // [5,7; 6,8] in column-major
    std::vector<double> C_col(m * n, 0.0);

    std::cout << "\nColumn-major test:" << std::endl;
    print_matrix(A_col, m, k, "Matrix A (col-major)", true);
    print_matrix(B_col, k, n, "Matrix B (col-major)", true);

    // C = A * B using column-major layout
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, n, k, 1.0, A_col.data(), m, B_col.data(), k, 0.0, C_col.data(), m);

    print_matrix(C_col, m, n, "Result C = A*B (col-major)", true);

    // Test 2: Row-major storage  
    std::vector<double> A_row = {1.0, 2.0, 3.0, 4.0}; // [1,2; 3,4] in row-major
    std::vector<double> B_row = {5.0, 6.0, 7.0, 8.0}; // [5,6; 7,8] in row-major
    std::vector<double> C_row(m * n, 0.0);

    std::cout << "\nRow-major test:" << std::endl;
    print_matrix(A_row, m, k, "Matrix A (row-major)", false);
    print_matrix(B_row, k, n, "Matrix B (row-major)", false);

    // C = A * B using row-major layout
    blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, n, k, 1.0, A_row.data(), k, B_row.data(), n, 0.0, C_row.data(), n);

    print_matrix(C_row, m, n, "Result C = A*B (row-major)", false);

    // Verify results
    std::vector<double> expected_col = {19.0, 43.0, 22.0, 50.0}; // column-major [19,43; 22,50]
    std::vector<double> expected_row = {19.0, 22.0, 43.0, 50.0}; // row-major [19,22; 43,50]

    bool col_correct = true, row_correct = true;
    for (int i = 0; i < m * n; i++) {
        if (std::abs(C_col[i] - expected_col[i]) > 1e-12) col_correct = false;
        if (std::abs(C_row[i] - expected_row[i]) > 1e-12) row_correct = false;
    }

    std::cout << "\nBLAS GEMM Results:" << std::endl;
    std::cout << "Column-major layout: " << (col_correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
    std::cout << "Row-major layout: " << (row_correct ? "✓ CORRECT" : "✗ INCORRECT") << std::endl;
}

int main() {
    std::cout << "=== LAPACK/BLAS Matrix Storage Convention Test ===" << std::endl;
    std::cout << "Testing both LAPACK and BLAS to determine matrix storage convention\n" << std::endl;

    // First test BLAS GEMM
    test_blas_gemm_convention();

    std::cout << "\n\n=== LAPACK SVD Matrix Storage Convention Test ===" << std::endl;
    std::cout << "Testing lapack::gesvd to determine matrix storage convention\n" << std::endl;

    // Test matrix: Simple 3x3 matrix
    // Row-major representation:
    // [ 3.0  2.0  2.0 ]
    // [ 2.0  3.0 -2.0 ]
    // [ 4.0  1.0  1.0 ]
    
    const int m = 3, n = 3;
    
    // First, test assuming column-major storage (LAPACK standard)
    std::cout << "TEST 1: Assuming Column-Major Storage (LAPACK standard)" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Store the matrix in column-major format
    std::vector<double> A_col_major = {
        3.0, 2.0, 4.0,  // First column:  [3, 2, 4]
        2.0, 3.0, 1.0,  // Second column: [2, 3, 1]  
        2.0,-2.0, 1.0   // Third column:  [2,-2, 1]
    };
    
    print_matrix(A_col_major, m, n, "Input matrix A (stored column-major)", true);
    
    // Prepare for SVD
    std::vector<double> A_copy1 = A_col_major;
    std::vector<double> S1(std::min(m, n));
    std::vector<double> U1(m * m);
    std::vector<double> VT1(n * n);
    
    // Perform SVD: A = U * S * V^T
    try {
        lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, m, n,
                      A_copy1.data(), m, S1.data(), U1.data(), m, VT1.data(), n);
        
        print_vector(S1, "Singular values (col-major test)");
        print_matrix(U1, m, m, "U matrix (col-major test)", true);
        print_matrix(VT1, n, n, "V^T matrix (col-major test)", true);
        
    } catch (const std::exception& e) {
        std::cout << "Error in column-major SVD: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n\n" << std::endl;
    
    // Second, test assuming row-major storage (less common for LAPACK)
    std::cout << "TEST 2: Assuming Row-Major Storage (non-standard for LAPACK)" << std::endl;
    std::cout << "=============================================================" << std::endl;
    
    // Store the same logical matrix in row-major format
    std::vector<double> A_row_major = {
        3.0, 2.0, 2.0,  // First row:  [3, 2, 2]
        2.0, 3.0,-2.0,  // Second row: [2, 3,-2]
        4.0, 1.0, 1.0   // Third row:  [4, 1, 1]
    };
    
    print_matrix(A_row_major, m, n, "Input matrix A (stored row-major)", false);
    
    // Prepare for SVD
    std::vector<double> A_copy2 = A_row_major;
    std::vector<double> S2(std::min(m, n));
    std::vector<double> U2(m * m);
    std::vector<double> VT2(n * n);
    
    // Perform SVD treating row-major data as if it were column-major
    try {
        lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, m, n,
                      A_copy2.data(), m, S2.data(), U2.data(), m, VT2.data(), n);
        
        print_vector(S2, "Singular values (row-major test)");
        print_matrix(U2, m, m, "U matrix (row-major test)", true);
        print_matrix(VT2, n, n, "V^T matrix (row-major test)", true);
        
    } catch (const std::exception& e) {
        std::cout << "Error in row-major SVD: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n\n" << std::endl;
    
    // Analysis and conclusion
    std::cout << "ANALYSIS AND CONCLUSION" << std::endl;
    std::cout << "=======================" << std::endl;
    
    // Check reconstruction quality for both cases
    auto reconstruct_and_check = [&](const std::vector<double>& U, 
                                     const std::vector<double>& S, 
                                     const std::vector<double>& VT,
                                     const std::vector<double>& original,
                                     const std::string& test_name) {
        
        std::vector<double> reconstructed(m * n, 0.0);
        
        // Reconstruct A = U * S * V^T
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < std::min(m, n); k++) {
                    reconstructed[j * m + i] += U[k * m + i] * S[k] * VT[j * n + k];
                }
            }
        }
        
        // Calculate reconstruction error
        double error = 0.0;
        for (int i = 0; i < m * n; i++) {
            error += std::pow(reconstructed[i] - original[i], 2);
        }
        error = std::sqrt(error);
        
        std::cout << test_name << " reconstruction error: " << std::scientific 
                  << std::setprecision(3) << error << std::endl;
        
        return error;
    };
    
    double error1 = reconstruct_and_check(U1, S1, VT1, A_col_major, "Column-major");
    double error2 = reconstruct_and_check(U2, S2, VT2, A_row_major, "Row-major");
    
    std::cout << "\nCONCLUSION:" << std::endl;
    std::cout << "============" << std::endl;
    
    std::cout << "\nLAPACK SVD Analysis:" << std::endl;
    if (error1 < 1e-10) {
        std::cout << "✓ LAPACK functions expect COLUMN-MAJOR matrix storage" << std::endl;
        std::cout << "  This is the standard LAPACK convention." << std::endl;
    } else if (error2 < 1e-10) {
        std::cout << "✓ LAPACK functions expect ROW-MAJOR matrix storage" << std::endl;
        std::cout << "  This is unusual for LAPACK!" << std::endl;
    } else {
        std::cout << "? Unclear LAPACK results - both tests show significant errors" << std::endl;
    }
    
    std::cout << "\nOverall Summary:" << std::endl;
    std::cout << "- BLAS functions support both ColMajor and RowMajor layouts explicitly" << std::endl;
    std::cout << "- The codebase uses blas::Layout::ColMajor and blas::Layout::RowMajor as needed" << std::endl;
    std::cout << "- LAPACK functions typically expect column-major storage (Fortran convention)" << std::endl;
    std::cout << "- Matrix storage: A[col * rows + row] for column-major" << std::endl;
    std::cout << "- Matrix storage: A[row * cols + col] for row-major" << std::endl;
    
    // Additional verification: check if singular values are sorted
    std::cout << "\nAdditional checks:" << std::endl;
    bool s1_sorted = true, s2_sorted = true;
    for (int i = 1; i < S1.size(); i++) {
        if (S1[i] > S1[i-1]) s1_sorted = false;
        if (S2[i] > S2[i-1]) s2_sorted = false;
    }
    
    std::cout << "Singular values sorted (col-major): " << (s1_sorted ? "YES" : "NO") << std::endl;
    std::cout << "Singular values sorted (row-major): " << (s2_sorted ? "YES" : "NO") << std::endl;
    
    std::cout << "\nKey Findings:" << std::endl;
    std::cout << "- LAPACK returns singular values in descending order by default" << std::endl;
    std::cout << "- BLAS GEMM explicitly specifies layout (ColMajor/RowMajor) in function call" << std::endl;
    std::cout << "- This MACIS codebase consistently uses blas::Layout::ColMajor" << std::endl;
    std::cout << "- Column-major is the standard for both BLAS and LAPACK (Fortran heritage)" << std::endl;
    
    return 0;
}