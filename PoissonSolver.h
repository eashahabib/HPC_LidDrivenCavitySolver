#pragma once
#ifndef POISSONSOLVER_H
#define POISSONSOLVER_H

#include "Functions.h" //basic functions used throughout the code 

//WITH SCALAPACK! //////////////////////////////////////////////////////////////////////////////////

class PoissonSolver
{
public:
    PoissonSolver();
    ~PoissonSolver();
    
    void SetDiagonal(double d0, double d1, double d2);
    
    void GenerateA_LapackLU(int srw, int scl); // generate the diagonal matrix in banded format with LU factorisation
    void SolveProblem_LU(int Nx, int Ny, int srw, int scl, double *w, double *psi); //solves the poisson problem with LU matrix from GenerateA_LU()
    
    void GenerateA_ScaLapack(int mpirank, int srw, int scl, int smallNBx, int smallNBy); // generate the diagonal matrix in banded format
    void SolveProblem_ScaLapack();
    
private: 
    double D0_val; //center diagonal value
    double D1_val; //1st super and sub diagonal value
    double D2_val; //2nd super and sub diagonal value
    
    double *A = nullptr;
    int    *ipiv = nullptr;
    

};

PoissonSolver::PoissonSolver(){
}

PoissonSolver::~PoissonSolver(){
    delete[] ipiv, A;
}

void PoissonSolver::SetDiagonal(double d0, double d1, double d2){
    D0_val = d0;
    D1_val = d1;
    D2_val = d2;
}

void PoissonSolver::GenerateA_LapackLU(int srw, int scl){
    int col = 2*scl+1+scl;
    A = new double[col*srw*scl];
    int i;
    for(i=0; i<srw*scl; i++){
        A[i*col + 0+scl] = d2;
        A[i*col + scl-1+scl] = d1;
        A[i*col + scl+scl] = d0;
        A[i*col + scl+1+scl] = d1;
        A[i*col + 2*scl+scl] = d2;
    }
    
    for(int i=0; i<srw*scl-(scl-1); i=i+scl){
        A[(i)*col + scl-1 +scl] = 0;
        A[(i+scl-1)*col + scl+1 +scl] = 0;
    }
    printMatrix( (2*srw+1+srw),srw*scl, A);
    
    int nsv = srw*scl;
    int ldh = 3*srw+1;
    int info;
    ipiv = new double[srw*scl];
    
    F77NAME(dgbtrf)(nsv, nsv, srw, srw, A, ldh, ipiv, &info);

    if (info) {
        cout << "Failed to LU factorise matrix" << endl;
    }
    
}

void PoissonSolver::SolveProblem_LU(int Nx, int Ny, int srw, int scl, double *w, double *psi){
//solving a system A u = f
    double *f = new double[srw*scl];
    for(int i=1; i<Nx-1; i++){
        for(int j=1; j<Ny-1; j++){
            f[(j-1)*srw+(i-1)] = w[j*Nx+i];
        }
    }
    
//    cout << "f" << endl;
//    printMatrix(srw, scl, f);
    
    int nsv = srw*scl;
    int kl = srw;         // Number of lower diagonals
    int ku = srw;         // Number of upper diagonals
    int nrhs = 1;
    int lda = 4*srw+1;
    //int ldb = nsv;
    int info;
    
    F77NAME(dgbtrs)('N', nsv, kl , ku, nrhs, A, lda, ipiv, f, nsv, &info);
    if (info) {
        cout << "Error in solve: " << info << endl;
    }
    
    for(int i=1; i<Nx-1; i++){
        for(int j=1; j<Ny-1; j++){
            //cout << f[(j-1)*Nx+(i-1)] << endl; 
            psi[j*Nx+i] = f[(j-1)*srw+(i-1)];
        }
    }
    delete[] f;
}

void PoissonSolver::GenerateA_ScaLapack(int mpirank, int srw, int scl, int smallNBx, int smallNBy){
    int col = 2*scl+1+2*srw; 
    int row = smallNBx*smallNBy;
    A = new double[col*row];
    
    for(int i=0; i<row; i++){
        A[i*col + 0+scl+srw] = d2;
        A[i*col + scl-1+scl+srw] = d1;
        A[i*col + scl+scl+srw] = d0;
        A[i*col + scl+1+scl+srw] = d1;
        A[i*col + 2*scl+2*srw] = d2;
    }
    
    int j = 0;
    for(int i=0+mpirank*row; i<row - (scl-1) +mpirank*row; i=i+scl){
        A[(i)*col + scl-1 +scl+srw] = 0;
        A[(i+scl-1)*col + scl+1 +scl+srw] = 0;
    }
}

void SolveProblem_ScaLapack(int ctx, int NBx, int NBy, int srw, int scl){
//solving a system A u = f
    
    int info; // Status value
    const int N = (srw)*(scl); // Total problem size
    const int NB = (NBx-2)*(NBy-2); // Blocking size (number of columns per process)
    const int BWL = srw; // Lower bandwidth
    const int BWU = srw; // Upper bandwidth
    const int NRHS = 1; // Number of RHS to solve
    const int JA = 1; // Start offset in matrix (to use just a submatrix)
    const int IB = 1; // Start offset in RHS vector (to use just a subvector)
    const int LA = (1 + 2*BWL + 2*BWU)*NB;
    const int LW = (NB+BWU)*(BWL+BWU)+6*(BWL+BWU)*(BWL+2*BWU) + max(NRHS*(NB+2*BWL+4*BWU), 1); // ScaLAPACK documentation
    double* work = new double[LW](); // Workspace
    double* A = new double[LA](); // Matrix banded storage
    
    SetDiagonal(a, b, c);
    GenerateA_ScaLapack(mpirank, srw, scl, NBx-2, NBy-2);
    
    //printMatrix(row, NB, A); 
    
    int* ipiv = new int [NB](); // Pivoting array
    double* x = new double[NB](); // In: RHS vector, Out: Solution;
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            x[(j-1)*(NBx-2)+(i-1)] = w[j*NBx+i];
        }
    }
    
    int desca[7]; // Descriptor for banded matrix
    desca[0] = 501; // Type
    desca[1] = ctx; // Context
    desca[2] = N; // Problem size
    desca[3] = NB; // Blocking of matrix
    desca[4] = 0; // Process row/column
    desca[5] = (1 + 2*BWL + 2*BWU); // Local leading dim
    desca[6] = 0; // Reserved
    int descb[7]; // Descriptor for RHS
    descb[0] = 502; // Type
    descb[1] = ctx; // Context
    descb[2] = N; // Problem size
    descb[3] = NB; // Blocking of matrix
    descb[4] = 0; // Process row/column
    descb[5] = NB; // Local leading dim
    descb[6] = 0; // Reserved
    
    //Perform the parallel solve.
    F77NAME(pdgbsv) (N, BWL, BWU, NRHS, A, JA, desca, ipiv, x, IB, descb, work, LW, &info);
    // Verify it completed successfully.
    if (info) {
        cout << "Error occurred in PDGBSV: " << info << endl;
    }
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+i] = x[(j-1)*(NBx-2)+(i-1)];
        }
    }
    
    delete[] work, x, desca, descb; 
    
}
#endif