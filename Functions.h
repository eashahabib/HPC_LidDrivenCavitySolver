#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#pragma once
#include <string>
using namespace std;
#include <iostream>

// This provides forward-declarations of the C++ iostream classes since
// we do not need the full definition in the header file.
#include <iosfwd>

#include <cblas.h>


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define F77NAME(x) x##_

extern "C" {
    /* Lapack declarations */
    // Performs LU factorisation of general banded matrix
    void F77NAME(dgbtrf)(const int& N, const int& M, const int& KL, 
                        const int& KU, double* AB, const int& LDAB,
                        int* IPIV, int* INFO);
        
    // Solves pre-factored system of equations
    void F77NAME(dgbtrs)(const char& TRANS, const int& N, const int& KL, 
                        const int& KU,
                        const int& NRHS, double* AB, const int& LDAB,
                        int* IPIV, double* B, const int& LDB, int* INFO);
                        
    // Solves the General Banded system
    void F77NAME(dgbsv)(const int& N, int& KL, int& KU, int& NRHS, double* AB,
                        int& LDAB, int *ipiv, double *B, int& LDB, int &info);
                        
    void F77NAME(dgels)(const char& TRANS, const int& M, const int& N,
                        const int& NRHS, double* A, const int& LDA,
                        double* B, const int& LDB, double* work, const int& lwork, 
                        int* INFO);
    
    /* Cblacs declarations */
    // Initialises the BLACS world communicator (calls MPI_Init if needed)
    void Cblacs_pinfo(int*, int*);
    // Get the default system context (i.e. MPI_COMM_WORLD)
    void Cblacs_get(int, int, int*);
    // Initialise a process grid of 1 rows and npe columns
    void Cblacs_gridinit(int*, const char*, int, int);
    // Frees the context handle we created with Cblacs_get()
    void Cblacs_gridexit(int);
    void Cblacs_exit(int);
    void Cblacs_barrier(int, const char*);
    // Get info about the grid to verify it is set up
    void Cblacs_gridinfo(int ctx, int *nrow, int *ncol, int *myrow, int *mycol);
    
    /* Scalapack declarations */
    void F77NAME(pdgbsv)(const int &N, const int &BWL, const int &BWU, const int &NRHS, 
                        double *A, const int &JA, 
                        int *desca, int *ipiv, double *x, const int &IB, int *descb, 
                        double *work, const int &LW, int *info);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void printMatrix(int row, int col, double* A);
void printMatrix(int row, int col, int* A);
void BandedMatrixLapackLU(int srw, int scl, double a, double b, double c, double *A, int *ipiv);
void BoundaryVorticity(int Nx, int Ny, double dx, double dy, double U, double *psi, double *w);
void InteriorVorticity(int Nx, int Ny, double dx, double dy, double *psi, double *w);
void NewInteriorVorticity(double Re, int Nx, int Ny, double dx, double dy, double dt, double *psi, double *w);
void PoissonProblem(int Nx, int Ny, int srw, int scl, double *A, double *w, double *psi, int *ipiv);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void printMatrix(int row, int col, double* A) {
    cout << endl;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            cout.width(6);
			cout << A[i+j*row] << "  ";
		}
		cout << endl;
	}
cout << "-----------------------------" << endl;
}

void printMatrix(int row, int col, int* A) {
    cout << endl;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            cout.width(6);
			cout << A[i+j*row] << "  ";
		}
		cout << endl;
	}
cout << "-----------------------------" << endl;
}

void BandedMatrixLapackLU(int srw, int scl, double a, double b, double c, double *A, int *ipiv){
    int col = 2*scl+1+scl; 
    int i;
    for(i=0; i<srw*scl; i++){
        A[i*col + 0+scl] = c;
        A[i*col + scl-1+scl] = b;
        A[i*col + scl+scl] = a;
        A[i*col + scl+1+scl] = b;
        A[i*col + 2*scl+scl] = c;
    }
    
    for(int i=0; i<srw*scl-(scl-1); i=i+scl){
        A[(i)*col + scl-1 +scl] = 0;
        A[(i+scl-1)*col + scl+1 +scl] = 0;
    }
    printMatrix( (2*srw+1+srw),srw*scl, A);
    
    int nsv = srw*scl;
    int ldh = 3*srw+1;
    int info;
    
    F77NAME(dgbtrf)(nsv, nsv, srw, srw, A, ldh, ipiv, &info);

    if (info) {
        cout << "Failed to LU factorise matrix" << endl;
    }
}

//Boundary Conditions for vorticity
void BoundaryVorticity(int Nx, int Ny, double dx, double dy, double U, double *psi, double *w){
    int i;
    for(i=0; i<Nx; i++){
        //top
        w[Nx*(Ny-1)+i] = (psi[Nx*(Ny-1)+i] - psi[Nx*(Ny-1)+i-Nx])*2/dy/dy - 2.0*U/dy;
        //bottom
        w[i] = (psi[i] - psi[i+Nx])*2/dy/dy;
    }
    for(i=0; i<Ny; i++){
        //left
        w[i*Nx] = (psi[i*Nx] - psi[i*Nx+1])*2/dx/dx;
        //right
        w[i*Nx+Nx-1] = (psi[i*Nx+Nx-1] - psi[i*Nx+Nx-2])*2/dx/dx;
    }
    
    //printMatrix( Nx, Ny, w);
}

void InteriorVorticity(int Nx, int Ny, double dx, double dy, double *psi, double *w){
    for(int i=1; i<Nx-1; i++){
        for(int j=1; j<Ny-1; j++){
            w[j*Nx+i] = -(psi[j*Nx+i+1] - 2*psi[j*Nx+i] + psi[j*Nx+i-1])/dx/dx - (psi[(j+1)*Nx+i] - 2*psi[j*Nx+i] + psi[(j-1)*Nx+i])/dy/dy;
            //cout<<w[j*Nx+i]<<endl;
        }
    }
    //printMatrix( Nx, Ny, w);
}

void NewInteriorVorticity(double Re, int Nx, int Ny, double dx, double dy, double dt, double *psi, double *w){
    //pringles.
    double t0, t1, t2, t3, t4;
    for(int i=1; i<Nx-1; i++){
        for(int j=1; j<Ny-1; j++){
            
            t0 = ( (w[j*Nx+i+1] - 2*w[j*Nx+i] + w[j*Nx+i-1])/dx/dx + (w[(j+1)*Nx+i] - 2*w[j*Nx+i] + w[(j-1)*Nx+i])/dy/dy )/Re;
            t1 = (psi[(j+1)*Nx+i] - psi[(j-1)*Nx+i])/2/dy;
            t2 = (w[j*Nx+i+1] - w[j*Nx+i-1])/2/dx;
            t3 = (psi[j*Nx+i+1] - psi[j*Nx+i-1])/2/dx;
            t4 = (w[(j+1)*Nx+i] - w[(j-1)*Nx+i])/2/dy;
            w[j*Nx+i] += (t0 - t1*t2 + t3*t4)*dt;
            
        }
    }
    cout << "w" << endl;
    printMatrix( Nx, Ny, w);
}

void PoissonProblem(int Nx, int Ny, int srw, int scl, double *A, double *w, double *psi, int *ipiv){
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
    int lda = 3*srw+1;
    //int ldb = nsv;
    int info;
    
//    cout << "A" << endl;
    //printMatrix( lda,nsv,  A);
    
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
//    cout << "psi" << endl;
    printMatrix(Nx, Ny, psi);
    delete f;
}

#endif // COMPLEX_H