#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#pragma once
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdlib.h> //for atof

using namespace std;

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
    void Cblacs_pinfo(int*, int*); // Initialises the BLACS world communicator (calls MPI_Init if needed)
    void Cblacs_get(int, int, int*);  // Get the default system context (i.e. MPI_COMM_WORLD)
    void Cblacs_gridinit(int*, const char*, int, int); // Initialise a process grid of npr rows and npe columns
    void Cblacs_gridexit(int); // Frees the context handle we created with Cblacs_get()
    void Cblacs_exit(int);
    void Cblacs_barrier(int, const char*);
    void Cblacs_gridinfo(int ctx, int *nrow, int *ncol, int *myrow, int *mycol); // Get info about the grid to verify it is set up
    
    int numroc_(int *N, int *Nb, int *iproc, int *isrcproc, int *nprocs);
    
    /* Scalapack declarations */
    void F77NAME(pdgbsv)(const int &N, const int &BWL, const int &BWU, const int &NRHS, 
                        double *A, const int &JA, 
                        int *desca, int *ipiv, double *x, const int &IB, int *descb, 
                        double *work, const int &LW, int *info);
                        
    void F77NAME(pdgbtrf)(const int &N, const int &BWL, const int &BWU, 
                        double *A, const int &JA, 
                        int *desca, int *ipiv, double *af, const int &laf,
                        double *work, const int &LW, int *info);
    
    void F77NAME(pdgbtrs)(const char, const int &N, const int &BWL, const int &BWU, const int &NRHS, 
                        double *A, const int &JA, 
                        int *desca, int *ipiv, double *x, const int &IB, int *descb, 
                        double *af, const int &laf,
                        double *work, const int &LW, int *info);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void printMatrix(int row, int col, double* A);
void printMatrix(int row, int col, int* A);
void Print2File(int argc, char *argv[], int Nx, int Ny, double *w, double *psi, string ext);
void Update(int mpirank, int Tmycol, int Tmyrow, int NBx, int NBy, int Px, int Py, double *w, double *top, double *bot, double *left, double *right);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void printMatrix(int row, int col, double* A) {
    cout << endl;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
            cout.width(10);
            cout.precision(5);
            //cout.fill(' ');
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
            cout.width(5);
			cout << A[i+j*row] << "  ";
		}
		cout << endl;
	}
cout << "-----------------------------" << endl;
}

void Print2File(int argc, char *argv[], int Nx, int Ny, double *w, double *psi, string ext = "txt"){
    string filename;
    for(int i=1; i<argc; i++){
            filename += argv[i];
        }
    filename += "." + ext;
    
    string delimiter;
    if(ext == "csv"){
        delimiter = ",";
    }else{
        delimiter = " ";
    }
    
    
    ofstream vfile(filename ,ios::trunc | ios::in);
    if(vfile.good()){
        for(int i=1; i<argc; i++){
            vfile << argv[i] << " ";
        }
        vfile << endl;
        vfile << "Vorticity" << delimiter << "StreamFunction" << endl;
        
        for(int i=0;i<Nx;i++){
            for(int j=0; j<Ny; j++){
                vfile.width(25);
                vfile.precision(20);
                vfile << w[i*Ny+j] << delimiter << psi[i*Ny+j] << endl; // outputting into the file
            }
            
        }
        vfile.close();
    }
    else{
        cout << "Failed to open"<<endl;
    }
}

void Update(int mpirank, int Tmycol, int Tmyrow, int NBx, int NBy, int Px, int Py, double *w, double *top, double *bot, double *left, double *right){
//    top = new double[NBx-2];
//    bot = new double[NBx-2];
//    left = new double[NBy-2];
//    right = new double[NBy-2];
    
    if(Px != 1){
        if(Tmyrow != Px-1 && Tmyrow != 0){ // Middle row
        
            for(int j=1; j<NBy-1; j++){
                left[j-1] = w[j*NBx+1];
                right[j-1] = w[j*NBx+(NBx-2)];
            }
            
            MPI_Send(left, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
            MPI_Send(right, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
            
        }else if(Tmyrow == Px-1){// Bottom row (Right)
            
            for(int j=1; j<NBy-1; j++){
                right[j-1] = w[j*NBx+1];
            }
            
            MPI_Send(right, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
            
        }else if(Tmyrow == 0){
            
            for(int j=1; j<NBy-1; j++){
                left[j-1] = w[j*NBx+(NBx-2)];
            }
            
            MPI_Send(left, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        }
    }
    
    if(Py != 1){
        if(Tmycol != Py-1 && Tmycol != 0){ // Middle column
            for(int i=1; i<NBx-1; i++){
                top[i-1] = w[(NBy-2)*NBx+i];
                bot[i-1] = w[1*NBx+i];
            }
            
            MPI_Send(top, NBx-2, MPI_DOUBLE, mpirank+Px, 0, MPI_COMM_WORLD);
            MPI_Send(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 0, MPI_COMM_WORLD);
            
        }else if(Tmycol == Py-1){// Right column (Top)
        
            for(int i=1; i<NBx-1; i++){
                bot[i-1] = w[1*NBx+i];
            }
            
            MPI_Send(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 0, MPI_COMM_WORLD);
            
        }else if(Tmycol == 0){ // Left Column (Bottom)
            
            for(int i=1; i<NBx-1; i++){
                top[i-1] = w[(NBy-2)*NBx+i];
            }
            
            MPI_Send(top, NBx-2, MPI_DOUBLE, mpirank+Px, 0, MPI_COMM_WORLD);
        }
    }
    
    if(Px != 1){
        
        if(Tmyrow != Px-1 && Tmyrow != 0){ // Middle row
        
            MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int j=1; j<NBy-1; j++){
                w[j*NBx+0] = left[j-1];
                w[j*NBx+(NBx-1)] = right[j-1];
            }
        }else if(Tmyrow == Px-1){// Bottom row (Right)
            
            MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int j=1; j<NBy-1; j++){
                w[j*NBx+0] = right[j-1];
            }
        }else if(Tmyrow == 0){  // Top Row (Left)
            
            MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int j=1; j<NBy-1; j++){
                w[j*NBx+(NBx-1)] = left[j-1];
            }
        }
    }
    
    if(Py != 1){
        if(Tmycol != Py-1 && Tmycol != 0){ // Middle column
        
            MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank+Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int i=1; i<NBx-1; i++){
                w[(NBy-1)*NBx+i] = top[i-1];
                w[0*NBx+i] = bot[i-1];
            }
        }else if(Tmycol == Py-1){// Righht most column (Top)
            
            MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int i=1; i<NBx-1; i++){
                w[0*NBx+i] = bot[i-1];
            }
        }else if(Tmycol == 0){ // Left most Column (Bottom)
            
            MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank+Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for(int i=1; i<NBx-1; i++){
                w[(NBy-1)*NBx+i] = top[i-1];
            }
        }
    }
        
//    delete [] top, bot, right, left;
}


#endif // COMPLEX_H