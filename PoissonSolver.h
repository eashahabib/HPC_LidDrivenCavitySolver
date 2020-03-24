#pragma once
#ifndef POISSONSOLVER_H
#define POISSONSOLVER_H

#include "Functions.h" //Basic functions and declarations used throughout the code 


class PoissonSolver
{
public:
    PoissonSolver();
    ~PoissonSolver();
    
//    void SetUpSystem(); //Setting up the solver for serial implementation
    void SetUpSystem(double D0, double D1, double D2, int nbx, int nby, int tmyrow, int tmycol, int px, int py); // Setting up the solver for parallel implementation
    
    void GenerateA_LapackLU(int srw, int scl); // generate the diagonal matrix in banded format with LU factorisation
    void SolveProblem_LU(int Nx, int Ny, int srw, int scl, double *w, double *psi); //solves the poisson problem with LU matrix from GenerateA_LU()
    
    void GenerateA_ScaLapack(int mpirank, int srw, int scl); // generate the diagonal matrix in banded format for Scalapack
    void ScaLapackSystemSetup(int ctx, int srw, int scl); // set up system and perform LU
    void SolveProblem_ScaLapack(double *w, double *psi); // Solve system using scalapack
    
    void ParallelJacobiSolver(int iter, int mpirank, double *w, double *psi, double *top, double *bot, double *left, double *right);
    
private: 
    double d0; //center diagonal value
    double d1; //1st super and sub diagonal value
    double d2; //2nd super and sub diagonal value
    
    double *A = nullptr;
    int    *ipiv = nullptr;
    int    *ipivot = nullptr;
    double *work = nullptr;
    double *af = nullptr;
    int desca[7];
    int descb[7];

     int N ; // Total problem size
     int NB; // Blocking size (number of columns per process)
     int BWL; // Lower bandwidth
     int BWU; // Upper bandwidth
     int NRHS; // Number of RHS to solve
     int JA; // Start offset in matrix (to use just a submatrix)
     int IB; // Start offset in RHS vector (to use just a subvector)
     int LA;
     int LW; // ScaLAPACK documentation
     int LAF;
     
     //Variables for the Jacobi System
     double *temp_jacobi = nullptr;
     
     //Variables for the Jacobi System
     int Tmyrow;
     int Tmycol;
     int Px;
     int Py;
     int NBx;
     int NBy;
     

};


#endif