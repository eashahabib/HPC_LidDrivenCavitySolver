#include "PoissonSolver.h"

PoissonSolver::PoissonSolver(){
}

PoissonSolver::~PoissonSolver(){
    delete[] A, ipiv, ipivot, work, af;
    free(desca);
    free(descb);
    
    delete[] temp_jacobi;
}

void PoissonSolver::SetUpSystem(double D0, double D1, double D2, int nbx, int nby, int tmyrow, int tmycol, int px, int py){
    d0 =  2.0*D0;
    d1 = -1.0*D1;
    d2 = -1.0*D2;
    
    NBx = nbx;
    NBy = nby;
    
    Tmyrow = tmyrow;
    Tmycol = tmycol;
    Px = px;
    Py = py;
    
    temp_jacobi = new double[(NBx-2)*(NBy-2)]();
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
    ipiv = new int[srw*scl];
    
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
    //delete[] f;
}

void PoissonSolver::GenerateA_ScaLapack(int mpirank, int srw, int scl){
    int col = 2*srw+1+2*srw; 
    int row = (NBx-2)*(NBy-2);
    A = new double[col*row]();
    
    for(int i=0; i<row; i++){
        A[i*col + 0+srw+srw] = d2;
        A[i*col + srw-1+srw+srw] = d1;
        A[i*col + srw+srw+srw] = d0;
        A[i*col + srw+1+srw+srw] = d1;
        A[i*col + 2*srw+2*srw] = d2;
    }
    
    int j = 0;
    for(int i=(mpirank*row)%srw; i<row - (srw-1); i=i+srw){
        A[(i)*col + srw-1 +srw+srw] = 0;
        A[(i+srw-1)*col + srw+1 +srw+srw] = 0;
    }
}

void PoissonSolver::ScaLapackSystemSetup(int ctx, int srw, int scl){
    N = (srw)*(scl); // Total problem size
    NB = (NBx-2)*(NBy-2); // Blocking size (number of columns per process)
    BWL = srw; // Lower bandwidth
    BWU = srw; // Upper bandwidth
    NRHS = 1; // Number of RHS to solve
    JA = 1; // Start offset in matrix (to use just a submatrix)
    IB = 1; // Start offset in RHS vector (to use just a subvector)
    LA = (1 + 2*BWL + 2*BWU)*NB;
    LW = (NB+BWU)*(BWL+BWU)+6*(BWL+BWU)*(BWL+2*BWU) + max(NRHS*(NB+2*BWL+4*BWU), 1); // ScaLAPACK documentation
//    LAF = ( (NB+BWU)*(BWL+BWU)+6*(BWL+BWU)*(BWL+2*BWU) );
    
//    int desca[7];// Descriptor for banded matrix
    desca[0] = 501; // Type
    desca[1] = ctx; // Context
    desca[2] = N; // Problem size
    desca[3] = NB; // Blocking of matrix
    desca[4] = 0; // Process row/column
    desca[5] = (1 + 2*BWL + 2*BWU); // Local leading dim
    desca[6] = 0; // Reserved
//    int descb[7];// Descriptor for RHS
    descb[0] = 502; // Type
    descb[1] = ctx; // Context
    descb[2] = N; // Problem size
    descb[3] = NB; // Blocking of matrix
    descb[4] = 0; // Process row/column
    descb[5] = NB; // Local leading dim
    descb[6] = 0; // Reserved
    
    work = new double[LW]();  //Workspace
    ipivot = new int[NB]();  //Pivoting array
//    af = new double[LAF]();
//    int info;
    
//    F77NAME(pdgbtrf)(N, BWL, BWU, A, JA, desca, ipivot, af, LAF, work, LW, &info);
//    
//    if (info) {
//        cout << "Error occurred in PDGBTRF: " << info << endl;
//    }
    
}

void PoissonSolver::SolveProblem_ScaLapack(double *w, double *psi){
//solving a system A u = f
    
    int info; // Status value
    
    double* x = new double[NB](); // In: RHS vector, Out: Solution;
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            x[(j-1)*(NBx-2)+(i-1)] = w[j*NBx+i];
        }
    }
    
    double* B = new double[LA]();    
    memcpy(B, A, LA*sizeof(double));
//    
//    work = new double[LW](); // Workspace
//    ipivot = new int[NB](); // Pivoting array
    
//    printMatrix((1 + 2*BWL + 2*BWU),NB, A);
//    printMatrix((1 + 2*BWL + 2*BWU),NB, B);
    
    //Perform the parallel solve.
    F77NAME(pdgbsv)(N, BWL, BWU, NRHS, B, JA, desca, ipivot, x, IB, descb, work, LW, &info);
//    F77NAME(pdgbtrs)('N', N, BWL, BWU, NRHS, B, JA, desca, ipivot, x, IB, descb, af, LAF, work, LW, &info);
    // Verify it completed successfully.
    if (info) {
        cout << "Error occurred in PDGBSV: " << info << endl;
    }
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+i] = x[(j-1)*(NBx-2)+(i-1)];
        }
    }
    
    delete [] B, x;
    
}

void PoissonSolver::ParallelJacobiSolver(int iter, int mpirank, double *w, double *psi, double *top, double *bot, double *left, double *right){
    
//    double err1, err2; 
    for(int k=0; k<iter; k++){//do{
//        err1 = err2;
        for(int i=1; i<NBx-1; i++){
            for(int j=1; j<NBy-1; j++){
                psi[j*NBx+i] = psi[j*NBx+i] + ( -(psi[j*NBx+i+1]+psi[j*NBx+i-1]-2*psi[j*NBx+i])*d1 - (psi[(j+1)*NBx+i]+psi[(j-1)*NBx+i]-2*psi[j*NBx+i])*d2 +w[j*NBx+i])/d0;
            }
        }
//        err2 = cblas_dnrm2((NBx-2)*(NBy-2), temp, 1);
//        for(int i=1; i<NBx-1; i++){
//            for(int j=1; j<NBy-1; j++){
//                psi[j*NBx+i] = temp_jacobi[(j-1)*(NBx-2)+(i-1)];
//            }
//        }
//        
        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, psi, top, bot, left, right);
    }//while(abs(err2-err1)>1e-10);

}