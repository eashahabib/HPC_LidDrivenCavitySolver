#define LIDDRIVENCAVITYJACOBI_H
#pragma once
/* preprocessor directive designed to cause the 
 * current source file to be included only once in a single compilation. */

#include "PoissonSolver.h"


class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetPartitionSize(int px, int py);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);
    

    void Initialise(int n, char *n_inputs[]);
    void Integrate(int &argc, char *argv[]);

    // Add any other public functions
    void BoundaryVorticity(); //Boundary Conditions for vorticity
    void BoundaryVorticity(int Tmycol, int Tmyrow); //Boundary Conditions for vorticity
    void InteriorVorticity();
//    void Update(int Tmyrow, int Tmycol, double *updated);
    void UpdateVorticity(int Tmyrow, int Tmycol);
    void NewInteriorVorticity();
//    void PoissonProblem(int mpirank, int ctx, double dx, double dy, int NBx, int NBy, int srw, int scl);
//    void UpdateStreamFunction(int myrow, int mycol);
//    void BandedMatrixScaLapack(int mpirank, int srw, int scl, int smallNBx, int smallNBy, double a, double b, double c);
    void PatchUp(int Tmyrow, int Tmycol);
    

private:
    double* w = nullptr; // vorticity
    double* psi = nullptr; // stream fuction

    double dt;
    double T;
    int    Nx;
    int    Ny;
    double Lx;
    double Ly;
    double Re;
    int    Px;
    int    Py;
    
    double U = 1.0;
    double dx;
    double dy;
    int    mpirank;
    int    NBx; // size of blocks in a partition
    int    NBy;
    int    srw;
    int    scl;
    
    double* W = nullptr; // vorticity
    double* PSI = nullptr; // stream fuction
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LidDrivenCavity::LidDrivenCavity(){ // constructor
}

LidDrivenCavity::~LidDrivenCavity(){ //destructor
    delete w;
    delete psi;
    delete W;
    delete PSI;
    
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen){
    Lx = xlen;
    Ly = ylen;
}

void LidDrivenCavity::SetGridSize(int nx, int ny){
    Nx = nx;
    Ny = ny;
}

void LidDrivenCavity::SetPartitionSize(int px, int py){
    Px = px;
    Py = py;
}

void LidDrivenCavity::SetTimeStep(double del_t){
    dt = del_t;
}

void LidDrivenCavity::SetFinalTime(double finalt){
    T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re){
    Re = re;
}

void LidDrivenCavity::Initialise(int n, char *n_inputs[]){
    for(int i=1; i<n; i=i+2){
        if (string(n_inputs[i]) == "--Lx" && string(n_inputs[i+2]) == "--Ly"){
            LidDrivenCavity::SetDomainSize( atof(n_inputs[i+1]), atof(n_inputs[i+3]) );
            //cout << "done 1" << endl;
        }
        else if(string(n_inputs[i]) == "--Nx" && string(n_inputs[i+2]) == "--Ny"){
            LidDrivenCavity::SetGridSize( atoi(n_inputs[i+1]), atoi(n_inputs[i+3]) );
            //cout << "done 2" << endl;
        }
        else if(string(n_inputs[i]) == "--Px" && string(n_inputs[i+2]) == "--Py"){
            LidDrivenCavity::SetPartitionSize( atoi(n_inputs[i+1]), atoi(n_inputs[i+3]) );
            //cout << "done 2" << endl;
        }
        else if(string(n_inputs[i]) == "--dt"){
            LidDrivenCavity::SetTimeStep(atof(n_inputs[i+1]) );
            //cout << "done 3" << endl;
        }
        else if(string(n_inputs[i]) == "--T"){
            LidDrivenCavity::SetFinalTime(atof(n_inputs[i+1]) );
            //cout << "done 4" << endl;
        }
        else if(string(n_inputs[i]) == "--Re"){
            LidDrivenCavity::SetReynoldsNumber( atof(n_inputs[i+1]) );
            //cout << "done 5" << endl;
        }
    }
    
    srw = Nx -2;
    scl = Ny -2;
    
    //2 row and col padding to hold for BCs for each partition block
    NBx = srw/Px+2; 
    NBy = scl/Py+2; 

    dx = Lx/(Nx-1);
    dy = Ly/(Ny-1);
    
    if (dt >= Re*dx*dy/4){
         cout << "dt " << dt << ">=" << Re*dx*dy/4 << endl;
         cout << "dt is too big! Please enter a smaller dt. ";
         return;
     }
    
}

void LidDrivenCavity::Integrate(int &argc, char *argv[]){
    
    int np, retval_rank, retval_size;
    MPI_Init(&argc, &argv);
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM) {
        cout << "Invalid communicator" << endl;
        return;
    } else if(np!= Px*Py){
        cout << "Invalid choice of processes!" << endl;
        return;
    }

     
    // ... Set up CBLACS grid for Scalapack for A
    int procrows = 1, proccols = Py*Px; // columns A will get divided into
    int ctx, myid, myrow, mycol, numproc, ncol, nrow;
    Cblacs_pinfo(&myid, &numproc);
    Cblacs_get(0, 0, &ctx);
    Cblacs_gridinit(&ctx, "Column-major", procrows, proccols);
    Cblacs_gridinfo( ctx, &nrow, &ncol, &myrow, &mycol);
    
    // ... Set up CBLACS grid for easy division of w and psi
    int prows = Px, pcols = Py;
    int Tctx, Tmyid, Tmyrow, Tmycol, Tnumproc, Tncol, Tnrow;
    Cblacs_pinfo(&Tmyid, &Tnumproc);
    Cblacs_get(0, 0, &Tctx);
    Cblacs_gridinit(&Tctx, "Column-major", prows, pcols);
    Cblacs_gridinfo(Tctx , &Tnrow, &Tncol, &Tmyrow, &Tmycol);
    

    double a0 = (1.0/dx/dx + 1.0/dy/dy), b1  = 1.0/dx/dx , c2 = 1.0/dy/dy;
    PoissonSolver *Sys = new PoissonSolver();
    Sys->SetDiagonal(a0, b1, c2);
//    Sys->GenerateA_ScaLapack(mpirank, srw, scl, NBx-2, NBy-2);
//    Sys->ScaLapackSystemSetup(ctx, NBx, NBy, srw, scl);
    

    //Initial Conditions
    w = new double[NBx*NBy]();
    psi = new double[NBx*NBy]();
    
    cout << "Rank: " << mpirank << " Row: " << Tmyrow << " Col: " << Tmycol << endl;
    
    for (int nt=1; nt<=T/dt; nt++){

        LidDrivenCavity::BoundaryVorticity(Tmycol, Tmyrow);
        
        LidDrivenCavity::InteriorVorticity();

//        LidDrivenCavity::UpdateVorticity(Tmyrow, Tmycol);
        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, w);
        
        LidDrivenCavity::NewInteriorVorticity();
        
        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, w);
        
        Sys->ParallelJacobiSolver(1000, mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, w, psi);
//        Sys->SolveProblem_ScaLapack(NBx, NBy, w, psi);
        
//        LidDrivenCavity::UpdateStreamFunction(Tmyrow, Tmycol);
        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, psi);
        
    }
    
    if(mpirank==0){
        W = new double[(Nx)*(Ny)]();
        PSI = new double[(Nx)*(Ny)]();
    }
    
    LidDrivenCavity::PatchUp(Tmyrow, Tmycol);
    
    if(mpirank == 0){
        cout << "psi" << endl;
        printMatrix(Nx, Ny, PSI);
        LidDrivenCavity::BoundaryVorticity();
        cout << "w" << endl;
        printMatrix(Nx, Ny, W);
        
        Print2File(argc, argv, Nx, Ny, w, psi);
        
        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, w);
        printMatrix(NBx, NBy, w);
    }
    
//    Cblacs_gridexit( ctx );
    Cblacs_gridexit( Tctx );
    MPI_Finalize(); 
}

void LidDrivenCavity::BoundaryVorticity(){
    for(int i=0; i<Nx; i++){
        //top
        W[Nx*(Ny-1)+i] = (PSI[Nx*(Ny-1)+i] - PSI[Nx*(Ny-1)+i-Nx])*2.0/dy/dy - 2.0*U/dy;
        //bottom
        W[i] = (PSI[i] - PSI[i+Nx])*2.0/dy/dy;
    }
    for(int i=0; i<Ny; i++){
        //left
        W[i*Nx] = (PSI[i*Nx] - PSI[i*Nx+1])*2.0/dx/dx;
        //right
        W[i*Nx+Nx-1] = (PSI[i*Nx+Nx-1] - PSI[i*Nx+Nx-2])*2.0/dx/dx;
    }
    
    //printMatrix( Nx, Ny, w);
}

void LidDrivenCavity::BoundaryVorticity(int Tmycol, int Tmyrow){
    if(Tmycol == Py-1){
        for(int i=0; i<NBx; i++){
            //top
            w[NBx*(NBy-1)+i] = (psi[NBx*(NBy-1)+i] - psi[NBx*(NBy-1)+i-NBx])*2/dy/dy - 2.0*U/dy;
        }
    }
    if(Tmycol == 0){
        for(int i=0; i<NBx; i++){
            //bottom
            w[i] = (psi[i] - psi[i+NBx])*2/dy/dy;
        }
    }
    if(Tmyrow == 0){
        for(int i=0; i<NBy; i++){
            //left
            w[i*NBx] = (psi[i*NBx] - psi[i*NBx+1])*2/dx/dx;
        }
    }
    if(Tmyrow == Px-1){
        for(int i=0; i<NBy; i++){
            //right
            w[i*NBx+NBx-1] = (psi[i*NBx+NBx-1] - psi[i*NBx+NBx-2])*2/dx/dx;
        }
    }
    
}

void LidDrivenCavity::InteriorVorticity(){
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+i] = -(psi[j*NBx+i+1] - 2.0*psi[j*NBx+i] + psi[j*NBx+i-1])/dx/dx - (psi[(j+1)*NBx+i] - 2.0*psi[j*NBx+i] + psi[(j-1)*NBx+i])/dy/dy;
        }
    }
}

void LidDrivenCavity::NewInteriorVorticity(){
    //pringles.
    double* t0 = new double[(NBx-2)*(NBy-2)]();
    double* t1 = new double[(NBx-2)*(NBy-2)]();
    double* t2 = new double[(NBx-2)*(NBy-2)]();
    double* t3 = new double[(NBx-2)*(NBy-2)]();
    double* t4 = new double[(NBx-2)*(NBy-2)]();
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            t0[(j-1)*(NBx-2)+(i-1)] = ( (w[j*NBx+i+1] - 2.0*w[j*NBx+i] + w[j*NBx+i-1])/dx/dx + (w[(j+1)*NBx+i] - 2.0*w[j*NBx+i] + w[(j-1)*NBx+i])/dy/dy )/Re;
            t1[(j-1)*(NBx-2)+(i-1)] = (psi[(j+1)*NBx+i] - psi[(j-1)*NBx+i])/2.0/dy;
            t2[(j-1)*(NBx-2)+(i-1)] = (w[j*NBx+i+1] - w[j*NBx+i-1])/2.0/dx;
            t3[(j-1)*(NBx-2)+(i-1)] = (psi[j*NBx+i+1] - psi[j*NBx+i-1])/2.0/dx;
            t4[(j-1)*(NBx-2)+(i-1)] = (w[(j+1)*NBx+i] - w[(j-1)*NBx+i])/2.0/dy;
        }
    }
    //appending inside the same loop resulted in incorrect results
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+i] += (t0[(j-1)*(NBx-2)+(i-1)] - t1[(j-1)*(NBx-2)+(i-1)]*t2[(j-1)*(NBx-2)+(i-1)] + t3[(j-1)*(NBx-2)+(i-1)]*t4[(j-1)*(NBx-2)+(i-1)])*dt;
        }
    }
//    cout << "w" << endl;
//    printMatrix( Nx, Ny, w);
    delete[] t0, t1, t2, t3, t4;
}

void LidDrivenCavity::PatchUp(int Tmyrow, int Tmycol){
    int Psize = (NBx-2)*(NBy-2);
    double *w_rel = new double[Psize];
    double *psi_rel = new double[Psize];
    
    double *W_in;
    double *PSI_in;
    
    if(mpirank==0){
        W_in = new double[(Nx-2)*(Ny-2)]();
        PSI_in = new double[(Nx-2)*(Ny-2)]();
    }
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            w_rel[(j-1)*(NBx-2)+i-1] = w[j*NBx+i];
            psi_rel[(j-1)*(NBx-2)+i-1] = psi[j*NBx+i];
        }
    }
    
    MPI_Gather(w_rel, Psize, MPI_DOUBLE, &W_in[mpirank*Psize], Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(psi_rel, Psize, MPI_DOUBLE, &PSI_in[mpirank*Psize], Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(mpirank == 0){
        int ind;
        for(int r=0; r<Px*Py ; r++){
            for(int i=0; i<NBx-2; i++){
                for(int j=0; j<NBy-2; j++){
                    ind = (j+1+ floor(r/Px) *(NBy-2))*Nx+(i+1)+ r%Px *(NBx-2);
//                    cout << "row" << r%Px << endl;
//                    cout << "col" << floor(r/Px) << endl;
                    W[ind] = W_in[j*(NBx-2)+i +mpirank*Psize];
                    PSI[ind] = PSI_in[j*(NBx-2)+i +mpirank*Psize];
                }
            }
        }
    }
    delete [] w_rel, psi_rel;
}

