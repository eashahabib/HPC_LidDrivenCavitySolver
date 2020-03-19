
#define LIDDRIVENCAVITYP_H
#pragma once
/* preprocessor directive designed to cause the 
 * current source file to be included only once in a single compilation. */

#include "Functions.h" //basic functions and declarations used throughout the code
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
    

    void Initialise(int n, string *n_inputs);
    void Integrate(int &argc, char *argv[]);

    // Add any other public functions
    void BoundaryVorticity(); //Boundary Conditions for vorticity
    void BoundaryVorticity(int Tmycol, int Tmyrow); //Boundary Conditions for vorticity
    void InteriorVorticity();
//    void Update(int Tmyrow, int Tmycol, double *updated);
    void UpdateVorticity(int myrow, int mycol);
    void NewInteriorVorticity();
//    void PoissonProblem(int mpirank, int ctx, double dx, double dy, int NBx, int NBy, int srw, int scl);
    void UpdateStreamFunction(int myrow, int mycol);
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

void LidDrivenCavity::Initialise(int n, string *n_inputs){
    for(int i=1; i<n; i=i+2){
        if (n_inputs[i] == "--Lx" && n_inputs[i+2] == "--Ly"){
            LidDrivenCavity::SetDomainSize( stod(n_inputs[i+1]), stod(n_inputs[i+3]) );
            //cout << "done 1" << endl;
        }
        else if(n_inputs[i] == "--Nx" && n_inputs[i+2] == "--Ny"){
            LidDrivenCavity::SetGridSize( stoi(n_inputs[i+1]), stoi(n_inputs[i+3]) );
            //cout << "done 2" << endl;
        }
        else if(n_inputs[i] == "--Px" && n_inputs[i+2] == "--Py"){
            LidDrivenCavity::SetPartitionSize( stoi(n_inputs[i+1]), stoi(n_inputs[i+3]) );
            //cout << "done 2" << endl;
        }
        else if(n_inputs[i] == "--dt"){
            LidDrivenCavity::SetTimeStep(stod(n_inputs[i+1]) );
            //cout << "done 3" << endl;
        }
        else if(n_inputs[i] == "--T"){
            LidDrivenCavity::SetFinalTime(stod(n_inputs[i+1]) );
            //cout << "done 4" << endl;
        }
        else if(n_inputs[i] == "--Re"){
            LidDrivenCavity::SetReynoldsNumber( stod(n_inputs[i+1]) );
            //cout << "done 5" << endl;
        }
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
    }

    int srw = Nx -2;
    int scl = Ny -2;
    
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
    

    double a0 = 2*(1/dx/dx + 1/dy/dy), b1  = -1/dx/dx , c2 = -1/dy/dy;
    PoissonSolver *Sys = new PoissonSolver();
    Sys->SetDiagonal(a0, b1, c2);
    Sys->GenerateA_ScaLapack(mpirank, srw, scl, NBx-2, NBy-2);
    Sys->ScaLapackSystemSetup(ctx, NBx, NBy, srw, scl);

    //Initial Conditions
    w = new double[NBx*NBy]();
    psi = new double[NBx*NBy]();
    
    cout << "Rank: " << mpirank << " Row: " << Tmyrow << " Col: " << Tmycol << endl;

//    for (int nt=1; nt<T/dt+1; nt++){
//        cout << nt << endl;
        LidDrivenCavity::BoundaryVorticity(Tmycol, Tmyrow);
        
        LidDrivenCavity::InteriorVorticity();

        LidDrivenCavity::UpdateVorticity(Tmyrow, Tmycol);
        
        LidDrivenCavity::NewInteriorVorticity();
        
////        PoissonProblem(mpirank, ctx, dx, dy, NBx, NBy, srw, scl, psi, w);
        Sys->SolveProblem_ScaLapack(NBx, NBy, w, psi);
        
        LidDrivenCavity::UpdateStreamFunction(Tmyrow, Tmycol);
        
//        printMatrix(NBx, NBy, w);
//    }
        printMatrix(NBx, NBy, w);    
    if(mpirank==0){
        W = new double[(Nx-2)*(Ny-2)]();
        PSI = new double[(Nx-2)*(Ny-2)]();
    }
    
    LidDrivenCavity::PatchUp(Tmyrow, Tmycol);
    
    Cblacs_gridexit( ctx );
    Cblacs_gridexit( Tctx );
    MPI_Finalize(); 
    
//     cout << "w" << endl;
//    printMatrix( Nx, Ny, w);
//    cout << "psi" << endl;
//    printMatrix( Nx, Ny, psi);
}

void LidDrivenCavity::BoundaryVorticity(){
    for(int i=0; i<Nx; i++){
        //top
        w[Nx*(Ny-1)+i] = (psi[Nx*(Ny-1)+i] - psi[Nx*(Ny-1)+i-Nx])*2/dy/dy - 2.0*U/dy;
        //bottom
        w[i] = (psi[i] - psi[i+Nx])*2/dy/dy;
    }
    for(int i=0; i<Ny; i++){
        //left
        w[i*Nx] = (psi[i*Nx] - psi[i*Nx+1])*2/dx/dx;
        //right
        w[i*Nx+Nx-1] = (psi[i*Nx+Nx-1] - psi[i*Nx+Nx-2])*2/dx/dx;
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
            w[j*NBx+i] = -(psi[j*NBx+i+1] - 2*psi[j*NBx+i] + psi[j*NBx+i-1])/dx/dx - (psi[(j+1)*NBx+i] - 2*psi[j*NBx+i] + psi[(j-1)*NBx+i])/dy/dy;
        }
    }
}

void LidDrivenCavity::UpdateVorticity(int Tmyrow, int Tmycol){
    if(Tmycol == Py-1 || Tmycol == 0){
        
    }else if(Tmycol != Py-1 || Tmycol != 0){ //not a top or bottom
        double *top = new double[NBx-2];
        double *bot = new double[NBx-2];
        
        for(int i=1; i<NBx-1; i++){
            top[i-1] = w[(NBy-2)*NBx+i];
            bot[i-1] = w[1*NBx+i];
        }
        
        MPI_Ssend(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        MPI_Ssend(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            w[(NBy-1)*NBx+i] = top[i-1];
            w[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == Py-1){// its a top
        double *bot = new double[NBx-2];
        
        for(int i=1; i<NBx-1; i++){
            bot[i-1] = w[1*NBx+i];
        }
        
        MPI_Ssend(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            w[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == 0){ // its a bottom
        double *top = new double[NBx-2];
        
        for(int i=1; i<NBx-1; i++){
            top[i-1] = w[(NBy-2)*NBx+i];
        }
        
        MPI_Ssend(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            w[(NBy-1)*NBx+i] = top[i-1];
        }
    }
    
    if(Tmycol == Py-1 || Tmycol == 0){
        
    }else if(Tmyrow != Px-1 || Tmyrow != 0){ //middle
        double *left = new double[NBy-2];
        double *right = new double[NBy-2];
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = w[j*NBx+1];
            right[j-1] = w[j*NBx+(NBx-2)];
        }
        
        MPI_Ssend(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD);
        MPI_Ssend(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD);
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+0] = left[j-1];
            w[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == Px-1){// its a left
        double *right = new double[NBy-2];
        
        for(int j=1; j<NBy-1; j++){
            right[j-1] = w[j*NBx+(NBx-2)];
        }
        
        MPI_Ssend(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD);
        
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == 0){ // its a right
        double *left = new double[NBy-2];
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = w[j*NBx+1];
        }
        
        MPI_Ssend(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD);
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+0] = left[j-1];
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
            t0[(j-1)*(NBx-2)+(i-1)] = ( (w[j*NBx+i+1] - 2*w[j*NBx+i] + w[j*NBx+i-1])/dx/dx + (w[(j+1)*NBx+i] - 2*w[j*NBx+i] + w[(j-1)*NBx+i])/dy/dy )/Re;
            t1[(j-1)*(NBx-2)+(i-1)] = (psi[(j+1)*Nx+i] - psi[(j-1)*Nx+i])/2/dy;
            t2[(j-1)*(NBx-2)+(i-1)] = (w[j*NBx+i+1] - w[j*NBx+i-1])/2/dx;
            t3[(j-1)*(NBx-2)+(i-1)] = (psi[j*NBx+i+1] - psi[j*NBx+i-1])/2/dx;
            t4[(j-1)*(NBx-2)+(i-1)] = (w[(j+1)*NBx+i] - w[(j-1)*NBx+i])/2/dy;
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
//    delete[] t0, t1, t2, t3, t4;
}

void LidDrivenCavity::UpdateStreamFunction(int Tmyrow, int Tmycol){
    if(Tmycol == Py-1 || Tmycol == 0){ // huhgu
        
    }else if(Tmycol != Py-1 || Tmycol != 0){ //not a top or bottom
        double *top = new double[NBx-2];
        double *bot = new double[NBx-2];
        
        for(int i=1; i<NBx-1; i++){
            top[i-1] = psi[(NBy-2)*NBx+i];
            bot[i-1] = psi[1*NBx+i];
        }
        
        MPI_Ssend(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        MPI_Ssend(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            psi[(NBy-1)*NBx+i] = top[i-1];
            psi[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == Py-1){// its a top
        double *bot = new double[NBx-2];
        
        for(int i=1; i<NBx-1; i++){
            bot[i-1] = psi[1*NBx+i];
        }
        
        MPI_Ssend(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            psi[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == 0){ // its a bottom
        double *top = new double[NBx-2];
        
        for(int i=1; i<NBx-1; i++){
            top[i-1] = psi[(NBy-2)*NBx+i];
        }
        
        MPI_Ssend(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            psi[(NBy-1)*NBx+i] = top[i-1];
        }
    }
    
    if(Tmycol == Py-1 || Tmycol == 0){
        
    }else if(Tmyrow != Px-1 || Tmyrow != 0){ //middle
        double *left = new double[NBy-2];
        double *right = new double[NBy-2];
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = psi[j*NBx+1];
            right[j-1] = psi[j*NBx+(NBx-2)];
        }
        
        MPI_Ssend(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD);
        MPI_Ssend(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD);
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+0] = left[j-1];
            psi[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == Px-1){// its a left
        double *right = new double[NBy-2];
        
        for(int j=1; j<NBy-1; j++){
            right[j-1] = psi[j*NBx+(NBx-2)];
        }
        
        MPI_Ssend(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD);
        
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == 0){ // its a right
        double *left = new double[NBy-2];
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = psi[j*NBx+1];
        }
        
        MPI_Ssend(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD);
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1*Px, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+0] = left[j-1];
        }
    }
}

void LidDrivenCavity::PatchUp(int Tmyrow, int Tmycol){
    int Psize = (NBx-2)*(NBy-2);
    double *w_rel = new double[Psize];
    double *psi_rel = new double[Psize];
    
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            w_rel[(j-1)*(NBx-2)+i-1] = w[j*NBx+i];
            psi_rel[(j-1)*(NBx-2)+i-1] = psi[j*NBx+i];
        }
    }
    
   // sent to badcode in Z2
    
    //printMatrix(NBx-1, NBy-1, w_rel);
    //printMatrix(NBx-1, NBy-1, psi_rel);
    MPI_Gather(w_rel, Psize, MPI_DOUBLE, W, Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(psi_rel, Psize, MPI_DOUBLE, PSI, Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //printMatrix(Nx, Ny, PSI);
    //delete[] w_rel, psi_rel;
}

//void LidDrivenCavity::PoissonProblem(int ctx, int NBx, int NBy, int srw, int scl){
////solving a system A u = f
//    
//    int info; // Status value
//    const int N = (srw)*(scl); // Total problem size
//    const int NB = (NBx-2)*(NBy-2); // Blocking size (number of columns per process)
//    const int BWL = srw; // Lower bandwidth
//    const int BWU = srw; // Upper bandwidth
//    const int NRHS = 1; // Number of RHS to solve
//    const int JA = 1; // Start offset in matrix (to use just a submatrix)
//    const int IB = 1; // Start offset in RHS vector (to use just a subvector)
//    const int LA = (1 + 2*BWL + 2*BWU)*NB;
//    const int LW = (NB+BWU)*(BWL+BWU)+6*(BWL+BWU)*(BWL+2*BWU) + max(NRHS*(NB+2*BWL+4*BWU), 1); // ScaLAPACK documentation
//    double* work = new double[LW](); // Workspace
//    double* A = new double[LA](); // Matrix banded storage
//    
//    cout << "N: " << N << endl;
//    cout << "NB: " << NB << endl;
//    
//    int row = (1 + 2*BWL + 2*BWU);
//    double a = 2*(1/dx/dx+1/dy/dy), b = -1/dy/dy, c = -1/dx/dx;
//    BandedMatrixScaLapack(mpirank, srw, scl, NBx-2, NBy-2, a, b, c, A);
//    
//    //printMatrix(row, NB, A); 
//    
//    int* ipiv = new int [NB](); // Pivoting array
//    double* x = new double[NB](); // In: RHS vector, Out: Solution;
//    
//    for(int i=1; i<NBx-1; i++){
//        for(int j=1; j<NBy-1; j++){
//            x[(j-1)*(NBx-2)+(i-1)] = w[j*NBx+i];
//        }
//    }
//    
//    int desca[7]; // Descriptor for banded matrix
//    desca[0] = 501; // Type
//    desca[1] = ctx; // Context
//    desca[2] = N; // Problem size
//    desca[3] = NB; // Blocking of matrix
//    desca[4] = 0; // Process row/column
//    desca[5] = row; // Local leading dim
//    desca[6] = 0; // Reserved
//    int descb[7]; // Descriptor for RHS
//    descb[0] = 502; // Type
//    descb[1] = ctx; // Context
//    descb[2] = N; // Problem size
//    descb[3] = NB; // Blocking of matrix
//    descb[4] = 0; // Process row/column
//    descb[5] = NB; // Local leading dim
//    descb[6] = 0; // Reserved
//    
//    //Perform the parallel solve.
//    F77NAME(pdgbsv) (N, BWL, BWU, NRHS, A, JA, desca, ipiv, x, IB, descb, work, LW, &info);
//    // Verify it completed successfully.
//    if (info) {
//        cout << "Error occurred in PDGBSV: " << info << endl;
//    }
//    
//    for(int i=1; i<NBx-1; i++){
//        for(int j=1; j<NBy-1; j++){
//            psi[j*NBx+i] = x[(j-1)*(NBx-2)+(i-1)];
//        }
//    }
//    
//    delete[] ipiv, work, x, desca, descb, A; 
//    
//}
//
//void LidDrivenCavity::BandedMatrixScaLapack(int mpirank, int srw, int scl, int smallNBx, int smallNBy, double a, double b, double c){
//    int col = 2*scl+1+2*srw; 
//    int row = smallNBx*smallNBy;
//    for(int i=0; i<row; i++){
//        A[i*col + 0+scl+srw] = c;
//        A[i*col + scl-1+scl+srw] = b;
//        A[i*col + scl+scl+srw] = a;
//        A[i*col + scl+1+scl+srw] = b;
//        A[i*col + 2*scl+2*srw] = c;
//    }
//    
//    int j = 0;
//    for(int i=0+mpirank*row; i<row - (scl-1) +mpirank*row; i=i+scl){
//        A[(i)*col + scl-1 +scl+srw] = 0;
//        A[(i+scl-1)*col + scl+1 +scl+srw] = 0;
//    }
//}
//
