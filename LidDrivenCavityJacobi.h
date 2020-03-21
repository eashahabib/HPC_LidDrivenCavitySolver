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
    void JacobiSolver(int iter, int Tmycol, int Tmyrow);
    

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

void LidDrivenCavity::JacobiSolver(int iter, int Tmycol, int Tmyrow){
    double *temp = new double[(NBx-2)*(NBy-2)]();
    double err1, err2; 
    for(int k=0; k<iter; k++){
    //do{
        err1 = err2;
        for(int i=1; i<NBx-1; i++){
            for(int j=1; j<NBy-1; j++){
                temp[(j-1)*(NBx-2)+(i-1)] = psi[j*NBx+i] + ((psi[j*NBx+i+1]+psi[j*NBx+i-1]-2*psi[j*NBx+i])/dx/dx + (psi[(j+1)*NBx+i]+psi[(j-1)*NBx+i]-2*psi[j*NBx+i])/dy/dy +w[j*NBx+i])/(2/dx/dx + 2/dy/dy);
            }
        }
        err2 = cblas_dnrm2((NBx-2)*(NBy-2), temp, 1);
        for(int i=1; i<NBx-1; i++){
            for(int j=1; j<NBy-1; j++){
                psi[j*NBx+i] = temp[(j-1)*(NBx-2)+(i-1)];
            }
        }
        LidDrivenCavity::UpdateStreamFunction(Tmyrow, Tmycol);
    }//while(abs(err2-err1)>1e-10);
    delete[] temp;
}

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
    

    //Initial Conditions
    w = new double[NBx*NBy]();
    psi = new double[NBx*NBy]();
    
    cout << "Rank: " << mpirank << " Row: " << Tmyrow << " Col: " << Tmycol << endl;
    
    for (int nt=1; nt<T/dt+1; nt++){

        LidDrivenCavity::BoundaryVorticity(Tmycol, Tmyrow);
        
        LidDrivenCavity::InteriorVorticity();

        LidDrivenCavity::UpdateVorticity(Tmyrow, Tmycol);
        
        LidDrivenCavity::NewInteriorVorticity();
        
        Sys->ParallelJacobiSolver(250, Tmycol, Tmyrow, NBx, NBy, w, psi);
        
//        LidDrivenCavity::JacobiSolver(100, Tmycol, Tmyrow);
        
        LidDrivenCavity::UpdateStreamFunction(Tmyrow, Tmycol);
        printMatrix(NBx, NBy, psi);
    }
    
    //printMatrix(NBx, NBy, w);    
    
    if(mpirank==0){
        W = new double[(Nx)*(Ny)]();
        PSI = new double[(Nx)*(Ny)]();
    }
    
    LidDrivenCavity::PatchUp(Tmyrow, Tmycol);
    
    if(mpirank == 0){
//        cout << "psi" << endl;
//        printMatrix(Nx, Ny, PSI);
        LidDrivenCavity::BoundaryVorticity();
//        cout << "w" << endl;
//        printMatrix(Nx, Ny, W);
    }
    
//    Cblacs_gridexit( ctx );
    Cblacs_gridexit( Tctx );
    MPI_Finalize(); 
}

void LidDrivenCavity::BoundaryVorticity(){
    for(int i=0; i<Nx; i++){
        //top
        W[Nx*(Ny-1)+i] = (PSI[Nx*(Ny-1)+i] - PSI[Nx*(Ny-1)+i-Nx])*2/dy/dy - 2.0*U/dy;
        //bottom
        W[i] = (PSI[i] - PSI[i+Nx])*2/dy/dy;
    }
    for(int i=0; i<Ny; i++){
        //left
        W[i*Nx] = (PSI[i*Nx] - PSI[i*Nx+1])*2/dx/dx;
        //right
        W[i*Nx+Nx-1] = (PSI[i*Nx+Nx-1] - PSI[i*Nx+Nx-2])*2/dx/dx;
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
    double *top = new double[NBx-2];
    double *bot = new double[NBx-2];
    double *left = new double[NBy-2];
    double *right = new double[NBy-2];
    
    if(Tmycol != Py-1 && Tmycol != 0){ //not a top or bottom
        for(int i=1; i<NBx-1; i++){
            top[i-1] = w[(NBy-2)*NBx+i];
            bot[i-1] = w[1*NBx+i];
        }
        
        MPI_Send(top, NBx-2, MPI_DOUBLE, mpirank+Px, 1, MPI_COMM_WORLD);
        MPI_Send(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 2, MPI_COMM_WORLD);
        
    }else if(Tmycol == Py-1){// its a top
        
        for(int i=1; i<NBx-1; i++){
            bot[i-1] = w[1*NBx+i];
        }
        
        MPI_Send(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 2, MPI_COMM_WORLD);
        
    }else if(Tmycol == 0){ // its a bottom
        
        for(int i=1; i<NBx-1; i++){
            top[i-1] = w[(NBy-2)*NBx+i];
        }
        
        MPI_Send(top, NBx-2, MPI_DOUBLE, mpirank+Px, 1, MPI_COMM_WORLD);
        
    }
    
    if(Tmyrow != Px-1 && Tmyrow != 0){ //middle
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = w[j*NBx+1];
            right[j-1] = w[j*NBx+(NBx-2)];
        }
        
        MPI_Send(left, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        MPI_Send(right, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
    }else if(Tmyrow == Px-1){// its a right
        
        for(int j=1; j<NBy-1; j++){
            right[j-1] = w[j*NBx+(NBx-2)];
        }
        
        MPI_Send(right, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        
    }else if(Tmyrow == 0){ // its a left
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = w[j*NBx+1];
        }
        
        MPI_Send(left, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
    }
    //////////////////////////////////////////////////////////////////
    if(Tmycol != Py-1 && Tmycol != 0){ //not a top or bottom
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank+Px, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            w[(NBy-1)*NBx+i] = top[i-1];
            w[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == Py-1){// its a top
        
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            w[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == 0){ // its a bottom
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank+Px, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            w[(NBy-1)*NBx+i] = top[i-1];
        }
    }
    ///////////////////////////////////////////////////////////////////
    
    if(Tmyrow != Px-1 && Tmyrow != 0){ //middle
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+0] = left[j-1];
            w[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == Px-1){// its a right
        
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            w[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == 0){ // its a left
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
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
            t1[(j-1)*(NBx-2)+(i-1)] = (psi[(j+1)*NBx+i] - psi[(j-1)*NBx+i])/2/dy;
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
    double *top = new double[NBx-2];
    double *bot = new double[NBx-2];
    double *left = new double[NBy-2];
    double *right = new double[NBy-2];
    
    if(Tmycol != Py-1 && Tmycol != 0){ //not a top or bottom
        for(int i=1; i<NBx-1; i++){
            top[i-1] = psi[(NBy-2)*NBx+i];
            bot[i-1] = psi[1*NBx+i];
        }
        
        MPI_Send(top, NBx-2, MPI_DOUBLE, mpirank+Px, 1, MPI_COMM_WORLD);
        MPI_Send(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 2, MPI_COMM_WORLD);
        
    }else if(Tmycol == Py-1){// its a top
        
        for(int i=1; i<NBx-1; i++){
            bot[i-1] = psi[1*NBx+i];
        }
        
        MPI_Send(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 2, MPI_COMM_WORLD);
        
    }else if(Tmycol == 0){ // its a bottom
        
        for(int i=1; i<NBx-1; i++){
            top[i-1] = psi[(NBy-2)*NBx+i];
        }
        
        MPI_Send(top, NBx-2, MPI_DOUBLE, mpirank+Px, 1, MPI_COMM_WORLD);
        
    }
    
    if(Tmyrow != Px-1 && Tmyrow != 0){ //middle
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = psi[j*NBx+1];
            right[j-1] = psi[j*NBx+(NBx-2)];
        }
        
        MPI_Send(left, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        MPI_Send(right, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
    }else if(Tmyrow == Px-1){// its a right
        
        for(int j=1; j<NBy-1; j++){
            right[j-1] = psi[j*NBx+(NBx-2)];
        }
        
        MPI_Send(right, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD);
        
    }else if(Tmyrow == 0){ // its a left
        
        for(int j=1; j<NBy-1; j++){
            left[j-1] = psi[j*NBx+1];
        }
        
        MPI_Send(left, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD);
        
    }
    //////////////////////////////////////////////////////////////////
    if(Tmycol != Py-1 && Tmycol != 0){ //not a top or bottom
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank+Px, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            psi[(NBy-1)*NBx+i] = top[i-1];
            psi[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == Py-1){// its a top
        
        MPI_Recv(bot, NBx-2, MPI_DOUBLE, mpirank-Px, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            psi[0*NBx+i] = bot[i-1];
        }
    }else if(Tmycol == 0){ // its a bottom
        
        MPI_Recv(top, NBx-2, MPI_DOUBLE, mpirank+Px, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int i=1; i<NBx-1; i++){
            psi[(NBy-1)*NBx+i] = top[i-1];
        }
    }
    ///////////////////////////////////////////////////////////////////
    
    if(Tmyrow != Px-1 && Tmyrow != 0){ //middle
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+0] = left[j-1];
            psi[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == Px-1){// its a right
        
        MPI_Recv(right, NBy-2, MPI_DOUBLE, mpirank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+(NBx-1)] = right[j-1];
        }
    }else if(Tmyrow == 0){ // its a left
        
        MPI_Recv(left, NBy-2, MPI_DOUBLE, mpirank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for(int j=1; j<NBy-1; j++){
            psi[j*NBx+0] = left[j-1];
        }
    }
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
    
    
    MPI_Gather(w_rel, Psize, MPI_DOUBLE, W_in, Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(psi_rel, Psize, MPI_DOUBLE, PSI_in, Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(mpirank == 0){
        for(int i=1; i<NBx-1; i++){
            for(int j=1; j<NBy-1; j++){
                W[j*NBx+i] = W_in[(j-1)*(NBx-2)+i-1];
                PSI[j*NBx+i] = PSI_in[(j-1)*(NBx-2)+i-1];
            }
        }
    }
    
}