
#define LIDDRIVENCAVITY_H
#pragma once
/* preprocessor directive designed to cause the 
 * current source file to be included only once in a single compilation. */

#include "Functions.h" //basic functions used throughout the code

class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);

    void Initialise(int n, string *n_inputs);
    void Integrate();

    // Add any other public functions

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
    double U = 1.0; 
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
LidDrivenCavity::LidDrivenCavity(){ // constructor
}

LidDrivenCavity::~LidDrivenCavity(){ //destructor
    delete[] w;
    delete[] psi;
    
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen){
    Lx = xlen;
    Ly = ylen;
}

void LidDrivenCavity::SetGridSize(int nx, int ny){
    Nx = nx;
    Ny = ny;
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

void LidDrivenCavity::Integrate()
{
     int srw = Nx -2;
     int scl = Ny -2;
     
     double dx = Lx/(Nx-1);
     double dy = Ly/(Ny-1);
     
     if (dt >= Re*dx*dy/4){
         cout << "dt " << dt << ">=" << Re*dx*dy/4 << endl;
         cout << "dt is too big! Please enter a smaller dt. ";
         return;
     }
     
     //const double U = 1.0;
     
     double *A = new double[(2*srw+1+srw)*srw*scl](); 
     int *ipiv = new int[srw*scl*srw*scl];
     double a = 2*(1/dx/dx + 1/dy/dy), b  = -1/dx/dx , c = -1/dy/dy;
     BandedMatrixLapackLU(srw, scl, a, b, c, A, ipiv); // should be 2*srw+1 by srw^2 plus srw rows for padding
     
     //Initial Conditions
     w = new double[Nx*Ny]();
     psi = new double[Nx*Ny]();
     
     for (int nt=1; nt<T/dt; nt++){
         BoundaryVorticity(Nx, Ny, dx, dy, U, psi, w);
         
         InteriorVorticity(Nx, Ny, dx, dy, psi, w);
         
         NewInteriorVorticity(Re, Nx, Ny, dx, dy, dt, psi, w);
         
         PoissonProblem(Nx, Ny, srw, scl, A, w, psi, ipiv);
     }
     //printMatrix( (2*srw+1+srw),srw*scl, A);
     
     cout << "w" << endl;
    printMatrix( Nx, Ny, w);
//    cout << "psi" << endl;
//    printMatrix( Nx, Ny, psi);
}