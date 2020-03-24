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
    

    void Initialise(int &argc, char *argv[]);
    void Integrate(int &argc, char *argv[]); //Solver for the Finite Difference method for the Lid Driven cavity

    // Add any other public functions
    void BoundaryVorticity(); //Calculation of vorticity boundary conditions at time t 
    void InteriorVorticity(); //Calculation of interior vorticity at time t
    void NewInteriorVorticity(); // Calculation of interior vorticity at time t+dt
    void PatchUp(); // Assembly of the processors
    void BoundaryVorticity_serial(); //Calculation of vorticity boundary conditions at time t for the whole process
    

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
    
    const double U = 1.0;
    double dx;
    double dy;
    int    mpirank;
    int    np;
    int    NBx; // size of blocks in a partition
    int    NBy;
    int    srw;
    int    scl;
    int    ctx, Tctx;
    int    myrow, mycol;
    int    Tmyrow, Tmycol;
    int    Psize;
    
    double* W = nullptr; // vorticity
    double* PSI = nullptr; // stream fuction
    
    // Communicator arrays
    double *work_t = nullptr;
    double *work_b = nullptr;
    double *work_l = nullptr;
    double *work_r = nullptr;
};