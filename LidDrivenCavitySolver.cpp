#include <mpi.h> // For MPI
using namespace std;

//#include "LidDrivenCavity.h" //Serial
//#include "LidDrivenCavityP.h" //Scalapack
#include "LidDrivenCavity.cpp" //Jacobi
#include "PoissonSolver.cpp"

int main(int argc, char *argv[]){
    // Command line input
//    int n = argc;
    int n=19; 
    //string n_input [n] = {argv};
    string n_input [n] = {" ","--Lx", "1.0", "--Ly", "1.0", "--Nx", "10", "--Ny", 
                            "10", "--Px", "1","--Py", "1", "--dt", "0.001","--T", "0.005", 
                            "--Re", "100.0"};
    
    
    // A new instance of the LidDrivenCavity class
    LidDrivenCavity* solver = new LidDrivenCavity();
    
    // Initialise the solver using the inputs
    solver->Initialise(argc, argv);

    // Run the solver
    solver->Integrate(argc, argv);

	return 0;
}

/* To run the code, use this command line syntax:
 --Lx 1.0 --Ly 1.0 --Nx 10 --Ny 10 --Px 1 --Py 1 --dt 0.001 --T 0.005 --Re 100.0

*/