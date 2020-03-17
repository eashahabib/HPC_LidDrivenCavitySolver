#include <mpi.h> // For MPI
using namespace std;

//#include "LidDrivenCavity.h"
#include "LidDrivenCavityP.h"

int main(int argc, char *argv[]){
    // Command line input
//    int n = argc;
    int n=19; 
    //string n_input [n] = {argv};
    string n_input [n] = {" ","--Lx", "1.0", "--Ly", "1.0", "--Nx", "12", "--Ny", 
                            "12", "--Px", "2","--Py", "2", "--dt", "0.1","--T", "2", 
                            "--Re", "100.0"};
    
    
    //ALL IS SUPPOSED TO BE COL MAJOR BUT I AM DUMB SO CHECK ON MISELV PLIS
    
    // Create a new instance of the LidDrivenCavity class
    LidDrivenCavity* solver = new LidDrivenCavity();
    
    // Configure the solver here...
    // ...
    
    solver->Initialise(n, n_input);
//    solver->Initialise(argc, argv);

    // Run the solver
    solver->Integrate(argc, argv);
    
    //delete

	return 0;
}