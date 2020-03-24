//functions for the classes
#include "LidDrivenCavity.h"

LidDrivenCavity::LidDrivenCavity(){ // constructor
}

LidDrivenCavity::~LidDrivenCavity(){ //destructor
    delete w;
    delete psi;

    delete[] W;
    delete[] PSI;
    
    delete[] work_t; 
    delete[] work_b;
    delete[] work_l; 
    delete[] work_r;
    
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

void LidDrivenCavity::Initialise(int &argc, char *argv[]){
    
    if (argc != 19){
        cout << "Insufficient input from the command line!" << endl;
        return;
    }
    
    for(int i=1; i<argc; i=i+2){
        if (string(argv[i]) == "--Lx" && string(argv[i+2]) == "--Ly"){
            LidDrivenCavity::SetDomainSize( atof(argv[i+1]), atof(argv[i+3]) );
        }
        else if(string(argv[i]) == "--Nx" && string(argv[i+2]) == "--Ny"){
            LidDrivenCavity::SetGridSize( atoi(argv[i+1]), atoi(argv[i+3]) );
        }
        else if(string(argv[i]) == "--Px" && string(argv[i+2]) == "--Py"){
            LidDrivenCavity::SetPartitionSize( atoi(argv[i+1]), atoi(argv[i+3]) );
        }
        else if(string(argv[i]) == "--dt"){
            LidDrivenCavity::SetTimeStep(atof(argv[i+1]) );
        }
        else if(string(argv[i]) == "--T"){
            LidDrivenCavity::SetFinalTime(atof(argv[i+1]) );
        }
        else if(string(argv[i]) == "--Re"){
            LidDrivenCavity::SetReynoldsNumber( atof(argv[i+1]) );
        }
    }
    
    int retval_rank, retval_size;
    MPI_Init(&argc, &argv);
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM) {
        cout << "Invalid communicator" << endl;
        MPI_Finalize(); 
        return;
    } else if(np!= Px*Py){
        cout << "Invalid choice of processes!" << endl;
        MPI_Finalize(); 
        return;
    }
//     
//    // ... Set up CBLACS grid for Scalapack for A
//    int procrows = 1, proccols = Py*Px; // columns A will get divided into
//    int myid, numproc, ncol, nrow;
//    Cblacs_pinfo(&myid, &numproc);
//    Cblacs_get(0, 0, &ctx);
//    Cblacs_gridinit(&ctx, "Column-major", procrows, proccols);
//    Cblacs_gridinfo( ctx, &nrow, &ncol, &myrow, &mycol);
    
    // ... Set up CBLACS grid for easy division of w and psi
    int prows = Px, pcols = Py; // process rows and process columns
    int Tmyid, Tnumproc, Tncol, Tnrow;
    Cblacs_pinfo(&Tmyid, &Tnumproc);
    Cblacs_get(0, 0, &Tctx);
    Cblacs_gridinit(&Tctx, "Column-major", prows, pcols);
    Cblacs_gridinfo(Tctx, &Tnrow, &Tncol, &Tmyrow, &Tmycol);
    
    srw = Nx - 2;
    scl = Ny - 2;
    
    //Determining size of each partition block
    NBx = srw/Px;
    NBy = scl/Px;
//    cout << "Rank: " << mpirank << " No. of rows: " << NBx << " No. of cols: " << NBy << endl;
    
    int iZero = 0; //helping variable
    int sBrw = numroc_(&srw, &NBx, &Tmyrow, &iZero, &prows); 
    int sBcl = numroc_(&scl, &NBy, &Tmycol, &iZero, &pcols);
    
    cout << "Rank: " << mpirank << " No. of rows: " << sBrw << " No. of cols: " << sBcl << endl;
    //2 row and col padding to hold for BCs for each partition block
    NBx = sBrw + 2;
    NBy = sBcl + 2;
    Psize = sBrw*sBcl;
    
    dx = Lx/(Nx-1);
    dy = Ly/(Ny-1);
    
    if (dt >= Re*dx*dy/4){
         cout << "dt " << dt << ">=" << Re*dx*dy/4 << endl;
         cout << "dt is too big! Please enter a smaller dt. " << endl;
         return;
     }
     
    //Initialise Arrays
    w = new double[NBx*NBy]();
    psi = new double[NBx*NBy]();
    
    work_t = new double[NBx-2];
    work_b = new double[NBx-2];
    work_l = new double[NBy-2];
    work_r = new double[NBy-2];
    
}

void LidDrivenCavity::Integrate(int &argc, char *argv[]){
    
    double a0 = (1.0/dx/dx + 1.0/dy/dy), b1  = 1.0/dx/dx , c2 = 1.0/dy/dy;
    PoissonSolver *Sys = new PoissonSolver();
    Sys->SetUpSystem(a0, b1, c2, NBx, NBy, Tmyrow, Tmycol, Px, Py);
//    Sys->GenerateA_ScaLapack(mpirank, srw, scl);
//    Sys->ScaLapackSystemSetup(ctx, srw, scl);
    
    for (int nt=1; nt<=T/dt; nt++){
        
        LidDrivenCavity::BoundaryVorticity();
        
        LidDrivenCavity::InteriorVorticity();

        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, w, work_t, work_b, work_l, work_r);
        
        LidDrivenCavity::NewInteriorVorticity();
        
        Update(mpirank, Tmycol, Tmyrow, NBx, NBy, Px, Py, w, work_t, work_b, work_l, work_r);
        
        Sys->ParallelJacobiSolver(1000, mpirank, w, psi, work_t, work_b, work_l, work_r);
//        Sys->SolveProblem_ScaLapack(NBx, NBy, w, psi);
        
    }
    
    if(mpirank==0){
        W = new double[(Nx)*(Ny)]();
        PSI = new double[(Nx)*(Ny)]();
    }
    
    //Gathers and assembles the vorticities and stream function
    LidDrivenCavity::PatchUp();
    
    if(mpirank == 0){
        cout << "psi" << endl;
        printMatrix(Nx, Ny, PSI);
        cout << "w" << endl;
        printMatrix(Nx, Ny, W);
        
        Print2File(argc, argv, Nx, Ny, W, PSI, "csv");
    }
    
//    Cblacs_gridexit( ctx );
    Cblacs_gridexit( Tctx );
    MPI_Finalize(); 
}


void LidDrivenCavity::BoundaryVorticity(){
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
    double* t0 = new double[Psize]();
    double* t1 = new double[Psize]();
    double* t2 = new double[Psize]();
    double* t3 = new double[Psize]();
    double* t4 = new double[Psize]();
    
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
    delete[] t0;
    delete[] t1;
    delete[] t2;
    delete[] t3;
    delete[] t4;
}

void LidDrivenCavity::PatchUp(){

    double *w_rel = new double[Psize];
    double *psi_rel = new double[Psize];
    
    double *W_in;
    double *PSI_in;
    
    int *dim;
    int *dim_1;
    int *dim_2;
    
    int *displs;
    int *displ_row;
    int *displ_col;
    
    if(mpirank==0){
        W_in = new double[(Nx-2)*(Ny-2)]();
        PSI_in = new double[(Nx-2)*(Ny-2)]();
        
        dim   = new int[np](); // stores Psize
        dim_1 = new int[np](); // stores row sizes
        dim_2 = new int[np](); // stores col sizes
        
        displs    = new int[np]();
        displ_row = new int[np]();
        displ_col = new int[np]();
    }
    
    for(int i=1; i<NBx-1; i++){
        for(int j=1; j<NBy-1; j++){
            w_rel[(j-1)*(NBx-2)+i-1] = w[j*NBx+i];
            psi_rel[(j-1)*(NBx-2)+i-1] = psi[j*NBx+i];
        }
    }
    
    MPI_Gather(&NBx, 1, MPI_INT, &dim_1[mpirank], 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&NBy, 1, MPI_INT, &dim_2[mpirank], 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if(mpirank==0){
        for(int i=0; i<np; i++){
            dim_1[i] -= 2;
            dim_2[i] -= 2;
            dim[i] = dim_1[i]*dim_2[i];
        }
        
        for(int i=1; i<np; i++){
            displs[i] = dim[i-1] + displs[i-1];
        }
        for(int i=1; i<Px; i++){
            displ_row[i] = dim_1[i-1] + displ_row[i-1];
        }
        for(int i=1; i<Py; i++){
            displ_col[i] = dim_2[(i-1)*Px] + displ_col[i-1];
        }
    }
    
//    MPI_Gather(w_rel, Psize, MPI_DOUBLE, &W_in[displs[mpirank]], Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(w_rel, Psize, MPI_DOUBLE, W_in, dim, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Gather(psi_rel, Psize, MPI_DOUBLE, &PSI_in[displs[mpirank]], Psize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(psi_rel, Psize, MPI_DOUBLE, PSI_in, dim, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if(mpirank == 0){
        int m=0;
        int ind;
        for(int r=0; r<np ; r++){
            for(int i=0; i<dim_1[r]; i++){
                for(int j=0; j<dim_2[r]; j++){
//                    ind = (j + 1 + floor(r/Px) *dim_2[r])*Nx + (i+1) + r%Px*(dim_1[r]);
                    ind = (j + 1 + displ_col[r/Px])*Nx + (i+1) + displ_row[r%Px];
                    
                    W[ind] = W_in[ j*dim_1[r] + i + displs[r] ];
                    PSI[ind] = PSI_in[ j*dim_1[r] + i + displs[r] ];
                }
            }
            LidDrivenCavity::BoundaryVorticity_serial();
        }
        
        delete [] W_in;
        delete[] PSI_in;
        delete [] dim;
        delete[] dim_1; 
        delete[] dim_2;
    }
    
    delete [] w_rel;
    delete[] psi_rel;
//    free(displs);
}

void LidDrivenCavity::BoundaryVorticity_serial(){
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