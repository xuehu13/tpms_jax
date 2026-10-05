// Frozen reference Gauss occupancy; Abaqus supplies native element kinematics.
// C ABI entry points for the installed Windows x64 Standard solver.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <vector>
#include "field_config.h"

namespace {
std::vector<double> kphi;
std::once_flag kloaded;
void kload() {
    FILE* f = std::fopen(K_FIELD_FILE, "rb");
    const size_t m = static_cast<size_t>(2*K_N)*(2*K_N)*(2*K_N);
    kphi.resize(m);
    if (!f || std::fread(kphi.data(), sizeof(double), m, f) != m) {
        std::fprintf(stderr, "TPMS reference field cannot be read\n"); std::abort();
    }
    std::fclose(f);
}
void ksdvini(double* state, double* coords, int* nstate, int* ncoords) {
    if (*nstate != 1 || *ncoords != 3) std::abort();
    std::call_once(kloaded, kload);
    int j[3];
    for (int a=0;a<3;++a) {
        const double x=coords[a]*K_N/K_L;
        const int cell=static_cast<int>(std::floor(x));
        const int high=(x-cell > .5);
        const double expected=cell+.5+(high ? 1 : -1)*.5/std::sqrt(3.);
        if (cell<0 || cell>=K_N || std::fabs(x-expected)>1e-7) {
            std::fprintf(stderr, "TPMS initial Gauss coordinate mismatch %.17g\n",x);
            std::abort();
        }
        j[a]=2*cell+high;
    }
    const int m=2*K_N;
    state[0]=kphi[(static_cast<size_t>(j[0])*m+j[1])*m+j[2]];
}
void kuhyper(double* i1,double* J,double* U,double* d1,double* d2,
             double* d3,int* nstate,double* state,int* nprops,double* props) {
    if (*nstate!=1 || *nprops!=3 || !std::isfinite(*J) || *J<=0) std::abort();
    const double s=props[2]+(1-props[2])*state[0];
    const double mu=props[0]*s, bulk=props[1]*s;
    U[1]=.5*mu*(*i1-3);
    U[0]=U[1]+.5*bulk*(*J-1)*(*J-1);
    for(int i=0;i<3;++i)d1[i]=0;
    for(int i=0;i<6;++i){d2[i]=0;d3[i]=0;}
    d1[0]=.5*mu; d1[2]=bulk*(*J-1); d2[2]=bulk;
}
}

extern "C" void sdvini(double* s,double* x,int* ns,int* nx,int*,int*,int*,int*) {
    ksdvini(s,x,ns,nx);
}
extern "C" void SDVINI(double* s,double* x,int* ns,int* nx,int* e,int* q,int* l,int* k) {
    sdvini(s,x,ns,nx,e,q,l,k);
}
#define K_UHYPER_ARGS double* i1,double* i2,double* J,double* U,double* d1,double* d2,double* d3,double* temp,int* elem,char* name,int* incmp,int* ns,double* s,int* nf,double* fv,double* dfv,int* np,double* p
extern "C" void uhyper(K_UHYPER_ARGS) { kuhyper(i1,J,U,d1,d2,d3,ns,s,np,p); }
extern "C" void UHYPER(K_UHYPER_ARGS) { uhyper(i1,i2,J,U,d1,d2,d3,temp,elem,name,incmp,ns,s,nf,fv,dfv,np,p); }
