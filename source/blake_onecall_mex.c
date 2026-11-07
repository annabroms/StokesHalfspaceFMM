/* Small MATLAB gateway plus a runtime guard on the actual Fortran FMM call. */
#include "mex.h"
#include <stdint.h>

void blake_onecall(int64_t*,int64_t*,const double*,const double*,const double*,
    const double*,const double*,const double*,double*,double*,int64_t*);
void __real_lfmm3d_(int64_t*,double*,int64_t*,double*,int64_t*,double*,int64_t*,double*,
    int64_t*,int64_t*,double*,double*,double*,int64_t*,double*,int64_t*,double*,double*,double*,int64_t*);
static int64_t calls, source_flag, target_flag;

void __wrap_lfmm3d_(int64_t *nd,double *eps,int64_t *ns,double *s,int64_t *ic,double *c,
    int64_t *id,double *d,int64_t *iper,int64_t *pg,double *p,double *g,double *h,
    int64_t *nt,double *x,int64_t *pgt,double *pt,double *gt,double *ht,int64_t *ier)
{
    /* The linker routes every lfmm3d invocation in this MEX through here. */
    ++calls;
    source_flag=*pg;
    target_flag=*pgt;
    if (calls!=1 || *nd!=4 || *pg!=0 || *pgt!=2)
        mexErrMsgIdAndTxt("blake:InvalidFMMCall","Expected one four-density call with flags (0,2).");
    __real_lfmm3d_(nd,eps,ns,s,ic,c,id,d,iper,pg,p,g,h,nt,x,pgt,pt,gt,ht,ier);
}

static void real_matrix(const mxArray *a, mwSize rows, mwSize columns)
{
    if (!mxIsDouble(a) || mxIsComplex(a) || mxIsSparse(a) || mxGetNumberOfDimensions(a)!=2 ||
        mxGetM(a)!=rows || mxGetN(a)!=columns)
        mexErrMsgIdAndTxt("blake:InvalidNativeInput","Unexpected native input type or shape.");
}

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    int64_t ns,nt,*stats;
    if (nrhs!=6 || nlhs!=3)
        mexErrMsgIdAndTxt("blake:InvalidNativeInput","Expected six inputs and three outputs.");
    ns=(int64_t)mxGetN(prhs[2]);
    nt=(int64_t)mxGetN(prhs[4]);
    real_matrix(prhs[0],3,2*ns);
    real_matrix(prhs[1],3,2*ns);
    real_matrix(prhs[2],1,ns);
    real_matrix(prhs[3],3,ns);
    real_matrix(prhs[4],3,nt);
    real_matrix(prhs[5],1,1);
    plhs[0]=mxCreateDoubleMatrix(3,nt,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(1,nt,mxREAL);
    plhs[2]=mxCreateNumericMatrix(1,4,mxINT64_CLASS,mxREAL);
    stats=(int64_t*)mxGetData(plhs[2]);
    calls=0; source_flag=0; target_flag=0;
    blake_onecall(&ns,&nt,mxGetPr(prhs[0]),mxGetPr(prhs[1]),mxGetPr(prhs[2]),
        mxGetPr(prhs[3]),mxGetPr(prhs[4]),mxGetPr(prhs[5]),mxGetPr(plhs[0]),
        mxGetPr(plhs[1]),stats);
    stats[1]=calls; stats[2]=source_flag; stats[3]=target_flag;
}
