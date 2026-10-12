#include <math.h>
#include <string.h>
#define N 8
#define P 5
static void mul(const double*a,const double*b,double*c){for(int i=0;i<N;i++)for(int j=0;j<N;j++){double z=0;for(int k=j;k<=i;k++)z+=a[i*N+k]*b[k*N+j];c[i*N+j]=z;}}
void taylor_snapshot(const double*t,double*p,double*j){
 double q[64]={0},dq[5][64]={{0}};double maxexit=0;
 for(int s=0;s<8;s++){for(int e=0;e<3;e++){int bit=1<<(2-e);if(s&bit)continue;int id=-1;if(e==1&&(s&4))id=3;if(e==2&&(s&2))id=4;double rate=exp(t[e]+(id>=0?t[id]:0));int dest=s|bit;q[dest*8+s]+=rate;q[s*8+s]-=rate;dq[e][dest*8+s]+=rate;dq[e][s*8+s]-=rate;if(id>=0){dq[id][dest*8+s]+=rate;dq[id][s*8+s]-=rate;}}if(-q[s*8+s]>maxexit)maxexit=-q[s*8+s];}
 int scale=0;while(maxexit>0.25){maxexit*=.5;scale++;}double fac=ldexp(1.,-scale);
 double a[64],da[5][64],v[64]={0},dv[5][64]={{0}},out[64]={0},der[5][64]={{0}},tmp[64],dt[64],dt2[64];
 for(int i=0;i<64;i++){a[i]=q[i]*fac;for(int k=0;k<5;k++)da[k][i]=dq[k][i]*fac;}for(int i=0;i<8;i++){v[i*8+i]=1;out[i*8+i]=1;}
 for(int it=1;it<=18;it++){
  for(int k=0;k<5;k++){mul(dv[k],a,dt);mul(v,da[k],dt2);for(int i=0;i<64;i++){dv[k][i]=(dt[i]+dt2[i])/it;der[k][i]+=dv[k][i];}}
  mul(v,a,tmp);for(int i=0;i<64;i++){v[i]=tmp[i]/it;out[i]+=v[i];}
 }
 for(int s=0;s<scale;s++){for(int k=0;k<5;k++){mul(der[k],out,dt);mul(out,der[k],dt2);for(int i=0;i<64;i++)der[k][i]=dt[i]+dt2[i];}mul(out,out,tmp);memcpy(out,tmp,sizeof(tmp));}
 for(int i=0;i<8;i++){p[i]=out[i*8];for(int k=0;k<5;k++)j[i*5+k]=der[k][i*8];}
}

void snapshot(const double*t,double*p,double*j){
 double rate[8][3]={{0}},nu=0;
 for(int s=0;s<8;s++){double total=0;for(int e=0;e<3;e++){int bit=1<<(2-e);if(s&bit)continue;int id=-1;if(e==1&&(s&4))id=3;if(e==2&&(s&2))id=4;rate[s][e]=exp(t[e]+(id>=0?t[id]:0));total+=rate[s][e];}if(total>nu)nu=total;}
 if(nu>80){taylor_snapshot(t,p,j);return;}
 double v[8]={1},dv[8][5]={{0}},next[8],nd[8][5];double weight=exp(-nu);
 for(int s=0;s<8;s++){p[s]=weight*v[s];for(int k=0;k<5;k++)j[s*5+k]=0;}
 for(int it=1;it<300;it++){
  memcpy(next,v,sizeof(v));memcpy(nd,dv,sizeof(dv));
  for(int s=0;s<8;s++)for(int e=0;e<3;e++)if(rate[s][e]>0){
   int dest=s|(1<<(2-e));double z=rate[s][e]/nu;int id=-1;if(e==1&&(s&4))id=3;if(e==2&&(s&2))id=4;
   next[dest]+=z*v[s];next[s]-=z*v[s];
   for(int k=0;k<5;k++){double flux=z*(dv[s][k]+((k==e||k==id)?v[s]:0));nd[dest][k]+=flux;nd[s][k]-=flux;}
  }
  weight*=nu/it;memcpy(v,next,sizeof(v));memcpy(dv,nd,sizeof(dv));
  for(int s=0;s<8;s++){p[s]+=weight*v[s];for(int k=0;k<5;k++)j[s*5+k]+=weight*dv[s][k];}
  if(it>nu+10&&weight<1e-17)break;
 }
}
void joint_fg(const double*x,const double*counts,const double*agg,const double*features,const int*widths,const int*qs,int env,double*out,double*g){
 double p[8],j[40];snapshot(x,p,j);int dim=5;for(int e=0;e<env;e++)dim+=qs[e];for(int k=0;k<dim;k++)g[k]=0;*out=0;int off=5;
 for(int e=0;e<env;e++){
  double w[8],sel[8],dsel[8][7]={{0}},z=0,mean[2]={0},wj[5]={0};int q=qs[e];
  for(int s=0;s<8;s++){double logw=0;for(int k=0;k<q;k++)logw+=features[e*16+s*2+k]*x[off+k];w[s]=exp(logw);z+=p[s]*w[s];for(int k=0;k<5;k++)wj[k]+=w[s]*j[s*5+k];}
  for(int s=0;s<8;s++){sel[s]=p[s]*w[s]/z;for(int k=0;k<q;k++)mean[k]+=sel[s]*features[e*16+s*2+k];}
  for(int s=0;s<8;s++){for(int k=0;k<5;k++)dsel[s][k]=w[s]*j[s*5+k]/z-sel[s]*wj[k]/z;for(int k=0;k<q;k++)dsel[s][5+k]=sel[s]*(features[e*16+s*2+k]-mean[k]);}
  for(int y=0;y<widths[e];y++){double py=0,dy[7]={0};for(int s=0;s<8;s++){double a=agg[e*64+y*8+s];py+=a*sel[s];for(int k=0;k<5+q;k++)dy[k]+=a*dsel[s][k];}py=fmax(py,1e-300);double c=counts[e*8+y];*out-=c*log(py);for(int k=0;k<5;k++)g[k]-=c*dy[k]/py;for(int k=0;k<q;k++)g[off+k]-=c*dy[5+k]/py;}
  off+=q;
 }
}
