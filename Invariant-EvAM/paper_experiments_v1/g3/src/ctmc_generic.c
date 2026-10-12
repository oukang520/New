#include <math.h>
#include <string.h>
#define M 16
#define P 10
static void mul(const double*a,const double*b,double*c,int n){for(int i=0;i<n;i++)for(int j=0;j<n;j++){double z=0;for(int k=j;k<=i;k++)z+=a[i*n+k]*b[k*n+j];c[i*n+j]=z;}}
void snapshot_generic(int d,int ne,const int*edges,const double*t,double time,double*prob,double*jac){
 int n=1<<d,p=d+ne;double q[M*M]={0},dq[P][M*M]={{0}};double mx=0;
 for(int s=0;s<n;s++){for(int e=0;e<d;e++){int bit=1<<(d-1-e);if(s&bit)continue;int ids[P]={0};ids[e]=1;double lograte=t[e];for(int k=0;k<ne;k++)if(edges[2*k]==e&&(s&(1<<(d-1-edges[2*k+1])))){lograte+=t[d+k];ids[d+k]=1;}double rate=exp(lograte)*time;int dest=s|bit;q[dest*n+s]+=rate;q[s*n+s]-=rate;for(int k=0;k<p;k++)if(ids[k]){dq[k][dest*n+s]+=rate;dq[k][s*n+s]-=rate;}}if(-q[s*n+s]>mx)mx=-q[s*n+s];}
 if(mx<=80){
 double v[M]={1},dv[M][P]={{0}},next[M],nd[M][P];double weight=exp(-mx);
 for(int i=0;i<n;i++){prob[i]=weight*v[i];for(int k=0;k<p;k++)jac[i*p+k]=0;}
 for(int it=1;it<300;it++){
 memcpy(next,v,sizeof(v));memcpy(nd,dv,sizeof(dv));
 for(int src=0;src<n;src++)for(int dst=src+1;dst<n;dst++)if(q[dst*n+src]>0){double z=q[dst*n+src]/mx;next[dst]+=z*v[src];next[src]-=z*v[src];for(int k=0;k<p;k++){double flux=z*dv[src][k]+dq[k][dst*n+src]*v[src]/mx;nd[dst][k]+=flux;nd[src][k]-=flux;}}
 weight*=mx/it;memcpy(v,next,sizeof(v));memcpy(dv,nd,sizeof(dv));for(int i=0;i<n;i++){prob[i]+=weight*v[i];for(int k=0;k<p;k++)jac[i*p+k]+=weight*dv[i][k];}
 if(it>mx+10&&weight<1e-17)break;
 }
 return;
 }
 int scale=0;while(mx>.25){mx*=.5;scale++;}double fac=ldexp(1.,-scale);
 double a[M*M],da[P][M*M],v[M*M]={0},dv[P][M*M]={{0}},out[M*M]={0},der[P][M*M]={{0}},tmp[M*M],dt[M*M],dt2[M*M];
 for(int i=0;i<n*n;i++){a[i]=q[i]*fac;for(int k=0;k<p;k++)da[k][i]=dq[k][i]*fac;}for(int i=0;i<n;i++){v[i*n+i]=1;out[i*n+i]=1;}
 for(int it=1;it<=18;it++){for(int k=0;k<p;k++){mul(dv[k],a,dt,n);mul(v,da[k],dt2,n);for(int i=0;i<n*n;i++){dv[k][i]=(dt[i]+dt2[i])/it;der[k][i]+=dv[k][i];}}mul(v,a,tmp,n);for(int i=0;i<n*n;i++){v[i]=tmp[i]/it;out[i]+=v[i];}}
 for(int s=0;s<scale;s++){for(int k=0;k<p;k++){mul(der[k],out,dt,n);mul(out,der[k],dt2,n);for(int i=0;i<n*n;i++)der[k][i]=dt[i]+dt2[i];}mul(out,out,tmp,n);memcpy(out,tmp,n*n*sizeof(double));}
 for(int i=0;i<n;i++){prob[i]=out[i*n];for(int k=0;k<p;k++)jac[i*p+k]=der[k][i*n];}
}
