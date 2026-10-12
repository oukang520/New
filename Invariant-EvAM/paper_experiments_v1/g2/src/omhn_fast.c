#include <math.h>
void absorption(const double*t,const double*omega,double*p,double*j){
 double q[8][8]={{0}},dq[5][8][8]={{{0}}},v[8]={0},dv[8][5]={{0}};
 for(int s=0;s<8;s++)for(int e=0;e<3;e++){
  int bit=1<<(2-e);if(s&bit)continue;int id=-1;if(e==1&&(s&4))id=3;if(e==2&&(s&2))id=4;
  int d=s|bit;double r=exp(t[e]+(id>=0?t[id]:0));q[d][s]+=r;q[s][s]-=r;dq[e][d][s]+=r;dq[e][s][s]-=r;if(id>=0){dq[id][d][s]+=r;dq[id][s][s]-=r;}
 }
 for(int i=0;i<8;i++){
  double h=exp(((i>>2)&1)*omega[0]+((i>>1)&1)*omega[1]+(i&1)*omega[2]),den=h-q[i][i],num=(i==0);
  for(int s=0;s<i;s++)num+=q[i][s]*v[s];v[i]=num/den;p[i]=h*v[i];
  for(int k=0;k<5;k++){double z=dq[k][i][i]*v[i];for(int s=0;s<i;s++)z+=dq[k][i][s]*v[s]+q[i][s]*dv[s][k];dv[i][k]=z/den;j[i*5+k]=h*dv[i][k];}
 }
}
