#include "Windows.h"
#include "mex.h"
#include "matrix.h"
//#include "LinAlg.h"
#include <vector>


class vec3d
{
private:
	double x,y,z;

public:

	vec3d(const double _x, const double _y, const double _z){x=_x; y=_y; z=_z;}
	vec3d(const vec3d& rhs){x=rhs.x; y=rhs.y; z=rhs.z;}
	
	double Norm() const {return sqrt(x*x+y*y+z*z);}
	vec3d& operator-=(const vec3d& rhs) {x-=rhs.x; y-=rhs.y; z-=rhs.z; return *this;} 
	vec3d operator-(const vec3d& rhs) const {vec3d ret(*this); ret-=rhs; return ret; } 
	vec3d Cross(const vec3d& Rhs) const { return vec3d(y*Rhs.z-z*Rhs.y,z*Rhs.x-x*Rhs.z, x*Rhs.y-y*Rhs.x);}
	vec3d& operator*=(const double& Rhs) {x*=Rhs;y*=Rhs;z*=Rhs; return *this;}
	vec3d operator*(const double& Rhs) const {vec3d Ret(*this); return Ret*=Rhs;}
	vec3d& operator+=(const vec3d& Rhs) {x+=Rhs.x;y+=Rhs.y;z+=Rhs.z; return *this;}
	vec3d operator+(const vec3d& Rhs) const {vec3d Ret(*this); return Ret+=Rhs;}



	double& operator[](const int i) 
	{
		if(i==0) 
			return x;
		if(i==1)
			return y;
		return z;
	}  



};

inline vec3d operator*(const double& Lhs,const vec3d&Rhs ) 
{
	return Rhs*Lhs;
}


void SubTri(std::vector<vec3d>& Qs, const vec3d& Q0, const vec3d& Q1, const vec3d& Q2, const double Thresh)
{
	vec3d v1=Q1-Q0;
	double l1=v1.Norm();
	vec3d v2=Q2-Q0;
	double l2=v2.Norm();

	//mexPrintf("Q0: %f,%f,%f\n",Q0[0],Q0[1],Q0[2]);
	//mexPrintf("Q1: %f,%f,%f\n",Q1[0],Q1[1],Q1[2]);
	//mexPrintf("Q2: %f,%f,%f\n",Q2[0],Q2[1],Q2[2]);

	
	double Area2=v1.Cross(v2).Norm();

	double Thr=Thresh*sqrt(l1*l2/Area2);

	double n1=floor(l1/Thr);
	double n2=floor(l2/Thr);

	for(double c1=0; c1<=n1; c1++)
	{
		for(double c2=0;c2<=n2;c2++)
		{
			double k1=(c1+0.5)/n1;
			double k2=(c2+0.5)/n2;
			if(k1+k2<1)
			{
				Qs.push_back(k1*v1+k2*v2+Q0);
			}
		}
	}



}


void mexFunction(int nOut, mxArray *Out[], int nIn, const mxArray *In[])
{
	size_t nQ=mxGetN(In[0]);
	size_t nTri=mxGetN(In[1]);
	if(mxGetM(In[0])!=3 || mxGetM(In[1])!=3)
		mexErrMsgTxt("Qm and Tri should have 3 rows, in MeshSupSamp(Qm,Tri,Thresh).") ;

	double Thresh=*mxGetPr(In[2]);

	double* pQ=mxGetPr(In[0]);
	std::vector<vec3d> Qs;
	Qs.reserve(nQ*2);
	for(size_t cQ=0;cQ<nQ;cQ++)
	{
		double x=*pQ;
		++pQ;
		double y=*pQ;
		++pQ;
		double z=*pQ;
		++pQ;

		Qs.push_back(vec3d(x,y,z));
	}

	double* pT=mxGetPr(In[1]);
	for(size_t cT=0;cT<nTri;cT++)
	{
		vec3d v0=Qs[*pT];
		++pT;
		vec3d v1=Qs[*pT];
		++pT;
		vec3d v2=Qs[*pT];
		++pT;

		SubTri(Qs,v0,v1,v2,Thresh);
	}


	Out[0] = mxCreateDoubleMatrix(3, Qs.size(), mxREAL) ;
	double* pOut=mxGetPr(Out[0]);
	for(auto Itt=Qs.begin();Itt!=Qs.end();++Itt)
	{
		*pOut=(*Itt)[0];
		++pOut;
		*pOut=(*Itt)[1];
		++pOut;
		*pOut=(*Itt)[2];
		++pOut;
	}

}