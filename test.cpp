#include<iostream>
#include<mlpack/core.hpp>
#include<mlpack/methods/ann/layer/layer.hpp>
#include<mlpack/methods/ann/ffn.hpp>
#include<mlpack/methods/linear_regression/linear_regression.hpp>
#include<mlpack/core/cv/metrics/r2_score.hpp>

using namespace std;
using namespace mlpack;
/*using namespace mlpack::ann;*/
using namespace mlpack::regression;

int main()
{
	arma::mat A=arma::randu(5,1);
	int i;
	int j;
	cout<<"A: "<<endl<<A<<endl;
	double B=arma::mean(arma::mean(A));
	cout<<"Mean: "<<endl<<arma::mean(A)<<endl;
	cout<<"B: "<<B<<endl;
	return 0;
}
