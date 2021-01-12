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
	int i;
	int j;
	data::DatasetInfo info;
	info.Type(0)=mlpack::data::Datatype::numeric;
	info.Type(1)=mlpack::data::Datatype::categorical;
	info.Type(2)=mlpack::data::Datatype::numeric;
	info.Type(3)=mlpack::data::Datatype::numeric;
	info.Type(4)=mlpack::data::Datatype::categorical;
	info.Type(5)=mlpack::data::Datatype::categorical;
	info.Type(6)=mlpack::data::Datatype::numeric;
	arma::mat trainData;
	arma::mat traindata;
	data::Load("insurance.csv",trainData,info);
	traindata=trainData.t();
	cout<<"The length of traindata is: "<<size(traindata)<<endl;
	cout<<traindata.row(5)<<endl;
	cout<<traindata.row(5)[2]<<endl;
	arma::mat X=arma::randu(size(traindata)[0],size(traindata)[1]-1);
	arma::mat Y=arma::randu(size(traindata)[0],1);
	for(i=0;i<size(traindata)[0];i++)
	{
		for(j=0;j<size(traindata)[1]-1;j++)
		{
			X.row(i)[j]=traindata.row(i)[j];
		}
	}
	cout<<"The size of X is: "<<size(X)<<endl;
	for(i=0;i<size(traindata)[0];i++)
	{
		Y.row(i)[0]=traindata.row(i)[6];
	}
	cout<<"The size of Y is: "<<size(Y)<<endl;
	X=X.t();
	Y=Y.t();
	LinearRegression lr(X,Y);
	arma::vec parameters = lr.Parameters();
	cout<<"The parameters of linear model are: "<<endl<<parameters<<endl;
	mlpack::cv::R2Score r2;
	cout<<"The r2 score is: "<<r2.Evaluate<LinearRegression,arma::mat,arma::mat>(lr,X,Y)<<endl;
	return 0;
}
