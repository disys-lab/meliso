#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include<map>
#include <vector>
#include <cmath>
#include <omp.h>

double *f;

double *d;

double *t;

int n;

double computeInterpolants(int n, int start, int end, double *f, double *d, double *t,int store_res){

    if(start == end){
        return d[start];
    }

    double f_2k = computeInterpolants(n-1,start+1,end,f,y,t,0);
    double f_1km1 = computeInterpolants(n-1,start,end-1,f,y,t,1);

    double f_res = double(f_2k - f_1km1)/(t[end] - t[start]);

    if(store_res){
        f[n] = f_res;
        f[n-1] = f_1km1;
    }

    return f_res;

}

double evaluatePolynomial(int n,double value, double *f, double *t){

    double res = 0;

    for (int i=0;i<n;i++){
        double prod = 1;
        for (int j =0; j<i ; j++){
            prod = prod *(value -t[j]);
        }

        res = res + prod*f[i];
    }

    return res;

}

int main(){

    n=6;

    f = (double*)malloc(n*sizeof(double));
    memset(f,0,n*sizeof(double));

    d = (double*)malloc(n*sizeof(double));
    memset(d,0,n*sizeof(double));

    t = (double*)malloc(n*sizeof(double));
    memset(t,0,n*sizeof(double));

    //error
    d[0] = -5.730373;
    d[1] = -10.876631;
    d[2] = -17.115587;
    d[3] = -23.553948;
    d[4] = -31.826230;
    d[5] = -42.553811;

    //output
    t[0] = -5.730373;
    t[1] = 2.323369;
    t[2] = 9.284413;
    t[3] = 16.046052;
    t[4] = 20.973770;
    t[5] = 23.446189;

    computeInterpolants(n-1,0,n-1,f,d,t,1);

    double y = 4.458242;
    double Output = -2.491162;
    //double real_output =0.94;
    double res_expected = Output -y;

    double res = evaluatePolynomial(n-1,Output,f,t);

    printf("res=%f,res_expected=%f\n",res,res_expected);
}