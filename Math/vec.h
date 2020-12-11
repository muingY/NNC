#ifndef VEC_H
#define VEC_H

#include "../LowLevel/Header.h"
#include "MathRoot.h"

class vec
{
public:
    vec();
    vec(std::vector<double> vecData_);
    vec(size_t length);
    ~vec();

    std::vector<double>* GetVector();

    vec operator+(vec& operation);
    double operator*(vec& operation);

private:
    std::vector<double> vecData;
};

#endif