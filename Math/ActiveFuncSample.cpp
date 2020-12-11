#include "ActiveFuncSample.h"

double ActiveFuncSample::IdentityFunction(double z)
{
    return z;
}
double ActiveFuncSample::StepFunction(double z)
{
    double x, y; 
    x = z;

    if (x < 0)
    {
        y = 0.0;
    }
    else
    {
        y = 1.0;
    }
    
    return y;
}
double ActiveFuncSample::SigmoidFunction(double z)
{
    double x, y; 
    x = z;

    double e = 2.71828182;
    y = 1.0 / (1.0 + pow(e, -x));

    return y;
}
double ActiveFuncSample::ReLU(double z)
{
    double x, y; 
    x = z;

    if (x < 0)
    {
        y = 0;
    }
    else
    {
        y = x;
    }
    
    return y;
}

double ActiveFuncSample_::IdentityFunction(double z)
{
    return 1.0;
}
double ActiveFuncSample_::StepFunction(double z)
{
    return 0.0;
}
double ActiveFuncSample_::SigmoidFunction(double z)
{
    return (ActiveFuncSample::SigmoidFunction(z) * (1.0 - ActiveFuncSample::SigmoidFunction(z)));
}
double ActiveFuncSample_::ReLU(double z)
{
    if (z < 0)
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
    return 0.0;
}