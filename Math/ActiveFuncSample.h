#ifndef ACTIVEFUNCSAMPLE_H
#define ACTIVEFUNCSAMPLE_H

#include "../LowLevel/Header.h"
#include "MathRoot.h"

namespace ActiveFuncSample
{
    double IdentityFunction(double z);
    double StepFunction(double z);
    double SigmoidFunction(double z);
    double ReLU(double z);
};

namespace ActiveFuncSample_
{
    double IdentityFunction(double z);
    double StepFunction(double z);
    double SigmoidFunction(double z);
    double ReLU(double z);
};

#endif