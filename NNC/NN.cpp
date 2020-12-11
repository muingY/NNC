#include "NN.h"
#include "../Math/ActiveFuncSample.h"

node::node()
{
    w = 1.0;
}
node::node(double w_)
{
    w = w_;
}
node::node(unit* endpoint_)
{
    w = 1.0;
    endpoint = endpoint_;
}

unit::unit()
{
    b = 0.0;
    ActFunc = ActiveFuncSample::IdentityFunction;
}
unit::unit(double b_)
{
    b = b_;
    ActFunc = ActiveFuncSample::IdentityFunction;
}
unit::unit(double b_, double (*ActFunc_)(double))
{
    b = b_;
    ActFunc = ActFunc_;
}
unit::unit(double (*ActFunc_)(double))
{
    b = 0.0;
    ActFunc = ActFunc_;
}