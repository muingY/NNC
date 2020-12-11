#ifndef UNIT_H
#define UNIT_H

#include "../LowLevel/Header.h"

class node;
class unit;

class node
{
public:
    node();
    node(double w_);
    node(unit* endpoint_);

    double w;
    unit* endpoint;
};

class unit
{
public:
    unit();
    unit(double b_);
    unit(double b_, double (*ActFunc_)(double));
    unit(double (*ActFunc_)(double));

    double b;
    double (*ActFunc)(double);

    double x;
    double z;
    double y;

    std::vector<node> nodeNet;
};

#endif