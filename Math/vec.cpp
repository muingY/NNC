#include "vec.h"

vec::vec()
{
    vecData.clear();
}
vec::vec(std::vector<double> vecData_)
{
    vecData = vecData_;
}
vec::vec(size_t length)
{
    vecData.clear();
    vecData.resize(length);
}
vec::~vec() 
{}

std::vector<double>* vec::GetVector()
{
    return &vecData;
}

vec vec::operator+(vec& operation)
{
    int n = vecData.size();

    if (operation.GetVector()->size() != n)
    {
        return vec();
    }
    for (int i = 0; i < n; i++)
    {
        vecData[i] += operation.GetVector()->at(i);
    }

    return vec(vecData);
}
double vec::operator*(vec& operation)
{
    int n = vecData.size();
    std::vector<double> SumTable;
    double result = 0;

    if (operation.GetVector()->size() != n)
    {
        return result;
    }
    for (int i = 0; i < n; i++)
    {
        SumTable.push_back(vecData.at(i) * operation.GetVector()->at(i));
    }
    for (int i = 0; i < n; i++)
    {
        result += SumTable.at(i);
    }

    return result;
}