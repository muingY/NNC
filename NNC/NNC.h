#ifndef NNC_H
#define NNC_H

#include "NN.h"

#define NNC_LAYER_INPUTLAYER  0
#define NNC_LAYER_HIDDENLAYER 1
#define NNC_LAYER_OUTPUTLAYER 2

class NNC
{
public:
    NNC();
    NNC(std::vector<std::vector<unit>> unitNet_, double LearningRate_ = 0.2);
    NNC(double LearningRate_);

    void InsertUnitLayer(std::vector<unit> InsertLayer);
    void InsertUnit(int layer, unit InsertUnit);
    
    bool AutoConnection();
    bool AutoInitNNData(double medium = 0.0, double standardDeviation = 2.0);

    bool SetLayerUnitActFunc(int layer, double (*ActFunc_)(double));
    bool SetLayerTypeUnitActFunc(int LayerType, double (*ActFunc_)(double));
    bool SetLearningRate(double LearningRate_);

    std::vector<double> GetOutputLayerData();

    bool FeedForward(std::vector<double> input);
    bool Backpropagation(std::vector<double> LearningData);

    bool PrintNNSum();

private:
    std::vector<std::vector<unit>> unitNet;

    double LearningRate;
    double C;
};

#endif