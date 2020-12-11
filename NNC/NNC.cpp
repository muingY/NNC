#include "NNC.h"
#include "../Math/ActiveFuncSample.h"
#include "../Math/vec.h"

NNC::NNC() 
{
    unitNet.clear();
    LearningRate = 0.2;
    C = 0.0;
}
NNC::NNC(std::vector<std::vector<unit>> unitNet_, double LearningRate_)
{
    unitNet = unitNet_;
    LearningRate = LearningRate_;
    C = 0.0;
}
NNC::NNC(double LearningRate_)
{
    unitNet.clear();
    LearningRate = LearningRate_;
    C = 0.0;
}

void NNC::InsertUnitLayer(std::vector<unit> InsertLayer)
{
    unitNet.push_back(InsertLayer);
}
void NNC::InsertUnit(int InsertLayerPos, unit InsertUnit)
{
    if (InsertLayerPos >= unitNet.size())
    {
        unitNet.resize(InsertLayerPos + 1);
    }

    unitNet.at(InsertLayerPos).push_back(InsertUnit);
}

bool NNC::AutoConnection()
{
    if (unitNet.size() < 2)
    {
        return false;
    }

    for (int l = 0; l < unitNet.size(); l++)
    {
        if (l == (unitNet.size() - 1))
        {
            break;
        }
        for (int j = 0; j < unitNet.at(l).size(); j++)
        {
            for (int nj = 0; nj < unitNet.at(l + 1).size(); nj++)
            {
                unitNet.at(l).at(j).nodeNet.push_back(node(&unitNet.at(l + 1).at(nj)));
            }
        }
    }

    return true;
}
bool NNC::AutoInitNNData(double medium, double standardDeviation)
{
    if (unitNet.size() < 1)
    {
        return false;
    }

    std::default_random_engine RandEngine;
    std::normal_distribution<double> dist(medium, standardDeviation);

    for (int l = 0; l < unitNet.size(); l++)
    {
        for (int j = 0; j < unitNet.at(l).size(); j++)
        {
            if (unitNet.at(l).at(j).ActFunc != ActiveFuncSample::IdentityFunction)
            {
                unitNet.at(l).at(j).b = dist(RandEngine);
                for (int nj = 0; nj < unitNet.at(l).at(j).nodeNet.size(); nj++)
                {
                    unitNet.at(l).at(j).nodeNet.at(nj).w = dist(RandEngine);
                }
            }
        }
    }

    return true;
}

bool NNC::SetLayerUnitActFunc(int layer, double (*ActFunc_)(double))
{
    if (!(layer >= 0 && layer < unitNet.size()))
    {
        return false;
    }

    for (int j = 0; j < unitNet.at(layer).size(); j++)
    {
        unitNet.at(layer).at(j).ActFunc = ActFunc_;
    }

    return true;
}
bool NNC::SetLayerTypeUnitActFunc(int LayerType, double (*ActFunc_)(double))
{
    if (unitNet.size() < 3)
    {
        return false;
    }

    switch (LayerType)
    {
    case NNC_LAYER_INPUTLAYER:
        for (int j = 0; j < unitNet.at(0).size(); j++)
        {
            unitNet.at(0).at(j).ActFunc = ActFunc_;
        }
        break;
    case NNC_LAYER_HIDDENLAYER:
        for (int l = 1; l < (unitNet.size() - 1); l++)
        {
            for (int j = 0; j < unitNet.at(l).size(); j++)
            {
                unitNet.at(l).at(j).ActFunc = ActFunc_;
            }
        }
        break;
    case NNC_LAYER_OUTPUTLAYER:
        for (int j = 0; j < unitNet.at(unitNet.size() - 1).size(); j++)
        {
            unitNet.at(unitNet.size() - 1).at(j).ActFunc = ActFunc_;
        }
        break;
    
    default:
        return false;
        break;
    }

    return true;
}
bool NNC::SetLearningRate(double LearningRate_)
{
    if (LearningRate_ < 0.0)
    {
        return false;
    }

    LearningRate = LearningRate_;

    return true;
}

std::vector<double> NNC::GetOutputLayerData()
{
    std::vector<double> OutputLayerData;

    for (int j = 0; j < unitNet.at(unitNet.size() - 1).size(); j++)
    {
        OutputLayerData.push_back(unitNet.at(unitNet.size() - 1).at(j).y);
    }

    return OutputLayerData;
}

bool NNC::FeedForward(std::vector<double> input)
{
    if (input.size() != unitNet.at(0).size())
    {
        return false;
    }
    for (int inj = 0; inj < unitNet.at(0).size(); inj++)
    {
        unitNet.at(0).at(inj).x = input.at(inj);
    }

    vec xPool;
    vec wPool;

    /* Input layer process */
    for (int j = 0; j < unitNet.at(0).size(); j++)
    {
        unitNet.at(0).at(j).z = unitNet.at(0).at(j).x;
        unitNet.at(0).at(j).y = unitNet.at(0).at(j).ActFunc(unitNet.at(0).at(j).z);
    }

    /* Hidden layer process */
    for (int l = 1; l < unitNet.size(); l++)
    {
        for (int j = 0; j < unitNet.at(l).size(); j++)
        {
            for (int exj = 0; exj < unitNet.at(l - 1).size(); exj++)
            {
                xPool.GetVector()->push_back(unitNet.at(l - 1).at(exj).y);
                wPool.GetVector()->push_back(unitNet.at(l - 1).at(exj).nodeNet.at(j).w);
            }
            unitNet.at(l).at(j).z = (xPool * wPool) + unitNet.at(l).at(j).b;
            unitNet.at(l).at(j).y = unitNet.at(l).at(j).ActFunc(unitNet.at(l).at(j).z);
            xPool = vec();
            wPool = vec();
        }
    }

    return true;
}
bool NNC::Backpropagation(std::vector<double> LearningData)
{
    if (LearningData.size() != unitNet.at(unitNet.size() - 1).size())
    {
        return false;
    }
    
    std::vector<std::vector<double>> unitNetError(unitNet.size());
    C = 0.0;
    for (int oj = 0; oj < unitNet.at(unitNet.size() - 1).size(); oj++)
    {
        C += (0.5 * ((LearningData.at(oj) - GetOutputLayerData().at(oj)) * (LearningData.at(oj) - GetOutputLayerData().at(oj))));
        
        if (unitNet.at(unitNet.size() - 1).at(oj).ActFunc == ActiveFuncSample::IdentityFunction)
        {
            unitNetError.at(unitNetError.size() - 1).push_back((GetOutputLayerData().at(oj) - LearningData.at(oj)) * ActiveFuncSample_::IdentityFunction(unitNet.at(unitNet.size() - 1).at(oj).z));
        }
        else if (unitNet.at(unitNet.size() - 1).at(oj).ActFunc == ActiveFuncSample::SigmoidFunction)
        {
            unitNetError.at(unitNetError.size() - 1).push_back((GetOutputLayerData().at(oj) - LearningData.at(oj)) * ActiveFuncSample_::SigmoidFunction(unitNet.at(unitNet.size() - 1).at(oj).z));
        }
        else if (unitNet.at(unitNet.size() - 1).at(oj).ActFunc == ActiveFuncSample::ReLU)
        {
            unitNetError.at(unitNetError.size() - 1).push_back((GetOutputLayerData().at(oj) - LearningData.at(oj)) * ActiveFuncSample_::ReLU(unitNet.at(unitNet.size() - 1).at(oj).z));
        }
        else if (unitNet.at(unitNet.size() - 1).at(oj).ActFunc == ActiveFuncSample::StepFunction)
        {
            unitNetError.at(unitNetError.size() - 1).push_back((GetOutputLayerData().at(oj) - LearningData.at(oj)) * ActiveFuncSample_::StepFunction(unitNet.at(unitNet.size() - 1).at(oj).z));
        }
    }

    for (int l = (unitNetError.size() - 2); l >= 0; l--)
    {
        vec NxError(unitNetError.at(l + 1));
        for (int j = 0; j < unitNet.at(l).size(); j++)
        {
            vec Exw;
            for (int nj = 0; nj < unitNet.at(l).at(j).nodeNet.size(); nj++)
            {
                Exw.GetVector()->push_back(unitNet.at(l).at(j).nodeNet.at(nj).w);
            }
            double a = (NxError * Exw);
            double b;
            if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::IdentityFunction)
            {
                b = ActiveFuncSample_::IdentityFunction(unitNet.at(l).at(j).z);
            }
            else if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::SigmoidFunction)
            {
                b = ActiveFuncSample_::SigmoidFunction(unitNet.at(l).at(j).z);
            }
            else if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::ReLU)
            {
                b = ActiveFuncSample_::ReLU(unitNet.at(l).at(j).z);
            }
            else if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::StepFunction)
            {
                b = ActiveFuncSample_::StepFunction(unitNet.at(l).at(j).z);
            }
            double ab(a * b);  // !!!!!
            unitNetError.at(l).push_back(a * b);
        }
    }

    for (int l = 0; l < unitNet.size(); l++)
    {
        for (int j = 0; j < unitNet.at(l).size(); j++)
        {
            double db = unitNet.at(l).at(j).b - unitNetError.at(l).at(j) * LearningRate;
            unitNet.at(l).at(j).b = db;

            for (int nj = 0; nj < unitNet.at(l).at(j).nodeNet.size(); nj++)
            {
                double dw = unitNet.at(l).at(j).nodeNet.at(nj).w - (unitNet.at(l).at(j).x * unitNetError.at(l).at(j)) * LearningRate;
                unitNet.at(l).at(j).nodeNet.at(nj).w = dw;
            }
        }
    }

    return true;
}

bool NNC::PrintNNSum()
{
    std::cout << "> Print Neuron Network Summary" << std::endl;
    std::cout << "  layer n = " << unitNet.size() << std::endl;
    int totalNn = 0;
    for (std::vector<unit> layerPool : unitNet)
    {
        totalNn += layerPool.size();
    }
    std::cout << "  total neuron n = " << totalNn << std::endl;
    std::cout << "  Learning Rate = " << LearningRate << std::endl << std::endl;

    for (int l = 0; l < unitNet.size(); l++)
    {
        std::cout << "  > layer " << l << std::endl;
        std::cout << "      unit n = " << unitNet.at(l).size() << std::endl;
        for (int j = 0; j < unitNet.at(l).size(); j++)
        {
            std::cout << "      unit l_" << l << " j_" << j << std::endl;
            std::cout << "          b = " << unitNet.at(l).at(j).b << std::endl;
            if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::IdentityFunction)
            {
                std::cout << "          ActFunc = IdentityFunction" << std::endl;
            }
            else if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::ReLU)
            {
                std::cout << "          ActFunc = ReLU" << std::endl;
            }
            else if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::SigmoidFunction)
            {
                std::cout << "          ActFunc = SigmoidFunction" << std::endl;
            }
            else if (unitNet.at(l).at(j).ActFunc == ActiveFuncSample::StepFunction)
            {
                std::cout << "          ActFunc = StepFunction" << std::endl;
            }
            for (int nj = 0; nj < unitNet.at(l).at(j).nodeNet.size(); nj++)
            {
                std::cout << "          node l_" << l << " t_" << nj << std::endl;
                std::cout << "              w = " << unitNet.at(l).at(j).nodeNet.at(nj).w << std::endl;
            }
        }
    }

    return true;
}