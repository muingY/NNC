#include "LowLevel/Header.h"
#include "Math/ActiveFuncSample.h"
#include "Math/vec.h"
#include "NNC/NN.h"
#include "NNC/NNC.h"

int main()
{
    std::cout << "< Neuron Network Control >" << std::endl;

    NNC nnc;
    /*
    nnc.InsertUnitLayer({unit(), unit(), unit(), unit()});
    nnc.InsertUnitLayer({unit(), unit(), unit()});
    nnc.InsertUnitLayer({unit(), unit(), unit(), unit()});
    nnc.InsertUnitLayer({unit()});
    3561
    */
    nnc.InsertUnitLayer({unit(), unit(), unit()});
    nnc.InsertUnitLayer({unit(), unit(), unit()});
    nnc.InsertUnitLayer({unit(), unit(), unit()});
    nnc.InsertUnitLayer({unit()});

    nnc.AutoConnection();
    nnc.SetLayerTypeUnitActFunc(NNC_LAYER_INPUTLAYER, ActiveFuncSample::IdentityFunction);
    nnc.SetLayerTypeUnitActFunc(NNC_LAYER_HIDDENLAYER, ActiveFuncSample::SigmoidFunction);
    nnc.SetLayerTypeUnitActFunc(NNC_LAYER_OUTPUTLAYER, ActiveFuncSample::SigmoidFunction);
    nnc.AutoInitNNData();

    nnc.PrintNNSum();

    nnc.FeedForward({3, 5, 1});

    std::vector<double> OutputLayerData = nnc.GetOutputLayerData();
    std::cout << std::endl << "> Print Output Layer Data" << std::endl;
    for (int oj = 0; oj < OutputLayerData.size(); oj++)
    {
        std::cout.precision(30);
        std::cout << "  out neuron" << oj << " = " << OutputLayerData.at(oj) << std::endl;
    }
    
    std::cout << std::endl;

    for (int i = 0; i < 1000; i++)
    {
        std::cout << "> Backpropagation " << i << std::endl;
        nnc.Backpropagation({0.5});
        nnc.FeedForward({3, 5, 1});
    }

    std::cout.precision(6);
    nnc.PrintNNSum();

    std::vector<double> OutputLayerData2 = nnc.GetOutputLayerData();
    std::cout << std::endl << "> Print Output Layer Data" << std::endl;
    for (int oj = 0; oj < OutputLayerData2.size(); oj++)
    {
        std::cout.precision(30);
        std::cout << "  out neuron" << oj << " = " << OutputLayerData2.at(oj) << std::endl;
    }

    // system("pause");
    return 0;
}