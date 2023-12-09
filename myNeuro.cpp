#include "myNeuro.h"
#include <typeinfo>

#define STRING(Value) #Value
bool is_optimizedM;
bool allow_optimisation_transform = false;
int iCycle;
int iCycleTotal;

myNeuro::myNeuro() {
    std::cout << "\n_________________________________ start myNeuro cpp\n";;
    inputNeurons = n1;
    outputNeurons = n3;
    nlCount = 2;
    errLimit = errLimitG;
    errOptinizationLimit = errOptinizationLimitG;
    list = (nnLay *) malloc((nlCount) * sizeof(nnLay));
    inputs = (float *) malloc((inputNeurons) * sizeof(float));
    targets = (float *) malloc((outputNeurons) * sizeof(float));
    list[0].setIO(n1, n2);
    list[1].setIO(n2, n3);
}

float **myNeuro::feedForwarding(bool mode_train) {
    list[0].toHiddenLayer(inputs);
    for (int i = 1; i < nlCount; i++)
        list[i].toHiddenLayer(list[i - 1].getHidden());
    if (mode_train) {
        return backPropagate();
    } else {
        for (int out = 0; out < outputNeurons; out++) {
            std::cout << "outputNeuron " + std::to_string(out) + ":";
            float outit = list[nlCount - 1].hidden[out];
            std::cout << std::to_string(outit) + "\n";
        }
        float **err3f;
        err3f = (float **) malloc(sizeof(float) * nlCount);
        return err3f;
    }
}

void myNeuro::optimiseWay() {
    if (!is_optimizedM)is_optimizedM = true;
}

float *myNeuro::processErrors(int i, bool &startOptimisation, bool showError = false, float totalE = 0.0) {
    float err1 = *list[i].getErrorsM();
    err1 = absF(err1);
    int lenLayer = list[i].getOutCount();
    startOptimisation = startOptimisation & (totalE < errOptinizationLimit);
    if (list[i].is_optimizedL != startOptimisation) list[i].is_optimizedL = startOptimisation;
    bool showBecuseOpt = (is_optimizedM || iCycle < 3);
    if (showError) {
        std::cout << " layer:" + std::to_string(i);
        std::cout << " error:" + std::to_string(err1);
        std::cout << " (optimisation:" + std::to_string((startOptimisation)) + ")";
        std::cout << " (len:" + std::to_string(lenLayer) + ")";
    }
    if (showError & !showBecuseOpt)std::cout << endl;
    if ((showError & showBecuseOpt) || is_optimizedM) {
        if (is_optimizedM)std::cout << "\n_________________!is_optimizedM!________________\n";
        std::cout << "\n";
        std::cout << " layer:" + std::to_string(i) + " ";
        printArray(list[i].getErrors(), i, lenLayer);
        std::cout << "\n";
    }
    return list[i].getErrors();
}

float **myNeuro::backPropagate() {
    bool showError = false;
    bool startOptimisation = true;
    if (rand() % 10000 == 9 || iCycle < 3) {
        showError = true;
    }
    float sum_out_error = list[nlCount - 1].calcOutError(targets, showError);
    float **err3 = static_cast<float **> (malloc((nlCount) * sizeof(float)));
    err3[nlCount - 1] = processErrors(nlCount - 1, startOptimisation, showError, sum_out_error);
    for (int i = nlCount - 2; i >= 0; i--) {
        list[i].calcHidError(list[i + 1].getErrors(), list[i + 1].getMatrix(),
                             list[i + 1].getInCount(), list[i + 1].getOutCount(), showError);
        err3[i] = processErrors(i, startOptimisation, showError);
    }
    if (showError) {
        std::cout << "\n";
    }
    if (startOptimisation) {
        optimiseWay();
    }
    for (int i = nlCount - 1; i > 0; i--)
        list[i].updMatrix(list[i - 1].getHidden());
    list[0].updMatrix(inputs);
    return err3;
}

float **myNeuro::train(float *in, float *targ, bool optimize) {
    inputs = in;
    targets = targ;
    float **resfeed = feedForwarding(true);
    if (optimize) {
        for (int i = (nlCount - 2); i >= 0; i--)
            sumFloatMD(i);
    }
    return resfeed;
}

void myNeuro::query(float *in) {
    std::cout << "\n_________________________________ start myNeuro cpp query\n";;
    inputs = in;
    feedForwarding(false);
    std::cout << "\n_________________________________ end myNeuro cpp query\n";;
}

void myNeuro::sumFloatMD(int inS) {
    float *errors = list[inS].getErrors();
    for (int i = 0; i < (list[inS].getOutCount() - 1); i++) {
        list[inS].errTmp[i] += errors[i];
    }
};

void myNeuro::optimize_layer(int inS) {
    int countOut = list[inS].getOutCount();
    if (rand() % 100 == 9)
        for (int inp = 0; inp < countOut; inp++) {
            if (absF(list[inS].errTmp[inp]) > 1) {
                std::cout << "truncMatrix layer:" << inS << " neuron:" << inp << " neuron val:" << list[inS].errTmp[inp]
                          << "  countOut:" << countOut << "\n";;
                list[inS].truncMatrixOut(inp);
                list[inS + 1].truncMatrixIn(inp);
            }
        }
}

void myNeuro::write_matrix_var1(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
    for (int i = nlCount - 1; i > 0; i--) {
        float **w1 = list[i].getMatrix();
        for (int i = 1; i < list[i].getOutCount(); ++i) {
            for (int j = 1; j < list[i].getInCount(); ++j) {
                file << w1[i][j] << " ";
            }
            file << endl;
        }
    }
    file.close();
}

void myNeuro::printArray(float *arr, int iList, int s) {
    std::string str_outErrors;
    for (int inp = 0; inp < s; inp++) {
        std::string type_s;
        std::string str_f;
        type_s = typeid(arr[inp]).name();
        str_f = 'f';
        if (type_s == str_f | type_s == "float") {
            int i2 = 0;
            float N = absF(arr[inp]);
            float Nf = absF(arr[inp]);
            while (N < 0.9 && i2 < 99) {
                N = N * 10;
                ++i2;
            }
            if (i2 == 99)i2 = 0;
            std::cout << i2 << ")";
            str_outErrors += to_string(inp) + '(' + to_string(Nf) + ')';
        } else {
            std::cout << (arr[inp]);
        }
        std::cout << ',';
    }
    cout << "\n" << fixed << str_outErrors;
}
