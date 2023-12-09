#ifndef MY_FUnC_HEADER_
#define MY_FUnC_HEADER_
extern bool is_optimizedM;
extern bool allow_optimisation_transform;
extern int iCycle;
extern int iCycleTotal;

static double absD(double N) {
    if (N < 0)N = N * -1;
    return N;
}

static float absF(float N) {
    if (N < 0)N = N * -1;
    return N;
}

#endif
#ifndef MYNEURO_H
#define MYNEURO_H

#include <iostream>
#include <math.h>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;
const string training_image_fn = "mnist/train-images.idx3-ubyte";
const string training_label_fn = "mnist/train-labels.idx1-ubyte";
const string model_fn = "model-neural-network.dat";
const string model_fn_opt = "model-neural-network-optimized.dat";
const string report_fn = "training-report.dat";
const int nTraining = 60000;
const int width = 28;
const int height = 28;
const int n1 = width * height;
const int n2 = 128;
const int n3 = 10;
const float errLimitG = 0.000005;
const float errOptinizationLimitG = 0.0001;

#include <sstream>
#include <string>

template<class T>
std::string toString(const T &value) {
    std::ostringstream os;
    os << value;
    return os.str();
}

#define learnRate 0.1
#define randWeight (( ((float)rand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))

class myNeuro {
public:
    myNeuro();

    struct nnLay {
        int in;
        int out;
        bool is_optimizedL;
        float *errTmp;
        float **matrix;
        float *hidden;
        float *errors;

        int getInCount() { return in; }

        int getOutCount() { return out; }

        float **getMatrix() { return matrix; }

        float *getErrorsM() { return errors; }

        void truncMatrixOut(int index) {
            if (out <= index)return;
            if (out <= 1)return;
            if (index == 0) {
                index = 1;
            }
            for (int ou = (index - 1); ou < (out - 1); ou++) {
                for (int hid = 0; hid < in; hid++) {
                    matrix[hid][ou] = matrix[hid][ou + 1];
                }
            }
            out = out - 1;
        };

        void truncMatrixIn(int index) {
            if (in <= index)return;
            if (in <= 1)return;
            if (index == 0) {
                index = 1;
            }
            for (int ou = 0; ou < out; ou++) {
                for (int hid = index - 1; hid < (in - 1); hid++) {
                    matrix[hid][ou] = matrix[hid + 1][ou];
                }
            }
            in = in - 1;
        };

        void updMatrix(float *enteredVal) {
            for (int ou = 0; ou < out; ou++) {
                for (int hid = 0; hid < in; hid++) {
                    matrix[hid][ou] += (learnRate * errors[ou] * enteredVal[hid]);
                }
                matrix[in][ou] += (learnRate * errors[ou]);
            }
        };

        void setIO(int inputs, int outputs) {
            in = inputs;
            out = outputs;
            std::cout << " in-out " + std::to_string(in) + " - " + std::to_string(out) + " \n ";
            std::cout << " randWeight " + std::to_string(randWeight) + " \n ";
            hidden = (float *) malloc((out) * sizeof(float));
            matrix = (float **) malloc((in + 1) * sizeof(float) * 2);
            for (int inp = 0; inp < in + 1; inp++) {
                try {
                    matrix[inp] = (float *) malloc(out * sizeof(float));
                }
                catch (const std::out_of_range &e) {
                    std::cout << "Out of Range error.1\n";;
                    std::cerr << e.what();
                } catch (const std::exception &e) {
                    std::cout << "Out of Range error.01\n";;
                    std::cerr << e.what();
                } catch (const std::string &e) {
                    std::cout << "Out of Range error.10\n";;
                } catch (...) {
                    std::cout << "Out of Range error.11\n";;
                }
            }
            for (int inp = 0; inp < in + 1; inp++) {
                for (int outp = 0; outp < out; outp++) {
                    try {
                        matrix[inp][outp] = randWeight;
                    }
                    catch (const std::out_of_range &e) {
                        std::cout << "Out of Range error.2\n";;
                        std::cerr << e.what();
                    } catch (const std::exception &e) {
                        std::cout << "Out of Range error.02\n";;
                        std::cerr << e.what();
                    } catch (const std::string &e) {
                        std::cout << "Out of Range error.20\n";;
                    } catch (...) {
                        std::cout << "Out of Range error.22\n";;
                    }
                }
            }
            errTmp = (float *) malloc((out) * sizeof(float));
            for (int i = 0; i < out; i++) { errTmp[i] = 0; };
        }

        void toHiddenLayer(float *inputs) {
            for (int hid = 0; hid < out; hid++) {
                float tmpS = 0.0;
                for (int inp = 0; inp < in; inp++) {
                    tmpS += inputs[inp] * matrix[inp][hid];
                }
                tmpS += matrix[in][hid];
                hidden[hid] = sigmoida(tmpS);
            }
        };

        float *getHidden() {
            return hidden;
        };

        float calcOutError(float *targets, bool &showError) {
            float errsum = 0.0;
            errors = (float *) malloc((out) * sizeof(float) * 4);
            for (int ou = 0; ou < out; ou++) {
                float eTmp = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);
                /* if (!isnan(eTmp)) std::cout << " - " + std::to_string(eTmp]) + " - " + std::to_string(out) + " \n ";*/
                errors[ou] = eTmp;
                errsum += absF(eTmp);
            }
            return errsum;
        };

        void calcHidError(float *targets, float **outWeights, int inS, int outS, bool &showError) {
            errors = (float *) malloc((inS) * sizeof(float) * 4);
            for (int hid = 0; hid < inS; hid++) {
                errors[hid] = 0.0;
                for (int ou = 0; ou < outS; ou++) {
                    errors[hid] += targets[ou] * outWeights[hid][ou];
                }
                errors[hid] *= sigmoidasDerivate(hidden[hid]);
            }
        };

        float *getErrors() {
            return errors;
        };

        float sigmoida(float val) {
            return (1.0 / (1.0 + exp(-val)));
        }

        float sigmoidasDerivate(float val) {
            return (val * (1.0 - val));
        };
    };

    float **feedForwarding(bool mode_train);

    float **backPropagate();

    void optimiseWay();

    float *processErrors(int i, bool &startOptimisation, bool showError, float totalE);

    float **train(float *in, float *targ, bool optimize);

    void query(float *in);

    void printArray(float *arr, int iList, int s);

    void write_matrix_var1(string file_name);

    void optimize_layer(int i);

    void sumFloatMD(int inS);

    int nlCount;
    struct nnLay *list;
private:
    int inputNeurons;
    int outputNeurons;
    float errLimit;
    float errOptinizationLimit;
    float *inputs;
    float *targets;
};

#endif
