#include "./Im2win_bchw.hpp"
#include "./Timer.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>

extern "C" void conv_withTimer1(
    HTensor1D &output, HTensor1D &input, HTensor1D &filter,
    HTensor1D &input_win, HTensor1D &filter_win,
    const size_t stride, Timer &T, double &elapsed1, double &elapsed2);

#define GIGA 1e9
using namespace std;


int main(int argc, char *argv[])
{
    // argv: B Cin H W Cout K stride
    size_t B      = atoi(argv[1]);
    size_t Cin    = atoi(argv[2]);
    size_t H      = atoi(argv[3]);
    size_t W      = atoi(argv[4]);
    size_t Cout   = atoi(argv[5]);
    size_t K      = atoi(argv[6]);
    size_t stride = atoi(argv[7]);

    tensorDimensions input  = {B, Cin, H, W};
    tensorDimensions filter = {Cout, Cin, K, K};

    Timer T;
    int number = 30;
    size_t out_h = (H - K) / stride + 1;
    size_t out_w = (W - K) / stride + 1;
    size_t operations = B * Cout * out_h * out_w * Cin * K * K * 2;

    HTensor1D input_d(input.batch, input.channel, input.height, input.width);
    HTensor1D filter_d(filter.batch, filter.channel, filter.height, filter.width);
    HTensor1D output_d(input.batch, filter.batch, out_h, out_w);
    HTensor1D input_win(0,0,0,0);
    HTensor1D filter_win(0,0,0,0);
    output_d.setZero();
    input_d.randomAssign();
    filter_d.randomAssign();

    image2window_pd_o1_h(input_win, input_d, filter_d, stride, 8, input_d.channel);
    filter2window_pd_o1_h(filter_win, filter_d, 8, filter_d.channel);

    double maxg = 0, e1 = 0, e2 = 0;
    double sum_e2 = 0;
    for (int i = 0; i < number; ++i) {
        e1 = 0; e2 = 0;
        output_d.setZero();
        conv_withTimer1(output_d, input_d, filter_d, input_win, filter_win, stride, T, e1, e2);
        double g = (operations / e2) / GIGA;
        sum_e2 += e2;
        maxg = g > maxg ? g : maxg;
    }
    double meang = (operations / (sum_e2/number)) / GIGA;
    // verify the kernel actually produced output
    half *op = output_d.getDataPtr();
    size_t nz = 0; double checksum = 0.0;
    size_t total = output_d.batch*output_d.channel*output_d.height*output_d.width;
    for (size_t i = 0; i < total; ++i) {
        float v = __half2float(op[i]);
        if (v != 0.0f) nz++;
        checksum += (double)v;
    }
    cout << "TFLOPS_best " << maxg/1000.0 << " TFLOPS_mean " << meang/1000.0 << " GFLOP " << operations/1e9
         << " nonzero " << nz << "/" << total
         << " checksum " << checksum << endl;
    return 0;
}
