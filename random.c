#include "random.h"
#include <stdlib.h>
#include <math.h>

void random_seed(unsigned int seed){
    srand(seed);
}

double random_normal(double mean, double standard_deviation){
    static int primed = 0;
    static double spare_value;

    if(primed){
        primed = 0;
        return mean + spare_value * standard_deviation;
    }

    double x = 0;
    double y = 0;
    double s = 0;
    do {
        x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        s = x * x + y * y;
    } while(s == 0 || s >= 1.0);

    x = x * sqrt(-2 * log(s) / s);
    y = y * sqrt(-2 * log(s) / s);

    primed = 1;
    spare_value = x;

    return mean + y * standard_deviation;
}
