#include "matrix.hpp"

int main(){
    Matrix a(2, 1);
    a(0, 0) = 1;
    a(1, 0) = 2;

    Matrix b(1, 2);
    b(0, 0) = 3;
    b(0, 1) = 4;

    Matrix c = a * b;

    a.put();
    b.put();
    c.put();

    return 0;
}
