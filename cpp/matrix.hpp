#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <memory>

class Matrix {
public:
    Matrix(int rows, int columns);

    void transpose();

    int rows() const;
    int columns() const;

    double& operator()(int x, int y);
    double operator()(int x, int y) const;

    Matrix operator*(const Matrix& rhs) const;

    void put();
private:
    int m_x;
    int m_y;
    std::unique_ptr<double[]> m_data;
    bool m_transposed;
};

#endif
