#include "matrix.hpp"
#include <cassert>

Matrix::Matrix(int rows, int columns){
    m_x = rows;
    m_y = columns;
    m_data = std::unique_ptr<double[]>(new double[m_x * m_y]);
    m_transposed = false;
}

void Matrix::transpose(){
    m_transposed = !m_transposed;
}

int Matrix::rows() const {
    if(m_transposed){
        return m_y;
    }
    return m_x;
}

int Matrix::columns() const {
    if(m_transposed){
        return m_x;
    }
    return m_y;
}

double& Matrix::operator()(int x, int y){
    assert(x >= 0 && x < m_x && y >= 0 && y < m_y);

    if(m_transposed){
        return m_data.get()[x + y * m_y];
    }
    return m_data.get()[y + x * m_y];
}

Matrix Matrix::operator*(const Matrix& rhs) const {
    const Matrix& lhs = (*this);
    assert(lhs.columns() == rhs.rows());

    Matrix res(lhs.rows(), rhs.columns());

    for(int x = 0; x < res.rows(); x++){
        for(int y = 0; y < res.columns(); y++){
            double sum = 0;
            for(int i = 0; i < rhs.rows(); i++){
                sum += rhs(i, y) * lhs(x, i);
            }
            res(x, y) = sum;
        }
    }

    return res;
}

double Matrix::operator()(int x, int y) const {
    assert(x >= 0 && x < m_x && y >= 0 && y < m_y);

    if(m_transposed){
        return m_data.get()[x + y * m_y];
    }
    return m_data.get()[y + x * m_y];
}

#include <iostream>

void Matrix::put(){
    for(int x = 0; x < m_x; x++){
        for(int y = 0; y < m_y; y++){
            std::cout << (*this)(x, y) << " ";
        }
        std::cout << std::endl;
    }
}
