/* Ozan Irsoy, 2013
 * 
 * Common functions and vector operator overloads
 */

#ifndef COMMON_CPP
#define COMMON_CPP

#include <cmath>
#include <vector>
#include <cassert>
#include <sstream>

#define uint unsigned int

using namespace std;

double str2double(const string& s) {
  istringstream i(s);
  double x;
  if (!(i >> x))
    return 0;
  return x;
} 

double dot(const vector<double> &x, const vector<double> &y) {
  assert(x.size() == y.size());
  double r=0;
  for(uint i = 0; i<x.size(); i++)
    r += x[i]*y[i];
  return r;
}

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

vector<double> sigmoid(vector<double> x) {
  vector<double> y(x);
  for (uint i=0; i<x.size(); i++)
    y[i] = 1 / (1 + exp(-x[i]));
  return y;
}

vector<double> softmax(vector<double> x) {
  vector<double> y(x);
  double sum = 0;
  for (vector<double>::iterator i = y.begin(); i != y.end(); i++) {
    *i = exp(*i);
    sum += *i;
  } 
  for (vector<double>::iterator i = y.begin(); i != y.end(); i++)
    *i /= sum;
  return y;
}

template <class T>
void shuffle(vector<T>& v) {  // KFY shuffle
  for (uint i=v.size()-1; i>0; i--) {
    uint j = (rand() % i);
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
  }
}

double urand(double min, double max) {
  return (double(rand())/RAND_MAX)*(max-min) + min;
}

template <class T>
vector<T>& operator+=(vector<T>& lhs, const vector<T>& rhs) {
  assert(lhs.size() == rhs.size());
  for (uint i=0; i<rhs.size(); i++)
    lhs[i] += rhs[i];
  return lhs;
}

template <class T>
const vector<T> operator+(const vector<T>& lhs, const vector<T>& rhs) {
  vector<T> result = lhs;
  result += rhs;
  return result;
}

template <class T>
vector<T>& operator-=(vector<T>& lhs, const vector<T>& rhs) {
  assert(lhs.size() == rhs.size());
  for (uint i=0; i<rhs.size(); i++)
    lhs[i] -= rhs[i];
  return lhs;
}

template <class T>
const vector<T> operator-(const vector<T>& lhs, const vector<T>& rhs) {
  vector<T> result = lhs;
  result -= rhs;
  return result;
}

template <class T>
vector<T>& operator*=(vector<T>& lhs, const vector<T>& rhs) {
  assert(lhs.size() == rhs.size());
  for (uint i=0; i<rhs.size(); i++)
    lhs[i] *= rhs[i];
  return lhs;
}

template <class T>
const vector<T> operator*(const vector<T>& lhs, const vector<T>& rhs) {
  vector<T> result = lhs;
  result *= rhs;
  return result;
}

template <class T>
vector<T>& operator*=(vector<T>& lhs, const T& rhs) {
  for (uint i=0; i<lhs.size(); i++)
    lhs[i] *= rhs;
  return lhs;
}

template <class T>
const vector<T> operator*(const vector<T>& lhs, const T& rhs) {
  vector<T> result = lhs;
  result *= rhs;
  return result;
}

template <class T>
const vector<T> operator*(const T& lhs, const vector<T>& rhs) {
  return rhs * lhs;
}

template <class T>
ostream& operator << (ostream& os, const vector<T>& v) {
  os << "[";
  for (typename vector<T>::const_iterator i = v.begin(); i != v.end(); i++)
    os << " " << *i;
  os << " ]";
  return os;
}

// index of max in a vector
uint argmax(const vector<double>& x) {
  double max = x[0];
  uint maxi = 0;
  for (uint i=1; i<x.size(); i++) {
    if (x[i] > max) {
      max = x[i];
      maxi = i;
    }
  }
  return maxi;
}

#endif
