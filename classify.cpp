#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <map>
#include "SoftTree.cpp"
#include "common.cpp"

map<string, double> classes;

void readFromFile(string filename, vector< vector<double> > &X, vector<double> &Y) {
  ifstream file;
  file.open(filename.c_str());
  assert(file.is_open());
  
  string word;  // class label might be any string
  vector<double> row;
  
  while(!file.eof())
  {
    file >> word;
    if (file.peek() == '\n') { // if line ends, 'val' is the response value
      if (classes.find(word) == classes.end())
        classes[word] = classes.size()-1;
      Y.push_back((double)(classes[word]));
      X.push_back(row);
      row.clear();
    } else {                  // else, it is an entry of the feature vector
      row.push_back(str2double(word));
    }
  }
}

void normalize(vector<vector<double> > &X, vector<vector<double> > &V, vector<vector<double> > &U)
{
  for (int dim=0; dim<X[0].size(); dim++)
  {
    double mean = 0;
    for (int i=0; i<X.size(); i++)
      mean += X[i][dim];
    mean /= (X.size());
    
    double std = 0;
    for (int i=0; i<X.size(); i++) {
      std += (X[i][dim]-mean)*(X[i][dim]-mean);
      X[i][dim] -= mean;
    }
    for (int i=0; i<V.size(); i++) {
      V[i][dim] -= mean;
    }
    for (int i=0; i<U.size(); i++) {
      U[i][dim] -= mean;
    }
    std /= (X.size()-1);
    std = sqrt(std);
    
    for (int i=0; i<X.size(); i++) {
      X[i][dim] /= std;
      if (isnan(X[i][dim])) cout << "nan" <<endl;
      //cout << X[i][dim] << " ";
    }
    for (int i=0; i<V.size(); i++) {
      V[i][dim] /= std;
      if (isnan(V[i][dim])) cout << "nan" <<endl;
      //cout << V[i][dim] << " ";
    }
    for (int i=0; i<U.size(); i++) {
      U[i][dim] /= std;
      if (isnan(U[i][dim])) cout << "nan" <<endl;
      //cout << U[i][dim] << " ";
    }
    //cout << endl;
  }
}

void normalize(vector<double> &X, vector<double> &V, vector<double> &U)
{
  double mean = 0;
  for (int i=0; i<X.size(); i++)
    mean += X[i];
  mean /= (X.size());
  
  double std = 0;
  for (int i=0; i<X.size(); i++) {
    std += (X[i]-mean)*(X[i]-mean);
    X[i] -= mean;
  }
  for (int i=0; i<V.size(); i++)
    V[i] -= mean;
  for (int i=0; i<U.size(); i++)
    U[i] -= mean;

  std /= (X.size()-1);
  std = sqrt(std);
  
  for (int i=0; i<X.size(); i++) {
    X[i] /= std;
    if (isnan(X[i])) cout << "nan" <<endl;
    //cout << X[i][dim] << " ";
  }
  for (int i=0; i<V.size(); i++) {
    V[i] /= std;
    if (isnan(V[i])) cout << "nan" <<endl;
    //cout << V[i][dim] << " ";
  }
  for (int i=0; i<U.size(); i++) {
    U[i] /= std;
    if (isnan(U[i])) cout << "nan" <<endl;
    //cout << U[i][dim] << " ";
  }
}

int main(int argc, char *argv[])
{
  assert(argc >= 3);
  uint fold = atoi(argv[2]);
  string name = (string)argv[1];
  
  uint m = (fold/2)+1;
  uint n = (fold%2)+1;

  vector< vector< double> > X, V, U;
  vector<double> Y, R, T;

  //srand(time(NULL)); // random seed
  srand(123457);

  cout.precision(5);  // # digits after decimal pt
  cout.setf(ios::fixed,ios::floatfield);
	
  string dataset = "data/" + name + "/" + name;
  string filename;
	
  filename = dataset
           + "-train-"
           + (char)(m+'0')
           + '-'
           + (char)(n+'0')
           + ".txt";
  
  readFromFile(filename, X, Y);
  
  filename = dataset
	   + "-validation-"
	   + (char)(m+'0')
	   + '-'
	   + (char)(n+'0')
	   + ".txt";
  
  readFromFile(filename, V, R);
  
  filename = dataset + "-test.txt";
  readFromFile(filename, U, T);
	
  normalize(X, V, U);
  //normalize(Y, R, T);
  
  SoftTree st(X, Y, V, R, 'c');
  
  double y;
  double acc=0;
  
  //ofstream outf("out");
  
  acc = 1-st.errRate(X, Y);
  
  cout << "SRT: ";
  cout << "n: " << st.size() << "\t";
  cout << "tra: " << acc << "\t";
  
  acc = 1-st.errRate(V, R);
  
  cout << "val: " << acc << "\t";
  
  acc = 1-st.errRate(U, T);
  
  cout << "tst: " << acc << "\t";
  cout << endl;
	
  return 0;
}
