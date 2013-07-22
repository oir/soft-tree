/* Ozan Irsoy, Bogazici University, 2012.
 * 
 * SoftTree.cpp implements Soft Regression Trees.
 * Internal nodes have linear sigmoid splits,
 * leaf nodes have constant values.
 * 
 * 
 * See,
 *   "Soft Decision Trees", O. Irsoy, O. T. Yildiz, E. Alpaydin, 
 *   ICPR 21, Tsukuba, Japan.
 * for details.
 * 
 * 
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <cassert>
#include "common.cpp"

#define HARDINIT true     // if true, optimization starts from hard tree parameters, else, randomly
#define MINALPHA 1        // starting range of learning rate
#define MAXALPHA 10       // ending range of learning rate
#define MAXEPOCH 25       // number of epochs in training
#define MAXRETRY 10       // number of restart of optimization from a starting point
#define PRETH 1e-3        // prepruning threshold

#define uint unsigned int

using namespace std;

class SoftTree;
class Node;

class Node
{
  public:
  
    Node();
    
  private:

    void learnParams(const vector<vector<double> > &, const vector<double> &, 
                      const vector<vector<double> > &, const vector<double> &, 
                      double, SoftTree*);
    void splitNode(const vector<vector<double> > &, const vector<double> &, 
                   const vector<vector<double> > &, const vector<double> &,
                   SoftTree*);
    double evaluate(const vector<double> &);
    uint size();
    void print(uint);
    void hardInit(const vector< vector <double> > &, const vector<double> &);
  
  // pointers & flags 
    Node* parent;
    Node* left;
    Node* right;
    bool isLeaf;
    bool isLeft;
              
  // params  
    vector<double> w;
    double w0; // if leaf, response value
               // if internal, bias of split

    double y;  // last computed response
    double v;  // last computed gating
     
  friend class SoftTree;
};

class SoftTree 
{
  public:
    SoftTree(const vector< vector<double> > &X, const vector<double> &Y, 
             const vector< vector<double> > &V, const vector<double> &R,
             char type = 'r');
    double evaluate(const vector<double> &);
    double meanSqErr(const vector<vector<double> > &, const vector<double> &);
    double errRate(const vector<vector<double> > &, const vector<double> &);
    uint size();
    void print();
    
  private:
    Node* root;
    char type; // 'r'egression or 'c'lassification

  friend class Node;
};

Node::Node()
{
  isLeaf = true;
  parent = NULL;
  left = NULL;
  right = NULL;
}

SoftTree::SoftTree(const vector< vector<double> > &X, const vector<double> &Y, 
                   const vector< vector<double> > &V, const vector<double> &R,
                   char type) {
  assert(type == 'r' || type == 'c');
  this->type = type;
  root = new Node;

  root->w = vector<double>(X[0].size());
  root->w0 = 0;
  for (uint i=0; i<Y.size(); i++)
    root->w0 += Y[i];
  root->w0 /= Y.size();
  
  root->splitNode(X, Y, V, R, this);
}

void Node::hardInit(const vector< vector< double > >& X, const vector< double >& Y)
{
  vector<double> sv(X.size()); // soft membership
  double total=0;
  
  assert(w.size() != 0);
  
  // (1) compute soft memberships
  for (uint j=0; j<X.size(); j++) {
    double t = 1;
    Node* m = this;
    Node* p;

    while(m->parent != NULL) {
      p = m->parent;
      if (m->isLeft)
        t *= sigmoid(dot(p->w, X[j]) + p->w0);
      else
        t *= (1-sigmoid(dot(p->w, X[j]) + p->w0));
      m = m->parent;
    }
    sv[j] = t;
    total += t;
  }
  
  if (total <= 1) { // not enough data, init randomly
    w = vector<double>(X[0].size());
    for (uint i=0; i<w.size(); i++)
      w[i] = urand(-0.005, 0.005);
    w0 = urand(-0.005, 0.005);
    left->w0 = urand(-0.005, 0.005);
    right->w0 = urand(-0.005, 0.005);
    return;
  }
  
  uint dim, bestDim=-1;
  double errBest = -1;
  double bestSplit;
  double bestw10, bestw20;
  vector<double> bestw1, bestw2;

  // (2) look for the best hard split
  for (dim=0; dim<X[0].size(); dim++)
  {
    vector<pair<double,uint> > f;
    
    for (uint i=0; i<X.size(); i++)
      f.push_back(make_pair(X[i][dim],i));
            
    sort(f.begin(), f.end());
    
    double sp;
    for (uint i=0; i<f.size()-1; i++) {
      
      if (f[i].first == f[i+1].first) continue;
      sp = 0.5*(f[i].first + f[i+1].first);

      double w10,w20,left,right,lsum,rsum;
                              
      w10 = w20 = lsum = rsum = 0;
      for (uint j=0; j<=i; j++) {
        w10 += Y[f[j].second]*sv[f[j].second];
        lsum += sv[f[j].second];
      }
      w10 /= lsum;
      
      for (uint j=i+1; j<f.size(); j++) {
        w20 += Y[f[j].second]*sv[f[j].second];
        rsum += sv[f[j].second];
      }
      w20 /= rsum;
     
      // weighted MSE for regression and
      // weighted Gini Impurity for classification 
      double errl = 0, errr = 0;
      for (uint j=0; j<=i; j++)
        errl += (w10 - Y[f[j].second])*(w10 - Y[f[j].second])*sv[f[j].second];
      errl /= lsum;
      for (uint j=i+1; j<f.size(); j++)
        errr += (w20 - Y[f[j].second])*(w20 - Y[f[j].second])*sv[f[j].second];
      errr /= rsum;
      
      double a = lsum/(lsum+rsum+0.0);
      double b = rsum/(lsum+rsum+0.0);
      
      if (a*errl + b*errr < errBest || errBest == -1) {
        bestSplit = sp;
        bestDim = dim;
        errBest = a*errl + b*errr;
        bestw10 = w10;
        bestw20 = w20;
        //cout << errbest << endl;
      }
    }
  }
  
  // (3) init params according to best hard split

  w = vector<double>(X[0].size());		
  for (uint i=0; i<w.size(); i++)
    w[i] = urand(-0.005, 0.005);
  w[bestDim] = -0.5;
  w0 = bestSplit*0.5;
  left->w0 = bestw10;
  right->w0 = bestw20;
  
  //cout << "bestsplit: " << bestsplit << endl;

  assert(w.size() != 0);
}

double SoftTree::evaluate(const vector<double> &x) {
  if (type == 'r')
    return root->evaluate(x);
  else if (type == 'c')
    return sigmoid(root->evaluate(x));
}

uint SoftTree::size() {
  return root->size();
}

uint Node::size() {
  if (isLeaf)
    return 1;
  else  
    return 1 + left->size() + right->size();
}

void Node::print(uint depth) {
  for (int i=0; i<depth; i++) cout << "__";
  if (!isLeaf)
    for (int i=0; i<w.size(); i++) cout << w[i] << ", ";
  cout << w0 << endl;
  if (!isLeaf) {
    left->print(depth+1);
    right->print(depth+1);
  }
}

void SoftTree::print() {
  root->print(1);
}

double Node::evaluate(const vector<double> &x) {
  if (isLeaf)
    y = w0;
  else {
    v = sigmoid(dot(w,x)+w0);
    y = v*(left->evaluate(x)) + (1-v)*(right->evaluate(x));
  }

  return y;
}

// This learns the parameters of current node and their potential children
void Node::learnParams(const vector< vector<double> > &X, const vector<double> &Y, 
			const vector< vector<double> > &V, const vector<double> &R,
                        double alpha, SoftTree* tree) {
  double u = 0.1;		// momentum weight
  double eps = 0.00001;
  
  uint e,i,j,temp;
  vector<uint> ix;

  vector<double> dw(X[0].size()); // grad of w
  vector<double> dwp(X[0].size()); // previous grad of w
          
  double dw10, dw20, dw0; // grads of w0
  double dw10p, dw20p, dw0p; // previous grads of w0

  for (i=0; i<Y.size(); i++) ix.push_back(i);

  for (e=0; e<MAXEPOCH; e++) {
    shuffle(ix);

    for (i=0; i<X.size(); i++) {
      j = ix[i];
      vector<double> x = X[j];
      double r = Y[j];
      double y = tree->evaluate(x);
      double d = y - r;

      double t = alpha*d;
      Node* m = this;
      Node* p;

      // compute negative gradient
      while(m->parent != NULL) {
        p = m->parent;
        if (m->isLeft)
          t *= p->v;
        else
          t *= (1 - p->v);
        m = m->parent;
      }

      dw = (-t*(left->y - right->y)*(v)*(1-v))*x;
      dw0 = -t*(left->y - right->y)*(v)*(1-v);
      dw10 = -t*(v);
      dw20 = -t*(1-v);
      
      // update params (params -= alpha*gradient)
      w += dw + u*dwp;              
      w0 += dw0 + u*dw0p;
      left->w0 += dw10 + u*dw10p;
      right->w0 += dw20 + u*dw20p;
      
      // update previous values
      dwp = dw;
      dw0p = dw0;
      dw10p = dw10;
      dw20p = dw20;

      alpha *= 0.9999;
    }
  }	
}

double SoftTree::meanSqErr(const vector< vector<double> > &X, 
                           const vector<double> &Y) {
  assert(type == 'r');
  double err=0, y;
  for (uint i=0; i<Y.size(); i++) {
    y = evaluate(X[i]);
    err += (Y[i]-y)*(Y[i]-y);
  }
  err /= Y.size();
  return err;
}

double SoftTree::errRate(const vector< vector<double> > &X, 
                         const vector<double> &Y) {
  assert(type == 'c');
  double err=0, y;
  for (uint i=0; i<Y.size(); i++) {
    y = evaluate(X[i]);
    err += (int)(Y[i] != (y > 0.5));
  }
  err /= Y.size();
  return err;
}

/* This splits the current node into two children recursively as follows:
 * (1) Create two children, then learn parameters of current node and its kids
 * (2) Measure error. If improved, keep the split, also recursively split
 *     the children. If not improved, revert the split (so that current node
 *     is a leaf) and stop the recursion at this node.
 */
void Node::splitNode(const vector< vector<double> > &X, const vector<double> &Y, 
                     const vector< vector<double> > &V, const vector<double> &R,
                     SoftTree* tree) {
  double err = 0;
  double y, r;
  char type = tree->type;

  if (type == 'r')
    err = tree->meanSqErr(V, R);
  else if (type == 'c')
    err = tree->errRate(V, R);

  Node temp;
  temp = *this;

  isLeaf = 0; // Make it an internal node
  w = vector<double>(X[0].size());

  // Create left child
  left = new Node;
  left->parent = this;
  left->isLeft = true;
          
  // Create right child
  right = new Node;
  right->parent = this;
  right->isLeft = false; 
  
  vector<double> bestw, bestwl, bestwr;
  double bestw0, bestw0l, bestw0r;
  double bestErr = 1e10;
  double newErr;
  
  // make MAXRETRY re-initializations and choose best
  double alpha;
  for (uint t=0; t<MAXRETRY; t++) {	
    if (HARDINIT) hardInit(X, Y);
    else {
      for (uint i=0; i<w.size(); i++)
        w[i] = urand(-0.005, 0.005);
      w0 = urand(-0.005, 0.005);
      left->w0 = urand(-0.005, 0.005);
      right->w0 = urand(-0.005, 0.005);
    }

    //alpha = MINALPHA + (t*(MAXALPHA-MINALPHA))/(float)MAXRETRY;
    alpha = (MAXALPHA+0.0) / pow(2,t+1);
    learnParams(X, Y, V, R, alpha, tree); 
    
    if (type == 'r')
      newErr = tree->meanSqErr(V, R);
    else if (type == 'c')
      newErr = tree->errRate(V, R);
    if (newErr < bestErr) {
      bestw = vector<double>(w);
      bestw0 = w0;
      bestw0l = left->w0;
      bestw0r = right->w0;
      bestErr = newErr;
    }
  }
  
  w = vector<double>(bestw);
  w0 = bestw0;
  left->w0 = bestw0l;
  right->w0 = bestw0r;
  
  //cout << "besterr: " << bestErr << " err: " << err << endl;

  // if the split is good enough, keep it,
  // continue splitting the children
  if (bestErr + PRETH < err) {
    left->splitNode(X, Y, V, R, tree);
    right->splitNode(X, Y, V, R, tree);
  } else {  // revert the split and stop recursion
    delete left;
    delete right;
    *this = temp;
  }
}




