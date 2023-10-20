
#include "gnn2df.h"

//typedef std::vector<std::vector<float> > f2vec;
//typedef std::vector<float> f1vec;


#define eulers 2.71828
t_data sigmoid(t_data x){
    t_data ret = (float)1./((float)1. + hls::pow((float)2.71828,(float)-x));
    return ret;
}

//#define DEBUG
#ifdef DEBUG
void printVec(f2vec vec, std::string name = ""){
  printf("--%s (%d,%d)--\n",name.c_str(),vec.size(),vec[0].size());
  for(hls::vector<float> v : vec){
    for(float f : v){
      printf("%6.2f ",f);
    }
    printf("\n");
  }
  printf("\n---\n\n");
}

void printVec(hls::vector<float> vec, std::string name = ""){
  printf("--%s (%d,)--\n",name.c_str(),vec.size());
  for(float f : vec){
    printf("%6.2f, ",f);
  }
  printf("\n---\n\n");
}

void printVec(hls::vector<int> vec, std::string name = ""){
  printf("--%s (%d,)--\n",name.c_str(),vec.size());
  for(int f : vec){
    printf("%3d, ",f);
  }
  printf("\n---\n\n");
}
#endif

// void readWeight(){

// }
#define N_MAX 21
#define N_FEA  2
#define E_MAX 42
#define E_FEA  2
//#define LATENT 8
/*
ec0:  a.size()=1  a[0].size()=24 w.size()=24  w[0].size()=8 b.size()=8
ec1:  a.size()=1  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
ed0:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
ed1:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
el0:  a.size()=42  a[0].size()=2 w.size()=2  w[0].size()=8 b.size()=8
el1:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
nc0:  a.size()=1  a[0].size()=16 w.size()=16  w[0].size()=8 b.size()=8
nc1:  a.size()=1  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
nl0:  a.size()=21  a[0].size()=2 w.size()=2  w[0].size()=8 b.size()=8
nl1:  a.size()=21  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
--o0 (42,8)--
o0:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
--o1 (42,1)--
o1:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=1 b.size()=1
--OUTPUT (42,)--
*/
/*
#define MY_PRAGMA(D) \
#pragma HLS UNROLL \
#pragma HLS ARRAY_PARTITION variable=a complete dim=D \
#pragma HLS ARRAY_PARTITION variable=w complete dim=D \
#pragma HLS ARRAY_PARTITION variable=b complete dim=D
*/
//------------------------------------------------------------------------------
/*
void Loop3(int dim,int is,int js,t_data **out, t_data **a, t_data **w, t_data *b, bool doMax = true) {

#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
#endif
	for (int i = 0; i < is; i++)
		for (int j = 0; j < js; j++)
			out[i][j] = 0.;
	//f2vec(21,8) out(is, hls::vector<float,8>(js, float(0.)));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			float sum = 0;
			for (int k = 0; k < dim; k++) {
				float prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			float bj = b[j];
			float outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
}



template <int SIZE_X, int SIZE_Y> void sum_func(int test_array[SIZE_X][SIZE_Y], int &sum)
{
  sum = 0;
  for(int i=0;i<SIZE_X;i++)
  for(int j=0;j<SIZE_Y;j++)
    sum+= test_array[i][j];
}

void test_sum(int a1[10][10], int a2[20][20], int &sum_a1, int &sum_a2)
{
  sum_func<10,10>(a1, sum_a1);
  sum_func<20,20>(a2, sum_a1);
}


*/
//------------------------------------------------------------------------------

#define USE_PRAGMA
/*
template <int X1, int Y1, int X2, int Y2, int X3, int Y3, int X4 >
f2vec(X1,Y1) dot_bias_max(f2vec(X2,Y2) a, f2vec(X3,Y3) w, f1vec(X4) b, bool doMax = true) {
	int dim = X3; //w.size();     2
	int is = X2; // a.size();    21
	int js = Y3;  // w[0].size();  8
	f2vec(X1,Y1) out;
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;
		}
	}
	return out;
}
//------------------------------------------------------------------------------
*/
//------------------------------------------------------------------------------
f2vec(21,8) dot_bias_max_nl0(f2vec(21,2) a, f2vec(2,8) w, f1vec(8) b, bool doMax = true) {
#ifdef USE_PRAGMA
  #pragma HLS PIPELINE
  #pragma HLS ARRAY_PARTITION variable=a complete dim=0
  #pragma HLS ARRAY_PARTITION variable=w complete dim=0
  #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	nl0:  a.size()=21  a[0].size()=2 w.size()=2  w[0].size()=8 b.size()=8
	int dim = 2; //w.size();
	int is = 21; // a.size();
	int js = 8; // w[0].size();

	//f2vec(21,8) out(is, hls::vector<float,8>(js, float(0.)));
	f2vec(21,8) out;
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
/*
	for (int i = 0; i < is; i++)
		for (int j = 0; j < js; j++)
			out[i][j] = 0.;
*/
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum) 0. ? (t_sum) 0. : outij;
			else
				out[i][j] = outij;

		}
	}

	return out;
}

//------------------------------------------------------------------------------
f2vec(21,8) dot_bias_max_nl1(f2vec(21,8) a, f2vec(8,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//nl1:  a.size()=21  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; // w.size();
	int is = 21; // a.size();
	int js = 8; // w[0].size();
	f2vec(21,8) out;
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
	//f2vec(21,8) out(is, hls::vector<float,8>(js, float(0.)));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(42,8) dot_bias_max_el0(f2vec(42,2) a, f2vec(2,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	el0:  a.size()=42  a[0].size()=2 w.size()=2  w[0].size()=8 b.size()=8
	int dim = 2; //w.size();
	int is = 42; // a.size();
	int js = 8; // w[0].size();
	f2vec(42,8) out;
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(42,8) dot_bias_max_el1(f2vec(42,8) a, f2vec(8,8) w, f1vec(8) b,bool doMax = true) { // el1, ed0, ed1, o0
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
///	el1:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; //w.size();
	int is = 42; // a.size();
	int js = 8; // w[0].size();
	f2vec(42,8) out;
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(1,8) dot_bias_max_ec0(f2vec(1,24) a, f2vec(24,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	ec0:  a.size()=1  a[0].size()=24 w.size()=24  w[0].size()=8 b.size()=8
	int dim = 24; // w.size();
	int is = 1; // a.size();
	int js = 8; // w[0].size();
	f2vec(1,8) out;
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(1,8) dot_bias_max_ec1(f2vec(1,8) a, f2vec(8,8) w, f1vec(8) b,bool doMax = true) {  // ec1, nc1
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	ec1:  a.size()=1  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; //w.size();
	int is = 1; // a.size();
	int js = 8; // w[0].size();
	f2vec(1,8) out;
// f2vec(1,8) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(1,8) dot_bias_max_nc0(f2vec(1,16) a, f2vec(16,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	nc0:  a.size()=1  a[0].size()=16 w.size()=16  w[0].size()=8 b.size()=8
	int dim = 16; //w.size();
	int is = 1; // a.size();
	int js = 8; // w[0].size();
	f2vec(1,8) out;
	// f2vec(1,8) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(1,8) dot_bias_max_nc1(f2vec(1,8) a, f2vec(8,8) w, f1vec(8) b,	bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	nc1:  a.size()=1  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; //w.size();
	int is = 1; // a.size();
	int js = 8; // w[0].size();
	f2vec(1,8) out;
// f2vec(1,8) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}
/*
//------------------------------------------------------------------------------
f2vec(42,8) dot_bias_max_ed0(f2vec(42,8) a, f2vec(8,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
//	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	ed0:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; // w.size();
	int is = 42; // a.size();
	int js = 8; // w[0].size();
	f2vec(42,8) out;
// f2vec(42,8) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(42,8) dot_bias_max_ed1(f2vec(42,8) a, f2vec(8,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
//	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	ed1:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; // w.size();
	int is = 42; // a.size();
	int js = 8; // w[0].size();
	f2vec(42,8) out;
// f2vec(42,8) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}

//------------------------------------------------------------------------------
f2vec(42,8) dot_bias_max_o0(f2vec(42,8) a, f2vec(8,8) w, f1vec(8) b,bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	o0:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=8 b.size()=8
	int dim = 8; // w.size();
	int is = 42; // a.size();
	int js = 8; // w[0].size();
	f2vec(42,8) out;
	//f2vec(42,8) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}
*/
//------------------------------------------------------------------------------
f2vec(42,1) dot_bias_max_o1(f2vec(42,8) a, f2vec(8,1) w, f1vec(1) b,bool doMax = true) {
#ifdef USE_PRAGMA
	#pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=a complete dim=0
    #pragma HLS ARRAY_PARTITION variable=w complete dim=0
    #pragma HLS ARRAY_PARTITION variable=b complete dim=0
#endif
//	o1:  a.size()=42  a[0].size()=8 w.size()=8  w[0].size()=1 b.size()=1
	int dim = 8; // w.size();
	int is = 42; // a.size();
	int js = 1; // w[0].size();
	f2vec(42,1) out;
// f2vec(42,1) out(a.size(), hls::vector<float>(w[0].size(), 0.f));
	for (int i = 0; i < is; i++) {
		for (int j = 0; j < js; j++) {
			t_sum sum = 0;
			for (int k = 0; k < dim; k++) {
				t_sum prod = a[i][k] * w[k][j];
				sum = sum + prod;
			}
			t_sum bj = b[j];
			t_sum outij = sum + bj;
			if (doMax)
				out[i][j] = outij < (t_sum)0. ? (t_sum)0. : outij;
			else
				out[i][j] = outij;

		}
	}
	return out;
}
//------------------------------------------------------------------------------


//void gnn2df(T_N_IN nodes, T_E_IN edges, T_RS_IN receivers, T_RS_IN senders,	T_OUT &output) {

//namespace node{
//  namespace encode{
	static f2vec(2,8)  node_encode_w0 = {
        {0.623410,   0.535574,  -0.536172,  -0.255458,   0.000485,  -0.544971,  -0.550192,   0.081384},
        {0.793420,   1.516701,  -0.143757,   0.062764,  -0.243151,   0.166429,   0.482118,   1.034274}};
    
   static f1vec(8) node_encode_b0 = {0.22467032,  0.23795536, -0.17338157,  0.21310022, -0.01426719,
          0.34945355,  0.30531732,  0.41244748};
   static f2vec(8,8) node_encode_w1 = {
      {  0.162971,  -0.217682,   0.433929,   0.252868,   0.206712,  -0.122244,  -0.019943,   0.321528}, 
      { -0.197780,   0.124628,   0.032819,  -0.232280,  -0.012289,   0.475019,   0.552135,   0.120046}, 
      {  0.044786,  -0.485079,  -0.019250,  -0.071276,  -0.255918,   0.424711,  -0.408793,  -0.497400}, 
      { -0.229350,   0.087952,  -0.216883,  -0.254318,   0.156743,   0.144111,   0.633330,   0.370692}, 
      { -0.203984,   0.103266,   0.254813,  -0.002166,   0.522760,   0.553758,  -0.020167,   0.339525}, 
      { -0.609030,   0.259029,   0.654661,   0.535689,   0.119899,   0.504081,   0.349107,   0.019953}, 
      {  0.078806,  -0.046590,   0.199496,   0.220353,   0.023317,   0.205367,  -0.268644,   0.503264}, 
      { -0.026789,  -0.168697,   0.544580,  -0.171773,  -0.013628,   0.538197,   0.144284,   0.119172}};

   static f1vec(8) node_encode_b1 = {0.        , -0.02582167,  0.32214115,  0.12011248,  0.18575091,
          0.38401905,  0.06826392,  0.21213902};
 // }

//  namespace core{
  static f2vec(16,8) node_core_w0 = {
      {  0.084406,  -0.148671,  -0.325332,   0.250550,   0.017577,   0.428243,  -0.245812,  -0.237573}, 
      { -0.430017,   0.228672,   0.192875,  -0.090507,  -0.106604,  -0.140162,   0.116334,  -0.106663}, 
      {  0.437169,   0.105303,   0.134714,   0.115132,  -0.500977,  -0.113565,  -0.021215,  -0.038846}, 
      {  0.276437,   0.230603,   0.175153,  -0.234099,   0.008146,   0.219241,  -0.081433,   0.105127}, 
      {  0.275521,   0.165247,   0.099685,  -0.071342,   0.024685,   0.268997,  -0.063704,  -0.220114}, 
      { -0.034866,  -0.000591,   0.061810,   0.057969,  -0.397260,  -0.018022,  -0.134502,  -0.508279}, 
      {  0.692577,   0.683659,  -0.256815,   0.657952,  -0.221842,   0.456848,   0.197116,   0.008471}, 
      {  0.079350,   0.279546,  -0.440483,  -0.058010,  -0.007756,   0.180442,  -0.014736,  -0.316673}, 
      {  0.278049,   0.169525,   0.428416,   0.174379,  -0.455944,  -0.137327,   0.353727,  -0.195589}, 
      {  0.128037,   0.065392,   0.018266,   0.116162,  -0.206013,   0.237506,   0.138715,   0.025260}, 
      {  0.581393,   0.061432,   0.197315,   0.293007,   0.056233,  -0.016885,  -0.027625,   0.428731}, 
      {  0.290008,   0.296308,  -0.167858,   0.053540,  -0.393956,  -0.211556,   0.009913,   0.151950}, 
      {  0.129799,   0.397346,   0.489612,   0.527037,   0.217497,  -0.087381,   0.208685,   0.034074}, 
      {  0.025310,   0.332284,   0.438564,   0.377162,  -0.134405,  -0.158273,   0.143364,   0.281581}, 
      {  0.196345,   0.051000,   0.094165,  -0.003100,  -0.065754,   0.078498,   0.085326,   0.267563}, 
      { -0.008587,   0.075338,  -0.131869,   0.377378,   0.188589,  -0.028307,  -0.053220,  -0.209675}};

  static f1vec(8) node_core_b0 = {0.34089228,  0.35576098,  0.22265642,  0.28979499, -0.12229158,
       -0.00516894,  0.2664694 ,  0.12260208};

  static f2vec(8,8) node_core_w1 = {
      { -0.005199,   0.479926,   0.432030,   0.006379,   0.387300,   0.153188,  -0.146924,  -0.061294}, 
      { -0.032001,  -0.378820,   0.128742,   0.010721,   0.266832,   0.169822,  -0.343196,   0.103808}, 
      {  0.245171,  -0.104822,  -0.546349,   0.279073,  -0.140113,  -0.549663,  -0.265515,   0.334790}, 
      {  0.124271,  -0.065085,   0.059043,   0.248686,   0.088054,   0.479606,  -0.343688,   0.262012}, 
      { -0.244144,   0.502510,   0.398393,  -0.210779,   0.172233,  -0.176887,   0.119239,  -0.505941}, 
      { -0.230195,   0.028460,   0.080391,  -0.081630,  -0.196116,   0.465759,  -0.008573,  -0.219062}, 
      {  0.658638,   0.055061,  -0.217387,   0.245319,   0.145767,  -0.331255,   0.047504,  -0.055018}, 
      {  0.655844,   0.150964,   0.045004,  -0.550541,   0.021373,  -0.242722,  -0.094893,   0.023638}};

  static f1vec(8) node_core_b1 = {0.32141953, -0.14381635,  0.37161407,  0.23082987,  0.41443494,
        0.00840067, -0.00491667,  0.11339182};
 // }

//}



//namespace edge{

//  namespace encode{
  static f2vec(2,8) edge_encode_w0 = {
      { -0.651344,   0.293017,   0.073951,  -0.069459,  -0.648597,   0.118594,  -0.002665,  -0.985578}, 
      { -0.040994,  -0.439637,  -0.201268,   0.324396,   0.298531,   0.942888,   1.475503,   1.041259}};
    
  static f1vec(8) edge_encode_b0 = {0.55086843, -0.0255258 ,  0.        ,  0.06049307, -0.17811596,
        -0.20042266, -0.18062403, -0.23274771};
  static f2vec(8,8) edge_encode_w1 = {
      {  0.853468,  -0.506859,   0.816709,   0.076465,  -0.371121,  -0.393894,  -0.461114,   0.637225}, 
      {  0.196163,   0.102353,   0.148788,   0.639337,   0.059285,   0.192088,  -0.496682,  -0.583844}, 
      { -0.004254,  -0.064800,   0.031763,  -0.081363,  -0.556825,   0.002140,  -0.244713,  -0.468528}, 
      {  0.352661,  -0.020092,  -0.067100,  -0.577917,  -0.566505,   0.683050,  -0.184611,  -0.086512}, 
      { -0.080244,   0.600278,   0.077019,  -0.123831,   0.422981,   0.537057,   0.833010,  -0.259715}, 
      {  0.056218,   0.248332,  -0.387582,  -0.079993,   0.569657,   0.058649,   0.129618,   0.166958}, 
      { -0.310403,   0.248441,   0.263546,  -0.615467,   0.380514,   0.141907,   0.450022,  -0.199812}, 
      { -0.107619,   0.544769,   0.105583,   0.392906,   0.589666,   0.398600,  -0.075473,  -0.383940}};

  static f1vec(8) edge_encode_b1 = {0.52754318, -0.15628274,  0.32447563, -0.05754587, -0.15938263,
          0.02293825, -0.12726679,  0.32849665};
 // }


 // namespace core{
  static f2vec(24,8) edge_core_w0 = {
      {  0.065881,  -0.035759,  -0.050764,   0.277215,  -0.055152,  -0.074379,   0.252505,  -0.163335}, 
      { -0.037734,  -0.306137,   0.425659,  -0.161344,   0.120674,  -0.528386,   0.124155,  -0.121209}, 
      {  0.104989,   0.208705,  -0.496409,   0.517723,  -0.028815,  -0.216574,   0.114963,  -0.078661}, 
      {  0.396166,  -0.083335,   0.049189,   0.203642,   0.058874,  -0.141510,   0.186218,  -0.244323}, 
      {  0.018042,  -0.207369,   0.405143,  -0.412340,   0.106142,  -0.036043,  -0.019658,  -0.087377}, 
      {  0.180537,  -0.236734,  -0.265374,  -0.015531,  -0.355697,  -0.339099,   0.401692,  -0.031497}, 
      {  0.192479,   0.142846,  -0.097693,   0.244264,   0.068058,  -0.039754,   0.292815,  -0.545233}, 
      {  0.028929,  -0.113524,  -0.430108,  -0.123487,  -0.212200,  -0.196742,  -0.079313,  -0.284033}, 
      {  0.369884,   0.340020,  -0.182987,   0.155458,   0.144475,  -0.028469,   0.389602,  -0.167919}, 
      { -0.061621,  -0.156102,  -0.256727,   0.154636,   0.148395,  -0.039386,  -0.347955,   0.009230}, 
      { -0.055340,  -0.001109,   0.193051,  -0.081738,   0.153977,   0.078023,   0.119666,   0.187008}, 
      { -0.115615,   0.245717,   0.042494,  -0.089638,   0.021329,  -0.060820,   0.036205,   0.512536}, 
      {  0.418877,   0.238273,   0.042397,  -0.065999,   0.250433,   0.220565,  -0.127868,   0.340493}, 
      {  0.099799,   0.068543,   0.131617,  -0.041201,   0.385281,  -0.255624,  -0.177288,   0.248237}, 
      { -0.156515,   0.189939,   0.165655,  -0.066490,   0.180007,  -0.225291,   0.028849,   0.227721}, 
      { -0.070967,  -0.107161,  -0.041789,   0.383868,  -0.031967,   0.128264,   0.108823,  -0.094973}, 
      {  0.051118,   0.187911,   0.478473,   0.004281,  -0.162946,  -0.054345,   0.113281,   0.397634}, 
      {  0.161796,   0.161842,   0.182797,  -0.271896,  -0.072600,   0.089079,   0.112366,   0.028274}, 
      {  0.141721,  -0.035209,  -0.181047,   0.231473,   0.009153,   0.117482,  -0.057005,  -0.073812}, 
      {  0.177976,   0.213057,   0.151500,   0.270443,   0.253637,  -0.013323,   0.107928,   0.049340}, 
      { -0.047485,  -0.026115,   0.221158,   0.404279,  -0.024741,   0.270464,   0.181773,  -0.180764}, 
      { -0.300630,  -0.221357,  -0.098348,   0.217913,  -0.109659,   0.425379,   0.055324,   0.055239}, 
      { -0.325316,  -0.033482,  -0.006736,   0.072322,  -0.060763,   0.079139,  -0.022574,   0.490435}, 
      {  0.086707,   0.004632,  -0.187948,  -0.043024,   0.186962,   0.075192,   0.041928,  -0.039342}};

  static f1vec(8) edge_core_b0 = {0.30956018,  0.10100008, -0.0188322 ,  0.32337107, -0.00766911,
       -0.03950361,  0.09155678,  0.15058496};
    
  static f2vec(8,8) edge_core_w1 = {
      {  0.059186,  -0.170182,   0.126985,  -0.269654,   0.366770,   0.612421,   0.339932,  -0.417722}, 
      {  0.531065,  -0.136909,   0.087287,   0.281241,  -0.549820,   0.696490,   0.453282,   0.197475}, 
      { -0.169233,   0.682399,   0.235486,  -0.291890,   0.629068,  -0.265151,  -0.220337,  -0.400352}, 
      {  0.541471,  -0.331300,  -0.048294,   0.353784,  -0.201587,   0.053022,   0.462093,   0.459846}, 
      {  0.180165,   0.555319,  -0.219402,   0.274569,   0.654826,  -0.017192,  -0.065970,   0.213922}, 
      {  0.177710,   0.068934,  -0.194174,  -0.119907,  -0.097233,   0.383379,  -0.148071,   0.307452}, 
      {  0.085893,  -0.114113,   0.282366,  -0.417762,  -0.252475,  -0.240307,   0.297676,  -0.477301}, 
      {  0.254987,   0.207432,  -0.523690,   0.380390,   0.115922,   0.481035,  -0.593300,   0.155919}};

  static f1vec(8) edge_core_b1 = {0.44017026, -0.11246487,  0.26106267,  0.23998186,  0.01266002,
      0.07869014,  0.25743163, -0.05308357};
 // }

 // namespace decode{
  static f2vec(8,8) edge_decode_w0 = {
      {  0.110199,  -0.235125,   0.036583,   0.352500,   0.323055,   0.472212,  -0.139470,  -0.250786}, 
      {  0.400588,  -0.040515,   0.106988,   0.482380,  -0.496441,  -0.212485,  -0.701767,   0.591675}, 
      { -0.056735,  -0.378461,  -0.410518,   0.009065,   0.041849,   1.019487,   0.247796,  -0.252977}, 
      { -0.313450,  -0.143677,   0.348079,  -0.335386,  -0.560533,   0.300216,  -0.082479,   0.457201}, 
      { -0.451821,   0.404020,   0.507771,  -0.057326,   0.214999,  -0.020568,  -0.567983,  -0.087899}, 
      { -0.434034,   0.169313,   0.174374,   0.021379,   0.163679,   0.221914,  -0.072882,   0.404135}, 
      { -0.138103,  -0.466507,  -0.087501,  -0.455973,   0.036669,   0.056646,   0.139309,  -0.052307}, 
      { -0.345609,  -0.516398,  -0.011759,  -0.326304,  -0.009386,  -0.189345,   0.177342,   0.693352}};

  static f1vec(8) edge_decode_b0 = {0.01334489, -0.0087637 , -0.06552379,  0.07167384, -0.10068713,
      0.29952601, -0.11466171, -0.1976240};

  static f2vec(8,8) edge_decode_w1 = {
      { -0.357006,  -0.303348,  -0.115575,   0.534785,   0.622493,  -0.083350,  -0.326241,   0.569312}, 
      {  0.543720,  -0.326854,  -0.162256,  -0.505668,  -0.028306,   0.385007,  -0.342444,  -0.283600}, 
      {  0.436393,  -0.033230,  -0.007183,  -0.725376,  -0.285786,  -0.598741,   0.052604,  -0.156654}, 
      {  0.778939,  -0.232739,  -0.067650,  -0.092554,  -0.101561,   0.098795,   0.044439,   0.404318}, 
      { -0.166370,   0.041557,  -0.372505,   0.185477,   0.044499,   0.246695,  -0.274501,   0.319857}, 
      { -0.057888,   0.348307,   0.181241,   0.100806,   0.036985,   0.584631,   0.655693,   0.173142}, 
      {  0.512927,   0.042112,   0.151168,  -0.525723,  -0.647701,  -0.456515,  -0.128175,   0.200855}, 
      {  0.291544,   0.025045,   0.220536,  -0.125305,   0.133760,   0.233607,  -0.364184,  -0.256512}};

  static f1vec(8) edge_decode_b1 = {-0.00537412,  0.41136006,  0.06230142,  0.16568036, -0.10269273,
        0.14929253,  0.335058  ,  0.32443451};
 // }

//  namespace output{

  static f2vec(8,8) edge_output_w0 = {
      { -0.068120,   0.151656,   0.424577,  -0.194033,  -0.388038,   0.216199,   0.223739,   0.059053}, 
      {  0.395512,  -0.213137,  -0.034443,   0.214977,   0.157334,   0.010685,  -0.273060,   0.313665}, 
      { -0.102404,   0.425419,  -0.644616,   0.037629,  -0.212343,   0.183648,   0.448192,  -0.053699}, 
      {  0.245274,  -0.015957,  -0.570109,  -0.493342,   0.293188,  -0.084116,  -0.536257,   0.081129}, 
      { -0.056800,  -0.361298,   0.492969,   0.677243,  -0.140231,  -0.143000,   0.193567,   0.541914}, 
      {  0.018095,   0.244007,  -0.018102,   0.229325,   0.220535,  -0.032628,  -0.032572,   0.029727}, 
      {  0.512055,   0.293777,  -0.254106,  -0.004845,   0.139933,   0.072927,  -0.183399,  -0.317442}, 
      {  0.422752,   0.117559,  -0.134684,   0.329881,   0.208191,   0.300702,  -0.360121,  -0.179776}};

  static f1vec(8) edge_output_b0 = {0.33593525,  0.02511275,  0.04698442, -0.2583392 ,  0.17215994,
        0.00113961, -0.01033456, -0.13181777};

  static f2vec(8,1) edge_output_w1 = {
    {  0.921449}, 
    {  0.207228}, 
    { -0.603713}, 
    { -0.146749}, 
    {  0.123575}, 
    {  0.168456}, 
    { -0.148655}, 
    { -0.054415}};

  static f1vec(1) edge_output_b1 = {0.02861931};
 // }


//}
//===============================================================================================
  // T_OUT  gnn2df(T_N_IN nodes, T_E_IN edges, T_RS_IN receivers, T_RS_IN senders) {
//void gnn2df(f2vec(21,2) nodes, f2vec(42,2) edges, T_RS_IN receivers, T_RS_IN senders,	T_OUT& output) {

void gnn2df(T_N_IN nodes, T_E_IN edges, T_RS_IN receivers, T_RS_IN senders,	T_OUT &output) {

//#pragma HLS disaggregate variable=nodes_in
//#pragma HLS disaggregate variable=edges_in

#pragma HLS INTERFACE s_axilite port=nodes
#pragma HLS INTERFACE s_axilite port=edges
#pragma HLS INTERFACE s_axilite port=receivers
#pragma HLS INTERFACE s_axilite port=senders
#pragma HLS INTERFACE s_axilite port=output
#pragma HLS INTERFACE s_axilite port=return
//#pragma HLS INTERFACE ap_vld port=nodes,edges,receivers,senders,output

	//	 T_E_IN edges = edges_in;


#pragma HLS ARRAY_RESHAPE variable=nodes complete dim=0
//#pragma HLS ARRAY_PARTITION variable=nodes complete dim=0

#pragma HLS ARRAY_RESHAPE variable=edges complete dim=0
//#pragma HLS ARRAY_PARTITION variable=edges complete dim=0

#pragma HLS ARRAY_RESHAPE variable=receivers complete dim=0
//#pragma HLS ARRAY_PARTITION variable=receivers complete dim=0

#pragma HLS ARRAY_RESHAPE variable=senders complete dim=0
//#pragma HLS ARRAY_PARTITION variable=senders complete dim=0

#pragma HLS ARRAY_RESHAPE variable=output complete dim=0
//#pragma HLS ARRAY_PARTITION variable=output complete dim=0


//#pragma HLS PIPELINE
//#pragma HLS DATAFLOW

	//for (int k=0; k<10; k++)                                                   output[k]  = k;
	                                                                             //output[10] =  0.33333;
	//f2vec(21,8) nl0 = dot_bias_max(nodes, node_encode_w0, node_encode_b0);     //output[11] =  nl0[5][5];
	f2vec(21,8) nl0 = dot_bias_max_nl0(nodes, node_encode_w0, node_encode_b0);   //output[11] =  nl0[5][5];

	f2vec(21,8) nl1 = dot_bias_max_nl1(nl0, node_encode_w1,	node_encode_b1);     //output[12] =  nl1[5][5];

	f2vec(42,8) el0 = dot_bias_max_el0(edges, edge_encode_w0, edge_encode_b0);   //output[13] =  el0[5][5];
	f2vec(42,8) el1 = dot_bias_max_el1(el0, edge_encode_w1,	edge_encode_b1);     //output[14] =  el1[5][5];



	int LATENT = 8;

	// for(int iter = 0; iter < 4; iter++){

	//f2vec(21,8) nodes_receive(nodes.size(), hls::vector<float>(LATENT, 0.f));
	f2vec(21,8) nodes_receive ;
	for (int i = 0; i < 21; i++)
#pragma HLS PIPELINE
		for (int j = 0; j < 8; j++)
#pragma HLS PIPELINE
			nodes_receive[i][j] = (t_data)0.;

	//f2vec(42,8) edges_update(edges.size(), hls::vector<float>(LATENT, 0.f));
	f2vec(42,8) edges_update  ;
	for (int i = 0; i < 42; i++)
#pragma HLS PIPELINE
		for (int j = 0; j < 8; j++)
#pragma HLS PIPELINE
			edges_update[i][j] = (t_data)0.;
    //------------------------------------------------------------------------
	for (int e = 0; e < 42; e++) { //  el1.size() = 42
		f1vec(24) in;
		//for(float ed : el1[e])  // j size = 8
		//  in.push_back(ed);
		for (int j = 0; j < 8; j++)
			in[j] = el1[e][j];

		//for(float r : nl1[receivers[e]])  // j size = 8
		//  in.push_back(r);
		for (int j = 0; j < 8; j++)
			in[8 + j] = nl1[receivers[e]][j];

		//for(float s : nl1[senders[e]])
		//  in.push_back(s);
		for (int j = 0; j < 8; j++)
			in[16 + j] = nl1[senders[e]][j];

		f2vec(1,24) in2;

		for (int j=1; j<24; j++) in2[0][j] = in[j];

		f2vec(1,8) ec0 = dot_bias_max_ec0(in2, edge_core_w0, edge_core_b0);
		f2vec(1,8) ec1 = dot_bias_max_ec1(ec0, edge_core_w1, edge_core_b1);
		edges_update[e] = ec1[0];
		for (int i = 0; i < 8; i++) {  //nodes_receive[receivers[e]].size() = 8
			nodes_receive[receivers[e]][i] += ec1[0][i];
		}
	}
    //-------------------------------------------------------------
	                                                                              // output[15] =  edges_update[5][5];

	//f2vec nodes_update(nodes.size(), hls::vector<float>(LATENT, 0.f));
	f2vec(21,8) nodes_update;
	for (int i = 0; i < 21; i++)
#pragma HLS PIPELINE
		for (int j = 0; j < 8; j++)
			nodes_update[i][j] = 0.;
    //------------------------------------------------------------
	for (int n = 0; n < 21; n++) {  // nodes.size() = 21
		f1vec(16) in ;
		//for(float nr : nodes_receive[n]) // 8
		//  in.push_back(nr);
		for (int j = 0; j < 8; j++)
			in[j] = nodes_receive[n][j];

		//for(float cn : nl1[n])
		//  in.push_back(cn);
		for (int j = 0; j < 8; j++)
			in[8 + j] = nl1[n][j];

		f2vec(1,16) in2;
		for (int j=1; j<16; j++) in2[0][j] = in[j];

		f2vec(1,8) nc0 = dot_bias_max_nc0(in2, node_core_w0, node_core_b0);
		f2vec(1,8) nc1 = dot_bias_max_ec1(nc0, node_core_w1, node_core_b1);          // _nc1
		nodes_update[n] = nc1[0];
	}
	//-----------------------------------------------------------
	// printVec(nodes_update,"nodes_update");
	// printVec(edges_update,"edges_update");
	                                                                                             //output[16] =  nodes_update[5][5];
	f2vec(42,8) ed0 = dot_bias_max_el1(edges_update, edge_decode_w0, edge_decode_b0); //_ed0   //output[17] =  ed0[5][5];

	f2vec(42,8) ed1 = dot_bias_max_el1(ed0, edge_decode_w1, edge_decode_b1);          //_ed1   //output[18] =  ed1[5][5];

	f2vec(42,8) o0 = dot_bias_max_el1(ed1, edge_output_w0, edge_output_b0);           //_o0    //output[19] =  o0[5][5];

	f2vec(42,1) o1 = dot_bias_max_o1(o0, edge_output_w1, edge_output_b1, false);                 //output[20] =  o1[5][0];

#ifdef DEBUG
  printVec(o1,"o1");
#endif
    //-------------------------------------------------------------
	//f1vec(42) output;
	for (int i = 0; i < 42; i++) {  // o1.size() = 42
//#pragma HLS UNROLL
		//output.push_back(sigmoid(o1[i][0]));
		output[i] = sigmoid(o1[i][0]);
		//output[i] = o1[i][0];
	}
    //-------------------------------------------------------------
#ifdef DEBUG
  printVec(output,"OUTPUT");
#endif

  //return output;
}
