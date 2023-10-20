#include <cmath>
#include "hls_math.h"
#include "ap_int.h"
#include "ap_fixed.h"
//#include <hls_vector.h>
//#include <array>
//#include <vector>
#include <stdio.h>
#include <string>
//#include <math.h>



/*  float
.812693 0.128698 0.0244462 0.399206 0.714628 0.0208942 0.698217 0.681283 0.0323975 0.773499 0.767962 0.14398 0.80937
0.131843 0.808928 0.448833 0.177469 0.142853 0.13078 0.807761 0.0705381 0.399384 0.0344007 0.773145 0.331402 0.103671
0.134954 0.092745 0.0895136 0.086384 0.169229 0.0277113 0.272068 0.250058 0.0365231 0.307136 0.802067 0.102717 0.0407987
0.280519 0.141732 0.12974
  ap_fixed<16, 6>
0.80957 0.130859 0.0244141 0.979492 0.716797 0.998047 0.700195 0.683594 0.998047 0.775391 0.769531 0.145508 0.806641
0.133789 0.806641 0.995117 0.983398 0.144531 0.132813 0.805664 0.0712891 0.400391 0.998047 0.775391 0.334961 0.105469
0.138672 0.0947266 0.0917969 0.0878906 0.171875 0.0283203 0.275391 0.253906 0.998047 0.311523 0.798828 0.104492 0.0410156
0.282227 0.143555 0.131836
*/

#define USE_FIXED
#ifdef USE_FIXED
  typedef ap_fixed<16, 7> t_data;
  typedef ap_fixed<16, 7> t_sum;
  typedef ap_int<16>      t_rs;
  typedef ap_int<16>      t_int;
#else
  typedef float t_data;
  typedef int   t_rs;
#endif

#define USE_HLS
#ifdef USE_HLS

	#include <hls_vector.h>

	typedef hls::vector<hls::vector<t_data, 2>, 21> T_N_IN;
	typedef hls::vector<hls::vector<t_data, 2>, 42> T_E_IN;
	typedef hls::vector<t_rs, 42> T_RS_IN;
	typedef hls::vector<hls::vector<t_data, 1>, 42> T_TRUE;
	typedef hls::vector<t_data, 42> T_OUT;

	#define f2vec(X,Y) hls::vector<hls::vector<t_data,Y>,X>
	#define f1vec(X)   hls::vector<t_data,X>

#else

	#include <vector>

	typedef std::vector<std::vector<float> > T_N_IN;
	typedef std::vector<std::vector<float> > T_E_IN;
	typedef std::vector<int>  T_RS_IN;
	typedef std::vector<std::vector<float> > T_TRUE;
	typedef std::vector<float> T_OUT;

	typedef std::vector<std::vector<float>> f2vec;
	typedef std::vector<float> f1vec;

	#define f2vec(X,Y) std::vector<std::vector<float> >
	#define f1vec(X)   std::vector<float>

#endif

void gnn2df(T_N_IN nodes, T_E_IN edges, T_RS_IN receivers,T_RS_IN senders,T_OUT &output);
 //  T_OUT gnn2df(T_N_IN nodes, T_E_IN edges, T_RS_IN receivers,T_RS_IN senders);

