#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #define __DEBUG__

# define MAX_ITCOUNT 200 // maximum number of iterations allowed for decoder

/* **********************************************************************
 *                                                                      *
 * sumprod(ch, vdeg, cdeg, intrlv, Nv, Nc, Nmsg, app)                   *
 * implements the sum product algorithm for decoding binary LDPC codes. *
 *                                                                      *
 * Arguments:                                                           *
 * ch contains the channel log-likelihood ratios                        *
 * vdeg contains the variable node degrees                              *
 * cdeg contains the constraint (check) node degrees                    *
 * intrlv contains the connections between variable nodes and           *
 *  constraint nodes (the ones in the LDPC matrix)                      *
 * Nv, Nc and Nmsg lengths of vdeg, cdeg and intrlv, respectively       *
 *                                                                      *
 * Outputs:                                                             *
 * app contains the final a-posteriori log-likelihood ratios after      *
 *  the algorithm completes                                             *
 * output value is the number of iterations or -1 if memory allocation  *
 *  failed.                                                             *
 *                                                                      *
 * Jossy, September 2018                                                *
 *                                                                      *
 ********************************************************************** */

int sumprod(double *ch, long *vdeg, long *cdeg, long *intrlv,
	    int Nv, int Nc, int Nmsg, double *app)
{
  double *msg;
  int j, k, imsg, stopflag, itcount;
  double aggr;
  
  // allocating memory to contain graph messages
  msg = calloc(Nmsg, sizeof(double));
  if (msg == NULL)
    return(-1);
  
  // main loop, will iterate until stopping criterion is fulfilled
  // (see below) or up to MAX_ITCOUNT iterations. 
  for (itcount = 0 ; itcount < MAX_ITCOUNT ; itcount++) {

    /* *****************************
     * variable node rule ("sum")  *
     ***************************** */
    // in the loops below, the variable imsg will run linearly through
    // all the node connections in the graph (with some jumps back for
    // re-processing)
    for (j = 0, imsg = 0 ; j < Nv ; j++) { // for all variable nodes
      // for all connections, initialise aggr as the channel value
      for (k = 0, aggr = ch[j] ; k < vdeg[j] ; k++, imsg++) 
        // note how the messages are read through the interleaver
        // during the variable node updates BUT NOT during the
        // constraint node updates
        aggr += msg[intrlv[imsg]]; // add in incoming message
      imsg -= vdeg[j]; // reset to re-process node messages
      for (k = 0; k < vdeg[j] ; k++, imsg++)
        // compute "extrinsic" message by subtracting incoming message
        msg[intrlv[imsg]] = aggr - msg[intrlv[imsg]];
      // the aggregate output is the app that we would use in our
      // final decisions once the iterations complete
      app[j] = aggr;
    }

    // stopping rule preparation
    stopflag = 0;

    /* ***********************************
     * constraint node rule ("product")  *
     *********************************** */
    for (j = 0, imsg = 0 ; j < Nc ; j++) { // for all constraint nodes
      // for all connections, initialise aggr as 1.0
      for (k = 0, aggr = 1.0 ; k < cdeg[j] ; k++, imsg++)
        // product of the tanh, temporarily stored in messages
        aggr *= (msg[imsg] = tanh(msg[imsg]/2.0));
      // the following stopping rule is not often used by practitioners (I wonder why?)
      // essentially, the overall aggr gives us an indication whether the constraint node
      // thinks that the sum of all variables is 0 or 1. Since this is a parity check,
      // we know that the sum must in fact be 0. Hence, if all the constraint nodes
      // estimate that 0 is more likely than 1, it means that all the parity-checks are
      // fulfilled. It is easy to show that the algorithm will only strengthen this belief
      // once in this position and never re-diverge out of a valid decision where all
      // constraint nodes agree that their parity-check equation is satisfied.
      // The implementation here uses a flag, initially set to 0. If any constraint node
      // believes its parity-check to be unsatisfied, it flags to 1 and hence prevents
      // the stopping criterion below to kick in. We check "stopflag == 0" for efficiency:
      // tanh is an expensive operation and there is no need to perform it once we know
      // that there has already been one unsatisfied constraint (C evaluates conditions
      // starting from the left, so will never compute atanh(aggr) if stopflag isn't 0.
      if (stopflag == 0 && 2.0 * atanh(aggr) <= 0.0)
        stopflag = 1;
      imsg -= cdeg[j]; // reset to re-process node messages
      // now compute the "extrinsic" rule for each outgoing message by dividing
      // by the corresponding tanh, which has temporarily been stored in the message
      for (k = 0; k < cdeg[j] ; k++, imsg++)
        msg[imsg] = 2.0 * atanh(aggr / msg[imsg]);
    }

    // stopping rule (at least one p.c. equation unfulfilled)
    if (!stopflag)
      break;
  }

  // free the pointer to dynamically allocated message array
  free(msg);

  return(itcount); // return number of iterations
}


/* **********************************************************************
 *                                                                      *
 * sumprod2(ch, vdeg, cdeg, intrlv, Nv, Nc, Nmsg, app)                  *
 *  works just like sumprod but uses a different (numerically more      *
 *  stable) implementation of the constraint node process. The          *
 *  calculation is 100% equivalent, i.e., if these were real numbers    *
 *  we would be computing the same thing, and we only get different     *
 *  results because of numerical inaccuracies. The operation in the     *
 *  sumprod is a product of many messsages followed by a division,      *
 *  which doesn't work at all well when the messages have different     *
 *  orders of magnitude.
 *                                                                      *
 * Jossy, September 2018                                                *
 *                                                                      *
 ********************************************************************** */

// the following 2 utility functions are used by sumprod2
// they are "prototyped" here so the compiler knows they in and ouputs,
// but the actual function definition will be given after sumprod2
double Lxor(double L1, double L2, int correction_flag); // LLR of XOR
double Lxfb(double *L, long dc, int correction_flag); // extrinsic LLR

int sumprod2(double *ch, long *vdeg, long *cdeg, long *intrlv,
	     int Nv, int Nc, int Nmsg, double *app)
{
  double *msg;
  int j, k, imsg, stopflag, itcount;
  double aggr;

  /*
  printf("Welcome to my C code!\n");
  printf("A few elements of my variable degrees:\n");
  for (k = 0 ; k < 80 ; k++)
    printf("%ld ", vdeg[k]);
  printf("\n");
  printf("A few elements of my check degrees:\n");
  for (k = 0 ; k < 80 ; k++)
    printf("%ld ", cdeg[k]);
  printf("\n");
  printf("A few elements of my interleaver:\n");
  for (k = 0 ; k < 80 ; k++)
    printf("%ld ", intrlv[k]);
  printf("\n");
  printf("A few elements of my channel observations:\n");
  for (k = 0 ; k < 80 ; k++)
    printf("%g ", ch[k]);
  printf("\n");
  */
  msg = calloc(Nmsg, sizeof(double));
  if (msg == NULL)
    return(-1);
  
  for (itcount = 0 ; itcount < MAX_ITCOUNT ; itcount++) {

    // var node (identical to sumprod)
    for (j = 0, imsg = 0 ; j < Nv ; j++) { 
      for (k = 0, aggr = ch[j] ; k < vdeg[j] ; k++, imsg++) 
        aggr += msg[intrlv[imsg]]; 
      imsg -= vdeg[j]; 
      for (k = 0; k < vdeg[j] ; k++, imsg++)
        msg[intrlv[imsg]] = aggr - msg[intrlv[imsg]];
      app[j] = aggr;
    }

    stopflag = 0;

    // constraint node 
    for (j = 0, imsg = 0 ; j < Nc ; j++) {
      // the constraint node is operated by calling a function that does pairwise
      // resolving on a "trellis" (see Lxfb for further explanations)
      // the flag 1 at the end instructs the function to oprate the "correction"
      // that makes this operation identical to the normal sum product rule (albeit
      // in a numerically more stable way than the product of tanh)
      // the min sum algorithm is the same operation without correction
      aggr = Lxfb(&(msg[imsg]), cdeg[j], 1);
      if (stopflag == 0 && aggr <= 0.0)
        stopflag = 1;
      imsg += cdeg[j];
    }

    if (!stopflag)
      break;
  }

  free(msg);
  
  /*  printf("Returning %d iterations\n");
  printf("C function over\n");
  */
  return(itcount); 
}

/* **********************************************************************
 *                                                                      *
 * Lxor(L1, L2, corr_flag) computes the LLR of the XOR (or modulo-2     *
 *  sum) of binary random variables given the LLRs L1 and L2 of the     *
 *  two variables. The function consists of a product of sign and       *
 *  minimisation of absolute values, followed by a two-step correction  *
 *  stage. Removing the correction stage, you obtain the min-sum        *
 *  approximation. The indicator corr_flag if set to zero bypasses      *
 *  the correction stage and performs the approximate calculation.      *
 *  Lxor(L1,L2) can be combined (both the corrected and approximate     *
 *  versions) to obtain the LLR of XORs of more than 2 random           *
 *  variables, e.g., Lxor(L1,Lxor(L2, L3)) gives the LLR of the XOR     *
 *  of three binary random variables.                                   *
 *                                                                      *
 *  This function is defined in the paper "Efficient Implementations    *
 *  of the Sum-Product Algorithm for Decoding LDPC Codes" by Hu,        *
 *  Elephteriou, Arnold, and Dholakia from IBM Research Zurich,         *
 *  presented at Globecom 2001. Beware however that there is a crucial  *
 *  typo in that paper: the two correction terms in Equation (5)        *
 *  are wrong (see two equations above (5) for the correct correction   *
 *  terms.
 *                                                                      *
 * Jossy, September 2018                                                *
 *                                                                      *
 ********************************************************************** */

double Lxor(double L1, double L2, int corr_flag)
{
  double L;

  // min rule, first multiply the signs
  if (signbit(L1) == signbit(L2))
    L = 1.0;
  else
    L = -1.0;
  L *= fmin(fabs(L1),fabs(L2)); // then the minimum of the absolute values
  // correction (this is done in the full sum product and skipped in the sum min)
  if (corr_flag) {
    L += log(1+exp(-fabs(L1+L2)));
    L -= log(1+exp(-fabs(L1-L2)));
  }

  return(L);
}

/* **********************************************************************
 *                                                                      *
 * Lxfb(L, dc, corr_flag) computes the eXtrinsic LLRs of dc binary      *
 *  random variables whose XOR is 0, i.e., for the j-th binary random   *
 *  variable it computes its LLR given observations of all binary       *
 *  variables except the j-th observation. The input vector L of length *
 *  dc gives the incoming individual LLRs of the dc binary random       *
 *  variables given their observations. The algorithm uses a forward-   *
 *  backward approach (closely related to the forward-backward          *
 *  algorithm used for hidden Markov models or the BCJR algorithm for   *
 *  convolutional codes): it computes a vector f (for forward) and b    *
 *  for backward as follows:                                            *
 *    fj = Lxor(L1, L2, ..., Lj)         and                            *
 *    bj = Lxor(Lj, L(j+1), Ldc)                                        *
 *  The results are then computed as                                    *
 *    Lxfb(Lj) = Lxor(b(j-1), f(j+1) for j=2...dc-1                     * 
 *    and Lxfb(1)=b2, Lxfb(dc)=f(dc-1)                                  *
 *  (note that we used user-friendly 1...dc indexing here in the        *
 *  but the function obviously uses C-compliant 0...dc-1 indexing)      *
 *                                                                      *
 * As in Lxor, corr_flag specifies whether to use the correction        *
 * to get the full sum product, or stop at minimisation and hence       *
 * use the min-sum approximation.                                       *
 *                                                                      *
 * The brute-force way to compute the extrinsic LLRs using the          *
 * pairwise operator Lxor would require dc-1 uses of Lxor for each      *
 * output, resulting in dc(dc-1) uses of Lxor. The present approach     *
 * requires only 3(dc-2) uses of Lxor. The method is described in the   *
 * paper "Efficient Implementations of the Sum-Product Algorithm for    *
 * Decoding LDPC Codes" by Hu, Elephteriou, Arnold, and Dholakia from   *
 * IBM Research Zurich, presented at Globecom 2001. The method is       *
 * described in the paper using the terminology of trellises.           *
 *                                                                      *
 * Jossy, September 2018                                                *
 *                                                                      *
 ********************************************************************** */

#define MAXDC 25 // some number larger than the maximum constraint node degree
// this is just to avoid having to dynamically allocated the required memory
// at every call, since the function is called very frequently

double Lxfb(double *L, long dc, int corr_flag)
{
  double f[MAXDC]; // forward values
  double b[MAXDC]; // backward values
  int k;

  // initilaize f[0] as L[0] and b[dc-1] as L[dc-1]
  for (k = 1, f[0] = L[0], b[dc-1] = L[dc-1] ; k < dc ; k++) {
    // compute the forward values as Lxor(f[k-1],L[k])
    f[k] = Lxor(f[k-1], L[k], corr_flag);
    // compute the backward values as Lxor(L[k],b[k+1])
    b[dc-k-1] = Lxor(b[dc-k], L[dc-k-1], corr_flag);
  }
  // set L[0] = b[1] ("product" of all except L[0])
  // and L[dc-1] = f[dc-2] ("product" of all except L[dc-1])
  for (k = 1, L[0] = b[1], L[dc-1] = f[dc-2] ; k < dc-1 ; k++)
    // all others are the "product" of f[k-1] with b[k+1]
    L[k] = Lxor(f[k-1], b[k+1], corr_flag);
  // return the overall (aggregated) non-extrinsic product (for stopping rule)
  return(b[0]);
}



/* **********************************************************************
 *                                                                      *
 * minsum(ch, vdeg, cdeg, intrlv, Nv, Nc, Nmsg, app)                    *
 * The min-sum algorithm is an approximation of the sum product         *
 * obtained by cutting out the correction in the constraint node        *
 * operation of sumprod2, so that the constraint node operation is      *
 * simply a product of the signs times the minimum of the absolute      *
 * values of the incoming messages. Its performance is slightly below   *
 * the sum product but it is numerically more stable and less           *
 * sensitive to the correct normalisation of the input log-likelihood   *
 * ratios (for example in a non-simulated communication setting, you    *
 * generally don't know the actual noise variance exactly: the sum      *
 * product requires you to estimate it accurately, whereas the sum min  *
 * will work as well if your estimate of the noise variance is wrong.)  *
 * The sum min can be improved by multiplying the result of the minimum *
 * by a heuristic correction factor.                                    *
 *                                                                      *
 * Jossy, September 2018                                                *
 *                                                                      *
 ********************************************************************** */

int minsum(double *ch, long *vdeg, long *cdeg, long *intrlv,
	   int Nv, int Nc, int Nmsg, double *app, double correction_factor)
{
  double *msg;
  int j, k, imsg, stopflag, itcount;
  double aggr;
  
  msg = calloc(Nmsg, sizeof(double));
  if (msg == NULL)
    return(-1);

  for (itcount = 0 ; itcount < MAX_ITCOUNT ; itcount++) {
    // var node (identical to sumprod)
    for (j = 0, imsg = 0 ; j < Nv ; j++) { 
      for (k = 0, aggr = ch[j] ; k < vdeg[j] ; k++, imsg++) 
        aggr += msg[intrlv[imsg]]; 
      imsg -= vdeg[j]; 
      for (k = 0; k < vdeg[j] ; k++, imsg++)
        msg[intrlv[imsg]] = aggr - msg[intrlv[imsg]];
      app[j] = aggr;
    }

    stopflag = 0;

    // constraint node 
    for (j = 0, imsg = 0 ; j < Nc ; j++, imsg += cdeg[j]) {
      // flag 0 no correction, so Lxfb will only approximate the sum product
      // operations by a minimum
      aggr = Lxfb(&(msg[imsg]), cdeg[j], 0); 
      if (stopflag == 0 && aggr <= 0.0)
        stopflag = 1;
      for (k = 0 ; k < cdeg[j]; k++)
        msg[imsg+k] *= correction_factor; // multiply result by a heuristic correction factor
    }

    if (!stopflag)
      break;
  }

  free(msg);

  return(itcount); 
}

/* **********************************************************************
 *                                                                      *
 * main()                                                               *
 * The following code is for testing the function of the decoders       *
 * directly in C. If using live or testing from python, comment out the *
 * command "#define __DEBUG__" at the beginning of the file and this    *
 * part of the code will be ignored.                                    *
 * The code assumes transmission of the all-zero codeword and           *
 * transmission over a binary symmetric channel. The input LLRs are     *
 * dimensioned to +2, -2 (where all the +2s are correct since this is   *
 * the all-zero vector, and the -2s correspond to channel errors). This *
 * corresponds approximately to the correct LLRs for a BSC with         *
 * crossover probability 0.1                                            *
 * The crossover probability can be specified as a command-line         *
 * argument (but the test will maintain the LLRs to +2/-2).             *
 *                                                                      *
 * Jossy, September 2018                                                *
 *                                                                      *
 ********************************************************************** */


#ifdef __DEBUG__

#include "ldpc802.16.81.h" // imports an LDPC decoder parameter from a file
// this is the 802.16 Wimax standard with z=81
#define Pe_default (0.09) // default if unspecified as a command line argument

int main(int argc, char *argv[])
{
  double ch[Nv]; // channel LLRs
  double app[Nv]; // decoder outputs
  int k, counterrors;
  int iterations;
  double Pe;

  // take the crossover probability from the comman line or detault
  if (argc == 1)
    Pe = Pe_default;
  else
    if (sscanf(argv[1], "%lf", &Pe) != 1)
      exit(-1);
    
  // generate the channel noise
  for (k = 0 , counterrors = 0 ; k < Nv ; k++) {
    ch[k] = 2.0-(double)codeword[k]*4.0;
    if (rand() < Pe * RAND_MAX) {
      ch[k] = -ch[k]; // channel error
      counterrors++;
    }
  }

    printf("Channel crossover probability: %g\n", Pe);
  printf("Entering sum product with %d/%d errors\n", counterrors, Nv);

  // call the decoder
  iterations = sumprod(ch, vdeg, cdeg, intrlv, Nv, Nc, Nmsg, app);

  // count the errors after the decoder
  for (counterrors = 0, k = 0 ; k < Nv ; k++)
    if ((app[k] < 0.0 && codeword[k] == 0) || (app[k] > 0.0 && codeword[k] == 1))
      counterrors++;

  printf("Exited sum product with exit code %d and %d/1944 errors\n", iterations, counterrors);
  
}

#endif // matches #ifdef __DEBUG__
