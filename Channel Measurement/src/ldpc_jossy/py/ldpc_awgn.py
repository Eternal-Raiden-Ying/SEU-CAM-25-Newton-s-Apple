import numpy as np
import sys

import ldpc

sim_param = [
    ("802.16","1/2",3,"A"),
    ("802.16","2/3",3,"A"),
    ("802.16","2/3",3,"B"),
    ("802.16","3/4",3,"A"),
    ("802.16","3/4",3,"B"),
    ("802.16","5/6",3,"A"),
    ("802.16","1/2",27,"A"),
    ("802.16","2/3",27,"A"),
    ("802.16","2/3",27,"B"),
    ("802.16","3/4",27,"A"),
    ("802.16","3/4",27,"B"),
    ("802.16","5/6",27,"A"),
    ("802.16","1/2",54,"A"),
    ("802.16","2/3",54,"A"),
    ("802.16","2/3",54,"B"),
    ("802.16","3/4",54,"A"),
    ("802.16","3/4",54,"B"),
    ("802.16","5/6",54,"A"),
    ("802.16","1/2",81,"A"),
    ("802.16","2/3",81,"A"),
    ("802.16","2/3",81,"B"),
    ("802.16","3/4",81,"A"),
    ("802.16","3/4",81,"B"),
    ("802.16","5/6",81,"A"),
    ("802.11n","1/2",27,"A"),
    ("802.11n","2/3",27,"A"),
    ("802.11n","3/4",27,"A"),
    ("802.11n","5/6",27,"A"),
    ("802.11n","1/2",54,"A"),
    ("802.11n","2/3",54,"A"),
    ("802.11n","3/4",54,"A"),
    ("802.11n","5/6",54,"A"),
    ("802.11n","1/2",81,"A"),
    ("802.11n","2/3",81,"A"),
    ("802.11n","3/4",81,"A"),
    ("802.11n","5/6",81,"A"),
]

def awgn(x, snr_db, spow = 1):

    sigma2 = spow / np.power(10,snr_db/10.0)
    noise = np.sqrt(sigma2)*np.random.randn(len(x))
    return (np.add(x, noise), sigma2)

def ch2llr(ch, sigma2):

    return 2.0/sigma2*ch

def bpsk(x):
    return 1.0 - 2.0*x



def sim(standard, rate, z, ptype='A', N_MEASUREMENTS=24, C_AWGN_OFFSET=1.0, P_STEP=100.0,
        MIN_ERRORS = 100, MAX_BLOCKS = 400000):

    if rate == "1/2":
        R = .5
    elif rate == "2/3":
        R = 0.6667
    elif rate == "3/4":
        R = 0.75
    elif rate == "5/6":
        R = 0.83333
    else:
        raise NameError("Rate unsupported")

    # start 1 dB above the SNR corresponding to AWGN rate
    # (which is above capacity since this is a BiAWGN)
    SNR = 10.0*np.log10(np.power(2,R)-1.0) + C_AWGN_OFFSET

    mycode = ldpc.code(standard, rate, z, ptype)
    K = mycode.K

    res = []

    for datapoint in range(N_MEASUREMENTS): 
        nbiterrors = 0
        nblockerrors = 0
        nblocks = 0
        nit_total = 0
        while nblockerrors < MIN_ERRORS:
            u = np.random.randint(0,2,K)
            x = mycode.encode(u)
            xm = bpsk(x)

            (y,sigma2) = awgn(xm, SNR)

            yl = ch2llr(y, sigma2)
            (app,nit) = mycode.decode(yl, 'sumprod2')
            xh = (app < 0.0)

            biterrors_thisblock = np.sum(x != xh)
            nbiterrors += biterrors_thisblock
            if biterrors_thisblock:
                nblockerrors += 1
            nblocks += 1
            nit_total += nit

            if nblocks >= MAX_BLOCKS:
                break
            
        f = open('data/results.txt', 'a')
        output = (standard, rate, z, SNR, nblocks, nblockerrors, nblocks*K, nbiterrors,
                  nit_total)
        f.write(str(output))
        f.write("\n")
        f.close()
    
        SNR += np.sqrt(P_STEP/nblocks) # heuristic SNR stepping method
        # the SNR will be increased in dB by a number that depends on the number
        # of blocks that were required to measure the current block error rate.
        # Since the measurement is set to observe at least MIN_ERRORS errors, the number
        # of blocks will be larger for smaller error rates (up to a maximum of 
        # MAX_BLOCKS), this heuristic will adopt smaller step sizes as the error
        # rate becomes smaller. The sqrt() was found to give a better step size
        # heuristically.
        
if __name__ == "__main__":
    if len(sys.argv) > 1: # args numbered 1 to 36 (better for grid engine calls
        argind = int(sys.argv[1])-1
    else:
        argind = 24
        
    sim(*sim_param[argind])
        
