import numpy as np
import csv
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

    
if len(sys.argv) > 1:
    save2file = True
    filename = sys.argv[1]
else:
    save2file = False


res = []
with open('data/results.csv', 'r') as f:
    rr = csv.reader(f)
    for row in rr:
        res.append(row)

#for fn in ['res1.csv', 'res2.csv']:
#    with open(fn, 'r') as f:
#        rr = csv.reader(f)
#        for row in rr:
#            res.append(row)

res = np.array(res)
std = res[:,0].astype(int)
R = res[:,1].astype(float)
tt = res[:,2].astype(int)
Z = res[:,3].astype(int)
EsNo = res[:,4].astype(float)
bler = res[:,6].astype(float)/res[:,5].astype(float)
ber = res[:,8].astype(float)/res[:,7].astype(float)
it = res[:,9].astype(float)/res[:,5].astype(float)
EbNo = EsNo - 10*np.log10(R)


SL = {0.5:3.1721, 0.667:4.0926, 0.75:4.6693, 0.833:5.3941} # Shannon limit

# display all performances by standard, rate and type for various Z on same plots

spl = 5
fign=0
for mystd in [11, 16]:
    # for display purposes: for 802.11n, will display all 4 rates on one page
    # while for 801.16, will display the 2 rates for which Type A&B codes exist
    # on one page and the 2 remaining rates on another page
    if mystd == 11:
        Rrange = [.5, .667, .75, .833]
    else:
        Rrange = [.667, .75, .5, .833]
        
    for myR in Rrange:
        for mytt in [0,1]:
            if (np.nonzero(np.logical_and(np.logical_and(std == mystd,
                                                         np.abs(R-myR)<.01),

                                          mytt == tt)))[0].size == 0:
                continue
            if spl == 5:
                plt.figure(fign, [10, 11])
                spl = 1
                fign = fign + 1
            plt.subplot(2,2,spl)
            #            plt.figure(fign)
            if mytt == 0:
                plt.title('802.%(st)d, R=%(rt).3g, Z=27,54,81' % {'st':mystd, 'rt':myR})
            else:
                plt.title('802.%(st)d, R=%(rt).3g, Type B, Z=27,54,81' % {'st':mystd, 'rt':myR})
            plt.yscale('log')
            plt.xlabel('Eb/No in dB')
            plt.ylabel('Bit Error Rate')
            plt.grid(True)
            for myZ in [27,54,81]:
                indx = np.nonzero(np.logical_and(np.logical_and(
                    np.logical_and(std == mystd, np.abs(R-myR)<.01),
                    tt == mytt), Z == myZ))[0]
                iindx = np.argsort(EbNo[indx])
                indx = indx[iindx]
                plt.plot(EbNo[indx], ber[indx], label='Z=%d'%myZ)
            plt.plot(np.repeat(SL[myR],2), plt.ylim(), 'r', label='Shannon limit')
            plt.legend()
            spl = spl + 1


# display by rate and Z for varying standard and type

for myZ in [27,54,81]:
    fign = fign + 1
    plt.figure(fign, [10,11])
    spl = 1
    for myR in [.5, .667, .75, .833]:
        plt.subplot(2,2,spl)
        plt.yscale('log')
        plt.xlabel('Eb/No in dB')
        plt.ylabel('Bit Error Rate')
        plt.grid(True)
        plt.title('Z=%(zz)d, R=%(rr).3f' % {'zz':myZ, 'rr':myR})
        indx = np.nonzero(np.logical_and(np.logical_and(
            std == 11, np.abs(R-myR)<.01),Z == myZ))[0]
        indx = indx[np.argsort(EbNo[indx])]
        plt.plot(EbNo[indx], ber[indx], label='802.11n')
        indx = np.nonzero(np.logical_and(np.logical_and(np.logical_and(
            std == 16, np.abs(R-myR)<.01),Z == myZ),tt==0))[0]
        indx = indx[np.argsort(EbNo[indx])]
        if (myR == .5 or myR == .833):
            plt.plot(EbNo[indx], ber[indx], label='802.16')
        else:
            plt.plot(EbNo[indx], ber[indx], label='802.16, Type A')
            indx = np.nonzero(np.logical_and(np.logical_and(np.logical_and(
                std == 16, np.abs(R-myR)<.01),Z == myZ),tt==1))[0]
            indx = indx[np.argsort(EbNo[indx])]
            plt.plot(EbNo[indx], ber[indx], label='802.16, Type B')
        plt.plot(np.repeat(SL[myR],2), plt.ylim(), 'r',  label='Shannon limit')
        plt.legend()
        spl = spl + 1

        
if save2file == True:
    multipage(filename)
else:
    plt.show()

    
