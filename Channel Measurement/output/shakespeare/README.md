**Here are some of the results when we try to transmit a txt (one of shakespeare's poem)**

In fact it's not very long, but transmit it is not that easy.  

We first transmit the bits from .txt file directly. 
And *data(TD without scrambler).png* shows the signal in time domain we transmit. 
The received signal is quite noisy and we could not extract anything.  

After that, we clipped the sharp peaks and re-normalized the signal, the BER of data actually
declined, but is still unacceptable, around 36%. When we draw the constellation distribution map
(in *the miss imag.png*), something interesting happened. The real part of constellation seems to 
be right, but all info in imag seems to be lost. And what we received seems to be a BPSK map instead of
a QPSK map. Quite strange!  

As we further explore, we find that the reason we lost info in imag is we clipped the signal.
In order to reserve the info, we should not clip it. And then, we are back to the starting point.

In order to solve that problem, we use a scrambler to destroy the periodicity of the original
bit (we suppose the sharp peaks are caused by the periodicity in bits, that's in freq domain,
so there will be an impluse in time domain. that makes sense, and this correspond to the details
of the first pic -- in each symbol, the peak occurs at a similar position).

And that idea does work! The results are shown in *constellation_pilot.png*, *constellation_txt_data.png*
and shakespeare-带中文.txt (By the way, the *shakespeare(未优化).txt*)

As to why scrambler can and how it can make it, see more at function comments in scrambler (utils.encode)


