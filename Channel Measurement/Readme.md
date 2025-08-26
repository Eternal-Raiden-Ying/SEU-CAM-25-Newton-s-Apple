in this version I removed files except source code, and the original directory is like this:

Channel Measurement   
\-- data        (to store original data, for emitter and for receiver when calculate BER)  
\-- output  
\-- \-- pic     (to store picture)  
\-- \-- file    (to store files decoded from received audio record)  
\-- record      (to store audio recording files(.npy))  
\-- src 
\-- \-- CLAP EXP  
\-- \-- ldpc_jossy  
\-- \-- \-- bin  
\-- \-- \-- \-- c_ldpc.dll  
\-- \-- \-- \-- results2csv.exe     (not used)  
\-- \-- \-- py  
\-- \-- \-- src  
\-- \-- \-- \-- c_ldpc.c  
\-- \-- \-- \-- ldpc802.16.81.h  
\-- \-- \-- \-- results2csv.c        (not used)  
\-- \-- OFDM  
\-- \-- utils  
\-- \-- receiver.py  
\-- \-- emitter.py  
\-- \-- record_signal.py  