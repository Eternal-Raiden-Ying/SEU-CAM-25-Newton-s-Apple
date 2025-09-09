1. install MSYS2  https://www.msys2.org/#installation
2. run the code in MSYS2  
    `pacman -S mingw-w64-ucrt-x86_64-gcc`
3. add root directory of gcc.exe to **system PATH**  
    someting like *xxx\MSYS2\ucrt64\bin*
4. run the code under **ldpc_jossy** directory  

`Windows:`  
    `mkdir bin`  
    `gcc -lm -shared -fPIC -o bin/c_ldpc.dll src/c_ldpc.c`  
    `gcc -o bin/results2csv src/results2csv.c`  

`Linux and MacOs:`  
    `mkdir bin`  
    `gcc -lm -shared -fPIC -o bin/c_ldpc.so src/c_ldpc.c`  
    `gcc -o bin/results2csv src/results2csv.c` 

5. make sure pth in py/ldpc.py is right  
for Windows, it needs to be `c_ldpc = ct.CDLL('./bin/c_ldpc.dll')` (3 times)  
for Linux and MacOs, it needs to be `c_ldpc = ct.CDLL('./bin/c_ldpc.so')` (3 times)