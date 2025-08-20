#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])

{

  FILE *fin, *fout;
  int line, d;
  char x[500];

  char std[12];
  char type;
  char outfilename[500];
  char infilename[500];
  char default_prefix[] = "data/results";
  char *prefix;
  
  int rnum,rden,z,nblock,nbler, nbit,nber, nit;
  double snr, rate;

  if (argc > 1) {
    if (argv[1][0] == '-') {
      printf("Usage: %s [infile_prefix]\n", argv[0]);
      printf("  infile must be of type .txt, default is data/results.txt\n");
      printf("  outfile will be .csv with same prefix, default data/results.csv\n");
      exit(0);
    }
    prefix = argv[1];
  }
  else
    prefix = default_prefix;
  
  sprintf(infilename, "%s.txt", prefix);
  sprintf(outfilename, "%s.csv", prefix);
  
  fin = fopen(infilename, "r");
  fout = fopen(outfilename, "w");

  if (fin == NULL || fout == NULL) {
    printf("Unable to open meaurement file %s\n", infilename);
    exit(-1);
  }
  else
    printf("Opened file %s for processing\n", infilename);
  
  line = 1;
  while (fgets(x, 500, fin) != NULL) {
    d = sscanf(x, "('%[^']', '%1d/%1d', %d, '%c', %lf, %d, %d, %d, %d, %d)",
	       std, &rnum, &rden, &z, &type, &snr, &nblock, &nbler, &nbit, &nber, &nit);
    printf("Scanned %d items in line %d\n", d, line);
    if (d != 11) {
      printf("Error reading line %d in results.txt\n", line);
      exit(-1);
    }
    /*    if (sscanf(x, "('802.1%[']s, '%1d/%1d', %d, %lf, %d, %d, %d, %d, %d)",
	       std, &rnum, &rden, &z, &snr, &nblock, &nbler, &nbit, &nber, &nit)
	!= 10) {
      printf("Error reading line %d\n", line);
      exit(-1);
      }*/
    if (std[5] == '6')
      fprintf(fout, "16, ");
    else
      fprintf(fout, "11, ");
    rate = ((double)rnum)/rden;
    fprintf(fout, "%g, ", rate);
    if (type == 'A')
      fprintf(fout, "0, ");
    else
      fprintf(fout, "1, ");
    fprintf(fout, "%d, %lg, %d, %d, %d, %d, %d\n", z, snr, nblock, nbler, nbit, nber, nit);
    line++;
  }
  fclose(fin);
  fclose(fout);

  printf("Finished scanning %d measurements, output written to %s\n", line, outfilename);
  
}
