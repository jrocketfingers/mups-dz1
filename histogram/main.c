#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "dump.h"
#include "utils.h"

#define UINT8_MAX 255

int main(int argc, char* argv[]) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }

  int numIterations;
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  pb_InitializeTimerSet(&timers);
  
  char *inputStr = "Input";
  char *outputStr = "Output";
  
  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);
  
  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);  

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));

  unsigned char** partial_histo;
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));
  
  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  unsigned int min = -1;
  unsigned int max = 0;
  #pragma omp parallel for default(none) shared(img, img_width, img_height) reduction(min: min) reduction(max: max)
  for(unsigned int i = 0; i < img_height * img_width; i++) {
    if(min > img[i])
      min = img[i];

    if(max < img[i])
      max = img[i];
  }

  printf("min: %d; max %d\n\n", min, max);

  memset(histo, 0, histo_height * histo_width * sizeof(unsigned char));

  unsigned int n_threads;

  #pragma omp parallel default(none) shared(n_threads, partial_histo)
  {
    #pragma omp single
    {
      n_threads = omp_get_num_threads();

      partial_histo = malloc(n_threads * sizeof(unsigned char *));
    }
  }

  unsigned int i;
  unsigned int thread_id;
  #pragma omp parallel default(none) shared(max, min, partial_histo, n_threads)
  {
    partial_histo[omp_get_thread_num()] = malloc((max - min) * sizeof(unsigned char));
  }

  int iter;
  for (iter = 0; iter < numIterations; iter++) {
    #pragma omp parallel default(none) shared(img, img_width, img_height, partial_histo, max, min) private(thread_id, i)
    {
      memset(partial_histo[omp_get_thread_num()], 0, (max - min) * sizeof(unsigned char));

      #pragma omp for schedule(static)
      for (i = 0; i < img_width * img_height; i++) {
        const unsigned int value = img[i];

        if (partial_histo[omp_get_thread_num()][value - min] < UINT8_MAX) {
          ++partial_histo[omp_get_thread_num()][value - min];
        }
      }
    }

    memset(histo, 0, histo_height * histo_width * sizeof(unsigned char));

    for(unsigned int thread = 0; thread < n_threads; thread++) {
      for(unsigned int value = min; value <= max; value++) {
        if(histo[value] + partial_histo[thread][value - min] <= UINT8_MAX)
          histo[value] += partial_histo[thread][value - min];
        else
          histo[value] = UINT8_MAX;
      }
    }
  }


  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;
}
