/*
//run the program in the background
nohup ./rknn_test model/PPO.rknn  &

//check the thread's pid
ps -T -p <pid> 

//check the npu load
cat /sys/kernel/debug/rknpu/load 

//check the frequency of core
sudo cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu1/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu3/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu4/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu5/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu6/cpufreq/cpuinfo_cur_freq
sudo cat /sys/devices/system/cpu/cpu7/cpufreq/cpuinfo_cur_freq

//initialize the pin
echo 92 > /sys/class/gpio/export  # Encoder A Phase 
echo in > /sys/class/gpio/gpio92/direction
echo falling > /sys/class/gpio/gpio92/edge
*/

/*-------------------------------------------
                  Includes
-------------------------------------------*/
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <memory.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <sched.h>
#include <pthread.h>
#include <wiringPi.h>

#include "ADS1x15.h"

using namespace std;


#define ADS1X15_ADDRESS     0x48
#define ENCODER_PINB        15
#define PWM_Pin             5
#define EN_Pin              7
#define DIR_Pin             8
ADS1115 ads;  /* Use this for the 16-bit version */
pthread_mutex_t mutex_PT;  //init the pthread_mutex_t
struct pollfd fds[1];


/*-------------------------------------------
                  Variables
-------------------------------------------*/
static volatile int key_value = 1;
/*-------------------------------------------
                  RKNN
-------------------------------------------*/
const int state_row = 1;  // input dimension
const int state_col = 3;   // input dimension
const int act_dim = 3;     // input dimension
float input_data[state_row][state_col] = {0.0 ,0.0 ,0.0}; // input matrix
int output_index = 0;    //index of the max value   

/*-------------------------------------------
                  states
-------------------------------------------*/
float theta_b = 0; 
float dtheta_b = 0; 
float dtheta_w= 0; 
int Maxon_PWM= 0; 

/*-------------------------------------------
                  ADS
-------------------------------------------*/
uint16_t adc0;
float volts0;


/*-------------------------------------------
                  Encoder
-------------------------------------------*/
float pulse = 0;  //count the pulse of the encoder
char forward_direction = '+';
char back_direction = '-';
char current_direction = 0; 
char direction_state = 0;
float pre_dtheta_b = 0; 


/*-------------------------------------------
                  Time testing
-------------------------------------------*/
double cur_time = 0;
double pre_time = 0;
double TimeUse = 0;
struct timeval StartTime;
struct timeval EndTime;


/*-------------------------------------------
                  Data Matrix
-------------------------------------------*/
static volatile int file_state = 1;
#define matrix_number 30000
int matrix_index = 0;  // current 
float time_matrix[matrix_number] = {0};
float theta_b_matrix[matrix_number] = {0};
float dtheta_b_matrix[matrix_number] = {0};
float dtheta_w_matrix[matrix_number] = {0};
float action_matrix[matrix_number] = {0};



void* Thread_50ms(void* arg);
void* File_thread(void* arg);
void* EXTI_thread(void* arg);
void encoder_info(double delta_t);
void T_start();
double T_end();
static void dump_tensor_attr(rknn_tensor_attr *attr);
static inline int64_t getCurrentTimeUs();

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int ret = 0;
rknn_context ctx = 0;
rknn_sdk_version sdk_ver;
rknn_input_output_num io_num;
rknn_tensor_attr input_attrs;
rknn_tensor_attr output_attrs;
int main(int argc, char *argv[])
{
   char *model_path = argv[1];
   // Load RKNN model
   int model_len = 0;
   ret = rknn_init(&ctx, &model_path[0], 0, 0, NULL);
   if (ret < 0){
      printf("rknn_init fail! ret=%d\n", ret);
   }

   // Get sdk and driver version
   ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
   if (ret != RKNN_SUCC){
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
   }
   printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

   // Get Model Input Output Info
   ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
   if (ret != RKNN_SUCC){
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
   }
   printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

   printf("input tensors:\n");
   memset(&input_attrs, 0, sizeof(rknn_tensor_attr));
   input_attrs.index = 0;
   ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs), sizeof(rknn_tensor_attr));
   if (ret < 0){
      printf("rknn_init error! ret=%d\n", ret);
      return -1;
   }
   input_attrs.type = RKNN_TENSOR_FLOAT32;
   input_attrs.size = input_attrs.size_with_stride = input_attrs.n_elems * sizeof(float);
   dump_tensor_attr(&input_attrs);

   printf("output tensors:\n");
   memset(&output_attrs, 0, sizeof(rknn_tensor_attr));
   output_attrs.index = 0;
   ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs), sizeof(rknn_tensor_attr));
   if (ret != RKNN_SUCC){
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
   }
   output_attrs.type = RKNN_TENSOR_FLOAT32;
   output_attrs.size = output_attrs.size_with_stride = output_attrs.n_elems * sizeof(float);
   dump_tensor_attr(&output_attrs);

   if (!ads.begin(1, ADS1X15_ADDRESS)) {
      while (1);
   }
   cout << "  ADS1115 init done!" << endl;

   //************** nonblocking I/O *******
   int flag1, flag2;
   if(flag1=(fcntl(STDIN_FILENO, F_GETFL, 0)) < 0)
   {
      perror("fcntl");
      return -1;
   }
   flag2 = flag1 | O_NONBLOCK;
   fcntl(STDIN_FILENO, F_SETFL, flag2); 
   //*************************************

   wiringPiSetup();
   pinMode(ENCODER_PINB, INPUT);
   pinMode(DIR_Pin, OUTPUT);
   pinMode(EN_Pin, OUTPUT);
   pinMode(PWM_Pin , PWM_OUTPUT);
   pwmSetMode(PWM_Pin,PWM_MODE_MS);
   //我们把pwm分为2400分，要5000hz的pwm频率，那时钟频率=24000000 Hz /(5000 * 2400)=2
   pwmSetClock(PWM_Pin,2);
   pwmSetRange(PWM_Pin,2400);
   pwmWrite(PWM_Pin,10);
   digitalWrite(EN_Pin,HIGH);

   //************** file *****************
   FILE *fd = fopen("./data.csv", "w+");
   if (fd == NULL){
      fprintf(stderr, "fopen() failed.\n");
      cout << "  Failed" << endl;
      exit(EXIT_FAILURE);
   }
   fprintf(fd, "time,theta_b,dtheta_b,dtheta_w,action\n");
   //*************************************

   pthread_t id1,id2,id3;
	int value;
   void *reVal, *reVa2, *reVa3;

   value = pthread_create(&id1, NULL, EXTI_thread, NULL);
   pthread_setname_np(id1, "EXTI_thread");
	if(value){
      cout << "  EXTI_thread is not created!" << endl;
      return -1;
	}

   value = pthread_create(&id2, NULL, File_thread, (void *)fd);
   pthread_setname_np(id2, "File_thread");
	if(value){
      cout << "  File_thread is not created!" << endl;
      return -1;
	}

   value = pthread_create(&id3, NULL, Thread_50ms, NULL);
   pthread_setname_np(id3, "Thread_50ms");
	if(value){
      cout << "  Thread_50ms is not created!" << endl;
      return -1;
	}

   pthread_join(id1, &reVal);
   pthread_join(id2, &reVa2);
   pthread_join(id3, &reVa3);
   
   cout << "  EXTI_thread exiting with status :" << reVal << "\n" << endl;
   cout << "  File_thread exiting with status :" << reVa2 << "\n" << endl;
   cout << "  Thread_50ms exiting with status :" << reVa3 << "\n" << endl;
   return 0;
}

static inline int64_t getCurrentTimeUs()
{
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
   printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
   "zp=%d, scale=%f\n",
   attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
   attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
   get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


void T_start()
{
    gettimeofday(&StartTime, NULL);  //measure the time
}

double T_end()
{
    gettimeofday(&EndTime, NULL);   //measurement ends
    TimeUse = 1000000*(EndTime.tv_sec-StartTime.tv_sec)+EndTime.tv_usec-StartTime.tv_usec;
    TimeUse /= 1000;  //the result is in the ms dimension
    return TimeUse;
}

void encoder_info(double delta_t)  //calculate the information of the encoder
{
    dtheta_b = pulse * 351.5625 / delta_t; //   n / ((delta_t / 1000) * 1024) * 360
   //  dtheta_b = 0.8 * pre_dtheta_b + 0.2 * dtheta_b; // low pass filter

    if(current_direction == forward_direction){
        theta_b = theta_b + pulse * 360 / 1024; //calculate the positive theta_b 
    }
    else{
        theta_b = theta_b - pulse * 360 / 1024; //calculate the negative theta_b 
        dtheta_b = -dtheta_b;
    }
    pulse = 0;    //set pulse to zero
   //  pre_dtheta_b = dtheta_b;
}


void* EXTI_thread(void* arg)
{
    int fd = open("/sys/class/gpio/gpio92/value",O_RDONLY);
    if(fd<0){
        perror("open '/sys/class/gpio/gpio92/value' failed!\n");  
        return (void *)-1;
    }
    fds[0].fd=fd;
    fds[0].events=POLLPRI;

    T_start();
    while(1){
        if(file_state ==0)pthread_exit(NULL); //exit the thread
        if(poll(fds,1,0)==-1){
            cout << "poll failed!\n" << endl;
            return (void *)-1;
        }
        if(fds[0].revents&POLLPRI){
            if(lseek(fd,0,SEEK_SET)==-1){
                cout << "lseek failed!\n" << endl;
                return (void *)-1;
            }
            char buffer[16];
            int len;
            if((len=read(fd,buffer,sizeof(buffer)))==-1){                
                cout << "read failed!\n" << endl;
                return (void *)-1;
            }
            else{
                if(digitalRead(ENCODER_PINB) == HIGH){  //if B pin is in the high place
                    current_direction = forward_direction;
                }
                else{
                    current_direction = back_direction;
                }
                pulse++;

                pre_time = cur_time;
                cur_time = T_end();
                encoder_info(cur_time - pre_time);
                

                printf("\r  The theta_b is: %.2f degrees, The dtheta_b is: %.2f degrees/s     ", theta_b, dtheta_b);   
                fflush(stdout); 
                
                
                pthread_mutex_lock(&mutex_PT);
                time_matrix[matrix_index] = cur_time;
                theta_b_matrix[matrix_index] = theta_b;
                dtheta_b_matrix[matrix_index] = dtheta_b;
                dtheta_w_matrix[matrix_index] = dtheta_w;
                action_matrix[matrix_index] = output_index;
                ++matrix_index;
                if(matrix_index > matrix_number){
                    cout << "\nMatrix number error!" << endl;
                    return (void *)-1;
                }
                pthread_mutex_unlock(&mutex_PT);

            }
        }
    }
}

void* File_thread(void* arg)
{
   FILE *fp = (FILE*)arg;
   char i = 0;
   while(1){
      i = getchar();
      if(i == 'r')
         key_value = 0;
      if(i == 'q'){
         file_state = 0;
         digitalWrite(EN_Pin,LOW); //close motor

         pthread_mutex_lock(&mutex_PT);
         int j = matrix_index;
         while(matrix_index--){
               fprintf(fp,"%.2f,%.2f,%.2f,%.2f,%.2f\n",  time_matrix[j - matrix_index],\
                                             theta_b_matrix[j - matrix_index],\
                                             dtheta_b_matrix[j - matrix_index],\
                                             dtheta_w_matrix[j - matrix_index],\
                                             action_matrix[j - matrix_index]);                        
         }
         cout << "  Data has been saved Over!!!" << endl;
         fclose(fp); 
         pthread_mutex_unlock(&mutex_PT);
         pthread_exit(NULL);
      }
   }
}

void* Thread_50ms(void* arg)
{
   
   // create input tensor memory
   rknn_tensor_mem *input_mems;
   input_mems = rknn_create_mem(ctx, input_attrs.size);

   // create output tensor memory
   rknn_tensor_mem *output_mems;
   output_mems = rknn_create_mem(ctx, output_attrs.size);

   // Set input tensor memory
   ret = rknn_set_io_mem(ctx, input_mems, &input_attrs);
   if (ret < 0){
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return NULL;
   }
   ret = rknn_set_io_mem(ctx, output_mems, &output_attrs);
   if (ret < 0){
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return NULL;
   }

   while(1){
      // 0~4V  -5000 rpm~5000 rpm
      adc0 = ads.readADC_SingleEnded(0);
      volts0 = ads.computeVolts(adc0);
      pthread_mutex_lock(&mutex_PT);
      dtheta_w = (volts0 - 2) / 2 * 5000 * 6; // degrees / s 
      pthread_mutex_unlock(&mutex_PT);

      
      input_data[0][0] = theta_b * M_PI / 180.0;
      input_data[0][1] = dtheta_b * M_PI / 180.0;
      input_data[0][2] = dtheta_w * M_PI / 180.0;

      input_data[0][0] = (input_data[0][0] - M_PI / 2) / (0.5 * M_PI);
      input_data[0][1] = input_data[0][1] / 2;
      input_data[0][2] = input_data[0][2] / 30;

      // copy input data to input tensor memory
      memcpy(input_mems->virt_addr, input_data, sizeof(input_data));

      // Run
      int64_t start_us = getCurrentTimeUs();
      ret = rknn_run(ctx, NULL);
      int64_t elapse_us = getCurrentTimeUs() - start_us;
      if (ret < 0){
         printf("rknn run error %d\n", ret);
         return NULL;
      }
         // printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
      

      // print the result
      float *buffer = (float *)(output_mems->virt_addr);
      output_index = max_element(buffer,buffer+3) - buffer;

      // for (int i = 0; i < state_col; i++){
      //    cout << input_data[0][i] << endl;
      // }
      // for (int i = 0; i < act_dim; i++){
      //    cout << buffer[i] << endl;
      // }
      // cout << output_index << endl;

      if(file_state == 0){
         // Destroy rknn memory
         rknn_destroy_mem(ctx, input_mems);
         rknn_destroy_mem(ctx, output_mems);
         // Destroy
         rknn_destroy(ctx);

         pthread_exit(NULL); //exit the thread
      }
      
      switch(output_index){
         case 0:
            Maxon_PWM = 2180;
            break;
         case 1:
            Maxon_PWM = 0;
            break;
         case 2:
            Maxon_PWM = -2180;   
            break;
         default:
            break;
      }
      
      // if(abs(theta_b)<2.5)Maxon_PWM=0;
      // Maxon_PWM = 2180; //max pwm
      if(Maxon_PWM > 0){
         digitalWrite(DIR_Pin,LOW);
         Maxon_PWM = Maxon_PWM * 0.8 + 240;
      }
      else{
         digitalWrite(DIR_Pin,HIGH);
         Maxon_PWM = Maxon_PWM * 0.8 - 240;
      }
      if(key_value == 1)
         Maxon_PWM = 240;
      pwmWrite(PWM_Pin,abs(Maxon_PWM));
      sleep(0.05);
   }
}
