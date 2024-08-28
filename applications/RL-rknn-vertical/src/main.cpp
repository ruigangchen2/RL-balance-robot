/*
//run the program in the background
nohup ./rknn_test model/PPO_efficient.rknn  &

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

*/

/*-------------------------------------------
                  Includes
-------------------------------------------*/
#include <sys/time.h>
#include <algorithm>
#include <wiringPi.h>
#include "rknn_api.h"
#include "ADS1x15.h"
#include "../imu/MotionSensor.h"

using namespace std;

#define ADS1X15_ADDRESS     0x48
#define PWM_Pin             5
#define EN_Pin              7
#define DIR_Pin             8
ADS1115 ads;  /* Use this for the 16-bit version */
pthread_mutex_t mutex_PT;  //init the pthread_mutex_t

/*-------------------------------------------
                  Variables
-------------------------------------------*/
int key_value = 1;
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
uint16_t adc0 = 0;
float volts0 = 0;
float error = 0;
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

void* Thread_adc(void* arg);
void* Thread_imu(void* arg);
void* Thread_action(void* arg);
void* Thread_file(void* arg);
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

   if (!ads.begin(3, ADS1X15_ADDRESS)) {
      while (1);
   }
   cout << "ADS1115 init done!" << endl;

	ms_open();
   cout << "IMU init done!" << endl;

   wiringPiSetup();
   pinMode(DIR_Pin, OUTPUT);
   pinMode(EN_Pin, OUTPUT);
   pinMode(PWM_Pin , PWM_OUTPUT);
   pwmSetMode(PWM_Pin,PWM_MODE_MS);
   //我们把pwm分为2400分，要5000hz的pwm频率，那时钟频率=24000000 Hz /(5000 * 2400)=2
   pwmSetClock(PWM_Pin,2);
   pwmSetRange(PWM_Pin,2400);
   pwmWrite(PWM_Pin,240);
   digitalWrite(EN_Pin,HIGH);
   digitalWrite(DIR_Pin,HIGH);


   //************** file *****************
   FILE *fd = fopen("./data.csv", "w+");
   if (fd == NULL){
      fprintf(stderr, "fopen() failed.\n");
      cout << "  Failed" << endl;
      exit(EXIT_FAILURE);
   }
   fprintf(fd, "time,theta_b,dtheta_b,dtheta_w,action\n");
   //*************************************

   pthread_t id1,id2,id3,id4;
	int value;
   void *reVa1, *reVa2, *reVa3, *reVa4;

   value = pthread_create(&id1, NULL, Thread_imu, (void *)fd);
   pthread_setname_np(id1, "Thread_imu");
	if(value){
      cout << "  Thread_imu is not created!" << endl;
      return -1;
	}

   value = pthread_create(&id2, NULL, Thread_file, (void *)fd);
   pthread_setname_np(id2, "Thread_file");
	if(value){
      cout << "  Thread_file is not created!" << endl;
      return -1;
	}

   value = pthread_create(&id3, NULL, Thread_action, NULL);
   pthread_setname_np(id3, "Thread_action");
	if(value){
      cout << "  Thread_action is not created!" << endl;
      return -1;
	}

   value = pthread_create(&id4, NULL, Thread_adc, (void *)fd);
   pthread_setname_np(id4, "Thread_adc");
	if(value){
      cout << "  Thread_adc is not created!" << endl;
      return -1;
	}

   pthread_join(id1, &reVa1);
   pthread_join(id2, &reVa2);
   pthread_join(id3, &reVa3);
   pthread_join(id4, &reVa4);
   
   cout << "  Thread_adc exiting with status :" << reVa1 << "\n" << endl;
   cout << "  Thread_imu exiting with status :" << reVa2 << "\n" << endl;
   cout << "  Thread_file exiting with status :" << reVa3 << "\n" << endl;
   cout << "  Thread_action exiting with status :" << reVa4 << "\n" << endl;
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



void* Thread_adc(void* arg)
{

   // for(int i = 0; i < 500; i++){
   //    adc0 = ads.readADC_SingleEnded(0);  // 10ms time consumed
   //    error += (-(ads.computeVolts(adc0) - 2.0) / 2.0 * 5000.0);
   //    // error += ads.computeVolts(adc0);
   // }
   // error = error / 500;
   // printf("adc error = %.2f\n", error);

   while(1){
      if(file_state == 0)pthread_exit(NULL); //exit the thread
      
      // 0~4V  -5000 rpm~5000 rpm
      adc0 = ads.readADC_SingleEnded(0);  // 10ms time consumed
      volts0 = ads.computeVolts(adc0);

      pthread_mutex_lock(&mutex_PT);
      dtheta_w = ((volts0 - 2.0) / 2.0 * 5000.0 * 6) + error; // degrees / s
      pthread_mutex_unlock(&mutex_PT);

   }
}

void* Thread_imu(void* arg)
{

   while(1){
      if(file_state == 0)pthread_exit(NULL); //exit the thread
      
      ms_update(); // 5ms update
      pthread_mutex_lock(&mutex_PT);
      theta_b = ypr[ROLL];
      dtheta_b = gyro[ROLL];
      pthread_mutex_unlock(&mutex_PT);
      // printf("\rroll = %.2f\tdroll = %.2f\tdwheel = %.2f", ypr[ROLL], gyro[ROLL], dtheta_w);
      // fflush(stdout); 
   }
}

void* Thread_file(void* arg)
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

void* Thread_action(void* arg)
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

   T_start();
   while(1){

      input_data[0][0] = theta_b * M_PI / 180.0;
      input_data[0][1] = dtheta_b * M_PI / 180.0;
      input_data[0][2] = dtheta_w * M_PI / 180.0;

      input_data[0][0] = (input_data[0][0] - M_PI / 2) / (0.5 * M_PI);
      input_data[0][1] = input_data[0][1] / 2;
      input_data[0][2] = input_data[0][2] / 30;

      // copy input data to input tensor memory
      memcpy(input_mems->virt_addr, input_data, sizeof(input_data));

      // Run
      // int64_t start_us = getCurrentTimeUs();
      ret = rknn_run(ctx, NULL);
      // int64_t elapse_us = getCurrentTimeUs() - start_us;
      if (ret < 0){
         printf("rknn run error %d\n", ret);
         return NULL;
      }
      // printf("Elapse Time = %.2fms\n", elapse_us / 1000.f);
   
      // print the result
      float *buffer = (float *)(output_mems->virt_addr);
      output_index = max_element(buffer,buffer+3) - buffer;
      printf("\rroll = %.2f\tdroll = %.2f\tdwheel = %.2f   output_index= %d", ypr[ROLL], gyro[ROLL], dtheta_w, output_index);
      fflush(stdout); 

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

      pthread_mutex_lock(&mutex_PT);
      time_matrix[matrix_index] = T_end();
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

      switch(output_index){
         case 0:
            Maxon_PWM = -2180;
            break;
         case 1:
            Maxon_PWM = 0;
            break;
         case 2:
            Maxon_PWM = 2180;   
            break;
         default:
            break;
      }
      
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
         Maxon_PWM = 0;

      pwmWrite(PWM_Pin,abs(Maxon_PWM));  // 5ms time consumed
      // sleep(0.015);
   }
}
