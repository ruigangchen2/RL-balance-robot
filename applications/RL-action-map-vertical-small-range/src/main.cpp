#include <iostream>
#include <iomanip>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <array>
#include <vector>

#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/time.h>

#include <sched.h>
#include <pthread.h>
#include <wiringPi.h>

#include <boost/asio.hpp>
#include <boost/system/error_code.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include "H5Cpp.h"
#include "ADS1x15.h"
#include "../imu/MotionSensor.h"


using namespace std;
using namespace H5;


/*
//run the program in the background
nohup ./run  & 

//check the thread's pid
ps -T -p <pid> 

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

#define ADS1X15_ADDRESS     0x48
#define PWM_Pin             5
#define EN_Pin              7
#define DIR_Pin             8
ADS1115 ads;  /* Use this for the 16-bit version */
pthread_mutex_t mutex_PT;  //init the pthread_mutex_t

float theta_b = 0; 
float pre_theta_b = 0; 
float dtheta_b = 0; 
float dtheta_w= 0; 

/***** ads variable *******/
uint16_t adc0;
float volts0;
/******************************/

/***** Testing time *****/
double TimeUse = 0;
double TimeIMU = 0;
struct timeval StartTime;
struct timeval EndTime;
struct timeval StartIMU;
struct timeval EndIMU;
/******************************/

/***** data matrix *****/
static volatile int file_state = 1;
static volatile int key_value = 1;
#define matrix_number 300000
int matrix_index = 0;  // current 
float time_matrix[matrix_number] = {0};
float theta_b_matrix[matrix_number] = {0};
float dtheta_b_matrix[matrix_number] = {0};
float dtheta_w_matrix[matrix_number] = {0};
float action_matrix[matrix_number] = {0};
/******************************/


void* Thread_imu(void* arg);
void* Thread_file(void* arg);
void* Thread_action(void* arg);
void* Thread_adc(void* arg);
void T_start();
double T_end();
double DegreesToRadians(double degrees);
double RadiansToDegrees(double radians);
double readAndParseData(boost::asio::serial_port& serial);
double getValue(const std::vector<double>& data, const hsize_t* dims, int i, int j, int k);
std::pair<std::vector<double>, std::vector<hsize_t>> readData(const std::string& filename, const std::string& datasetname);
int argmin(double y0, const std::vector<double>& x);
std::vector<double> create_linspace(double start, double end, int N);


/***** RL related *****/
float action = 0;
float action_pre = 0;
int state1 = 0;
int state2 = 0;
int state3 = 0;
int state1_pre = 0;
int state2_pre = 0;
int state3_pre = 0;
std::vector<double> RL_theta_b = create_linspace(-20, 20, 40); 
std::vector<double> RL_dtheta_b = create_linspace(RadiansToDegrees(-2), RadiansToDegrees(2), 50);
std::vector<double> RL_dtheta_w = create_linspace(RadiansToDegrees(-500), RadiansToDegrees(500), 500);
const std::string filename = "table/action-table-large-range";
const std::string datasetname = "working_save";
auto dataAndDims = readData(filename, datasetname);
std::vector<double> table_data = dataAndDims.first;
std::vector<hsize_t> dims = dataAndDims.second;

int state1_small_range = 0;
int state2_small_range = 0;
int state3_small_range = 0;
int state1_pre_small_range = 0;
int state2_pre_small_range = 0;
int state3_pre_small_range = 0;
std::vector<double> RL_theta_b_small_range = create_linspace(-5, 5, 40); 
std::vector<double> RL_dtheta_b_small_range = create_linspace(RadiansToDegrees(-1), RadiansToDegrees(1), 50);
std::vector<double> RL_dtheta_w_small_range = create_linspace(RadiansToDegrees(-15), RadiansToDegrees(15), 200);
const std::string filename_small_range = "table/action-table-small-range";
const std::string datasetname_small_range = "working_save";
auto dataAndDims_small_range = readData(filename_small_range, datasetname_small_range);
std::vector<double> table_data_small_range = dataAndDims_small_range.first;
std::vector<hsize_t> dims_small_range = dataAndDims_small_range.second;
/******************************/

 
int main()
{
    ms_open();
    cout << "IMU init done!" << endl;

    if (!ads.begin(3, ADS1X15_ADDRESS))while(1);
    cout << "ADS1115 init done!" << endl;

    wiringPiSetup();
    pinMode(DIR_Pin, OUTPUT);
    pinMode(EN_Pin, OUTPUT);
    pinMode(PWM_Pin , PWM_OUTPUT);
    pwmSetMode(PWM_Pin,PWM_MODE_MS);    
    pwmSetClock(PWM_Pin,2);//我们把pwm分为2400分，要5000hz的pwm频率，那时钟频率=24000000 Hz / (5000 * 2400) = 2
    pwmSetRange(PWM_Pin,2400);
    pwmWrite(PWM_Pin,240);
    digitalWrite(EN_Pin,HIGH);
    // digitalWrite(DIR_Pin,HIGH);

//************** file *****************
    FILE *fd = fopen("./data.csv", "w+");
    if (fd == NULL){
        fprintf(stderr, "fopen() failed.\n");
        cout << "Failed" << endl;
        exit(EXIT_FAILURE);
    }
    fprintf(fd, "time,theta_b,dtheta_b,dtheta_w\n");
//*************************************

    pthread_t id1,id2,id3,id4;
	int value;
    void *reVa1, *reVa2, *reVa3, *reVa4;

    value = pthread_create(&id1, NULL, Thread_imu, NULL);
    pthread_setname_np(id1, "Thread_imu");
	if(value){
        cout << "Thread_imu is not created!" << endl;
        return -1;
	}

    value = pthread_create(&id2, NULL, Thread_file, (void *)fd);
    pthread_setname_np(id2, "Thread_file");
	if(value){
        cout << "Thread_file is not created!" << endl;
        return -1;
	}

    value = pthread_create(&id3, NULL, Thread_action, NULL);
    pthread_setname_np(id3, "Thread_action");
	if(value){
        cout << "Thread_action is not created!" << endl;
        return -1;
	}

    value = pthread_create(&id4, NULL, Thread_adc, NULL);
    pthread_setname_np(id4, "Thread_adc");
	if(value){
        cout << "Thread_adc is not created!" << endl;
        return -1;
	}

    pthread_join(id1, &reVa1);
    pthread_join(id2, &reVa2);
    pthread_join(id3, &reVa3);
    pthread_join(id4, &reVa4);
   
    cout << "Thread_imu exiting with status :" << reVa1 << "\n" << endl;
    cout << "Thread_file exiting with status :" << reVa2 << "\n" << endl;
    cout << "Thread_action exiting with status :" << reVa3 << "\n" << endl;
    cout << "Thread_adc exiting with status :" << reVa4 << "\n" << endl;
    return 0;
}



//创建一个等差数组
std::vector<double> create_linspace(double start, double end, int N) 
{
    double step = (end - start) / (N - 1);
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) {
        x[i] = start + i*step;
    }
    return x;
}

//找到最近的索引
int argmin(double y0, const std::vector<double>& x) 
{
    auto result = std::min_element(x.begin(), x.end(), [&](double a, double b) {
        return std::abs(a - y0) < std::abs(b - y0);
    });
    return std::distance(x.begin(), result);     // Index
}


// 读取数据
std::pair<std::vector<double>, std::vector<hsize_t>> readData(const std::string& filename, const std::string& datasetname)
{
    H5File file(filename, H5F_ACC_RDONLY);
    DataSet dataset = file.openDataSet(datasetname);

    DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data(), NULL);
    
    hsize_t total_size = 1;
    for(int i = 0; i < rank; i++)
    {
        total_size *= dims[i];
    }

    std::vector<double> data(total_size);
    dataset.read(data.data(), PredType::NATIVE_DOUBLE);

    return std::make_pair(data, dims);
}

// 获取数组具体的值
double getValue(const std::vector<double>& data, const hsize_t* dims, int i, int j, int k)
{
    int index = k + dims[2] * (j + dims[1] * i);
    return data[index];
}

double DegreesToRadians(double degrees) 
{
    return degrees * M_PI / 180.0;
}

double RadiansToDegrees(double radians) 
{
    return radians * 180.0 / M_PI;
}


double readAndParseData(boost::asio::serial_port& serial) 
{
    // 创建一个缓冲区来存储返回的数据.
    std::array<uint8_t, 128> buf;
    boost::system::error_code ec;
    size_t len = serial.read_some(boost::asio::buffer(buf), ec);
    if(ec) {
        std::cerr << "Error on reading: " << ec.message() << "\n";
        return 0; // or some error code
    }

    // 解析接收到的数据
    uint16_t encoder = (static_cast<uint16_t>(buf[2]) << 8) | buf[1];
    uint16_t encoderRaw = (static_cast<uint16_t>(buf[4]) << 8) | buf[3]; 
    uint16_t encoderOffset = (static_cast<uint16_t>(buf[6]) << 8) | buf[5];
    uint8_t checksum = buf[7];

    return static_cast<double>(encoderOffset);
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
         cout << "Data has been saved Over!!!" << endl;
         fclose(fp); 
         pthread_mutex_unlock(&mutex_PT);
         pthread_exit(NULL);
      }
   }
}



void* Thread_imu(void* arg)
{
    gettimeofday(&StartIMU, NULL);  //measure the time
    while(1){
        if(file_state == 0)pthread_exit(NULL); //exit the thread
        

        ms_update(); // 5ms update
        gettimeofday(&EndIMU, NULL);   //measurement ends
        TimeIMU = 1000000*(EndIMU.tv_sec-StartIMU.tv_sec)+EndIMU.tv_usec-StartIMU.tv_usec;
        TimeIMU /= 1000000;
        gettimeofday(&StartIMU, NULL);  //measure the time
        
        pthread_mutex_lock(&mutex_PT);
        theta_b = ypr[ROLL];
        dtheta_b = (theta_b - pre_theta_b)/TimeIMU;
        pre_theta_b = theta_b;
        pthread_mutex_unlock(&mutex_PT);
        
    }
}

void* Thread_adc(void* arg)
{

    while(1){
        if(file_state == 0)pthread_exit(NULL); //exit the thread

        // 0~4V  -5000 rpm~5000 rpm
        adc0 = ads.readADC_SingleEnded(0);  // 10ms time consumed
        volts0 = ads.computeVolts(adc0);

        pthread_mutex_lock(&mutex_PT);
        dtheta_w = ((volts0 - 2.0) / 2.0 * 5000.0 * 6); // degrees / s
        pthread_mutex_unlock(&mutex_PT);

    }
}

void* Thread_action(void* arg)
{
    T_start();
    while(1){
        if(file_state == 0)pthread_exit(NULL); //exit the thread
        // 0~4V  -5000 rpm~5000 rpm

        if(abs(theta_b) > 5){
            state1 = argmin(theta_b,RL_theta_b);
            state2 = argmin(dtheta_b,RL_dtheta_b);
            state3 = argmin(dtheta_w,RL_dtheta_w);

            if(state1 != state1_pre && state2 != state2_pre){
                action = getValue(table_data, dims.data(), state1, state2, state3);
                if (action == -2){
                    // cout<< "invalidate state"<<endl;
                    action = 0;
                }
                if (action!=action_pre){
                    if(action == 0 && key_value == 0){
                        pwmWrite(PWM_Pin,240);
                        action_pre = action;
                    }
                    else if(action == 1 && key_value == 0){
                        digitalWrite(DIR_Pin,HIGH);
                        pwmWrite(PWM_Pin,1984); // 90%  pwm*0.8+240  1984   0.07Nm
                        action_pre = action;
                    }
                    else if(action == -1 && key_value == 0){
                        digitalWrite(DIR_Pin,LOW);
                        pwmWrite(PWM_Pin,1984); // 90% pwm*0.8+240  1984    0.07Nm
                        action_pre = action;
                    }
                }     
            }
        }
        else if(abs(theta_b) < 5){
            state1_small_range = argmin(theta_b,RL_theta_b_small_range);
            state2_small_range = argmin(dtheta_b,RL_dtheta_b_small_range);
            state3_small_range = argmin(dtheta_w,RL_dtheta_w_small_range);

            if(state1_small_range != state1_pre_small_range && state2_small_range != state2_pre_small_range){
                action = getValue(table_data_small_range, dims_small_range.data(), state1_small_range, state2_small_range, state3_small_range);
                if (action == -2){
                    // cout<< "invalidate state"<<endl;
                    action = 0;
                }
                if (action!=action_pre){
                    if(action == 0 && key_value == 0){
                        pwmWrite(PWM_Pin,240);
                        action_pre = action;
                    }
                    else if(action == 1 && key_value == 0){
                        digitalWrite(DIR_Pin,HIGH);
                        pwmWrite(PWM_Pin,738); // 26%  pwm*0.8+240  738   0.02Nm
                        action_pre = action;
                    }
                    else if(action == -1 && key_value == 0){
                        digitalWrite(DIR_Pin,LOW);
                        pwmWrite(PWM_Pin,738); // 26% pwm*0.8+240  738   0.02Nm
                        action_pre = action;
                    }
                }     
            }
        }



        printf("\rtheta_b: %.2f,dtheta_b: %.2f,dtheta_w:%.2f,action index:%.2f   ", theta_b, dtheta_b,dtheta_w,action);   
        fflush(stdout); 
        

        pthread_mutex_lock(&mutex_PT);
        time_matrix[matrix_index] = T_end();
        theta_b_matrix[matrix_index] = theta_b;
        dtheta_b_matrix[matrix_index] = dtheta_b;
        dtheta_w_matrix[matrix_index] = dtheta_w;
        action_matrix[matrix_index] = action;
        ++matrix_index;
        if(matrix_index > matrix_number){
            cout << "\nMatrix number error!" << endl;
            return (void *)-1;
        }
        pthread_mutex_unlock(&mutex_PT);

        
        usleep(1000); // 1ms
    }
}
