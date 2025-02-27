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

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cstring>
#include <array>
#include <vector>

#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <sys/time.h>

#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sched.h>
#include <sys/types.h>
#include <pthread.h>
#include <wiringPi.h>

#include <boost/asio.hpp>
#include <boost/system/error_code.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <linux/spi/spidev.h>
#include "H5Cpp.h"
#include "ADS1220.h"
#include "../imu/MotionSensor.h"

using namespace std;
using namespace H5;

// #define CPU_BIND

#define SPI_DEV_PATH    "/dev/spidev4.1"
#define DRDY_PIN 13
#define CS_PIN 15

#define PWM_Pin             5
#define EN_Pin              7
#define DIR_Pin             8
pthread_mutex_t mutex_PT;  //init the pthread_mutex_t


float theta_b = 0; 
float pre_theta_b = 0; 
float dtheta_b = 0; 
float dtheta_w= 0; 

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

/*********** SPI  ***********/
unsigned char tx_buffer[10000];
unsigned char rx_buffer[10000];
int fd_spi;                          // SPI 控制引脚的设备文件描述符
uint32_t mode_spi = SPI_MODE_1;      //用于保存 SPI 工作模式
uint8_t bits_spi = 8;                // 接收、发送数据位数
uint32_t speed_spi = 500000;         // 发送速度
uint16_t delay_spi;              //保存延时时间
/******************************/

int spi_init(void);
int transfer_SPI(int fd_spi, uint8_t const *tx, uint8_t const *rx, size_t len);
void CS_UP(void);
void CS_DOWN(void);
void TRANSMIT(uint8_t data);
uint8_t RECEIVE(void);
void DELAY(uint32_t us);
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

int state1_0_1nm = 0;
int state2_0_1nm = 0;
int state3_0_1nm = 0;
int state1_pre_0_1nm = 0;
int state2_pre_0_1nm = 0;
int state3_pre_0_1nm = 0;
std::vector<double> RL_theta_b_0_1nm = create_linspace(-3, 3, 40); 
std::vector<double> RL_dtheta_b_0_1nm = create_linspace(RadiansToDegrees(-0.5), RadiansToDegrees(0.5), 50);
std::vector<double> RL_dtheta_w_0_1nm = create_linspace(RadiansToDegrees(-2), RadiansToDegrees(2), 200);
const std::string filename_0_1nm = "table/0.1nm";
const std::string datasetname_0_1nm = "working_save";
auto dataAndDims_0_1nm = readData(filename_0_1nm, datasetname_0_1nm);
std::vector<double> table_data_0_1nm = dataAndDims_0_1nm.first;
std::vector<hsize_t> dims_0_1nm = dataAndDims_0_1nm.second;

/******************************/

ADS1220_Handler_t Handler = {};
ADS1220_Parameters_t Parameters = {};

int main()
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    ms_open();
    cout << "IMU init done!" << endl;

    int ret;
    ret = spi_init();
    if( -1 == ret  ){
        printf("spi_init error\n");
        exit(-1);
    }

    wiringPiSetup();
    pinMode (CS_PIN, OUTPUT);   
	pinMode (DRDY_PIN, INPUT);
    pinMode(DIR_Pin, OUTPUT);
    pinMode(EN_Pin, OUTPUT);
    pinMode(PWM_Pin , PWM_OUTPUT);
    pwmSetMode(PWM_Pin,PWM_MODE_MS);    
    pwmSetClock(PWM_Pin,2);//我们把pwm分为2400分，要5000hz的pwm频率，那时钟频率=24000000 Hz / (5000 * 2400) = 2
    pwmSetRange(PWM_Pin,2400);
    pwmWrite(PWM_Pin,240);
    digitalWrite(EN_Pin,HIGH);
    digitalWrite(DIR_Pin,HIGH);

    Handler.ADC_CS_HIGH = CS_UP;
    Handler.ADC_CS_LOW = CS_DOWN;
    Handler.ADC_Transmit = TRANSMIT;
    Handler.ADC_Receive = RECEIVE;
    Handler.ADC_Delay_US = DELAY;
    
    Parameters.PGAdisable = 1;
    Parameters.InputMuxConfig = P0NAVSS;
    Parameters.DataRate = _1000_SPS_;
    // Parameters.FIRFilter = S50or60Hz;
    Parameters.VoltageRef = AnalogSupply;
    // Passing Parameters as NULL to use default configurations.
    ADS1220_Init(&Handler, &Parameters);

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

    // pthread_attr_destroy(&attr);

    return 0;
}



//创建一个等差数组
std::vector<double> create_linspace(double start, double end, int N) 
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

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
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    auto result = std::min_element(x.begin(), x.end(), [&](double a, double b) {
        return std::abs(a - y0) < std::abs(b - y0);
    });
    return std::distance(x.begin(), result);     // Index
}


// 读取数据
std::pair<std::vector<double>, std::vector<hsize_t>> readData(const std::string& filename, const std::string& datasetname)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

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
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    int index = k + dims[2] * (j + dims[1] * i);
    return data[index];
}

double DegreesToRadians(double degrees) 
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    return degrees * M_PI / 180.0;
}

double RadiansToDegrees(double radians) 
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    return radians * 180.0 / M_PI;
}


double readAndParseData(boost::asio::serial_port& serial) 
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

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
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    gettimeofday(&StartTime, NULL);  //measure the time
}

double T_end()
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    gettimeofday(&EndTime, NULL);   //measurement ends
    TimeUse = 1000000*(EndTime.tv_sec-StartTime.tv_sec)+EndTime.tv_usec-StartTime.tv_usec;
    TimeUse /= 1000;  //the result is in the ms dimension
    return TimeUse;
}
    

void* Thread_file(void* arg)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    FILE *fp = (FILE*)arg;
    char i = 0;
    while(1){
        i = getchar();
        if(i == 'r'){
            key_value = 0;
            digitalWrite(EN_Pin,HIGH);
        }
        else if(i == 's'){
            key_value = 1;
            digitalWrite(EN_Pin,LOW);
        }
        else if(i == 'q'){
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
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

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
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    while(1){
        if(file_state == 0)pthread_exit(NULL); //exit the thread

        // 0~4V  -6710 rpm~6710 rpm
        // Default conversion mode is Single-shot
        ADS1220_StartSync(&Handler);

        // // GPIO_DRDY_GROUP and GPIO_DRDY_PIN depend on your schematic.
        while(digitalRead(DRDY_PIN) == 1){}
        
        int32_t ADC_Data;
        ADS1220_ReadData(&Handler, &ADC_Data);

        float voltage;
        voltage = ADCValueToVoltage(ADC_Data, 5.170, 1);
        
        // 2.048 is internal voltage reference and is used as default config.
        // 1 is default adc gain value and it must be equivalent to the gain config in ADS1220_Parameters_t.
        // printf("voltage : %f | velocity : %f\r\n", voltage, (voltage - 2) * 4000 / 2);
        
        pthread_mutex_lock(&mutex_PT);
        dtheta_w = ((voltage - 2.0) / 2.0 * 6710.0 * 6.0); // degrees / s
        pthread_mutex_unlock(&mutex_PT);

    }
}

void* Thread_action(void* arg)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

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
        else if(abs(theta_b) < 5 && abs(theta_b) > 0){
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
    
        // else if(abs(theta_b) < 3){
        //     state1_0_1nm = argmin(theta_b,RL_theta_b_0_1nm);
        //     state2_0_1nm = argmin(dtheta_b,RL_dtheta_b_0_1nm);
        //     state3_0_1nm = argmin(dtheta_w,RL_dtheta_w_0_1nm);

        //     if(state1_0_1nm != state1_pre_0_1nm && state2_0_1nm != state2_pre_0_1nm){
        //         action = getValue(table_data_0_1nm, dims_0_1nm.data(), state1_0_1nm, state2_0_1nm, state3_0_1nm);
        //         if (action == -2){
        //             // cout<< "invalidate state"<<endl;
        //             action = 0;
        //         }
        //         if (action!=action_pre){
        //             if(action == 0 && key_value == 0){
        //                 pwmWrite(PWM_Pin,240);
        //                 action_pre = action;
        //             }
        //             else if(action == 1 && key_value == 0){
        //                 digitalWrite(DIR_Pin,HIGH);
        //                 pwmWrite(PWM_Pin,489); // 12.97%  pwm*0.8+240  489   0.01Nm
        //                 action_pre = action;
        //             }
        //             else if(action == -1 && key_value == 0){
        //                 digitalWrite(DIR_Pin,LOW);
        //                 pwmWrite(PWM_Pin,489); // 12.97% pwm*0.8+240  489   0.01Nm
        //                 action_pre = action;
        //             }
        //         }     
        //     }
        // }

        printf("\rtheta_b: %.2f | dtheta_b: %.2f | dtheta_w:%.2f | action index:%.2f   ", theta_b, dtheta_b,dtheta_w,action);   
        fflush(stdout);         

        // pthread_mutex_lock(&mutex_PT);
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
        // pthread_mutex_unlock(&mutex_PT);

        usleep(5000); // 5ms
    }
}


int transfer(int fd_spi, uint8_t const *tx, uint8_t const *rx, uint32_t len)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif

    int ret;
    struct spi_ioc_transfer tr = {
        .tx_buf = (unsigned long )tx,
        .rx_buf = (unsigned long )rx,
        .len = len,
        .speed_hz = speed_spi,
        .delay_usecs = delay_spi,
        .bits_per_word =bits_spi,
    };
    ret = ioctl(fd_spi,SPI_IOC_MESSAGE(1),&tr);
    if( ret == -1 ){
        return -1;
    }

    return 0;
}

int spi_init(void)
{
    int ret;
    fd_spi = open(SPI_DEV_PATH,O_RDWR);
    if(fd_spi < 0){
        perror("/dev/.1");
        return -1;
    }

    //set the mode of SPI
    ret = ioctl(fd_spi,SPI_IOC_WR_MODE,&mode_spi);
    if( ret == -1){
        printf("SPI_IOC_WR_MODE error......\n ");
        goto fd_close;
    }

    ret = ioctl(fd_spi,SPI_IOC_RD_MODE,&mode_spi);
    if( ret == -1){
        printf("SPI_IOC_RD_MODE error......\n ");
        goto fd_close;
    }

    //set the communication length of SPI
    ret = ioctl(fd_spi,SPI_IOC_RD_BITS_PER_WORD,&bits_spi);
    if( ret == -1){
        printf("SPI_IOC_RD_BITS_PER_WORD error......\n ");
        goto fd_close;
    }
    ret = ioctl(fd_spi,SPI_IOC_WR_BITS_PER_WORD,&bits_spi);
    if( ret == -1){
        printf("SPI_IOC_WR_BITS_PER_WORD error......\n ");
        goto fd_close;
    }

    //set the frequenct of SPI
    ret = ioctl(fd_spi,SPI_IOC_WR_MAX_SPEED_HZ,&speed_spi);
    if( ret == -1){
        printf("SPI_IOC_WR_MAX_SPEED_HZ error......\n ");
        goto fd_close;
    }
    ret = ioctl(fd_spi,SPI_IOC_RD_MAX_SPEED_HZ,&speed_spi);
    if( ret == -1){
        printf("SPI_IOC_RD_MAX_SPEED_HZ error......\n ");
        goto fd_close;
    }

    printf("spi mode: 0x%x\n", mode_spi);
    printf("bits per word: %d\n", bits_spi);
    printf("max speed: %d Hz (%d KHz)\n", speed_spi, speed_spi / 1000);
    return 0;

    fd_close:
        close(fd_spi);
        return -1;
}

void CS_UP(void)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif
    // GPIO_CS_GROUP and GPIO_CS_PIN depend on your schematic.
    // HAL_GPIO_WritePin(GPIO_CS_GROUP, GPIO_CS_PIN, GPIO_PIN_SET);
    digitalWrite(CS_PIN, HIGH);
}
void CS_DOWN(void)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif
    // GPIO_CS_GROUP and GPIO_CS_PIN depend on your schematic.
    // HAL_GPIO_WritePin(GPIO_CS_GROUP, GPIO_CS_PIN, GPIO_PIN_RESET);
    digitalWrite(CS_PIN, LOW);
}
void TRANSMIT(uint8_t data)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif
    // SPI_DRIVER depends on your configuration.
    // HAL_SPI_Transmit (SPI_DRIVER, &data, sizeof(uint8_t), HAL_MAX_DELAY);
    transfer(fd_spi, &data, rx_buffer, sizeof(data));
}
uint8_t RECEIVE(void)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif
    uint8_t dataR = 0;
    // SPI_DRIVER depends on your configuration.
    // HAL_SPI_Receive (SPI_DRIVER, &dataR, sizeof(uint8_t), HAL_MAX_DELAY);
    transfer(fd_spi, tx_buffer, &dataR, sizeof(dataR));
    return dataR;
}
void DELAY(uint32_t us)
{
    #ifndef CPU_BIND
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7,&mask);
    if (sched_setaffinity(0,sizeof(mask),&mask)<0){
        printf("affinity set fail!");
    }
    #endif
    // DELAY_US depends on your code.
    usleep(us);
}
