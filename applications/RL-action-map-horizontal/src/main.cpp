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

//initialize the pin
echo 92 > /sys/class/gpio/export  # Encoder A Phase 
echo in > /sys/class/gpio/gpio92/direction
echo falling > /sys/class/gpio/gpio92/edge
*/

#define ADS1X15_ADDRESS     0x48
#define encoderB_pin        15
#define PWM_Pin             5
#define EN_Pin              7
#define DIR_Pin             8
ADS1115 ads;  /* Use this for the 16-bit version */
pthread_mutex_t mutex_PT;  //init the pthread_mutex_t
struct pollfd fds[1];

float theta_b = 0; 
float dtheta_b = 0; 
float dtheta_w= 0; 
int Maxon_PWM= 0; 

/***** ads variable *******/
uint16_t adc0;
float volts0;
/******************************/

/***** AB encoder variable *****/
float pulse = 0;  //count the pulse of the encoder
char forward_direction = '+';
char back_direction = '-';
char current_direction = 0; 
char direction_state = 0;
float pre_dtheta_b = 0; 
/******************************/

/***** Testing time *****/
double cur_time = 0;
double pre_time = 0;
double TimeUse = 0;
struct timeval StartTime;
struct timeval EndTime;
/******************************/

/***** data matrix *****/
static volatile int file_state = 1;
#define matrix_number 30000
int matrix_index = 0;  // current 
float time_matrix[matrix_number] = {0};
float theta_b_matrix[matrix_number] = {0};
float dtheta_b_matrix[matrix_number] = {0};
float dtheta_w_matrix[matrix_number] = {0};
float action_matrix[matrix_number] = {0};
/******************************/

void* Thread_50ms(void* arg);
void* File_thread(void* arg);
void* EXTI_thread(void* arg);
void encoder_info(double delta_t);
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

std::vector<double> RL_theta_b = create_linspace(-90, 90, 20);
std::vector<double> RL_dtheta_b = create_linspace(RadiansToDegrees(-2), RadiansToDegrees(2), 50);
std::vector<double> RL_dtheta_w = create_linspace(RadiansToDegrees(-30), RadiansToDegrees(30), 100);
const std::string filename = "table/action table";
const std::string datasetname = "working_save";
auto dataAndDims = readData(filename, datasetname);
std::vector<double> table_data = dataAndDims.first;
std::vector<hsize_t> dims = dataAndDims.second;
/******************************/

 
int main()
{
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
    pinMode(encoderB_pin, INPUT);
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
    fprintf(fd, "time,theta_b,dtheta_b,dtheta_w\n");
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

void encoder_info(double delta_t)  //calculate the information of the encoder
{
    dtheta_b = pulse * 351.5625 / delta_t; //   n / ((delta_t / 1000) * 1024) * 360
    // dtheta_b = 0.8 * pre_dtheta_b + 0.2 * dtheta_b;

    if(current_direction == forward_direction){
        theta_b = theta_b + pulse * 360 / 1024; //calculate the positive theta_b 
    }
    else{
        theta_b = theta_b - pulse * 360 / 1024; //calculate the negative theta_b 
        dtheta_b = -dtheta_b;
    }
    pulse = 0;    //set pulse to zero
    // pre_dtheta_b = dtheta_b;
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
                if(digitalRead(encoderB_pin) == HIGH){  //if B pin is in the high place
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
                action_matrix[matrix_index] = action;
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
    while(1){
        if(file_state == 0)pthread_exit(NULL); //exit the thread
        // 0~4V  -5000 rpm~5000 rpm
        adc0 = ads.readADC_SingleEnded(0);
        volts0 = ads.computeVolts(adc0);
        pthread_mutex_lock(&mutex_PT);
        dtheta_w = (volts0 - 2) / 2 * 5000 * 6; // degrees / s 
        pthread_mutex_unlock(&mutex_PT);

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
                if(action==0){
                    Maxon_PWM = 0;
                    action_pre = action;
                }
                else if(action==1){
                    Maxon_PWM = -2180;
                    action_pre = action;
                }
                else if(action==-1){
                    Maxon_PWM = 2180;
                    action_pre = action;
                }
            }     
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
        pwmWrite(PWM_Pin,abs(Maxon_PWM));

        sleep(0.05);
    }
}
