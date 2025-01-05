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
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>

#include <sched.h>
#include <pthread.h>
#include <wiringPi.h>
#include <linux/spi/spidev.h>
#include "ADS1220.h"
#include "../imu/MotionSensor.h"

using namespace std;


#define SPI_DEV_PATH    "/dev/spidev4.1"
#define DRDY_PIN 13
#define CS_PIN 15

#define PWM_Pin             5
#define EN_Pin              7
#define DIR_Pin             8
pthread_mutex_t mutex_PT;  //init the pthread_mutex_t

float theta_limit = 40.0;
float theta_b = 0; 
float pre_theta_b = 0; 
float dtheta_b = 0; 
float dtheta_w= 0; 

/********** PID ***************/
float theta_b_KP = 400;
float theta_b_KD = 20;
float theta_w_KD = 0.1;
float PID_output = 0;
char PID_flag = 0;

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


ADS1220_Handler_t Handler = {};
ADS1220_Parameters_t Parameters = {};

int main()
{
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
    Parameters.DataRate = _600_SPS_;
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
    return 0;
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
               fprintf(fp,"%.2f,%.2f,%.2f,%.2f\n",  time_matrix[j - matrix_index],\
                                                    theta_b_matrix[j - matrix_index],\
                                                    dtheta_b_matrix[j - matrix_index],\
                                                    dtheta_w_matrix[j - matrix_index]);                    
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

        // 0~4V  -6710 rpm~6710 rpm
        // Default conversion mode is Single-shot
        ADS1220_StartSync(&Handler);

        // // GPIO_DRDY_GROUP and GPIO_DRDY_PIN depend on your schematic.
        while(digitalRead(DRDY_PIN) == 1){}
        
        int32_t ADC_Data;
        ADS1220_ReadData(&Handler, &ADC_Data);

        float voltage;
        voltage = ADCValueToVoltage(ADC_Data, 5.185, 1);
        
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
    T_start();
    while(1){
        if(file_state == 0)pthread_exit(NULL); //exit the thread

        if(abs(theta_b) > theta_limit){
            PID_flag = 0;
        }

        if(abs(theta_b) < theta_limit && key_value == 0){
            PID_flag = 1;
            PID_output = theta_b_KP * theta_b + theta_b_KD * dtheta_b + theta_w_KD * dtheta_w;
            if(PID_output > 0){
                PID_output = (int)(PID_output * 0.8 + 240);
                digitalWrite(DIR_Pin,HIGH);
            }
            else{
                PID_output = (int)(PID_output * 0.8 - 240);
                digitalWrite(DIR_Pin,LOW);
            }
            PID_output = abs(PID_output);
            if(PID_output >= 1920)PID_output = 1920;
            pwmWrite(PWM_Pin,PID_output);
        }
        if(PID_flag == 0) pwmWrite(PWM_Pin,240);


        printf("\rtheta_b: %.2f | dtheta_b: %.2f | dtheta_w:%.2f   ", theta_b, dtheta_b,dtheta_w);   
        fflush(stdout);         

        pthread_mutex_lock(&mutex_PT);
        time_matrix[matrix_index] = T_end();
        theta_b_matrix[matrix_index] = theta_b;
        dtheta_b_matrix[matrix_index] = dtheta_b;
        dtheta_w_matrix[matrix_index] = dtheta_w;
        ++matrix_index;
        if(matrix_index > matrix_number){
            cout << "\nMatrix number error!" << endl;
            return (void *)-1;
        }
        pthread_mutex_unlock(&mutex_PT);

        usleep(1000); // 1ms
    }
}


int transfer(int fd_spi, uint8_t const *tx, uint8_t const *rx, uint32_t len)
{
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
    // GPIO_CS_GROUP and GPIO_CS_PIN depend on your schematic.
    // HAL_GPIO_WritePin(GPIO_CS_GROUP, GPIO_CS_PIN, GPIO_PIN_SET);
    digitalWrite(CS_PIN, HIGH);
}
void CS_DOWN(void)
{
    // GPIO_CS_GROUP and GPIO_CS_PIN depend on your schematic.
    // HAL_GPIO_WritePin(GPIO_CS_GROUP, GPIO_CS_PIN, GPIO_PIN_RESET);
    digitalWrite(CS_PIN, LOW);
}
void TRANSMIT(uint8_t data)
{
    // SPI_DRIVER depends on your configuration.
    // HAL_SPI_Transmit (SPI_DRIVER, &data, sizeof(uint8_t), HAL_MAX_DELAY);
    transfer(fd_spi, &data, rx_buffer, sizeof(data));
}
uint8_t RECEIVE(void)
{
    uint8_t dataR = 0;
    // SPI_DRIVER depends on your configuration.
    // HAL_SPI_Receive (SPI_DRIVER, &dataR, sizeof(uint8_t), HAL_MAX_DELAY);
    transfer(fd_spi, tx_buffer, &dataR, sizeof(dataR));
    return dataR;
}
void DELAY(uint32_t us)
{
    // DELAY_US depends on your code.
    usleep(us);
}
