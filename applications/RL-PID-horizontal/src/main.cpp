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
#include "ADS1x15.h"

using namespace std;


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
/******************************/

void* Thread_50ms(void* arg);
void* File_thread(void* arg);
void* EXTI_thread(void* arg);
void encoder_info(double delta_t);
void T_start();
double T_end();

 
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
                fprintf(fp,"%.2f,%.2f,%.2f,%.2f\n",  time_matrix[j - matrix_index],\
                                                theta_b_matrix[j - matrix_index],\
                                                dtheta_b_matrix[j - matrix_index],\
                                                dtheta_w_matrix[j - matrix_index]);                        
            }
            cout << "  Data has been saved Over!!!" << endl;
            fclose(fp); 
            pthread_mutex_unlock(&mutex_PT);
            pthread_exit(NULL);
        }
    }
}

float B_P = 200;
float B_D = 10;
float W_P = 0.1;

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

        Maxon_PWM = B_P * (0-theta_b) + B_D * (0 - dtheta_b) + W_P * dtheta_w;
        if(Maxon_PWM > 0){
            digitalWrite(DIR_Pin,LOW);
            Maxon_PWM = Maxon_PWM * 0.8 + 240;
        }
        else{
            digitalWrite(DIR_Pin,HIGH);
            Maxon_PWM = Maxon_PWM * 0.8 - 240;
        }

        if(abs(Maxon_PWM) > 2160)Maxon_PWM = 2160; //90% PWM
        pwmWrite(PWM_Pin,abs(Maxon_PWM));

        sleep(0.05);
    }
}
