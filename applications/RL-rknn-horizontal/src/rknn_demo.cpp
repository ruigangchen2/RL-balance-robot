/*-------------------------------------------
                  Includes
-------------------------------------------*/
#include "rknn_api.h"
#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;


/*-------------------------------------------
                  RKNN
-------------------------------------------*/
const int state_row = 1;  // input dimension
const int state_col = 3;   // input dimension
const int act_dim = 3;     // input dimension
float input_data[state_row][state_col] = {0.0 ,0.0 ,0.0}; // input matrix
int output_index = 0;    //index of the max value   

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
      return 1;
   }
   ret = rknn_set_io_mem(ctx, output_mems, &output_attrs);
   if (ret < 0){
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return 1;
   }

 
   input_data[0][0] = 80;
   input_data[0][1] = 0;
   input_data[0][2] = -3000;

   // copy input data to input tensor memory
   memcpy(input_mems->virt_addr, input_data, sizeof(input_data));

   // Run
   int64_t start_us = getCurrentTimeUs();
   ret = rknn_run(ctx, NULL);
   int64_t elapse_us = getCurrentTimeUs() - start_us;
   if (ret < 0){
      printf("rknn run error %d\n", ret);
      return 1;
   }
   printf("Elapse Time = %.2fms\n", elapse_us / 1000.f);
   
   // print the result
   float *buffer = (float *)(output_mems->virt_addr);
   output_index = max_element(buffer,buffer+3) - buffer;

   for (int i = 0; i < state_col; i++){
      cout << input_data[0][i] << endl;
   }
   for (int i = 0; i < act_dim; i++){
      cout << buffer[i] << endl;
   }
   cout << output_index << endl;

   // Destroy rknn memory
   rknn_destroy_mem(ctx, input_mems);
   rknn_destroy_mem(ctx, output_mems);
   // Destroy
   rknn_destroy(ctx);

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

