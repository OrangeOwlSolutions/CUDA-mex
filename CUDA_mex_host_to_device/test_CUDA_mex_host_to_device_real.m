clear all
close all
clc

a = 1 : 10;

b = CUDA_mex_host_to_device_real(a);

clear mex
