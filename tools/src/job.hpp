#pragma once

void cpu_read_cpumem();
void cpu_write_cpumem();
void gpu_read_cpumem_kernel();
void gpu_write_cpumem_kernel();
void gpu_read_gpumem_kernel();
void gpu_write_gpumem_kernel();
void gpu_read_cpumem_memcpy();
void gpu_write_cpumem_memcpy();
void gpu_read_gpumem_memcpy();
void gpu_write_gpumem_memcpy();
void nic_read_cpumem();
void nic_write_cpumem();
void nic_read_gpumem();
void nic_write_gpumem();
void nic_sharp_allreduce();

void dummy();