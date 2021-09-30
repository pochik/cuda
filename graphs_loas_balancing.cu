#include <cfloat>
#include <chrono>
#include <cuda_profiler_api.h>
#include <iostream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////////

#define SAFE_CALL( CallInstruction ) { \
    cudaError_t cuerr = CallInstruction; \
    if(cuerr != cudaSuccess) { \
         printf("CUDA error: %s at call \"" #CallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		 throw "error in CUDA API function, aborting..."; \
    } \
}

#define SAFE_KERNEL_CALL( KernelCallInstruction ){ \
    KernelCallInstruction; \
    cudaError_t cuerr = cudaGetLastError(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel launch: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel launch, aborting..."; \
    } \
    cuerr = cudaDeviceSynchronize(); \
    if(cuerr != cudaSuccess) { \
        printf("CUDA error in kernel execution: %s at kernel \"" #KernelCallInstruction "\"\n", cudaGetErrorString(cuerr)); \
		throw "error in CUDA kernel execution, aborting..."; \
    } \
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////

void __global__ gather(int *ptrs, int *connections, int *outgoing_ids, int vertices_count, int *data, int *result)
{
    const long long src_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (src_id < vertices_count)
    {
        const int first_edge_ptr = ptrs[src_id];
        const int connections_count = connections[src_id];
        //connections_count = ptrs[src_id + 1] - ptrs[src_id];

        for(register int cur_edge = 0; cur_edge < connections_count; cur_edge++)
        {
            // first_edge_ptr + cur_edge - индекс текущего ребра в массивах
            int dst_id = outgoing_ids[first_edge_ptr + cur_edge];
            int val = data[dst_id];
            result[first_edge_ptr + cur_edge] = val;

            // данную программу можно легко переделать во многие графовые алгоритмы, например:
            /* BFS
            int src_level = data[src_id];
            int dst_level = data[dst_id];
            if((src_level == current_level) && (dst_level == UNVISITED_VERTEX))
            {
                data[dst_id] = current_level + 1;
            }
            */

            /* SSSP
            float weight = outgoing_weights[first_edge_ptr + cur_edge];
            float src_weight = data[src_id];
            float dst_weight = data[dst_id];

            if(dst_weight > src_weight + weight)
            {
                data[dst_id] = src_weight + weight;
            }
            */
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
    int vertices_count = 1024*1024;

    int *ptrs = new int[vertices_count];
    int *data = new int[vertices_count];
    int *connections = new int[vertices_count];

    int pos = 0;
    for(int i = 0; i < vertices_count; i++) // TODO (bonus) граф с несколькими "большими" вершинами
    {
        ptrs[i] = pos;
        connections[i] = 16 + rand()%32;
        pos += connections[i];

        data[i] = rand();
    }

    int edges_count = pos;
    int *outgoing_ids = new int[edges_count];
    int *result = new int[edges_count];
    for(int i = 0; i < edges_count; i++)
    {
        outgoing_ids[i] = rand()%vertices_count;
    }

    int *dev_ptrs; int *dev_connections; int *dev_outgoing_ids; int *dev_data; int *dev_result;
    cudaMalloc((void**)&dev_ptrs, vertices_count*sizeof(int));
    cudaMalloc((void**)&dev_connections, vertices_count*sizeof(int));
    cudaMalloc((void**)&dev_data, vertices_count*sizeof(int));

    cudaMalloc((void**)&dev_outgoing_ids, edges_count*sizeof(int));
    cudaMalloc((void**)&dev_result, edges_count*sizeof(int));

    SAFE_CALL(cudaMemcpy(dev_ptrs, ptrs, vertices_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_connections, connections, vertices_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_data, data, vertices_count * sizeof(int), cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(dev_outgoing_ids, outgoing_ids, edges_count * sizeof(int), cudaMemcpyHostToDevice));

    dim3 compute_threads(1024);
    dim3 compute_blocks((vertices_count - 1) / compute_threads.x + 1);

    for(int i = 0; i < 5; i++)
    {
        auto start = std::chrono::steady_clock::now();
        SAFE_KERNEL_CALL((gather <<< compute_blocks, compute_threads >>> (dev_ptrs, dev_connections, dev_outgoing_ids, vertices_count, dev_data, dev_result)));
        auto end = std::chrono::steady_clock::now(); // TODO почему работает данный замер веремени?
        std::chrono::duration<double> elapsed_seconds = end-start;
        cout << "time: " << (elapsed_seconds.count())*1000.0 << " ms" << endl;
        cout << "bandwidth: " << 3.0*sizeof(int)*edges_count/((elapsed_seconds.count())*1e9) << " GB/s" << endl << endl;
    }

    int *copy_device_result = new int[edges_count];

    // TODO copy

    for(int src_id = 0; src_id < vertices_count; src_id++)
    {
        const int first_edge_ptr = ptrs[src_id];
        const int connections_count = connections[src_id];
        for(register int cur_edge = 0; cur_edge < connections_count; cur_edge++)
        {
            int dst_id = outgoing_ids[first_edge_ptr + cur_edge];
            int val = data[dst_id];
            result[first_edge_ptr + cur_edge] = val;
        }
    }


    // TODO check
    int errors_count = 0;
    for(int i = 0; i < edges_count; i++)
    {
        if(result[i] != copy_device_result[i])
        {
            errors_count++;
        }
    }
    cout << errors_count << endl;


    // TODO какие 3 недостатка у текущей версии ядра?

    // TODO отпрофилировать текущую версию, сделать выводы о её производитлеьности

    // TODO сделать оптимизированную версию ядра

    // TODO (bonus) реализовать базовую версию BFS алгоритма (выделить структуры данных и реализовать сам алгоритм)

    cudaFree(dev_data);
    cudaFree(dev_ptrs);
    cudaFree(dev_connections);
    cudaFree(dev_result);
    cudaFree(dev_outgoing_ids);

    delete[]result;
    delete[]data;
    delete[]ptrs;
    delete[]outgoing_ids;
    delete[]connections;
    
    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////
