#include <iostream>
#include <fstream>

#define dtype int

const int TILE_DIM = 16;
const int REPEAT = 100;

__global__ void naive_transpose(dtype* matrix, dtype* transposed_matrix, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width)
        transposed_matrix[row * width + col] = matrix[col * width + row];
}

__global__ void transpose_coalesced(dtype* matrix, dtype* transposed_matrix, int width) {
    __shared__ dtype tile[TILE_DIM][TILE_DIM];

    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;

    // load the matrix into shared memory
    if (row < width && col < width) {
        tile[threadIdx.y][threadIdx.x] = matrix[col * width + row];
    }

    __syncthreads();

    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;

    if (row < width && col < width) {
        transposed_matrix[col * width + row] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose_coalesced_no_bank_conflicts(dtype* matrix, dtype* transposed_matrix, int width) {
    __shared__ dtype tile[TILE_DIM][TILE_DIM + 1];

    int row = blockIdx.x * TILE_DIM + threadIdx.x;
    int col = blockIdx.y * TILE_DIM + threadIdx.y;

    // load the matrix into shared memory
    if (row < width && col < width) {
        tile[threadIdx.y][threadIdx.x] = matrix[col * width + row];
    }

    __syncthreads();

    row = blockIdx.y * TILE_DIM + threadIdx.x;
    col = blockIdx.x * TILE_DIM + threadIdx.y;

    if (row < width && col < width) {
        transposed_matrix[col * width + row] = tile[threadIdx.x][threadIdx.y];
    }
}

dtype* create_matrix(int size) {
    int total_size = size * size;
    dtype* matrix = (dtype*)malloc(total_size * sizeof(dtype));
    for (int i = 0; i < total_size; i++) {
        matrix[i] = (dtype)(rand() % 100);
    }
    return matrix;
}

bool check_transpose(const dtype *gt, const dtype *result, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (gt[i * size + j] != result[i * size + j]) {
                return false;
            }
        }
    }
    return true;
}

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

int main(int argc, char const *argv[])
{
    // if the argc is less than 2, then print the usage and exit
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " specify a matrix dimension" << std::endl;
        return 1;
    }
    
    /*
    -------- DEVICE PROPERTIES --------
    */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n");
    printf("Device name: %s\n", prop.name);
    printf("Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %.1f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);    
    
    // Setup problem
    const int input_dimension = atoi(argv[1]);
    if (input_dimension < 1) {
        std::cerr << "\nERROR: Matrix dimension must be greater than 0" << std::endl;
        return 1;
    }
    const int side_length = 1 << input_dimension;
    printf("\nMatrix size: %d X %d ---> ", side_length, side_length);
    const int mem_size = side_length * side_length * sizeof(dtype);
    
    // File to store the results
    std::ofstream file("csv/results.csv", std::ios::app);


    // Define a new type to create an array of pointers to transposition functions
    typedef void (*transpose_func)(dtype*, dtype*, int);
    transpose_func transpose_functions[] = {naive_transpose, transpose_coalesced, transpose_coalesced_no_bank_conflicts};
    const char* function_names[] = {"Naive Transpose", "Coalesced Transpose", "Coalesced Transpose no Bank Conflicts"};
    uint16_t length = sizeof(transpose_functions) / sizeof(transpose_functions[0]);

    // Define the kernel launch parameters
    dim3 block(TILE_DIM, TILE_DIM, 1);
    dim3 grid(
        (side_length + TILE_DIM - 1) / TILE_DIM,
        (side_length + TILE_DIM - 1) / TILE_DIM,
        1
    );

    const int grid_size = (side_length + TILE_DIM - 1) / TILE_DIM;
    printf("%dx%d tiles ", grid_size, grid_size);
    printf("with %dx%d threads per block\n", TILE_DIM, TILE_DIM);

    // Iterate over the functions
    for (uint16_t f = 0; f < length; f++) {

        dtype* matrix = create_matrix(side_length);
        dtype* transposed_matrix = (dtype*)malloc(mem_size);
        dtype *ground_truth = (dtype*) malloc(mem_size);
        
        // calculate the ground truth
        for (int i = 0; i < side_length; i++) {
            for (int j = 0; j < side_length; j++) {
                ground_truth[j * side_length + i] = matrix[i * side_length + j];
            }
        }

        dtype* d_matrix;
        dtype* d_transposed_matrix;


        cudaMalloc(&d_matrix, mem_size);
        cudaMalloc(&d_transposed_matrix, mem_size);

        cudaMemcpy(d_matrix, matrix, mem_size, cudaMemcpyHostToDevice);
        cudaMemset(d_transposed_matrix, 0, mem_size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;

        warm_up_gpu<<<1, 1>>>();

        cudaEventRecord(start, 0);
        for (int i = 0; i < REPEAT; i++) 
            transpose_functions[f]<<<grid, block>>>(d_matrix, d_transposed_matrix, side_length);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        cudaDeviceSynchronize();

        cudaMemcpy(transposed_matrix, d_transposed_matrix, mem_size, cudaMemcpyDeviceToHost);

        if (check_transpose(ground_truth, transposed_matrix, side_length)) {
            printf("\nAlgorithm: %s\n", function_names[f]);
            printf("Bandwidth: %20.2f GB/s\n", 2 * mem_size * 1e-6 * REPEAT / ms );
            printf("Status: correct\n");
            printf("--------------------------------\n");
            file << function_names[f] << "," << side_length << "," << TILE_DIM << "," << 2 * mem_size * 1e-6 * REPEAT / ms << std::endl;
        } else {
            printf("Matrix transposition is incorrect\n");
            printf("--------------------------------\n");
        }


        cudaFree(d_matrix);
        cudaFree(d_transposed_matrix);
        free(matrix);
        free(transposed_matrix);
    }
    
    return 0;
}