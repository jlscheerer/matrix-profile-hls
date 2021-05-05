#include "MatrixProfile.hpp"
#include "kernel/MatrixProfileKernel.hpp"

#include "hls_stream.h"

using hls::stream;

void MemoryToStream(const unsigned int *memory, stream<unsigned int> &stream, int size){
    MemoryToStream:
    for(int i = 0; i < size; ++i){
        stream << memory[i];
    }
}

void AddStage(stream<unsigned int> &streamIn, stream<unsigned int> &streamOut, int size){
    AddStage:
    for(int i = 0; i < size; ++i){
        const unsigned int element = streamIn.read();
        const unsigned int result = element + 1;
        streamOut << result;
    }
}

void StreamToMemory(stream<unsigned int> &stream, unsigned int *memory, int size){
    StreamToMemory:
    for(int i = 0; i < size; ++i){
        memory[i] = stream.read();
    }
}

extern "C"{
void MatrixProfileKernelTLF(const unsigned int *memoryIn, unsigned int *memoryOut, int size) {
    #pragma HLS DATAFLOW
    stream<unsigned int> streams[NumberOfStages + 1];
    
    // read memory into initial stream
    MemoryToStream(memoryIn, streams[0], size);

    // connect successive compute units
    for(int k = 0; k < NumberOfStages; ++k){
        #pragma HLS UNROLL
        AddStage(streams[k], streams[k+1], size);
    }

    // write final stream back to memory
    StreamToMemory(streams[NumberOfStages], memoryOut, size);
}
}
