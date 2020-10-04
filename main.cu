#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

// -----------
#include <climits>         // for MAX_INT
#include <bits/stdc++.h>   // for sorting
// -----------

#define THREADS_DIM 16


using namespace std;

void KNN(ArffData* dataset, int k, int* predictions)
{
    // predictions is the array where you have to return the class predicted (integer) for the dataset instances
    // int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    
    // The following two lines show the syntax to retrieve the attribute values and the class value for a given instance in the dataset
    // float attributeValue = dataset->get_instance(instanceIndex)->get(attributeIndex)->operator float();
    // int classValue =  dataset->get_instance(instanceIndex)->get(dataset->num_attributes() - 1)->operator int32();
    
    // Implement the KNN here, fill the predictions array

    for(int i = 0; i < dataset->num_instances(); i++)
    {

        // getNeighbors()
        int neighbors[5];
        tuple<int, double>* distances = (tuple<int, double>*)malloc(dataset->num_instances() * sizeof(tuple<int, double>));
        for(int j = 0; j < dataset->num_instances(); j++)
        {

            // map(dataset, (train) => (train, distance(train)))
            if(j == i)
            {
                distances[j] = tuple<int, double>(j, INT_MAX);
                continue;
            }

            long squaredSum = 0;
            for(int y = 0; y < dataset->num_attributes() - 1; y++)
            {
                squaredSum += pow(dataset->get_instance(i)->get(y)->operator float() - dataset->get_instance(j)->get(y)->operator float(),  2);
            }

            distances[j] = tuple<int, double>(j, sqrt(squaredSum));
        }

        // distances.sort()
        sort(distances, distances + dataset->num_instances(), [](tuple<int, double> a, tuple<int, double> b) {
            return get<1>(a) < get<1>(b);
        });

        // distances.take(5)
        for(int x = 0; x < k; x++)
        {
            neighbors[x] = get<0>(distances[x]);
        }

        // map(neighbors, (x) => neighbors.class)
        int outputValues[k];
        for(int j = 0; j < k; j++)
        {
            outputValues[j] = dataset->get_instance(neighbors[j])->get(dataset->num_attributes() - 1)->operator int32();
        }

        // mode()
        map<int, int> histogram;

        int mode_count = 0;
        int mode = -1;
        for(int a = 0; a < k; a++) 
        {
            int element = outputValues[a];
            histogram[element]++;
            if(histogram[element] > mode_count)
            {
                mode_count = histogram[element];
                mode = element;
            }
        }

        predictions[i] = mode;
        free(distances);
    }
}

__global__ void KNN_GPU(
    float* dataset, 
    int rows, 
    int columns, 
    int maximumClass, 
    int k, 
    int* predictions,

    int* outputValues,
    int* outputValueMapping,
    int* neighbors,
    double* neighborDistances
)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x; // Some combination of threadId and blockId

    if(row < rows)
    {

        // getNeighbors()
        for(int j = 0; j < rows; j++)
        {

            // map(dataset, (train) => (train, distance(train)))
            if(j == row)
            {
                continue;
            }

            double squaredSum = 0;
            for(int y = 0; y < columns - 2; y++)
            {
                squaredSum += (
                    (dataset[row * columns + y] - dataset[j * columns + y]) *
                    (dataset[row * columns + y] - dataset[j * columns + y])
                );
            }

            double sqrtOfSquaredSum = sqrt(squaredSum);

            // modified insertion sort
            int lastLargerIndex = -1;
            for(int idx = k - 1; idx >= 0; idx--)
            {

                // printf("%d : %f > %f = %d\n", idx, neighborDistances[idx], sqrtOfSquaredSum, neighborDistances[idx] > sqrtOfSquaredSum);
                if(neighborDistances[idx + (row * k)] > sqrtOfSquaredSum && idx != 0) 
                {
                    lastLargerIndex = idx;
                }
                else if(neighborDistances[idx + (row * k)] > sqrtOfSquaredSum && idx == 0)
                {
                    neighbors[idx + (row * k)] = j;
                    neighborDistances[idx + (row * k)] = sqrtOfSquaredSum;
                    lastLargerIndex = -1;
                }
                else if(neighborDistances[idx + (row * k)] < sqrtOfSquaredSum && lastLargerIndex != -1)
                {
                    neighbors[lastLargerIndex + (row * k)] = j;
                    neighborDistances[lastLargerIndex + (row * k)] = sqrtOfSquaredSum;
                    lastLargerIndex = -1;
                    break;
                }
            }
        }

        // map(neighbors, (x) => neighbors.class)
        memset(outputValues + (row * k), 0, k * sizeof(int));

        for(int j = 0; j < k; j++)
        {
            outputValues[j + (row * k)] = (int)dataset[(neighbors[j + (row * k)] * columns) + columns - 1];
        }

        printf(
            "%d: %d %d %d %d %d\n", 
            row, 
            outputValues[(row * k) + 0], 
            outputValues[(row * k) + 1],
            outputValues[(row * k) + 2],
            outputValues[(row * k) + 3],
            outputValues[(row * k) + 4]
        );


        // mode()
        memset(outputValueMapping + (row * (maximumClass+1)), 0, k * sizeof(int));

        int mode = 0;
        int modeCount = -1;
        for(int blah = 0; blah < k; blah++)
        {
            int outputValue = outputValues[blah + (row * k)];
            
            outputValueMapping[outputValue + (row * (maximumClass+1))] += 1;

            if(outputValueMapping[outputValue + (row * (maximumClass+1))] > modeCount)
            {
                modeCount = outputValueMapping[outputValue + (row * (maximumClass+1))];
                mode = outputValue; 
            }
        }

        predictions[row] = mode;
        // printf("%d %d\n", row, mode);
    }
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
    
    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    
    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    
    return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Usage: ./main datasets/datasetFile.arff k" << endl;
        exit(0);
    }

    // Get k
    int k = atoi(argv[2]);

    // Open the dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();

    // Allocate Memory
    int* predictionsHostCPU;
    int* predictionsHost;
    float* datasetArrayHost;
    float* datasetArrayDevice;
    int* predictionsDevice;

    cudaMalloc(&predictionsDevice, dataset->num_instances() * sizeof(int));
    cudaMalloc(&datasetArrayDevice, dataset->num_instances() * dataset->num_attributes() * sizeof(float));
    cudaMallocHost(&datasetArrayHost, dataset->num_instances() * dataset->num_attributes() * sizeof(float));
    cudaMallocHost(&predictionsHostCPU, dataset->num_instances() * sizeof(int));
    cudaMallocHost(&predictionsHost, dataset->num_instances() * sizeof(int));

    int gridDim = (dataset->num_instances() + THREADS_DIM - 1) / THREADS_DIM;

    dim3 blockSize (THREADS_DIM, 1);
    dim3 gridSize (gridDim, 1);

    for(int i = 0; i < dataset->num_instances(); i++)
    {
        for(int j = 0; j < dataset->num_attributes(); j++)
        {
            datasetArrayHost[i * dataset->num_attributes() + j] = dataset->get_instance(i)->get(j)->operator float();
        }
    }

    cudaMemcpy(datasetArrayDevice, datasetArrayHost, dataset->num_instances() * dataset->num_attributes() * sizeof(float), cudaMemcpyHostToDevice);

    // --------------------------- CPU ---------------
   
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    // Get the class predictions
    KNN(dataset, k, predictionsHostCPU);
    // Compute the confusion matrix
    int* confusionMatrix = computeConfusionMatrix(predictionsHostCPU, dataset);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, dataset);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

   printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);

    // ----------------------------- GPU -------------------------
    int maximum = 0;
    for(int blah = 0; blah < dataset->num_instances(); blah++)
    {
        int val = dataset->get_instance(blah)->get(dataset->num_attributes() - 1)->operator int32(); 
        if(val > maximum) { maximum = val; }
    }

    /***********************************************************/

    int* outputValues;
    int* outputValueMapping;
    int* neighbors;
    double* neighborDistances;

    int* d_outputValues;
    int* d_outputValueMapping;
    int* d_neighbors;
    double* d_neighborDistances;

    cudaMallocHost(&outputValues, k * dataset->num_instances() * sizeof(int));
    cudaMallocHost(&outputValueMapping, (maximum+1) * dataset->num_instances() * sizeof(int));
    cudaMallocHost(&neighbors, k * dataset->num_instances() * sizeof(int));
    cudaMallocHost(&neighborDistances, k * dataset->num_instances() * sizeof(double));

    cudaMalloc(&d_outputValues, k * dataset->num_instances() * sizeof(int));
    cudaMalloc(&d_outputValueMapping, (maximum+1) * dataset->num_instances() * sizeof(int));
    cudaMalloc(&d_neighbors, k * dataset->num_instances() * sizeof(int));
    cudaMalloc(&d_neighborDistances, k * dataset->num_instances() * sizeof(double));

    memset(outputValues, 0, k * dataset->num_instances() * sizeof(int));
    memset(outputValueMapping, 0, k * dataset->num_instances() * sizeof(int));
    memset(neighbors, 0, k * dataset->num_instances() * sizeof(int));
    uninitialized_fill(neighborDistances, neighborDistances + dataset->num_instances() * k, FLT_MAX);

    cudaMemcpy(
        d_neighbors, 
        neighbors, 
        k * dataset->num_instances() * sizeof(int), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_neighborDistances, 
        neighborDistances, 
        k * dataset->num_instances() * sizeof(double), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_outputValues, 
        outputValues, 
        k * dataset->num_instances() * sizeof(int), 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        d_outputValueMapping, 
        outputValueMapping, 
        (maximum+1) * dataset->num_instances() * sizeof(int), 
        cudaMemcpyHostToDevice
    );

    /***********************************************************/

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    // Get the class predictions
    KNN_GPU<<< gridSize, blockSize >>>(
        datasetArrayDevice, 
        dataset->num_instances(), 
        dataset->num_attributes(), 
        maximum, 
        k, 
        predictionsDevice,

        d_outputValues,
        d_outputValueMapping,
        d_neighbors,
        d_neighborDistances
    );

    cudaMemcpy(predictionsHost, predictionsDevice, dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < dataset->num_instances(); i++)
    {
        if(predictionsHost[i] == predictionsHostCPU[i])
        {
            cout << predictionsHost[i] << " ";
        }
        else
        {
            cout << "[" << predictionsHostCPU[i] << ", " << predictionsHost[i] << "] ";
        }
    }
    cout << endl;

    // Compute the confusion matrix
    int* confusionMatrixGPU = computeConfusionMatrix(predictionsHost, dataset);
    // Calculate the accuracy
    float accuracyGPU = computeAccuracy(confusionMatrixGPU, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diffGPU = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier on the GPU for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diffGPU, accuracyGPU);

    /***********************************/

    cudaFreeHost(outputValues);
    cudaFreeHost(outputValueMapping);
    cudaFreeHost(neighbors);
    cudaFreeHost(neighborDistances);

    cudaFree(d_outputValues);
    cudaFree(d_outputValueMapping);
    cudaFree(d_neighbors);
    cudaFree(d_neighborDistances);


    /************************************/

    // Free memory
    cudaFree(predictionsDevice);
    cudaFree(datasetArrayDevice);
    cudaFreeHost(predictionsHost);
    cudaFreeHost(datasetArrayHost);
    cudaFreeHost(predictionsHostCPU);
}
