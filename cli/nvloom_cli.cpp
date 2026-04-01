/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvloom.h"
#include "util.h"
#include "testcases.h"

#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <memory>

#define NVLOOM_VERSION "2.0.0"
#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif

bool richOutput = false;
int gpuToRackSamples = 3;
std::string csvInputFile;
std::string csvOutputFile;
int iterations = NvLoom::getDefaultIterationCount();
int latencyIterations = NvLoom::getDefaultLatencyIterationCount();
bool enableProfiling = false;
char bandwidthUnit = 'G';

bool shouldContinue(boost::program_options::variables_map &vm, int iteration, std::chrono::time_point<std::chrono::high_resolution_clock> startTime) {
    if (vm["repeat"].defaulted() && vm["duration"].defaulted()) {
        return false;
    }

    if (!vm["repeat"].defaulted()) {
        return iteration + 1 < vm["repeat"].as<int>();
    }

    if (!vm["duration"].defaulted()) {
        auto duration = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
        int shouldContinue = (duration < vm["duration"].as<int>()) ? 1 : 0;
        MPI_Bcast(&shouldContinue, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return shouldContinue;
    }

    ASSERT(0);
    return false;
}

AllocatorStrategy getAllocatorStrategy(std::string allocatorStrategyString) {
    if (allocatorStrategyString == "reuse"){
        return ALLOCATOR_STRATEGY_REUSE;
    } else if (allocatorStrategyString == "unique") {
        return ALLOCATOR_STRATEGY_UNIQUE;
    } else if (allocatorStrategyString == "cudapool") {
        return ALLOCATOR_STRATEGY_CUDA_POOLS;
    }
    throw std::runtime_error("Unknown allocator strategy: " + allocatorStrategyString);
}

char getBandwidthUnit(char bandwidthUnitChar) {
    char originalBandwidthUnitChar = bandwidthUnitChar;
    bandwidthUnitChar = std::tolower(bandwidthUnitChar);
    if (bandwidthUnitChar != 'k' && bandwidthUnitChar != 'm' && bandwidthUnitChar != 'g' && bandwidthUnitChar != 't') {
        throw std::runtime_error("Invalid bandwidth unit: " + std::to_string(originalBandwidthUnitChar));
    }
    return bandwidthUnitChar;
}

int run_program(int argc, char **argv) {
    boost::program_options::options_description opts("nvloom CLI");
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> suitesToRun;
    std::string allocatorStrategyString;
    std::string bufferSizeStr = std::to_string(NvLoom::getDefaultBufferSizeInMiB()) + "M";
    size_t bufferSizeInBytes;
    bool listTestcases = false;
    int repeat = 1;
    int duration = -1;
    AllocatorStrategy allocatorStrategy;

    std::string suitesOptionDescription("Suite(s) to run (by name): all-to-one, rack-aware-all-to-one, egm, egm-gpu-to-rack, fabric-stress, gpu-to-rack, multicast, pairwise, rack-to-rack, latency, pairwise-tma, multicast-tma");
    opts.add_options()
        ("help,h", "Produce help message")
        ("bufferSize,b", boost::program_options::value<std::string>(&bufferSizeStr)->default_value(bufferSizeStr), "Buffer size in bytes. Also accepts k, m, g, t suffixes for kibibytes, mebibytes, gibibytes and tebibytes respectively. Letter case is ignored. Byte count must be divisible by 128 bytes.")
        ("latencyIterations", boost::program_options::value<int>(&latencyIterations)->default_value(latencyIterations), "Number of iterations for latency testcases, must be at least 1024")
        ("testcase,t", boost::program_options::value<std::vector<std::string>>(&testcasesToRun)->multitoken(), "Testcase(s) to run (by name)")
        ("suite,s", boost::program_options::value<std::vector<std::string>>(&suitesToRun)->multitoken(), suitesOptionDescription.c_str())
        ("listTestcases,l", boost::program_options::bool_switch(&listTestcases)->default_value(listTestcases), "List testcases")
        ("richOutput,r", boost::program_options::bool_switch(&richOutput)->default_value(richOutput), "Rich output")
        ("allocatorStrategy,a", boost::program_options::value<std::string>(&allocatorStrategyString)->default_value("reuse"), "Allocator strategy: choose between unique, reuse and cudapool")
        ("gpuToRackSamples", boost::program_options::value<int>(&gpuToRackSamples)->default_value(gpuToRackSamples), "Number of per-rack samples to use in gpu_to_rack testcases")
        ("csvInputFile", boost::program_options::value<std::string>(&csvInputFile)->default_value(csvInputFile), "CSV input file")
        ("csvOutputFile", boost::program_options::value<std::string>(&csvOutputFile)->default_value(csvOutputFile), "CSV output file")
        ("iterations,i", boost::program_options::value<int>(&iterations)->default_value(iterations), "Number of copy iterations within the testcase to run, not including the warmup iteration")
        ("repeat,c", boost::program_options::value<int>(&repeat)->default_value(repeat), "Number of times to repeat each testcase")
        ("duration,d", boost::program_options::value<int>(&duration)->default_value(duration), "Duration of each testcase in seconds")
        ("enableProfiling", boost::program_options::bool_switch(&enableProfiling)->default_value(enableProfiling), "Enable profiling by disabling spinkernels. This may reduce accuracy of performance results.")
        ("bandwidthUnit", boost::program_options::value<char>(&bandwidthUnit)->default_value(bandwidthUnit), "Unit of reported bandwidths. Acceptable values are K, M, G, T for KB/s, MB/s, GB/s and TB/s respectively. Letter case is ignored.")
        ;

    boost::program_options::variables_map vm;
    try {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opts), vm);
        boost::program_options::notify(vm);

        bufferSizeInBytes = getBufferSizeInBytes(bufferSizeStr);
        if (bufferSizeInBytes % 128 != 0) {
            throw std::runtime_error("Buffer size must be divisible by 128 bytes");
        }
        allocatorStrategy = getAllocatorStrategy(allocatorStrategyString);

        if (latencyIterations < 1024) {
            throw std::runtime_error("Latency iterations must be at least 1024");
        }

        if (!vm["repeat"].defaulted() && !vm["duration"].defaulted()) {
            throw std::runtime_error("Cannot specify both repeat and duration");
        }

        bandwidthUnit = getBandwidthUnit(bandwidthUnit);
    } catch (const std::exception& e) {
        std::cerr << "Couldn't parse command line arguments properly:\n";
        std::cerr << e.what() << '\n' << '\n';
        std::cerr << opts << "\n";
        return 1;
    }

    if (vm.count("help")) {
        OUTPUT << opts << "\n";
        return 0;
    }

    OUTPUT << "nvloom_cli " << NVLOOM_VERSION << std::endl;
    OUTPUT << "git commit: " << GIT_COMMIT << std::endl;
    OUTPUT << "NVCC version: " << getNvccVersion() << std::endl;
    OUTPUT << "CUDA version: " << getCudaVersion() << std::endl;
    OUTPUT << "Driver version: " << getDriverVersion() << std::endl;
    OUTPUT << "Allocation strategy: " << allocatorStrategyString << std::endl;
    OUTPUT << "Buffer size: " << bufferSizeInBytes << " bytes" << std::endl;
    OUTPUT << "Iteration count: " << iterations << std::endl;

    auto [testcases, suites] = buildTestcases(allocatorStrategy);

    if (listTestcases) {
        OUTPUT << "Available testcases: " << std::endl;
        for (auto const& [testcaseName, testcaseFunction] : testcases) {
            OUTPUT << testcaseName << std::endl;
        }
        return 0;
    }

    std::map<std::string, std::vector<int> > rackToProcessMap;
    int localDevice = discoverRanks(rackToProcessMap);
    NvLoom::initialize(localDevice, rackToProcessMap);
    if (enableProfiling) {
        NvLoom::disableSpinKernels();
    }

    std::set<std::string> testcasesToRunSet;

    try {
        for (auto suite: suitesToRun) {
            if (suites.count(suite) == 0) {
                throw std::runtime_error("No such suite as \"" + suite + "\"");
            }
            for (auto testcase: suites[suite]) {
                testcasesToRunSet.insert(testcase);
            }
        }

        for (auto testcase: testcasesToRun) {
            if (testcases.count(testcase) == 0) {
                throw std::runtime_error("No such testcase as \"" + testcase + "\"");
            }
            testcasesToRunSet.insert(testcase);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    if (testcasesToRunSet.size() == 0) {
        for (auto& [testcaseName, testcase] : testcases) {
            testcasesToRunSet.insert(testcaseName);
        }
    }

    for (auto testcase : testcasesToRunSet) {
        int iterationCount = 0;
        auto loopStartTime = std::chrono::high_resolution_clock::now();
        while (true) {
            std::string testcaseName = testcase;
            if (!vm["repeat"].defaulted() || !vm["duration"].defaulted()) {
                testcaseName += "_iter_" + std::to_string(iterationCount);
            }

            OUTPUT << "Running " << testcaseName << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
            try {
                testcases[testcase]->filterRun(bufferSizeInBytes);
            } catch (const std::exception& e) {
                std::cerr << "Error running testcase " << testcaseName << ": " << e.what() << std::endl;
                return 1;
            }
            auto endTime = std::chrono::high_resolution_clock::now();

            bool shouldContinueIteration = shouldContinue(vm, iterationCount, loopStartTime);
            if (!shouldContinueIteration) {
                // We're only clearing the pools on last iteration of the loop
                // But we still want to include the time it took to clear the pools in the output
                clearAllocationPools();
            }

            OUTPUT << "ExecutionTime " << testcaseName << " " << std::chrono::duration<double>(endTime - startTime).count() << " s" << std::endl;
            OUTPUT << "Done " << testcaseName << std::endl;
            OUTPUT << std::endl;

            if (!shouldContinueIteration) {
                break;
            }

            iterationCount++;
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    std::cout << std::unitbuf;
    int ret = run_program(argc, argv);

    NvLoom::finalize();
    return ret;
}
