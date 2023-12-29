#ifndef SRC_CPU_TIMER_H
#define SRC_CPU_TIMER_H
#pragma once
#include <chrono>

struct CpuTimer
{
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint start;
    TimePoint stop;

    void Start()
    {
        start = Clock::now();
    }

    void Stop()
    {
        stop = Clock::now();
    }

    float Elapsed()
    {
        std::chrono::duration<float, std::milli> duration = stop - start;
        return duration.count();
    }
};

#endif