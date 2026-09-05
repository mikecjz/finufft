#include <cufinufft.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t NX = 200;
constexpr int64_t NY = 200;
constexpr float EPS = 1.0e-6f;
constexpr int N_REPEATS = 20;
constexpr int N_SPOKES = 5;
constexpr int SAMPLES_PER_SPOKE = 400;
constexpr int N_ARMS = 2;
constexpr int SAMPLES_PER_ARM = N_SPOKES * SAMPLES_PER_SPOKE / N_ARMS;
constexpr int SPIRAL_TURNS = 4;
constexpr float PI = 3.14159265358979323846f;

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cufinufft(int status, const char *operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) +
                             " failed with cuFINUFFT error " +
                             std::to_string(status));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)),
               "cudaMalloc");
  }

  ~DeviceBuffer() { cudaFree(data_); }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t size() const { return count_; }

  void copy_from_host(const std::vector<T> &source) {
    if (source.size() != count_) {
      throw std::runtime_error("Host/device buffer size mismatch");
    }
    check_cuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy host to device");
  }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

class CudaEvent {
public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() { cudaEventDestroy(event_); }
  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;
  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_{};
};

class NufftPlan {
public:
  NufftPlan(int64_t m, float *d_x, float *d_y) {
    const int64_t modes[2] = {NX, NY};
    check_cufinufft(
        cufinufftf_makeplan(1, 2, modes, +1, 1, EPS, &plan_, nullptr),
        "cufinufftf_makeplan");
    try {
      check_cufinufft(cufinufftf_setpts(plan_, m, d_x, d_y, nullptr, 0,
                                        nullptr, nullptr, nullptr),
                      "cufinufftf_setpts");
    } catch (...) {
      cufinufftf_destroy(plan_);
      plan_ = nullptr;
      throw;
    }
  }

  ~NufftPlan() {
    if (plan_ != nullptr) {
      cufinufftf_destroy(plan_);
    }
  }

  NufftPlan(const NufftPlan &) = delete;
  NufftPlan &operator=(const NufftPlan &) = delete;

  void execute(cuFloatComplex *d_c, cuFloatComplex *d_f) {
    check_cufinufft(cufinufftf_execute(plan_, d_c, d_f),
                    "cufinufftf_execute");
  }

private:
  cufinufftf_plan plan_ = nullptr;
};

struct Trajectory {
  std::string name;
  std::vector<float> x;
  std::vector<float> y;
};

Trajectory make_radial_trajectory() {
  Trajectory trajectory{"5-spoke radial", {}, {}};
  trajectory.x.resize(N_SPOKES * SAMPLES_PER_SPOKE);
  trajectory.y.resize(N_SPOKES * SAMPLES_PER_SPOKE);

  for (int spoke = 0; spoke < N_SPOKES; ++spoke) {
    const float angle = static_cast<float>(spoke) * PI / N_SPOKES;
    for (int sample = 0; sample < SAMPLES_PER_SPOKE; ++sample) {
      const float r = -PI + 2.0f * PI * sample / SAMPLES_PER_SPOKE;
      const std::size_t index = spoke * SAMPLES_PER_SPOKE + sample;
      trajectory.x[index] = std::cos(angle) * r;
      trajectory.y[index] = std::sin(angle) * r;
    }
  }
  return trajectory;
}

Trajectory make_spiral_trajectory() {
  Trajectory trajectory{"2-arm spiral", {}, {}};
  trajectory.x.resize(N_ARMS * SAMPLES_PER_ARM);
  trajectory.y.resize(N_ARMS * SAMPLES_PER_ARM);

  for (int arm = 0; arm < N_ARMS; ++arm) {
    const float phase = 2.0f * PI * arm / N_ARMS;
    for (int sample = 0; sample < SAMPLES_PER_ARM; ++sample) {
      const float t = static_cast<float>(sample) / SAMPLES_PER_ARM;
      const float radius = PI * t;
      const float angle = 2.0f * PI * SPIRAL_TURNS * t + phase;
      const std::size_t index = arm * SAMPLES_PER_ARM + sample;
      trajectory.x[index] = radius * std::cos(angle);
      trajectory.y[index] = radius * std::sin(angle);
    }
  }
  return trajectory;
}

struct Statistics {
  double first;
  double mean;
  double median;
  double standard_deviation;
};

Statistics summarize(const std::vector<double> &samples) {
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();

  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  const double median = sorted.size() % 2 == 0
                            ? 0.5 * (sorted[middle - 1] + sorted[middle])
                            : sorted[middle];

  double squared_error = 0.0;
  for (double value : samples) {
    const double difference = value - mean;
    squared_error += difference * difference;
  }

  return {samples.front(), mean, median,
          std::sqrt(squared_error / samples.size())};
}

void print_statistics(const char *label, const Statistics &statistics) {
  std::cout << "  " << label << "\n"
            << "    single run: " << statistics.first << " ms\n"
            << "    mean:       " << statistics.mean << " ms\n"
            << "    median:     " << statistics.median << " ms\n"
            << "    std dev:    " << statistics.standard_deviation << " ms\n";
}

Statistics time_execute_only(NufftPlan &plan, cuFloatComplex *d_c,
                             cuFloatComplex *d_f) {
  plan.execute(d_c, d_f);
  check_cuda(cudaDeviceSynchronize(), "warm-up synchronization");

  CudaEvent start;
  CudaEvent end;
  std::vector<double> times_ms;
  times_ms.reserve(N_REPEATS);

  for (int repeat = 0; repeat < N_REPEATS; ++repeat) {
    check_cuda(cudaEventRecord(start.get()), "record start event");
    plan.execute(d_c, d_f);
    check_cuda(cudaEventRecord(end.get()), "record end event");
    check_cuda(cudaEventSynchronize(end.get()), "synchronize end event");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start.get(), end.get()),
               "cudaEventElapsedTime");
    times_ms.push_back(elapsed_ms);
  }
  return summarize(times_ms);
}

void execute_with_new_plan(int64_t m, float *d_x, float *d_y,
                           cuFloatComplex *d_c, cuFloatComplex *d_f) {
  NufftPlan plan(m, d_x, d_y);
  plan.execute(d_c, d_f);
}

Statistics time_end_to_end(int64_t m, float *d_x, float *d_y,
                           cuFloatComplex *d_c, cuFloatComplex *d_f) {
  execute_with_new_plan(m, d_x, d_y, d_c, d_f);
  check_cuda(cudaDeviceSynchronize(), "end-to-end warm-up synchronization");

  std::vector<double> times_ms;
  times_ms.reserve(N_REPEATS);
  for (int repeat = 0; repeat < N_REPEATS; ++repeat) {
    check_cuda(cudaDeviceSynchronize(), "pre-timing synchronization");
    const auto start = std::chrono::steady_clock::now();
    execute_with_new_plan(m, d_x, d_y, d_c, d_f);
    check_cuda(cudaDeviceSynchronize(), "post-timing synchronization");
    const auto end = std::chrono::steady_clock::now();
    times_ms.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  return summarize(times_ms);
}

void benchmark(const Trajectory &trajectory) {
  const int64_t m = static_cast<int64_t>(trajectory.x.size());
  DeviceBuffer<float> d_x(m);
  DeviceBuffer<float> d_y(m);
  DeviceBuffer<cuFloatComplex> d_c(m);
  DeviceBuffer<cuFloatComplex> d_f(NX * NY);
  d_x.copy_from_host(trajectory.x);
  d_y.copy_from_host(trajectory.y);

  std::mt19937 generator(0);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<cuFloatComplex> coefficients(m);
  for (auto &coefficient : coefficients) {
    coefficient = make_cuFloatComplex(normal(generator), normal(generator));
  }
  d_c.copy_from_host(coefficients);

  Statistics execute_only{};
  {
    NufftPlan reusable_plan(m, d_x.get(), d_y.get());
    execute_only = time_execute_only(reusable_plan, d_c.get(), d_f.get());
  }
  const Statistics end_to_end =
      time_end_to_end(m, d_x.get(), d_y.get(), d_c.get(), d_f.get());

  std::cout << trajectory.name << " (" << m << " samples):\n";
  print_statistics("execute only (reused plan):", execute_only);
  print_statistics("end to end (new plan each run):", end_to_end);
  std::cout << '\n';
}

} // namespace

int main() {
  try {
    std::cout << std::fixed << std::setprecision(3);
    benchmark(make_radial_trajectory());
    benchmark(make_spiral_trajectory());
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

