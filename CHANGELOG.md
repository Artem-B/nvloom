# Changelog

## [1.4.0] - 2025-12-08

### Added
- Benchmarks of multicast broadcast operations executed by Copy Engines
- Unicast and multicast TMA testcases
- Heatmap plotter now has an option to skip plotting data values

### Changed
- All-to-one testcases warn if individual results are significantly assymetric
- Column separators in gpu_to_rack testcases are only printed in "rich output mode"
- Deprecated CUDA 12.X, support will be removed in next release
- Deprecated 570 and 575 drivers, support will be removed in next release

### Fixed
- Row separator handling in heatmap plotter

## [1.3.0] - 2025-10-02

### Added
- CSV defined benchmarks
- Memory access latency benchmarks
- Sample Dockerfile to build nvloom

### Changed
- Retry mechanism for CUDA multicast allocations was removed

### Fixed
- Freeing MNNVL memory did not have enough MPI barriers, leading to race conditions in extremely rare edge-cases
- Benchmarking algorithm sometimes would record "end event" twice. This had no impact on benchmark results.

## [1.2.0] - 2025-07-21

### Added
- Multicast reductions benchmarks
- Option to specify iteration count (-i/--iterations)
- Option to repeat a testcase for a specified number of iterations (-c/--repeat)
- Option to repeat a testcase for a specified number of seconds (-d/--duration)
- CUDA Stream Ordered Memory Allocator was added as a new allocator option (-a cudapool)

### Changed
- Caching multicast allocations is now much faster, thanks to multicast-specific memory pool

## [1.1.0] - 2025-05-22

### Added

- Heatmap plotter
- Support for CUDA Error Log Management
- Retry mechanism for CUDA multicast allocations
- Nvloom_cli argument to set number of samples in gpu-to-rack testcases
- Nvloom_cli now prints its version, git commit it was built from, and specified buffer size
- Nvloom_cli now prints units when reporting results
- Native compilation for sm_103 on CUDA 12.9 toolkits

### Changed

- Expanded README.md
- Rack-to-rack are now both unidir and bidir, and bidir rack-to-rack are symmetry-optimized.

### Fixed

- Bug where requesting allocations over 4 GiB would fail with CUDA_OUT_OF_MEMORY

## [1.0.0] - 2025-03-17

### Added

- Initial release
