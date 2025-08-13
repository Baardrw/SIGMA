# Results from seed line without cg
5 warps per block, 280000000 total measurements
```
=== BENCHMARK RESULTS ===

--- Kernel Performance (average of 10 runs) ---
Average kernel time:     87.536 ms
Min kernel time:         83.466 ms
Max kernel time:         95.287 ms
Standard deviation:      3.700 ms
Coefficient of variation: 4.227%

--- Performance Metrics ---
Measurements per second: -1.706e+09
Buckets per second:      2.283e+08
Est. memory bandwidth:   86.747 GB/s

--- GPU Utilization ---
Blocks launched:         2000000
Threads per block:       320
Total threads:           640000000
Thread utilization:      43.8%
Test completed successfully!
```


# Results from seed line with cg size 16
5 warps per block, 280000000 total measurements
```

=== BENCHMARK RESULTS ===

--- Timing Breakdown ---
Setup time:              344.820 ms
Data loading time:       2351.207 ms
Host to Device copy:     621.457 ms
Device to Host copy:     21.274 ms
Total time:              3864.259 ms

--- Kernel Performance (average of 10 runs) ---
Average kernel time:     46.267 ms
Min kernel time:         45.201 ms
Max kernel time:         50.508 ms
Standard deviation:      1.938 ms
Coefficient of variation: 4.189%

--- Performance Metrics ---
Measurements per second: -3.228e+09
Buckets per second:      4.319e+08
Est. memory bandwidth:   164.112 GB/s

--- GPU Utilization ---
Blocks launched:         2000000
Threads per block:       160
Total threads:           320000000
Thread utilization:      87.5%
```

# Python 
12,281 seconds 3.41 hours

266978