# 18/08 

2D functional multi bucket per warp + overflow handling

CG size 16

=== BENCHMARK RESULTS ===

--- Timing Breakdown ---
Setup time:              323.220 ms
Data loading time:       992.992 ms
Host to Device copy:     121.686 ms
Device to Host copy:     4.379 ms
Total time:              2071.666 ms

--- Kernel Performance (average of 10 runs) ---
Average kernel time:     56.090 ms
Min kernel time:         54.065 ms
Max kernel time:         68.839 ms
Standard deviation:      4.369 ms
Coefficient of variation: 7.790%

--- Performance Metrics ---
Measurements per second: 4.990e+08
Buckets per second:      3.564e+07
Est. memory bandwidth:   13.543 GB/s