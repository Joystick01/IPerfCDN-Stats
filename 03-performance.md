=== PHASE 3: PERFORMANCE ANALYSIS AND VISUALIZATIONS ===

=== LATENCY EXTRACTION AND ANALYSIS ===
Extracting latency data...
✓ Extracted latency data for 321,846 measurements
✓ Extracted hop-by-hop data for 3,316,363 responsive hops

=== LATENCY DISTRIBUTION ANALYSIS ===
End-to-End Latency Statistics by Service Provider and Protocol:
                           count    mean    std  median   min      max     p95
protocol service_provider                                                     
IPv4     Akamai CDN        22989  145.58  75.41  161.14  0.00   485.37  248.88
         Berkeley NTP      22989  159.28  82.21  158.46  0.00   428.89  313.17
         Cloudflare CDN    22989    1.92   4.70    1.42  0.00   183.64    4.82
         Cloudflare DNS    22989    1.72   2.05    1.36  0.34   182.56    4.77
         Google DNS        22989    3.69   7.08    1.30  0.00    81.36   21.96
         Heise             22989  147.90  89.17  156.04  0.00  2350.10  280.75
         Quad9 DNS         22989    2.83   4.21    1.57  0.00   206.65   13.85
IPv6     Akamai CDN        22989  144.48  77.30  161.37  0.00   618.27  246.72
         Berkeley NTP      22989  149.85  73.43  149.54  0.00   336.69  267.74
         Cloudflare CDN    22989    2.04   4.46    1.58  0.00   184.47    4.78
         Cloudflare DNS    22989    2.04   4.44    1.58  0.00   184.12    4.73
         Google DNS        22989    5.60  12.01    1.72  0.00    71.84   28.28
         Heise             22989  147.83  87.64  156.12  0.00  1473.70  280.75
         Quad9 DNS         22989    3.09   3.87    1.78  0.00   160.24   13.93

Latency Per Hop Statistics:
                       count  mean   std  median
protocol service_type                           
IPv4     Anycast       91941  0.38  0.79    0.20
         Unicast       45960  8.96  5.36    8.93
         Unicast CDN   22988  7.81  4.20    8.36
IPv6     Anycast       91948  0.38  1.08    0.20
         Unicast       45927  8.55  5.19    8.21
         Unicast CDN   22952  8.63  4.97    9.59

=== REGIONAL PERFORMANCE ANALYSIS ===
Regional Performance Summary:
                         end_to_end_latency_mean  end_to_end_latency_std  \
protocol region                                                            
IPv4     af-south-1                        93.18                  110.75   
         ap-east-1                         80.32                   88.34   
         ap-northeast-1                    82.92                  103.08   
         ap-south-1                        80.28                   98.25   
         ap-southeast-2                    98.24                  119.71   
         ca-central-1                      42.52                   48.87   
         eu-central-1                      23.25                   53.56   
         eu-north-1                        32.93                   56.43   
         sa-east-1                         82.56                   94.74   
         us-west-1                         45.18                   67.98   
IPv6     af-south-1                        86.52                   98.30   
         ap-east-1                         79.83                   88.07   
         ap-northeast-1                    82.86                  101.74   
         ap-south-1                        81.20                   95.18   
         ap-southeast-2                    98.09                  119.25   
         ca-central-1                      40.07                   45.08   
         eu-central-1                      22.29                   51.04   
         eu-north-1                        31.06                   51.78   
         sa-east-1                         82.59                   93.94   
         us-west-1                         45.48                   68.03   

                         end_to_end_latency_median  hop_count_mean  \
protocol region                                                      
IPv4     af-south-1                          21.97           13.74   
         ap-east-1                           13.86           12.57   
         ap-northeast-1                       2.85           11.29   
         ap-south-1                           1.92           12.90   
         ap-southeast-2                       1.22           12.10   
         ca-central-1                         1.32           13.73   
         eu-central-1                         1.40           10.45   
         eu-north-1                           4.85           11.04   
         sa-east-1                            1.12           10.92   
         us-west-1                            1.95            9.87   
IPv6     af-south-1                          21.98           16.12   
         ap-east-1                           13.93           12.36   
         ap-northeast-1                       3.01           11.75   
         ap-south-1                           2.15           13.68   
         ap-southeast-2                       1.56           12.84   
         ca-central-1                         1.71           15.79   
         eu-central-1                         1.44           10.38   
         eu-north-1                           4.85           11.67   
         sa-east-1                            2.32           10.94   
         us-west-1                            2.44           10.46   

                         latency_per_hop_mean  
protocol region                                
IPv4     af-south-1                      4.98  
         ap-east-1                       4.44  
         ap-northeast-1                  5.50  
         ap-south-1                      4.27  
         ap-southeast-2                  5.74  
         ca-central-1                    2.21  
         eu-central-1                    1.23  
         eu-north-1                      2.08  
         sa-east-1                       5.11  
         us-west-1                       3.35  
IPv6     af-south-1                      4.29  
         ap-east-1                       5.01  
         ap-northeast-1                  5.53  
         ap-south-1                      4.53  
         ap-southeast-2                  5.69  
         ca-central-1                    1.95  
         eu-central-1                    1.12  
         eu-north-1                      1.82  
         sa-east-1                       5.74  
         us-west-1                       3.23  

=== TEMPORAL ANALYSIS ===
Hourly Latency Patterns (Mean End-to-End Latency by Hour):
hour                       0       1       2       3       4       5       6   \
protocol service_type                                                           
IPv4     Anycast         2.52    2.49    2.48    2.48    2.51    2.47    2.62   
         Unicast       153.51  152.77  153.17  153.47  153.45  153.28  153.17   
         Unicast CDN   145.25  144.92  145.36  145.75  145.34  145.54  145.93   
IPv6     Anycast         3.18    3.15    3.19    3.13    3.18    3.15    3.21   
         Unicast       148.14  148.52  147.98  148.67  148.34  149.09  148.51   
         Unicast CDN   144.58  144.40  144.41  144.82  144.19  144.79  144.33   

hour                       7       8       9   ...      14      15      16  \
protocol service_type                          ...                           
IPv4     Anycast         2.52    2.55    2.48  ...    2.52    2.50    2.64   
         Unicast       153.51  153.17  153.02  ...  154.95  153.61  154.73   
         Unicast CDN   145.51  145.43  145.79  ...  145.42  145.75  145.75   
IPv6     Anycast         3.17    3.14    3.14  ...    3.16    3.15    3.28   
         Unicast       148.64  150.42  149.08  ...  149.63  148.72  148.57   
         Unicast CDN   144.47  144.67  145.13  ...  144.70  144.71  144.64   

hour                       17      18      19      20      21      22      23  
protocol service_type                                                          
IPv4     Anycast         2.70    2.67    2.62    2.50    2.51    2.51    2.50  
         Unicast       153.00  154.05  153.12  153.22  153.09  153.88  152.94  
         Unicast CDN   145.81  145.98  145.91  145.99  145.51  145.08  145.35  
IPv6     Anycast         3.44    3.47    3.36    3.12    3.10    3.12    3.09  
         Unicast       148.40  149.74  147.70  148.91  148.58  148.85  148.37  
         Unicast CDN   144.75  144.46  144.07  144.24  144.24  143.93  144.16  

[6 rows x 24 columns]

Daily Latency Patterns (Mean End-to-End Latency by Day):
day_name               Friday  Monday  Saturday  Sunday  Thursday  Tuesday  \
protocol service_type                                                        
IPv4     Anycast         2.49    2.50      2.48    2.52      2.54     2.72   
         Unicast       153.51  153.65    153.81  153.31    153.07   153.20   
         Unicast CDN   145.56  145.36    145.42  145.63    145.54   145.73   
IPv6     Anycast         3.24    3.29      3.29    3.31      2.73     3.63   
         Unicast       148.30  148.80    148.63  148.67    148.47   148.86   
         Unicast CDN   144.70  144.19    144.60  144.50    144.53   144.16   

day_name               Wednesday  
protocol service_type             
IPv4     Anycast            2.50  
         Unicast          154.50  
         Unicast CDN      145.78  
IPv6     Anycast            3.01  
         Unicast          150.01  
         Unicast CDN      144.62  

=== SERVICE PROVIDER STABILITY ANALYSIS ===
Service Provider Stability Analysis (CV = Coefficient of Variation %):
                           mean_latency  std_latency  cv_latency  \
protocol service_provider                                          
IPv4     Akamai CDN              145.58        75.41       51.80   
         Berkeley NTP            159.28        82.21       51.62   
         Cloudflare CDN            1.92         4.70      245.17   
         Cloudflare DNS            1.72         2.05      118.82   
         Google DNS                3.69         7.08      192.00   
         Heise                   147.90        89.17       60.29   
         Quad9 DNS                 2.83         4.21      148.98   
IPv6     Akamai CDN              144.48        77.30       53.50   
         Berkeley NTP            149.85        73.43       49.00   
         Cloudflare CDN            2.04         4.46      218.32   
         Cloudflare DNS            2.04         4.44      218.08   
         Google DNS                5.60        12.01      214.39   
         Heise                   147.83        87.64       59.29   
         Quad9 DNS                 3.09         3.87      125.39   

                           mean_hop_count  std_hop_count  cv_hop_count  
protocol service_provider                                               
IPv4     Akamai CDN                 18.61           3.46         18.60  
         Berkeley NTP               19.91           3.85         19.33  
         Cloudflare CDN              8.93           1.69         18.92  
         Cloudflare DNS              8.76           1.73         19.78  
         Google DNS                  6.36           0.57          8.92  
         Heise                      13.90           3.00         21.60  
         Quad9 DNS                   6.55           2.07         31.57  
IPv6     Akamai CDN                 16.79           3.73         22.20  
         Berkeley NTP               21.47           4.00         18.63  
         Cloudflare CDN              9.11           2.03         22.27  
         Cloudflare DNS             10.16           2.11         20.75  
         Google DNS                  7.48           2.60         34.81  
         Heise                      13.71           2.45         17.84  
         Quad9 DNS                   9.46           2.13         22.50  

=== HOP-BY-HOP LATENCY ANALYSIS ===
Average Latency by Hop Position (first 10 hops):
hop_position              1      2      3      4      5      6       7   \
protocol service_type                                                     
IPv4     Anycast       13.35  15.05  12.61   7.08  17.46  15.19    3.15   
         Unicast       14.14  12.74   8.25  72.19  11.48  24.92  105.12   
         Unicast CDN   12.69  39.38  16.73  10.24   9.62  67.73   57.84   
IPv6     Anycast        7.47  12.71   6.26   8.67   8.27  13.40    3.94   
         Unicast        7.91  16.02   3.11  75.82   7.87  12.74   89.51   
         Unicast CDN    7.49  50.89   3.49   3.64   3.33  18.07   61.23   

hop_position              8      9       10  
protocol service_type                        
IPv4     Anycast        4.41   3.84    8.67  
         Unicast       70.97  92.36  104.38  
         Unicast CDN   55.70  71.13   92.44  
IPv6     Anycast        4.56   4.47    3.47  
         Unicast       63.38  76.03   88.70  
         Unicast CDN   24.08  43.83   73.50  

=== PERFORMANCE ANOMALY DETECTION ===
High-Latency Measurements (>1000ms):
Total anomalies: 9 (0.00%)
Anomalies by Service Provider:
                           count  mean_latency  max_latency
protocol service_provider                                  
IPv4     Heise                 5       1451.70       2350.1
IPv6     Heise                 4       1314.22       1473.7

=== CONTINENTAL INFRASTRUCTURE QUALITY ===
Continental Infrastructure Quality:
                                    end_to_end_latency                 \
                                                  mean    std  median   
protocol continent     service_type                                     
IPv4     Africa        Anycast                    7.47  10.11    1.87   
                       Unicast                  233.26  80.24  251.82   
                       Unicast CDN              155.86   3.05  156.29   
         Asia-Pacific  Anycast                    2.36   5.02    1.37   
                       Unicast                  191.57  66.71  182.50   
                       Unicast CDN              205.50  32.33  213.57   
         Europe        Anycast                    2.37   1.68    1.78   
                       Unicast                   86.84  75.58   29.59   
                       Unicast CDN               13.49  12.12   24.08   
         North America Anycast                    1.53   1.10    1.61   
                       Unicast                   84.78  54.98   91.94   
                       Unicast CDN              131.23  18.55  139.72   
         South America Anycast                    0.69   0.60    0.47   
                       Unicast                  193.23   9.57  194.22   
                       Unicast CDN              188.69   8.34  188.18   
IPv6     Africa        Anycast                    7.34  10.03    1.90   
                       Unicast                  210.47  57.83  171.26   
                       Unicast CDN              155.31   3.28  155.36   
         Asia-Pacific  Anycast                    3.56   9.82    1.50   
                       Unicast                  187.78  65.33  160.56   
                       Unicast CDN              208.64  30.53  206.30   
         Europe        Anycast                    2.68   1.59    1.88   
                       Unicast                   81.88  70.33   85.26   
                       Unicast CDN               12.24  11.06   18.87   
         North America Anycast                    1.80   0.76    1.76   
                       Unicast                   85.04  54.97   92.57   
                       Unicast CDN              122.12  27.03  132.90   
         South America Anycast                    1.37   0.71    1.23   
                       Unicast                  193.13   9.31  194.45   
                       Unicast CDN              186.38   1.25  186.05   

                                    hop_count latency_per_hop  
                                         mean            mean  
protocol continent     service_type                            
IPv4     Africa        Anycast           8.86            1.15  
                       Unicast          20.93           10.90  
                       Unicast CDN      18.93            8.49  
         Asia-Pacific  Anycast           7.53            0.38  
                       Unicast          17.47           11.60  
                       Unicast CDN      20.42           10.19  
         Europe        Anycast           7.52            0.30  
                       Unicast          14.68            4.77  
                       Unicast CDN      15.79            0.87  
         North America Anycast           7.97            0.21  
                       Unicast          15.63            5.69  
                       Unicast CDN      19.45            7.22  
         South America Anycast           6.52            0.11  
                       Unicast          17.63           11.34  
                       Unicast CDN      15.07           12.66  
IPv6     Africa        Anycast          12.53            0.58  
                       Unicast          21.37            9.98  
                       Unicast CDN      20.00            7.77  
         Asia-Pacific  Anycast           8.76            0.50  
                       Unicast          18.28           10.97  
                       Unicast CDN      17.00           12.39  
         Europe        Anycast           8.04            0.32  
                       Unicast          16.02            4.08  
                       Unicast CDN      12.99            0.83  
         North America Anycast           9.70            0.21  
                       Unicast          16.56            5.22  
                       Unicast CDN      19.98            6.85  
         South America Anycast           7.51            0.19  
                       Unicast          16.28           13.01  
                       Unicast CDN      13.95           13.43  

=== SAVING RESULTS ===
✓ Latency analysis saved to ../results/latency_analysis.csv
✓ Hop-by-hop analysis saved to ../results/hop_by_hop_analysis.csv
✓ Stability analysis saved to ../results/stability_analysis.csv

=== PHASE 3 COMPLETE ===
Ready to proceed with Phase 4: Advanced Analysis and Visualizations