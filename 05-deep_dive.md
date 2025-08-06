=== PHASE 5: DEEP DIVE ANALYSES AND ADVANCED VISUALIZATIONS ===

=== DEEP DIVE 1: THE ANYCAST LATENCY PARADOX ===

Anycast Paradox Statistical Analysis:
  Cloudflare DNS: r = -0.0242 (p = 2.19e-07, n = 45,978)
  Google DNS: r = 0.2549 (p = 0.00e+00, n = 45,978)
  Quad9 DNS: r = -0.1362 (p = 3.51e-189, n = 45,978)
  Cloudflare CDN: r = 0.0195 (p = 2.98e-05, n = 45,978)

Possible Explanations:
  1. Geographic proximity: Shorter paths to farther anycast nodes
  2. Infrastructure quality: Longer paths through higher-quality networks
  3. Load balancing: Traffic engineering optimizing for latency over hop count
  4. Network topology: Dense interconnection reducing latency despite hop count

=== DEEP DIVE 2: IPv4 vs IPv6 INFRASTRUCTURE COMPARISON ===

QoS Metrics Summary:
             path_avg_jitter                               path_avg_loss  \
                       count   mean    std median      max         count   
service_type                                                               
Anycast               183912  3.034  4.741  2.072  177.402        183912   
Unicast                91956  2.717  3.627  1.761  129.211         91956   
Unicast CDN            45978  2.547  3.663  1.649   65.670         45978   

                                              
                mean     std  median     max  
service_type                                  
Anycast       14.575  14.651  12.500  70.000  
Unicast       15.716  12.230  16.000  65.833  
Unicast CDN   15.085   9.796  16.667  54.000  

=== DEEP DIVE 4: GOOGLE'S PERFORMANCE EXCELLENCE ===

Google DNS Performance Excellence Analysis:
  Mean latency: 4.645ms
  Latency std dev: 9.906ms
  Mean hop count: 6.92
  Hop count std dev: 1.97
  Latency CV: 213.25%
  Hop count CV: 28.42%

=== PHASE 5 COMPLETE ===
All deep dive analyses completed with advanced visualizations
Key findings ready for paper integration