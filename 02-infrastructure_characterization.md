=== PHASE 2: INFRASTRUCTURE CHARACTERIZATION ===

=== HOP COUNT ANALYSIS ===
Extracting hop count data...
✓ Extracted hop data for 321,846 measurements

Hop Count Statistics by Protocol:
           count   mean   std  min  max  median
protocol                                       
IPv4      160923  11.86  5.82    2   30    10.0
IPv6      160923  12.60  5.44    4   30    11.0

Hop Count Statistics by Service Type:
                       count   mean   std  min  max  median
protocol service_type                                      
IPv4     Anycast       91956   7.65  2.01    2   18     7.0
         Unicast       45978  16.91  4.57    5   27    17.0
         Unicast CDN   22989  18.61  3.46   12   30    19.0
IPv6     Anycast       91956   9.05  2.44    4   19     9.0
         Unicast       45978  17.59  5.10    6   30    17.0
         Unicast CDN   22989  16.79  3.73    8   25    15.0

Hop Count Statistics by Service Provider:
                           count   mean   std  min  max  median
protocol service_provider                                      
IPv4     Akamai CDN        22989  18.61  3.46   12   30    19.0
         Berkeley NTP      22989  19.91  3.85   11   27    21.0
         Cloudflare CDN    22989   8.93  1.69    6   18     9.0
         Cloudflare DNS    22989   8.76  1.73    6   18     9.0
         Google DNS        22989   6.36  0.57    6   14     6.0
         Heise             22989  13.90  3.00    5   22    14.0
         Quad9 DNS         22989   6.55  2.07    2   14     7.0
IPv6     Akamai CDN        22989  16.79  3.73    8   25    15.0
         Berkeley NTP      22989  21.47  4.00   11   30    21.0
         Cloudflare CDN    22989   9.11  2.03    6   18     9.0
         Cloudflare DNS    22989  10.16  2.11    7   19     9.0
         Google DNS        22989   7.48  2.60    5   17     6.0
         Heise             22989  13.71  2.45    6   23    14.0
         Quad9 DNS         22989   9.46  2.13    4   18     9.0

=== REGIONAL HOP COUNT PATTERNS ===
Hop Count Statistics by Region:
                         count   mean   std  min  max
protocol region                                      
IPv4     af-south-1      16100  13.74  6.52    5   27
         ap-east-1       16093  12.57  6.66    4   25
         ap-northeast-1  16058  11.29  5.30    5   26
         ap-south-1      16107  12.90  6.31    5   25
         ap-southeast-2  16079  12.10  6.02    4   25
         ca-central-1    16107  13.73  6.66    4   30
         eu-central-1    16086  10.45  5.18    2   22
         eu-north-1      16100  11.04  4.09    6   22
         sa-east-1       16093  10.92  5.45    5   22
         us-west-1       16100   9.87  3.77    6   22
IPv6     af-south-1      16100  16.12  5.59    5   30
         ap-east-1       16093  12.36  4.91    4   25
         ap-northeast-1  16058  11.75  5.07    6   26
         ap-south-1      16107  13.68  6.58    5   30
         ap-southeast-2  16079  12.84  4.61    4   22
         ca-central-1    16107  15.79  5.65    4   25
         eu-central-1    16086  10.38  5.06    5   25
         eu-north-1      16100  11.67  4.67    6   24
         sa-east-1       16093  10.94  4.80    6   23
         us-west-1       16100  10.46  3.27    6   17

=== ASN ANALYSIS ===
Extracting ASN data...
✓ Extracted ASN data for 1,816,271 hop records
✓ Extracted path data for 321,834 measurements

Most Common ASNs (Top 15):
  AS20940: 385,169 occurrences
  AS16509: 285,049 occurrences
  AS25: 229,266 occurrences
  AS13335: 173,547 occurrences
  AS2152: 160,783 occurrences
  AS15169: 141,784 occurrences
  AS12306: 132,621 occurrences
  AS2914: 91,884 occurrences
  AS19281: 45,966 occurrences
  AS201011: 38,732 occurrences
  AS6453: 26,880 occurrences
  AS135217: 26,619 occurrences
  AS3356: 24,559 occurrences
  AS6939: 14,421 occurrences
  AS3257: 12,097 occurrences

ASN Usage by Protocol (Top 10 per protocol):

IPv4 Top ASNs:
  AS20940: 196,648 occurrences
  AS16509: 176,764 occurrences
  AS25: 114,890 occurrences
  AS13335: 86,741 occurrences
  AS15169: 77,019 occurrences
  AS12306: 73,548 occurrences
  AS2152: 73,036 occurrences
  AS2914: 60,589 occurrences
  AS6453: 26,869 occurrences
  AS3356: 24,511 occurrences

IPv6 Top ASNs:
  AS20940: 188,521 occurrences
  AS25: 114,376 occurrences
  AS16509: 108,285 occurrences
  AS2152: 87,747 occurrences
  AS13335: 86,806 occurrences
  AS15169: 64,765 occurrences
  AS12306: 59,073 occurrences
  AS2914: 31,295 occurrences
  AS135217: 26,619 occurrences
  AS19281: 22,986 occurrences

=== ASN PATH DIVERSITY ===
AS Path Length Statistics:
           count  mean   std  min  max  median
protocol                                      
IPv4      160918  6.08  4.06    1   18     4.0
IPv6      160916  5.21  3.67    1   22     4.0

Unique AS Paths per Service Provider:
protocol  service_provider
IPv4      Akamai CDN           64
          Berkeley NTP        235
          Cloudflare CDN       24
          Cloudflare DNS       28
          Google DNS           10
          Heise                35
          Quad9 DNS            16
IPv6      Akamai CDN           70
          Berkeley NTP        157
          Cloudflare CDN       46
          Cloudflare DNS       50
          Google DNS           48
          Heise               147
          Quad9 DNS            65
Name: as_path, dtype: int64

=== TIER 1 PROVIDER ANALYSIS ===
Tier 1 Provider Usage:

IPv4:
  AS2914 (NTT): 60,589 occurrences
  AS6453 (TATA): 26,869 occurrences
  AS3356 (Level3/Lumen): 24,511 occurrences
  AS1299 (Telia): 7,417 occurrences
  AS3257 (GTT): 6,071 occurrences
  AS174 (Cogent): 105 occurrences
  AS5511 (Orange): 47 occurrences
  AS6762 (Sparkle): 9 occurrences

IPv6:
  AS2914 (NTT): 31,295 occurrences
  AS6939 (Hurricane Electric): 14,421 occurrences
  AS3257 (GTT): 6,026 occurrences
  AS174 (Cogent): 290 occurrences
  AS3356 (Level3/Lumen): 48 occurrences
  AS6453 (TATA): 11 occurrences
  AS5511 (Orange): 10 occurrences

=== INFRASTRUCTURE QUALITY INDICATORS ===
Calculating infrastructure quality metrics...
✓ Calculated quality metrics for 321,846 measurements

Infrastructure Quality by Service Type:
                      path_avg_latency               path_max_latency          \
                                  mean    std median             mean  median   
protocol service_type                                                           
IPv4     Anycast                 10.44  34.23   3.54            53.51   10.93   
         Unicast                102.38  65.96  95.52           261.91  162.02   
         Unicast CDN             82.88  48.65  93.66           276.04  183.11   
IPv6     Anycast                  7.23  25.31   3.35            32.57    9.92   
         Unicast                 93.13  59.26  92.06           231.15  156.47   
         Unicast CDN             71.77  41.59  79.61           242.83  174.89   

                      path_avg_loss        path_avg_jitter         
                               mean    max            mean median  
protocol service_type                                              
IPv4     Anycast              13.27  70.00            3.22   1.93  
         Unicast              11.22  65.83            2.81   1.82  
         Unicast CDN          21.61  54.00            2.74   1.75  
IPv6     Anycast              15.88  66.67            2.85   2.17  
         Unicast              20.21  62.50            2.62   1.71  
         Unicast CDN           8.56  45.22            2.35   1.59  

Saving intermediate results...
✓ Results saved to ../results/ directory

=== PHASE 2 COMPLETE ===
Ready to proceed with Phase 3: Performance Analysis and Visualizations