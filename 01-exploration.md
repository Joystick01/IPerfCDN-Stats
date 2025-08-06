=== PHASE 1: DATA EXPLORATION AND QUALITY ASSESSMENT ===

Loading datasets...
✓ IPv4 dataset loaded: 160,923 records
✓ IPv6 dataset loaded: 160,923 records

Service Provider Mapping:
  1.1.1.1                             → Cloudflare DNS  (Anycast)
  8.8.8.8                             → Google DNS      (Anycast)
  9.9.9.9                             → Quad9 DNS       (Anycast)
  2.16.241.219                        → Akamai CDN      (Unicast CDN)
  104.16.123.96                       → Cloudflare CDN  (Anycast)
  193.99.144.85                       → Heise           (Unicast)
  169.229.128.134                     → Berkeley NTP    (Unicast)
  2606:4700:4700::1111                → Cloudflare DNS  (Anycast)
  2001:4860:4860::8888                → Google DNS      (Anycast)
  2620:fe::fe:9                       → Quad9 DNS       (Anycast)
  2a02:26f0:3500:1b::1724:a393        → Akamai CDN      (Unicast CDN)
  2606:4700::6810:7b60                → Cloudflare CDN  (Anycast)
  2a02:2e0:3fe:1001:7777:772e:2:85    → Heise           (Unicast)
  2607:f140:ffff:8000:0:8006:0:a      → Berkeley NTP    (Unicast)

=== DATASET OVERVIEW ===
IPv4 Records: 160,923
IPv6 Records: 160,923
Total Records: 321,846

IPv4 Dataset Structure:
id                          object
utctime             datetime64[ns]
bitpattern                  object
src                         object
psize                        int32
dst                         object
tos                          int32
tests                        int32
region                      object
hubs                        object
service_provider            object
service_type                object
protocol                    object
dtype: object

IPv6 Dataset Structure:
id                          object
utctime             datetime64[ns]
bitpattern                  object
src                         object
psize                        int32
dst                         object
tos                          int32
tests                        int32
region                      object
hubs                        object
service_provider            object
service_type                object
protocol                    object
dtype: object

=== TEMPORAL COVERAGE ===
IPv4 Time Range:
  Start: 2025-05-27 12:59:06.053865
  End:   2025-06-20 14:31:15.563100
  Duration: 24 days 01:32:09.509235

IPv6 Time Range:
  Start: 2025-05-27 12:59:06.053865
  End:   2025-06-20 14:31:15.563100
  Duration: 24 days 01:32:09.509235

=== REGIONAL COVERAGE ===
IPv4 Measurements per Region:
  af-south-1     :  16,100 measurements
  ap-east-1      :  16,093 measurements
  ap-northeast-1 :  16,058 measurements
  ap-south-1     :  16,107 measurements
  ap-southeast-2 :  16,079 measurements
  ca-central-1   :  16,107 measurements
  eu-central-1   :  16,086 measurements
  eu-north-1     :  16,100 measurements
  sa-east-1      :  16,093 measurements
  us-west-1      :  16,100 measurements

IPv6 Measurements per Region:
  af-south-1     :  16,100 measurements
  ap-east-1      :  16,093 measurements
  ap-northeast-1 :  16,058 measurements
  ap-south-1     :  16,107 measurements
  ap-southeast-2 :  16,079 measurements
  ca-central-1   :  16,107 measurements
  eu-central-1   :  16,086 measurements
  eu-north-1     :  16,100 measurements
  sa-east-1      :  16,093 measurements
  us-west-1      :  16,100 measurements

=== SERVICE PROVIDER COVERAGE ===
IPv4 Measurements per Service Provider:
  Heise          :  22,989 measurements (Unicast)
  Quad9 DNS      :  22,989 measurements (Anycast)
  Berkeley NTP   :  22,989 measurements (Unicast)
  Google DNS     :  22,989 measurements (Anycast)
  Akamai CDN     :  22,989 measurements (Unicast CDN)
  Cloudflare DNS :  22,989 measurements (Anycast)
  Cloudflare CDN :  22,989 measurements (Anycast)

IPv6 Measurements per Service Provider:
  Quad9 DNS      :  22,989 measurements (Anycast)
  Google DNS     :  22,989 measurements (Anycast)
  Cloudflare DNS :  22,989 measurements (Anycast)
  Berkeley NTP   :  22,989 measurements (Unicast)
  Heise          :  22,989 measurements (Unicast)
  Akamai CDN     :  22,989 measurements (Unicast CDN)
  Cloudflare CDN :  22,989 measurements (Anycast)

=== DATA QUALITY ASSESSMENT ===
IPv4 Hubs Analysis:
  Records with empty hubs: 0
  Records with hop data: 160,923
  Hop count range: 2 - 30
  Mean hop count: 11.86
  Total hops analyzed: 1,908,628

IPv6 Hubs Analysis:
  Records with empty hubs: 0
  Records with hop data: 160,923
  Hop count range: 4 - 30
  Mean hop count: 12.60
  Total hops analyzed: 2,027,376

=== MISSING DATA ANALYSIS ===
IPv4 Missing Data:

IPv6 Missing Data:

=== BASIC MEASUREMENT STATISTICS ===
IPv4 Packet Size Statistics:
  Mean: 64.00 bytes
  Range: 64 - 64 bytes
IPv4 Tests per Measurement:
  Mean: 5.00
  Range: 5 - 5

IPv6 Packet Size Statistics:
  Mean: 64.00 bytes
  Range: 64 - 64 bytes
IPv6 Tests per Measurement:
  Mean: 5.00
  Range: 5 - 5

=== SAMPLE DATA INSPECTION ===
IPv4 Sample Record:
  id: 3a3ecb3e-f2aa-47b1-a5bc-f390fd815375
  utctime: 2025-06-09 18:00:56.182862
  bitpattern: 0x00
  src: ip-10-0-0-88.ca-central-1.compute.internal
  psize: 64
  dst: 193.99.144.85
  tos: 0
  tests: 5
  region: ca-central-1
  service_provider: Heise
  service_type: Unicast
  protocol: IPv4

IPv6 Sample Record:
  id: 7aace2bf-4a0c-4c09-8386-1e6e2796cb08
  utctime: 2025-06-12 00:46:15.935049
  bitpattern: 0x00
  src: ip-10-0-0-5.ap-east-1.compute.internal
  psize: 64
  dst: 2620:fe::fe:9
  tos: 0
  tests: 5
  region: ap-east-1
  service_provider: Quad9 DNS
  service_type: Anycast
  protocol: IPv6

Sample Hubs Structure (first record, first 3 hops):
  Hop 1: {'Loss%': 0.0, 'Rcv': 5, 'Jint': 149.7, 'count': 1, 'Jttr': 110.98, 'Javg': 31.499, 'Drop': 0, 'Last': 113.01, 'Avg': 34.936, 'StDev': 47.718, 'Wrst': 113.01, 'Gmean': 11.952, 'host': 'ec2-52-60-0-83.ca-central-1.compute.amazonaws.com (52.60.0.83)', 'Snt': 5, 'Jmax': 110.98, 'Best': 2.028, 'ASN': 'AS16509'}
  Hop 2: {'Loss%': 100.0, 'Rcv': 0, 'Jint': 0.0, 'count': 2, 'Jttr': 0.0, 'Javg': 0.0, 'Drop': 5, 'Last': 0.0, 'Avg': 0.0, 'StDev': 0.0, 'Wrst': 0.0, 'Gmean': 0.0, 'host': '???', 'Snt': 5, 'Jmax': 0.0, 'Best': 0.0, 'ASN': 'AS???'}
  Hop 3: {'Loss%': 100.0, 'Rcv': 0, 'Jint': 0.0, 'count': 3, 'Jttr': 0.0, 'Javg': 0.0, 'Drop': 5, 'Last': 0.0, 'Avg': 0.0, 'StDev': 0.0, 'Wrst': 0.0, 'Gmean': 0.0, 'host': '???', 'Snt': 5, 'Jmax': 0.0, 'Best': 0.0, 'ASN': 'AS???'}

=== PHASE 1 COMPLETE ===
Ready to proceed with Phase 2: Infrastructure Characterization