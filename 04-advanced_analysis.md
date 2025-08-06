=== PHASE 4: ADVANCED ANALYSIS AND VISUALIZATIONS ===

✓ Loaded previous analysis results

=== VISUALIZATION 1: LATENCY DISTRIBUTIONS ===

✓ Latency distribution plot saved

=== VISUALIZATION 2: HOPCOUNT vs LATENCY ANALYSIS ===

Correlation Analysis (Pearson r):
  Anycast: r = -0.012 (p < 1.41e-07)
  Unicast: r = 0.466 (p < 0.00e+00)
  Unicast CDN: r = 0.274 (p < 0.00e+00)
✓ Hop count vs latency analysis completed

=== VISUALIZATION 3: REGIONAL INFRASTRUCTURE QUALITY ===

✓ Regional analysis visualization completed

=== ADVANCED ANALYSIS: ANYCAST INFRASTRUCTURE REVERSE ENGINEERING ===
Developing Anycast Infrastructure Reverse Engineering Methodology...

--- Cloudflare DNS Infrastructure Analysis ---
  Unique penultimate hop hosts: 188
  Unique penultimate ASNs: 6
  Regional latency variation (std): 1.76ms
  Hop count coefficient of variation: 21.69%
  Top penultimate ASNs (potential anycast hosts):
    AS13335: 36723 measurements
    AS???: 6892 measurements
    AS16509: 2210 measurements
    AS135217: 143 measurements
    AS3356: 6 measurements
  Estimated anycast server locations: 10

--- Google DNS Infrastructure Analysis ---
  Unique penultimate hop hosts: 780
  Unique penultimate ASNs: 3
  Regional latency variation (std): 3.19ms
  Hop count coefficient of variation: 28.42%
  Top penultimate ASNs (potential anycast hosts):
    AS15169: 45371 measurements
    AS???: 606 measurements
    AS135217: 1 measurements
  Estimated anycast server locations: 10

--- Quad9 DNS Infrastructure Analysis ---
  Unique penultimate hop hosts: 44
  Unique penultimate ASNs: 5
  Regional latency variation (std): 1.05ms
  Hop count coefficient of variation: 31.90%
  Top penultimate ASNs (potential anycast hosts):
    AS???: 36172 measurements
    AS7195: 4598 measurements
    AS49544: 4590 measurements
    AS135217: 476 measurements
    AS152144: 142 measurements
  Estimated anycast server locations: 10

--- Cloudflare CDN Infrastructure Analysis ---
  Unique penultimate hop hosts: 195
  Unique penultimate ASNs: 8
  Regional latency variation (std): 2.46ms
  Hop count coefficient of variation: 20.72%
  Top penultimate ASNs (potential anycast hosts):
    AS13335: 36750 measurements
    AS???: 6843 measurements
    AS16509: 2233 measurements
    AS135217: 141 measurements
    AS3356: 6 measurements
  Estimated anycast server locations: 10

✓ Anycast infrastructure analysis completed

=== ROUTING STABILITY ANALYSIS ===
Analyzing routing path stability...

--- IPv4 Routing Stability ---
  Average unique paths per region-destination: 9.19
  Average stability ratio: 0.996
  Most stable paths:
    af-south-1→Cloudflare DNS: 1.000 (1 unique paths)
    eu-central-1→Quad9 DNS: 1.000 (1 unique paths)
    ap-south-1→Google DNS: 1.000 (1 unique paths)
  Most variable paths:
    ap-south-1→Berkeley NTP: 0.974 (61 unique paths)
    eu-north-1→Berkeley NTP: 0.977 (54 unique paths)
    ca-central-1→Akamai CDN: 0.980 (46 unique paths)

--- IPv6 Routing Stability ---
  Average unique paths per region-destination: 13.64
  Average stability ratio: 0.995
  Most stable paths:
    eu-north-1→Cloudflare CDN: 0.998 (6 unique paths)
    ap-southeast-2→Cloudflare CDN: 0.998 (6 unique paths)
    eu-north-1→Google DNS: 0.997 (7 unique paths)
  Most variable paths:
    af-south-1→Berkeley NTP: 0.979 (50 unique paths)
    ap-south-1→Berkeley NTP: 0.986 (34 unique paths)
    ap-northeast-1→Berkeley NTP: 0.988 (29 unique paths)

✓ Routing stability analysis completed

=== TIER 1 TRANSIT PROVIDER DETAILED ANALYSIS ===
Analyzing Tier 1 transit provider usage patterns...

--- IPv4 Tier 1 Provider Usage ---
  Tier 1 Usage by Service Provider:
    Berkeley NTP:
      AS3356 (Level3/Lumen): 11,526 paths (50.1%)
      AS6453 (TATA): 6,853 paths (29.8%)
      AS2914 (NTT): 5,212 paths (22.7%)
      AS1299 (Telia): 2,271 paths (9.9%)
      AS174 (Cogent): 29 paths (0.1%)
    Heise:
      AS2914 (NTT): 4,892 paths (21.3%)
      AS3257 (GTT): 2,024 paths (8.8%)
      AS3356 (Level3/Lumen): 16 paths (0.1%)
    Akamai CDN:
      AS6453 (TATA): 460 paths (2.0%)
      AS174 (Cogent): 1 paths (0.0%)
    Cloudflare DNS:
      AS1299 (Telia): 18 paths (0.1%)
      AS3356 (Level3/Lumen): 5 paths (0.0%)
      AS174 (Cogent): 4 paths (0.0%)
      AS5511 (Orange): 3 paths (0.0%)
      AS6762 (Sparkle): 3 paths (0.0%)
    Cloudflare CDN:
      AS5511 (Orange): 20 paths (0.1%)
      AS3356 (Level3/Lumen): 3 paths (0.0%)
      AS1299 (Telia): 1 paths (0.0%)

--- IPv6 Tier 1 Provider Usage ---
  Tier 1 Usage by Service Provider:
    Heise:
      AS2914 (NTT): 4,889 paths (21.3%)
      AS3257 (GTT): 2,009 paths (8.7%)
      AS3356 (Level3/Lumen): 7 paths (0.0%)
    Berkeley NTP:
      AS6939 (Hurricane Electric): 10,213 paths (44.4%)
    Akamai CDN:
      AS174 (Cogent): 72 paths (0.3%)
      AS6453 (TATA): 1 paths (0.0%)
    Cloudflare DNS:
      AS3356 (Level3/Lumen): 7 paths (0.0%)
      AS5511 (Orange): 3 paths (0.0%)
    Cloudflare CDN:
      AS3356 (Level3/Lumen): 7 paths (0.0%)
      AS5511 (Orange): 3 paths (0.0%)
      AS174 (Cogent): 2 paths (0.0%)
    Quad9 DNS:
      AS2914 (NTT): 1 paths (0.0%)

=== PHASE 4 COMPLETE ===
All advanced analyses completed and visualizations saved to ../results/
Ready for paper structure and writing phase