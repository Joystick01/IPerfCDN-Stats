=== FIXED ANYCAST INFRASTRUCTURE REVERSE ENGINEERING ===

=== METHOD 1: PENULTIMATE HOP ANALYSIS ===

--- Cloudflare DNS Penultimate Hop Analysis ---
  Unique penultimate hosts found: 187
  Top 5 penultimate hosts:
    cloudflare.cinx.net.za: 4588 measurements
    2620:107:4008:169::2: 1122 measurements
    99.83.91.47: 1117 measurements
    2620:107:4008:164::2: 1117 measurements
    99.83.91.49: 1093 measurements
  Hosts serving multiple continents: 1
    These suggest regional server consolidation:
    ???: serves 4 continents (North America, Europe, Africa, Asia-Pacific)

  ASN-based analysis:
  Unique penultimate ASNs: 6
  Top 3 penultimate ASNs:
    AS13335: 36723 measurements
    AS???: 6892 measurements
    AS16509: 2210 measurements

--- Google DNS Penultimate Hop Analysis ---
  Unique penultimate hosts found: 780
  Top 5 penultimate hosts:
    ???: 593 measurements
    142.250.238.147: 324 measurements
    2001:4860:0:1::769d: 315 measurements
    142.250.58.131: 311 measurements
    2001:4860:0:1::829: 294 measurements
  Hosts serving multiple continents: 1
    These suggest regional server consolidation:
    ???: serves 5 continents (Europe, Asia-Pacific, North America, South America, Africa)

  ASN-based analysis:
  Unique penultimate ASNs: 3
  Top 3 penultimate ASNs:
    AS15169: 45371 measurements
    AS???: 606 measurements
    AS135217: 1 measurements

--- Quad9 DNS Penultimate Hop Analysis ---
  Unique penultimate hosts found: 36
  Top 5 penultimate hosts:
    as42.sfmix.org: 4600 measurements
    woodynet.cinx.net.za: 4599 measurements
    pch1.peer.qix.ca: 4597 measurements
    as42.nsw.ix.asn.au: 4001 measurements
    ???: 2379 measurements
  Hosts serving multiple continents: 2
    These suggest regional server consolidation:
    ???: serves 3 continents (Asia-Pacific, North America, Europe)
    hosted-by.i3d.net: serves 2 continents (Asia-Pacific, North America)

  ASN-based analysis:
  Unique penultimate ASNs: 5
  Top 3 penultimate ASNs:
    AS???: 36172 measurements
    AS7195: 4598 measurements
    AS49544: 4590 measurements

--- Cloudflare CDN Penultimate Hop Analysis ---
  Unique penultimate hosts found: 194
  Top 5 penultimate hosts:
    cloudflare.cinx.net.za: 4543 measurements
    99.83.91.47: 1130 measurements
    2620:107:4008:164::2: 1120 measurements
    2620:107:4008:169::2: 1113 measurements
    99.83.91.49: 1103 measurements
  Hosts serving multiple continents: 1
    These suggest regional server consolidation:
    ???: serves 5 continents (Europe, North America, Africa, South America, Asia-Pacific)

  ASN-based analysis:
  Unique penultimate ASNs: 8
  Top 3 penultimate ASNs:
    AS13335: 36750 measurements
    AS???: 6843 measurements
    AS16509: 2233 measurements

=== METHOD 2: LATENCY-BASED GEOGRAPHIC CLUSTERING ===

--- Cloudflare DNS Geographic Latency Clustering ---
  Regional latency statistics:
    af-south-1: 1.88±0.29ms (n=4600.0)
    ap-east-1: 1.99±9.98ms (n=4598.0)
    ap-northeast-1: 2.42±0.34ms (n=4588.0)
    ap-south-1: 0.97±2.62ms (n=4602.0)
    ap-southeast-2: 1.25±0.20ms (n=4594.0)
    ca-central-1: 1.45±1.16ms (n=4602.0)
    eu-central-1: 1.24±0.53ms (n=4596.0)
    eu-north-1: 4.49±0.81ms (n=4600.0)
    sa-east-1: 1.18±0.67ms (n=4598.0)
    us-west-1: 1.92±1.01ms (n=4600.0)
  Optimal cluster count (servers): 2
  Silhouette score: 0.715
  Regional server assignments:
    af-south-1: Server 1 (1.88ms)
    ap-east-1: Server 1 (1.99ms)
    ap-northeast-1: Server 1 (2.42ms)
    ap-south-1: Server 1 (0.97ms)
    ap-southeast-2: Server 1 (1.25ms)
    ca-central-1: Server 1 (1.45ms)
    eu-central-1: Server 1 (1.24ms)
    eu-north-1: Server 2 (4.49ms)
    sa-east-1: Server 1 (1.18ms)
    us-west-1: Server 1 (1.92ms)

--- Google DNS Geographic Latency Clustering ---
  Regional latency statistics:
    af-south-1: 24.14±5.12ms (n=4600.0)
    ap-east-1: 1.09±0.24ms (n=4598.0)
    ap-northeast-1: 2.46±0.85ms (n=4588.0)
    ap-south-1: 9.71±21.54ms (n=4602.0)
    ap-southeast-2: 0.92±0.36ms (n=4594.0)
    ca-central-1: 1.25±0.23ms (n=4602.0)
    eu-central-1: 1.04±1.49ms (n=4596.0)
    eu-north-1: 3.04±1.02ms (n=4600.0)
    sa-east-1: 0.88±0.76ms (n=4598.0)
    us-west-1: 1.91±0.27ms (n=4600.0)
  Optimal cluster count (servers): 2
  Silhouette score: 0.779
  Regional server assignments:
    af-south-1: Server 2 (24.14ms)
    ap-east-1: Server 1 (1.09ms)
    ap-northeast-1: Server 1 (2.46ms)
    ap-south-1: Server 1 (9.71ms)
    ap-southeast-2: Server 1 (0.92ms)
    ca-central-1: Server 1 (1.25ms)
    eu-central-1: Server 1 (1.04ms)
    eu-north-1: Server 1 (3.04ms)
    sa-east-1: Server 1 (0.88ms)
    us-west-1: Server 1 (1.91ms)

--- Quad9 DNS Geographic Latency Clustering ---
  Regional latency statistics:
    af-south-1: 1.55±0.89ms (n=4600.0)
    ap-east-1: 14.01±4.30ms (n=4598.0)
    ap-northeast-1: 2.75±0.36ms (n=4588.0)
    ap-south-1: 1.56±1.49ms (n=4602.0)
    ap-southeast-2: 0.98±0.46ms (n=4594.0)
    ca-central-1: 1.34±0.68ms (n=4602.0)
    eu-central-1: 1.61±0.22ms (n=4596.0)
    eu-north-1: 2.91±0.94ms (n=4600.0)
    sa-east-1: 0.88±0.82ms (n=4598.0)
    us-west-1: 2.00±0.35ms (n=4600.0)
  Optimal cluster count (servers): 2
  Silhouette score: 0.839
  Regional server assignments:
    af-south-1: Server 1 (1.55ms)
    ap-east-1: Server 2 (14.01ms)
    ap-northeast-1: Server 1 (2.75ms)
    ap-south-1: Server 1 (1.56ms)
    ap-southeast-2: Server 1 (0.98ms)
    ca-central-1: Server 1 (1.34ms)
    eu-central-1: Server 1 (1.61ms)
    eu-north-1: Server 1 (2.91ms)
    sa-east-1: Server 1 (0.88ms)
    us-west-1: Server 1 (2.00ms)

--- Cloudflare CDN Geographic Latency Clustering ---
  Regional latency statistics:
    af-south-1: 2.05±2.19ms (n=4600.0)
    ap-east-1: 2.49±13.55ms (n=4598.0)
    ap-northeast-1: 2.48±0.52ms (n=4588.0)
    ap-south-1: 0.97±2.17ms (n=4602.0)
    ap-southeast-2: 1.31±0.87ms (n=4594.0)
    ca-central-1: 1.49±1.13ms (n=4602.0)
    eu-central-1: 1.33±1.18ms (n=4596.0)
    eu-north-1: 4.53±0.81ms (n=4600.0)
    sa-east-1: 1.18±0.66ms (n=4598.0)
    us-west-1: 1.98±1.51ms (n=4600.0)
  Optimal cluster count (servers): 4
  Silhouette score: 0.707
  Regional server assignments:
    af-south-1: Server 3 (2.05ms)
    ap-east-1: Server 2 (2.49ms)
    ap-northeast-1: Server 2 (2.48ms)
    ap-south-1: Server 1 (0.97ms)
    ap-southeast-2: Server 1 (1.31ms)
    ca-central-1: Server 1 (1.49ms)
    eu-central-1: Server 1 (1.33ms)
    eu-north-1: Server 4 (4.53ms)
    sa-east-1: Server 1 (1.18ms)
    us-west-1: Server 3 (1.98ms)

=== METHOD 3: PATH SIMILARITY ANALYSIS ===

--- Cloudflare DNS Path Similarity Analysis ---
  Total unique AS paths: 72
  Most common paths (top 5):
    Path 1: 15513 measurements
      Serves regions: sa-east-1, eu-central-1, ap-east-1, us-west-1, ap-southeast-2, eu-north-1, ap-south-1, ap-northeast-1, ca-central-1
      Avg latency: 2.07ms
      AS path: 13335→13335

    Path 2: 12144 measurements
      Serves regions: eu-central-1, sa-east-1, us-west-1, ap-east-1, ap-southeast-2, ap-northeast-1, ap-south-1, eu-north-1
      Avg latency: 1.69ms
      AS path: 16509→16509→13335→13335

    Path 3: 4535 measurements
      Serves regions: af-south-1, ca-central-1, ap-east-1, sa-east-1
      Avg latency: 2.22ms
      AS path: 13335

    Path 4: 3201 measurements
      Serves regions: eu-north-1, ap-northeast-1, sa-east-1, eu-central-1, ap-south-1, ca-central-1
      Avg latency: 2.68ms
      AS path: 16509→16509→16509→13335→13335

    Path 5: 3010 measurements
      Serves regions: ap-east-1, eu-central-1, sa-east-1, us-west-1, ap-south-1, eu-north-1, ap-northeast-1
      Avg latency: 1.28ms
      AS path: 16509→13335→13335

  Major routing paths (>5% traffic): 5

--- Google DNS Path Similarity Analysis ---
  Total unique AS paths: 52
  Most common paths (top 5):
    Path 1: 12236 measurements
      Serves regions: sa-east-1, eu-north-1, eu-central-1, ap-east-1, ca-central-1, us-west-1, ap-northeast-1, ap-southeast-2, ap-south-1, af-south-1
      Avg latency: 2.96ms
      AS path: 15169→15169→15169

    Path 2: 11235 measurements
      Serves regions: eu-central-1, eu-north-1, ap-southeast-2, us-west-1, ap-south-1, sa-east-1
      Avg latency: 1.95ms
      AS path: 15169→15169→15169→15169

    Path 3: 8128 measurements
      Serves regions: ap-south-1, ap-east-1, af-south-1, sa-east-1, ca-central-1, ap-southeast-2, ap-northeast-1, eu-north-1, eu-central-1
      Avg latency: 7.68ms
      AS path: 16509→15169→15169→15169

    Path 4: 6046 measurements
      Serves regions: ca-central-1, ap-northeast-1, ap-southeast-2, af-south-1, eu-central-1
      Avg latency: 3.11ms
      AS path: 16509→16509→15169→15169→15169

    Path 5: 4723 measurements
      Serves regions: eu-north-1, ap-east-1, ap-south-1, sa-east-1, eu-central-1, ap-northeast-1, us-west-1, ca-central-1, ap-southeast-2, af-south-1
      Avg latency: 5.59ms
      AS path: 15169→15169

  Major routing paths (>5% traffic): 5

--- Quad9 DNS Path Similarity Analysis ---
  Total unique AS paths: 75
  Most common paths (top 5):
    Path 1: 20884 measurements
      Serves regions: ap-south-1, eu-central-1, af-south-1, ap-southeast-2, ca-central-1, us-west-1
      Avg latency: 1.47ms
      AS path: 19281

    Path 2: 7539 measurements
      Serves regions: eu-north-1, ca-central-1, us-west-1, ap-east-1, ap-south-1, ap-southeast-2, eu-central-1
      Avg latency: 3.18ms
      AS path: 16509→19281

    Path 3: 5383 measurements
      Serves regions: eu-north-1, ap-east-1, ca-central-1, ap-southeast-2
      Avg latency: 10.94ms
      AS path: 16509→16509→19281

    Path 4: 3263 measurements
      Serves regions: ap-northeast-1
      Avg latency: 2.74ms
      AS path: 16509→49544→19281

    Path 5: 1246 measurements
      Serves regions: ap-northeast-1
      Avg latency: 2.75ms
      AS path: 16509→16509→49544→19281

  Major routing paths (>5% traffic): 4

--- Cloudflare CDN Path Similarity Analysis ---
  Total unique AS paths: 62
  Most common paths (top 5):
    Path 1: 17747 measurements
      Serves regions: ap-southeast-2, sa-east-1, eu-central-1, ap-northeast-1, ap-east-1, eu-north-1, us-west-1, ap-south-1, ca-central-1
      Avg latency: 1.97ms
      AS path: 13335→13335

    Path 2: 10072 measurements
      Serves regions: ap-northeast-1, eu-north-1, us-west-1, eu-central-1, ap-east-1, sa-east-1, ap-south-1, ap-southeast-2, af-south-1
      Avg latency: 1.93ms
      AS path: 16509→16509→13335→13335

    Path 3: 4497 measurements
      Serves regions: af-south-1, sa-east-1, ca-central-1, ap-east-1, ap-northeast-1
      Avg latency: 2.26ms
      AS path: 13335

    Path 4: 3198 measurements
      Serves regions: us-west-1, sa-east-1, eu-central-1, ap-south-1, ap-east-1, eu-north-1
      Avg latency: 1.31ms
      AS path: 16509→13335→13335

    Path 5: 2888 measurements
      Serves regions: ap-northeast-1, eu-north-1, sa-east-1, ap-south-1, ca-central-1, af-south-1
      Avg latency: 2.99ms
      AS path: 16509→16509→16509→13335→13335

  Major routing paths (>5% traffic): 5

=== FINAL ANYCAST SERVER ESTIMATES ===
Service Provider Anycast Server Estimates:
============================================================

Cloudflare DNS:
  Total measurements analyzed: 45,978
  Unique penultimate hosts: 188
  Unique penultimate ASNs: 6
  Latency-based clustering: 2 servers
  Major routing paths: 5
  → ESTIMATED SERVERS: 5

Google DNS:
  Total measurements analyzed: 45,978
  Unique penultimate hosts: 780
  Unique penultimate ASNs: 3
  Latency-based clustering: 2 servers
  Major routing paths: 5
  → ESTIMATED SERVERS: 5

Quad9 DNS:
  Total measurements analyzed: 45,978
  Unique penultimate hosts: 44
  Unique penultimate ASNs: 5
  Latency-based clustering: 2 servers
  Major routing paths: 4
  → ESTIMATED SERVERS: 4

Cloudflare CDN:
  Total measurements analyzed: 45,978
  Unique penultimate hosts: 195
  Unique penultimate ASNs: 8
  Latency-based clustering: 4 servers
  Major routing paths: 5
  → ESTIMATED SERVERS: 5

=== METHODOLOGY VALIDATION ===
Key improvements over previous approach:
1. Uses actual network infrastructure indicators (penultimate hops)
2. Applies geographic latency clustering with validation
3. Analyzes routing path diversity patterns
4. Combines multiple independent methods
5. No longer assumes one server per measurement region

Note: These estimates represent minimum server counts.
Actual deployments may have additional servers not detectable from our vantage points.