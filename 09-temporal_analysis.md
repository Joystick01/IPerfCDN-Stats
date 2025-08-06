=== TIMEZONE-NORMALIZED TEMPORAL ANALYSIS ===

=== TIMEZONE MAPPING FOR AWS REGIONS ===
Region Timezone Mappings:
  us-west-1       → US/Pacific
  ca-central-1    → Canada/Eastern
  eu-central-1    → Europe/Berlin
  eu-north-1      → Europe/Stockholm
  af-south-1      → Africa/Johannesburg
  ap-east-1       → Asia/Hong_Kong
  ap-south-1      → Asia/Kolkata
  ap-northeast-1  → Asia/Tokyo
  ap-southeast-2  → Australia/Sydney
  sa-east-1       → America/Sao_Paulo

=== CONVERTING TO LOCAL TIMES ===
Converting timestamps to regional local times...
✓ Converted 321,846 measurements to local times

=== SAMPLE TIMEZONE CONVERSIONS ===
ca-central-1   : UTC 18:xx → Local 14:xx (Δ+20h)
eu-north-1     : UTC 20:xx → Local 22:xx (Δ+2h)
ap-south-1     : UTC 17:xx → Local 22:xx (Δ+5h)
eu-central-1   : UTC 08:xx → Local 10:xx (Δ+2h)
ap-south-1     : UTC 17:xx → Local 22:xx (Δ+5h)
ap-northeast-1 : UTC 23:xx → Local 08:xx (Δ+9h)
ca-central-1   : UTC 12:xx → Local 08:xx (Δ+20h)
eu-north-1     : UTC 21:xx → Local 23:xx (Δ+2h)
ap-southeast-2 : UTC 02:xx → Local 12:xx (Δ+10h)
eu-north-1     : UTC 00:xx → Local 02:xx (Δ+2h)

=== COMPARATIVE TEMPORAL ANALYSIS ===

=== STATISTICAL COMPARISON: UTC vs LOCAL TIME ===
Temporal Variability Analysis:
  UTC hourly pattern std deviation: 0.179ms
  Local hourly pattern std deviation: 0.154ms
  Local time shows less variability

=== BUSINESS HOURS IMPACT ANALYSIS ===
Business Hours Performance Impact:
Region               | Off-Hours      | Business Hours | Difference
----------------------------------------------------------------------
us-west-1       |    45.38ms    |      45.24ms   |  -0.14ms (Lower)
ca-central-1    |    41.23ms    |      41.40ms   |  +0.18ms (Higher)
eu-central-1    |    22.77ms    |      22.77ms   |  -0.00ms (Lower)
eu-north-1      |    31.99ms    |      32.01ms   |  +0.02ms (Higher)
af-south-1      |    89.83ms    |      89.89ms   |  +0.06ms (Higher)
ap-east-1       |    80.24ms    |      79.80ms   |  -0.44ms (Lower)
ap-south-1      |    80.69ms    |      80.83ms   |  +0.15ms (Higher)
ap-northeast-1  |    83.16ms    |      82.45ms   |  -0.71ms (Lower)
ap-southeast-2  |    98.46ms    |      97.68ms   |  -0.78ms (Lower)
sa-east-1       |    82.54ms    |      82.63ms   |  +0.09ms (Higher)

=== SERVICE TYPE LOCAL TIME ANALYSIS ===
Service Type Local Time Statistics (sample hours):

Anycast:
  Peak hour: 2:00 (3.10ms)
  Low hour:  5:00 (2.77ms)
  Peak/Low ratio: 1.12x

Unicast:
  Peak hour: 20:00 (152.75ms)
  Low hour:  10:00 (150.70ms)
  Peak/Low ratio: 1.01x

Unicast CDN:
  Peak hour: 17:00 (146.22ms)
  Low hour:  2:00 (144.39ms)
  Peak/Low ratio: 1.01x

=== WEEKEND vs WEEKDAY ANALYSIS (LOCAL TIME) ===
Weekend Effect Analysis (Local Time):
Region-Service                    | Weekday    | Weekend    | Weekend Effect
--------------------------------------------------------------------------------
us-west-1-Anycast      |    1.96ms |    1.92ms |  -2.0% (Lower)
us-west-1-Unicast      |   81.18ms |   81.41ms |  +0.3% (Higher)
eu-central-1-Anycast      |    1.31ms |    1.29ms |  -1.1% (Lower)
eu-central-1-Unicast      |   76.46ms |   76.36ms |  -0.1% (Lower)
ap-northeast-1-Anycast      |    2.53ms |    2.52ms |  -0.2% (Lower)
ap-northeast-1-Unicast      |  173.65ms |  173.44ms |  -0.1% (Lower)

=== TIMEZONE NORMALIZATION COMPLETE ===
Key Findings:
✓ Converted all measurements to regional local times
✓ Analyzed business hours vs off-hours impact
✓ Compared UTC vs local temporal patterns
✓ Examined weekend effects using local time

This analysis reveals whether our 'stable temporal performance'
was masking important local usage pattern effects.

