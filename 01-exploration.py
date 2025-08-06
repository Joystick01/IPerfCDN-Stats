import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=== PHASE 1: DATA EXPLORATION AND QUALITY ASSESSMENT ===")
print()

# Load datasets
print("Loading datasets...")
try:
    df_ipv4 = pd.read_parquet("../data/IPv4.parquet")
    df_ipv6 = pd.read_parquet("../data/IPv6.parquet")
    print(f"✓ IPv4 dataset loaded: {len(df_ipv4):,} records")
    print(f"✓ IPv6 dataset loaded: {len(df_ipv6):,} records")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    exit()

print()

# Service Provider Mapping
service_mapping = {
    # IPv4 Addresses
    '1.1.1.1': 'Cloudflare DNS',
    '8.8.8.8': 'Google DNS', 
    '9.9.9.9': 'Quad9 DNS',
    '2.16.241.219': 'Akamai CDN',
    '104.16.123.96': 'Cloudflare CDN',
    '193.99.144.85': 'Heise',
    '169.229.128.134': 'Berkeley NTP',
    # IPv6 Addresses  
    '2606:4700:4700::1111': 'Cloudflare DNS',
    '2001:4860:4860::8888': 'Google DNS',
    '2620:fe::fe:9': 'Quad9 DNS', 
    '2a02:26f0:3500:1b::1724:a393': 'Akamai CDN',
    '2606:4700::6810:7b60': 'Cloudflare CDN',
    '2a02:2e0:3fe:1001:7777:772e:2:85': 'Heise',
    '2607:f140:ffff:8000:0:8006:0:a': 'Berkeley NTP'
}

# Service type mapping
service_type_mapping = {
    'Cloudflare DNS': 'Anycast',
    'Google DNS': 'Anycast', 
    'Quad9 DNS': 'Anycast',
    'Cloudflare CDN': 'Anycast',
    'Akamai CDN': 'Unicast CDN',
    'Heise': 'Unicast',
    'Berkeley NTP': 'Unicast'
}

# Add service provider and type columns
df_ipv4['service_provider'] = df_ipv4['dst'].map(service_mapping)
df_ipv4['service_type'] = df_ipv4['service_provider'].map(service_type_mapping)
df_ipv4['protocol'] = 'IPv4'

df_ipv6['service_provider'] = df_ipv6['dst'].map(service_mapping)
df_ipv6['service_type'] = df_ipv6['service_provider'].map(service_type_mapping)
df_ipv6['protocol'] = 'IPv6'

print("Service Provider Mapping:")
for ip, provider in service_mapping.items():
    service_type = service_type_mapping[provider]
    print(f"  {ip:<35} → {provider:<15} ({service_type})")
print()

# Basic dataset information
print("=== DATASET OVERVIEW ===")
print(f"IPv4 Records: {len(df_ipv4):,}")
print(f"IPv6 Records: {len(df_ipv6):,}")
print(f"Total Records: {len(df_ipv4) + len(df_ipv6):,}")
print()

print("IPv4 Dataset Structure:")
print(df_ipv4.dtypes)
print()

print("IPv6 Dataset Structure:")  
print(df_ipv6.dtypes)
print()

# Time range analysis
print("=== TEMPORAL COVERAGE ===")
for name, df in [("IPv4", df_ipv4), ("IPv6", df_ipv6)]:
    if 'utctime' in df.columns:
        # Convert utctime if needed
        if df['utctime'].dtype == 'object':
            df['utctime'] = pd.to_datetime(df['utctime'])
        
        print(f"{name} Time Range:")
        print(f"  Start: {df['utctime'].min()}")
        print(f"  End:   {df['utctime'].max()}")
        print(f"  Duration: {df['utctime'].max() - df['utctime'].min()}")
        print()

# Regional coverage
print("=== REGIONAL COVERAGE ===")
regions_ipv4 = df_ipv4['region'].value_counts().sort_index()
regions_ipv6 = df_ipv6['region'].value_counts().sort_index()

print("IPv4 Measurements per Region:")
for region, count in regions_ipv4.items():
    print(f"  {region:<15}: {count:>7,} measurements")

print()
print("IPv6 Measurements per Region:")
for region, count in regions_ipv6.items():
    print(f"  {region:<15}: {count:>7,} measurements")
print()

# Service Provider Coverage
print("=== SERVICE PROVIDER COVERAGE ===")
print("IPv4 Measurements per Service Provider:")
service_ipv4 = df_ipv4['service_provider'].value_counts()
for provider, count in service_ipv4.items():
    service_type = service_type_mapping[provider]
    print(f"  {provider:<15}: {count:>7,} measurements ({service_type})")

print()
print("IPv6 Measurements per Service Provider:")
service_ipv6 = df_ipv6['service_provider'].value_counts()
for provider, count in service_ipv6.items():
    service_type = service_type_mapping[provider]
    print(f"  {provider:<15}: {count:>7,} measurements ({service_type})")
print()

# Data Quality Assessment
print("=== DATA QUALITY ASSESSMENT ===")

def analyze_hubs_data(df, protocol_name):
    print(f"{protocol_name} Hubs Analysis:")
    
    # Check for empty hubs
    # Safe way to check for empty arrays without triggering pandas array comparison error
    empty_hubs = 0
    total_hop_count = 0
    hop_counts = []
    
    for idx, hubs in enumerate(df['hubs'].values):
        if hubs is None or len(hubs) == 0:
            empty_hubs += 1
        else:
            hop_count = len(hubs)
            hop_counts.append(hop_count)
            total_hop_count += hop_count
    
    print(f"  Records with empty hubs: {empty_hubs:,}")
    print(f"  Records with hop data: {len(hop_counts):,}")
    if hop_counts:
        print(f"  Hop count range: {min(hop_counts)} - {max(hop_counts)}")
        print(f"  Mean hop count: {np.mean(hop_counts):.2f}")
        print(f"  Total hops analyzed: {total_hop_count:,}")
    print()

analyze_hubs_data(df_ipv4, "IPv4")
analyze_hubs_data(df_ipv6, "IPv6")

# Missing data analysis
print("=== MISSING DATA ANALYSIS ===")
print("IPv4 Missing Data:")
ipv4_missing = df_ipv4.isnull().sum()
for col, missing in ipv4_missing.items():
    if missing > 0:
        print(f"  {col}: {missing:,} ({missing/len(df_ipv4)*100:.2f}%)")

print()
print("IPv6 Missing Data:")
ipv6_missing = df_ipv6.isnull().sum()
for col, missing in ipv6_missing.items():
    if missing > 0:
        print(f"  {col}: {missing:,} ({missing/len(df_ipv6)*100:.2f}%)")
print()

# Basic statistics
print("=== BASIC MEASUREMENT STATISTICS ===")
for name, df in [("IPv4", df_ipv4), ("IPv6", df_ipv6)]:
    print(f"{name} Packet Size Statistics:")
    if 'psize' in df.columns:
        print(f"  Mean: {df['psize'].mean():.2f} bytes")
        print(f"  Range: {df['psize'].min()} - {df['psize'].max()} bytes")
    
    print(f"{name} Tests per Measurement:")
    if 'tests' in df.columns:
        print(f"  Mean: {df['tests'].mean():.2f}")
        print(f"  Range: {df['tests'].min()} - {df['tests'].max()}")
    print()

# Sample data inspection
print("=== SAMPLE DATA INSPECTION ===")
print("IPv4 Sample Record:")
sample_ipv4 = df_ipv4.head(1)
for col in sample_ipv4.columns:
    if col != 'hubs':  # Skip hubs for now due to complexity
        print(f"  {col}: {sample_ipv4[col].iloc[0]}")

print()
print("IPv6 Sample Record:")
sample_ipv6 = df_ipv6.head(1)
for col in sample_ipv6.columns:
    if col != 'hubs':  # Skip hubs for now due to complexity  
        print(f"  {col}: {sample_ipv6[col].iloc[0]}")

print()
print("Sample Hubs Structure (first record, first 3 hops):")
if len(df_ipv4) > 0 and df_ipv4['hubs'].iloc[0] is not None:
    sample_hubs = df_ipv4['hubs'].iloc[0]
    for i, hop in enumerate(sample_hubs[:3]):  # Show first 3 hops
        print(f"  Hop {i+1}: {dict(hop)}")

print()
print("=== PHASE 1 COMPLETE ===")
print("Ready to proceed with Phase 2: Infrastructure Characterization")