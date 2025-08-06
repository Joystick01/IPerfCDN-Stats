import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

print("=== PHASE 2: INFRASTRUCTURE CHARACTERIZATION ===")
print()

# Load datasets with service mappings (from Phase 1)
df_ipv4 = pd.read_parquet("../data/IPv4.parquet")
df_ipv6 = pd.read_parquet("../data/IPv6.parquet")

# Service Provider Mapping
service_mapping = {
    # IPv4 Addresses
    '1.1.1.1': 'Cloudflare DNS', '8.8.8.8': 'Google DNS', '9.9.9.9': 'Quad9 DNS',
    '2.16.241.219': 'Akamai CDN', '104.16.123.96': 'Cloudflare CDN',
    '193.99.144.85': 'Heise', '169.229.128.134': 'Berkeley NTP',
    # IPv6 Addresses  
    '2606:4700:4700::1111': 'Cloudflare DNS', '2001:4860:4860::8888': 'Google DNS',
    '2620:fe::fe:9': 'Quad9 DNS', '2a02:26f0:3500:1b::1724:a393': 'Akamai CDN',
    '2606:4700::6810:7b60': 'Cloudflare CDN', '2a02:2e0:3fe:1001:7777:772e:2:85': 'Heise',
    '2607:f140:ffff:8000:0:8006:0:a': 'Berkeley NTP'
}

service_type_mapping = {
    'Cloudflare DNS': 'Anycast', 'Google DNS': 'Anycast', 'Quad9 DNS': 'Anycast',
    'Cloudflare CDN': 'Anycast', 'Akamai CDN': 'Unicast CDN', 
    'Heise': 'Unicast', 'Berkeley NTP': 'Unicast'
}

# Add service columns
for df in [df_ipv4, df_ipv6]:
    df['service_provider'] = df['dst'].map(service_mapping)
    df['service_type'] = df['service_provider'].map(service_type_mapping)

df_ipv4['protocol'] = 'IPv4'
df_ipv6['protocol'] = 'IPv6'

print("=== HOP COUNT ANALYSIS ===")

def extract_hop_data(df, protocol_name):
    """Extract hop count and latency data from hubs arrays"""
    hop_data = []
    
    for idx, row in df.iterrows():
        hubs = row['hubs']
        if hubs is not None and len(hubs) > 0:
            hop_count = len(hubs)
            
            # Calculate end-to-end latency (average of final hop)
            final_hop = hubs[-1]
            end_to_end_latency = final_hop.get('Avg', 0) if final_hop.get('Avg') is not None else 0
            
            hop_data.append({
                'protocol': protocol_name,
                'region': row['region'],
                'service_provider': row['service_provider'],
                'service_type': row['service_type'],
                'hop_count': hop_count,
                'end_to_end_latency': end_to_end_latency,
                'measurement_id': row['id']
            })
    
    return pd.DataFrame(hop_data)

# Extract hop data for both protocols
print("Extracting hop count data...")
hop_data_ipv4 = extract_hop_data(df_ipv4, 'IPv4')
hop_data_ipv6 = extract_hop_data(df_ipv6, 'IPv6')
hop_data_combined = pd.concat([hop_data_ipv4, hop_data_ipv6], ignore_index=True)

print(f"✓ Extracted hop data for {len(hop_data_combined):,} measurements")
print()

# Hop count statistics by protocol
print("Hop Count Statistics by Protocol:")
hop_stats_by_protocol = hop_data_combined.groupby('protocol')['hop_count'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(2)
print(hop_stats_by_protocol)
print()

# Hop count statistics by service type
print("Hop Count Statistics by Service Type:")
hop_stats_by_service_type = hop_data_combined.groupby(['protocol', 'service_type'])['hop_count'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(2)
print(hop_stats_by_service_type)
print()

# Hop count statistics by service provider
print("Hop Count Statistics by Service Provider:")
hop_stats_by_provider = hop_data_combined.groupby(['protocol', 'service_provider'])['hop_count'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(2)
print(hop_stats_by_provider)
print()

print("=== REGIONAL HOP COUNT PATTERNS ===")
# Hop count by region and protocol
hop_stats_by_region = hop_data_combined.groupby(['protocol', 'region'])['hop_count'].agg([
    'count', 'mean', 'std', 'min', 'max'
]).round(2)
print("Hop Count Statistics by Region:")
print(hop_stats_by_region)
print()

print("=== ASN ANALYSIS ===")

def extract_asn_data(df, protocol_name):
    """Extract ASN information from hubs arrays"""
    asn_data = []
    path_data = []
    
    for idx, row in df.iterrows():
        hubs = row['hubs']
        if hubs is not None and len(hubs) > 0:
            asn_path = []
            
            for hop_idx, hop in enumerate(hubs):
                asn = hop.get('ASN', 'AS???')
                host = hop.get('host', '???')
                loss_pct = hop.get('Loss%', 0)
                avg_latency = hop.get('Avg', 0) if hop.get('Avg') is not None else 0
                
                # Clean up ASN format
                if asn and asn != 'AS???':
                    asn_clean = asn.replace('AS', '') if asn.startswith('AS') else asn
                    asn_path.append(asn_clean)
                    
                    asn_data.append({
                        'protocol': protocol_name,
                        'region': row['region'],
                        'service_provider': row['service_provider'],
                        'service_type': row['service_type'],
                        'asn': asn_clean,
                        'hop_position': hop_idx + 1,
                        'host': host,
                        'loss_percent': loss_pct,
                        'avg_latency': avg_latency,
                        'measurement_id': row['id']
                    })
            
            # Store AS path
            if asn_path:
                path_data.append({
                    'protocol': protocol_name,
                    'region': row['region'],
                    'service_provider': row['service_provider'],
                    'service_type': row['service_type'],
                    'as_path': ' -> '.join(asn_path),
                    'as_path_length': len(asn_path),
                    'measurement_id': row['id']
                })
    
    return pd.DataFrame(asn_data), pd.DataFrame(path_data)

print("Extracting ASN data...")
asn_data_ipv4, path_data_ipv4 = extract_asn_data(df_ipv4, 'IPv4')
asn_data_ipv6, path_data_ipv6 = extract_asn_data(df_ipv6, 'IPv6')

asn_data_combined = pd.concat([asn_data_ipv4, asn_data_ipv6], ignore_index=True)
path_data_combined = pd.concat([path_data_ipv4, path_data_ipv6], ignore_index=True)

print(f"✓ Extracted ASN data for {len(asn_data_combined):,} hop records")
print(f"✓ Extracted path data for {len(path_data_combined):,} measurements")
print()

# Most common ASNs
print("Most Common ASNs (Top 15):")
top_asns = asn_data_combined['asn'].value_counts().head(15)
for asn, count in top_asns.items():
    print(f"  AS{asn}: {count:,} occurrences")
print()

# ASN usage by protocol
print("ASN Usage by Protocol (Top 10 per protocol):")
for protocol in ['IPv4', 'IPv6']:
    print(f"\n{protocol} Top ASNs:")
    protocol_asns = asn_data_combined[asn_data_combined['protocol'] == protocol]['asn'].value_counts().head(10)
    for asn, count in protocol_asns.items():
        print(f"  AS{asn}: {count:,} occurrences")

print()

# ASN diversity analysis
print("=== ASN PATH DIVERSITY ===")
print("AS Path Length Statistics:")
path_length_stats = path_data_combined.groupby('protocol')['as_path_length'].agg([
    'count', 'mean', 'std', 'min', 'max', 'median'
]).round(2)
print(path_length_stats)
print()

# Unique AS paths per service provider
print("Unique AS Paths per Service Provider:")
unique_paths_by_provider = path_data_combined.groupby(['protocol', 'service_provider'])['as_path'].nunique()
print(unique_paths_by_provider)
print()

print("=== TIER 1 PROVIDER ANALYSIS ===")
# Known Tier 1 ASNs (major ones)
tier1_asns = {
    '174': 'Cogent', '701': 'Verizon', '1299': 'Telia', '3320': 'Deutsche Telekom',
    '3257': 'GTT', '6453': 'TATA', '6762': 'Sparkle', '1239': 'Sprint',
    '3356': 'Level3/Lumen', '5511': 'Orange', '2914': 'NTT', '6939': 'Hurricane Electric'
}

# Find Tier 1 usage
tier1_usage = asn_data_combined[asn_data_combined['asn'].isin(tier1_asns.keys())]

if len(tier1_usage) > 0:
    print("Tier 1 Provider Usage:")
    tier1_counts = tier1_usage.groupby(['protocol', 'asn']).size().reset_index(name='count')
    tier1_counts['provider_name'] = tier1_counts['asn'].map(tier1_asns)
    
    for protocol in ['IPv4', 'IPv6']:
        protocol_tier1 = tier1_counts[tier1_counts['protocol'] == protocol]
        if len(protocol_tier1) > 0:
            print(f"\n{protocol}:")
            for _, row in protocol_tier1.sort_values('count', ascending=False).iterrows():
                print(f"  AS{row['asn']} ({row['provider_name']}): {row['count']:,} occurrences")
else:
    print("No major Tier 1 providers found in paths")

print()

print("=== INFRASTRUCTURE QUALITY INDICATORS ===")

def calculate_quality_metrics(df, protocol_name):
    """Calculate infrastructure quality metrics from hubs data"""
    quality_metrics = []
    
    for idx, row in df.iterrows():
        hubs = row['hubs']
        if hubs is not None and len(hubs) > 0:
            # Calculate metrics across the path
            avg_latencies = [hop.get('Avg', 0) for hop in hubs if hop.get('Avg') is not None and hop.get('Avg') > 0]
            loss_rates = [hop.get('Loss%', 0) for hop in hubs if hop.get('Loss%') is not None]
            jitter_values = [hop.get('Javg', 0) for hop in hubs if hop.get('Javg') is not None and hop.get('Javg') > 0]
            
            if avg_latencies:  # Only if we have valid latency data
                quality_metrics.append({
                    'protocol': protocol_name,
                    'region': row['region'],
                    'service_provider': row['service_provider'],
                    'service_type': row['service_type'],
                    'path_avg_latency': np.mean(avg_latencies),
                    'path_max_latency': max(avg_latencies),
                    'path_avg_loss': np.mean(loss_rates) if loss_rates else 0,
                    'path_max_loss': max(loss_rates) if loss_rates else 0,
                    'path_avg_jitter': np.mean(jitter_values) if jitter_values else 0,
                    'responsive_hops': len([l for l in avg_latencies if l > 0]),
                    'total_hops': len(hubs),
                    'measurement_id': row['id']
                })
    
    return pd.DataFrame(quality_metrics)

print("Calculating infrastructure quality metrics...")
quality_ipv4 = calculate_quality_metrics(df_ipv4, 'IPv4')
quality_ipv6 = calculate_quality_metrics(df_ipv6, 'IPv6')
quality_combined = pd.concat([quality_ipv4, quality_ipv6], ignore_index=True)

print(f"✓ Calculated quality metrics for {len(quality_combined):,} measurements")
print()

# Quality by service type
print("Infrastructure Quality by Service Type:")
quality_by_type = quality_combined.groupby(['protocol', 'service_type']).agg({
    'path_avg_latency': ['mean', 'std', 'median'],
    'path_max_latency': ['mean', 'median'],
    'path_avg_loss': ['mean', 'max'],
    'path_avg_jitter': ['mean', 'median']
}).round(2)
print(quality_by_type)
print()

# Save intermediate results for visualization
print("Saving intermediate results...")
hop_data_combined.to_csv('../results/hop_count_analysis.csv', index=False)
asn_data_combined.to_csv('../results/asn_analysis.csv', index=False)
path_data_combined.to_csv('../results/path_analysis.csv', index=False)
quality_combined.to_csv('../results/quality_metrics.csv', index=False)

print("✓ Results saved to ../results/ directory")
print()
print("=== PHASE 2 COMPLETE ===")
print("Ready to proceed with Phase 3: Performance Analysis and Visualizations")