import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12
})

print("=== PHASE 3: PERFORMANCE ANALYSIS AND VISUALIZATIONS ===")
print()

# Load datasets
df_ipv4 = pd.read_parquet("../data/IPv4.parquet")
df_ipv6 = pd.read_parquet("../data/IPv6.parquet")

# Service Provider Mapping
service_mapping = {
    '1.1.1.1': 'Cloudflare DNS', '8.8.8.8': 'Google DNS', '9.9.9.9': 'Quad9 DNS',
    '2.16.241.219': 'Akamai CDN', '104.16.123.96': 'Cloudflare CDN',
    '193.99.144.85': 'Heise', '169.229.128.134': 'Berkeley NTP',
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

# Add service columns and ensure datetime
for df in [df_ipv4, df_ipv6]:
    df['service_provider'] = df['dst'].map(service_mapping)
    df['service_type'] = df['service_provider'].map(service_type_mapping)
    df['utctime'] = pd.to_datetime(df['utctime'])

df_ipv4['protocol'] = 'IPv4'
df_ipv6['protocol'] = 'IPv6'

print("=== LATENCY EXTRACTION AND ANALYSIS ===")

def extract_detailed_latency_data(df, protocol_name):
    """Extract comprehensive latency data from hubs arrays"""
    latency_data = []
    hop_by_hop_data = []
    
    for idx, row in df.iterrows():
        hubs = row['hubs']
        if hubs is not None and len(hubs) > 0:
            # End-to-end latency (final hop)
            final_hop = hubs[-1]
            end_latency = final_hop.get('Avg', 0) if final_hop.get('Avg') is not None else 0
            
            # Calculate additional metrics
            all_latencies = [hop.get('Avg', 0) for hop in hubs if hop.get('Avg') is not None and hop.get('Avg') > 0]
            all_best = [hop.get('Best', 0) for hop in hubs if hop.get('Best') is not None and hop.get('Best') > 0]
            all_worst = [hop.get('Wrst', 0) for hop in hubs if hop.get('Wrst') is not None and hop.get('Wrst') > 0]
            all_jitter = [hop.get('Javg', 0) for hop in hubs if hop.get('Javg') is not None and hop.get('Javg') > 0]
            all_loss = [hop.get('Loss%', 0) for hop in hubs if hop.get('Loss%') is not None]
            
            # Store measurement-level data
            latency_data.append({
                'protocol': protocol_name,
                'region': row['region'],
                'service_provider': row['service_provider'],
                'service_type': row['service_type'],
                'timestamp': row['utctime'],
                'hop_count': len(hubs),
                'end_to_end_latency': end_latency,
                'path_avg_latency': np.mean(all_latencies) if all_latencies else 0,
                'path_best_latency': np.mean(all_best) if all_best else 0,
                'path_worst_latency': np.mean(all_worst) if all_worst else 0,
                'path_avg_jitter': np.mean(all_jitter) if all_jitter else 0,
                'path_avg_loss': np.mean(all_loss) if all_loss else 0,
                'latency_per_hop': end_latency / len(hubs) if len(hubs) > 0 else 0,
                'measurement_id': row['id']
            })
            
            # Store hop-by-hop data
            for hop_idx, hop in enumerate(hubs):
                avg_lat = hop.get('Avg', 0)
                if avg_lat is not None and avg_lat > 0:  # Only include responsive hops
                    hop_by_hop_data.append({
                        'protocol': protocol_name,
                        'region': row['region'],
                        'service_provider': row['service_provider'],
                        'service_type': row['service_type'],
                        'hop_position': hop_idx + 1,
                        'hop_latency': avg_lat,
                        'hop_best': hop.get('Best', 0),
                        'hop_worst': hop.get('Wrst', 0),
                        'hop_jitter': hop.get('Javg', 0),
                        'hop_loss': hop.get('Loss%', 0),
                        'hop_host': hop.get('host', ''),
                        'hop_asn': hop.get('ASN', ''),
                        'measurement_id': row['id']
                    })
    
    return pd.DataFrame(latency_data), pd.DataFrame(hop_by_hop_data)

print("Extracting latency data...")
latency_ipv4, hops_ipv4 = extract_detailed_latency_data(df_ipv4, 'IPv4')
latency_ipv6, hops_ipv6 = extract_detailed_latency_data(df_ipv6, 'IPv6')

latency_combined = pd.concat([latency_ipv4, latency_ipv6], ignore_index=True)
hops_combined = pd.concat([hops_ipv4, hops_ipv6], ignore_index=True)

print(f"✓ Extracted latency data for {len(latency_combined):,} measurements")
print(f"✓ Extracted hop-by-hop data for {len(hops_combined):,} responsive hops")
print()

print("=== LATENCY DISTRIBUTION ANALYSIS ===")

# End-to-end latency statistics
print("End-to-End Latency Statistics by Service Provider and Protocol:")
latency_stats = latency_combined.groupby(['protocol', 'service_provider'])['end_to_end_latency'].agg([
    'count', 'mean', 'std', 'median', 'min', 'max',
    lambda x: np.percentile(x, 95)  # 95th percentile
]).round(2)
latency_stats.columns = ['count', 'mean', 'std', 'median', 'min', 'max', 'p95']
print(latency_stats)
print()

# Latency per hop analysis
print("Latency Per Hop Statistics:")
latency_per_hop_stats = latency_combined[latency_combined['latency_per_hop'] > 0].groupby([
    'protocol', 'service_type'])['latency_per_hop'].agg([
    'count', 'mean', 'std', 'median'
]).round(2)
print(latency_per_hop_stats)
print()

print("=== REGIONAL PERFORMANCE ANALYSIS ===")

# Regional latency analysis
regional_performance = latency_combined.groupby(['protocol', 'region']).agg({
    'end_to_end_latency': ['mean', 'std', 'median'],
    'hop_count': ['mean'],
    'latency_per_hop': ['mean']
}).round(2)
regional_performance.columns = ['_'.join(col).strip() for col in regional_performance.columns]
print("Regional Performance Summary:")
print(regional_performance)
print()

print("=== TEMPORAL ANALYSIS ===")

# Add time-based columns for temporal analysis
latency_combined['hour'] = latency_combined['timestamp'].dt.hour
latency_combined['day_of_week'] = latency_combined['timestamp'].dt.dayofweek
latency_combined['day_name'] = latency_combined['timestamp'].dt.day_name()
latency_combined['date'] = latency_combined['timestamp'].dt.date

# Hourly patterns
print("Hourly Latency Patterns (Mean End-to-End Latency by Hour):")
hourly_patterns = latency_combined.groupby(['protocol', 'service_type', 'hour'])['end_to_end_latency'].mean().unstack('hour').round(2)
print(hourly_patterns)
print()

# Daily patterns
print("Daily Latency Patterns (Mean End-to-End Latency by Day):")
daily_patterns = latency_combined.groupby(['protocol', 'service_type', 'day_name'])['end_to_end_latency'].mean().unstack('day_name').round(2)
print(daily_patterns)
print()

print("=== SERVICE PROVIDER STABILITY ANALYSIS ===")

# Calculate coefficient of variation (CV) as stability metric
def calculate_stability_metrics(group):
    return pd.Series({
        'mean_latency': group['end_to_end_latency'].mean(),
        'std_latency': group['end_to_end_latency'].std(),
        'cv_latency': group['end_to_end_latency'].std() / group['end_to_end_latency'].mean() * 100,
        'mean_hop_count': group['hop_count'].mean(),
        'std_hop_count': group['hop_count'].std(),
        'cv_hop_count': group['hop_count'].std() / group['hop_count'].mean() * 100 if group['hop_count'].mean() > 0 else 0
    })

stability_analysis = latency_combined.groupby(['protocol', 'service_provider']).apply(calculate_stability_metrics).round(2)
print("Service Provider Stability Analysis (CV = Coefficient of Variation %):")
print(stability_analysis)
print()

print("=== HOP-BY-HOP LATENCY ANALYSIS ===")

# Latency accumulation by hop position
print("Average Latency by Hop Position (first 10 hops):")
hop_position_latency = hops_combined[hops_combined['hop_position'] <= 10].groupby([
    'protocol', 'service_type', 'hop_position'
])['hop_latency'].mean().unstack('hop_position').round(2)
print(hop_position_latency)
print()

print("=== PERFORMANCE ANOMALY DETECTION ===")

# Identify high-latency measurements (> 1000ms)
high_latency_threshold = 1000  # 1 second
anomalies = latency_combined[latency_combined['end_to_end_latency'] > high_latency_threshold]

print(f"High-Latency Measurements (>{high_latency_threshold}ms):")
print(f"Total anomalies: {len(anomalies):,} ({len(anomalies)/len(latency_combined)*100:.2f}%)")

if len(anomalies) > 0:
    anomaly_summary = anomalies.groupby(['protocol', 'service_provider']).agg({
        'end_to_end_latency': ['count', 'mean', 'max']
    }).round(2)
    anomaly_summary.columns = ['count', 'mean_latency', 'max_latency']
    print("Anomalies by Service Provider:")
    print(anomaly_summary)
print()

print("=== CONTINENTAL INFRASTRUCTURE QUALITY ===")

# Map regions to continents
region_to_continent = {
    'us-west-1': 'North America',
    'ca-central-1': 'North America', 
    'eu-central-1': 'Europe',
    'eu-north-1': 'Europe',
    'af-south-1': 'Africa',
    'ap-east-1': 'Asia-Pacific',
    'ap-south-1': 'Asia-Pacific',
    'ap-northeast-1': 'Asia-Pacific',
    'ap-southeast-2': 'Asia-Pacific',
    'sa-east-1': 'South America'
}

latency_combined['continent'] = latency_combined['region'].map(region_to_continent)

continental_performance = latency_combined.groupby(['protocol', 'continent', 'service_type']).agg({
    'end_to_end_latency': ['mean', 'std', 'median'],
    'hop_count': ['mean'],
    'latency_per_hop': ['mean']
}).round(2)

print("Continental Infrastructure Quality:")
print(continental_performance)
print()

# Save all results for visualization
print("=== SAVING RESULTS ===")
latency_combined.to_csv('../results/latency_analysis.csv', index=False)
hops_combined.to_csv('../results/hop_by_hop_analysis.csv', index=False)
stability_analysis.to_csv('../results/stability_analysis.csv')

print("✓ Latency analysis saved to ../results/latency_analysis.csv")
print("✓ Hop-by-hop analysis saved to ../results/hop_by_hop_analysis.csv") 
print("✓ Stability analysis saved to ../results/stability_analysis.csv")
print()

print("=== PHASE 3 COMPLETE ===")
print("Ready to proceed with Phase 4: Advanced Analysis and Visualizations")