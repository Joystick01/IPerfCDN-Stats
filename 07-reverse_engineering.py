import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

print("=== FIXED ANYCAST INFRASTRUCTURE REVERSE ENGINEERING ===")
print()

# Load raw data
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

for df in [df_ipv4, df_ipv6]:
    df['service_provider'] = df['dst'].map(service_mapping)
    df['service_type'] = df['service_provider'].map(service_type_mapping)

df_ipv4['protocol'] = 'IPv4'
df_ipv6['protocol'] = 'IPv6'

def extract_anycast_signatures(df, service_name, protocol_name):
    """Extract signatures that can identify different anycast server instances"""
    
    service_data = df[df['service_provider'] == service_name].copy()
    signatures = []
    
    for _, row in service_data.iterrows():
        hubs = row['hubs']
        if hubs is not None and len(hubs) > 0:
            # Get final hop info
            final_hop = hubs[-1]
            final_latency = final_hop.get('Avg', 0)
            final_host = final_hop.get('host', '')
            final_asn = final_hop.get('ASN', '')
            
            # Get penultimate hop info (key for anycast identification)
            penultimate_hop = hubs[-2] if len(hubs) > 1 else {}
            penult_host = penultimate_hop.get('host', '')
            penult_asn = penultimate_hop.get('ASN', '')
            penult_latency = penultimate_hop.get('Avg', 0)
            
            # Create AS path signature (important for routing identification)
            as_path = []
            for hop in hubs:
                asn = hop.get('ASN', '').replace('AS', '')
                if asn and asn != '???':
                    as_path.append(asn)
            as_path_str = '→'.join(as_path) if as_path else ''
            
            signatures.append({
                'measurement_id': row['id'],
                'region': row['region'],
                'protocol': protocol_name,
                'service': service_name,
                'final_latency': final_latency,
                'final_host': final_host,
                'final_asn': final_asn,
                'penult_host': penult_host,
                'penult_asn': penult_asn,
                'penult_latency': penult_latency,
                'hop_count': len(hubs),
                'as_path': as_path_str,
                'as_path_length': len(as_path)
            })
    
    return pd.DataFrame(signatures)

print("=== METHOD 1: PENULTIMATE HOP ANALYSIS ===")

def analyze_penultimate_hops(service_name):
    """Analyze penultimate hops to identify anycast server locations"""
    
    print(f"\n--- {service_name} Penultimate Hop Analysis ---")
    
    # Combine IPv4 and IPv6 data
    ipv4_sigs = extract_anycast_signatures(df_ipv4, service_name, 'IPv4')
    ipv6_sigs = extract_anycast_signatures(df_ipv6, service_name, 'IPv6')
    all_sigs = pd.concat([ipv4_sigs, ipv6_sigs], ignore_index=True)
    
    if len(all_sigs) == 0:
        print(f"  No data found for {service_name}")
        return None
    
    # Method 1A: Group by penultimate host
    penult_hosts = all_sigs[all_sigs['penult_host'] != ''].copy()
    
    if len(penult_hosts) > 0:
        # Clean up hostnames to identify unique servers
        penult_hosts['clean_host'] = penult_hosts['penult_host'].apply(lambda x: x.split('(')[0].strip())
        
        # Count unique penultimate hosts per region
        host_region_matrix = penult_hosts.groupby(['region', 'clean_host']).size().unstack(fill_value=0)
        
        print(f"  Unique penultimate hosts found: {len(host_region_matrix.columns)}")
        print(f"  Top 5 penultimate hosts:")
        top_hosts = penult_hosts['clean_host'].value_counts().head(5)
        for host, count in top_hosts.items():
            print(f"    {host}: {count} measurements")
        
        # Method 1B: Look for patterns suggesting different servers
        # If the same penultimate host serves multiple distant regions, it suggests fewer servers
        # If different penultimate hosts serve nearby regions, it suggests more servers
        
        region_to_continent = {
            'us-west-1': 'North America', 'ca-central-1': 'North America', 
            'eu-central-1': 'Europe', 'eu-north-1': 'Europe', 'af-south-1': 'Africa',
            'ap-east-1': 'Asia-Pacific', 'ap-south-1': 'Asia-Pacific', 
            'ap-northeast-1': 'Asia-Pacific', 'ap-southeast-2': 'Asia-Pacific',
            'sa-east-1': 'South America'
        }
        
        penult_hosts['continent'] = penult_hosts['region'].map(region_to_continent)
        
        # Check if same penultimate host serves multiple continents (indicates fewer servers)
        host_continent_spread = penult_hosts.groupby('clean_host')['continent'].nunique()
        multi_continent_hosts = host_continent_spread[host_continent_spread > 1]
        
        print(f"  Hosts serving multiple continents: {len(multi_continent_hosts)}")
        if len(multi_continent_hosts) > 0:
            print("    These suggest regional server consolidation:")
            for host, continent_count in multi_continent_hosts.head(3).items():
                continents = penult_hosts[penult_hosts['clean_host'] == host]['continent'].unique()
                print(f"    {host}: serves {continent_count} continents ({', '.join(continents)})")
    
    # Method 1C: ASN-based clustering
    print(f"\n  ASN-based analysis:")
    penult_asns = all_sigs[all_sigs['penult_asn'] != '']['penult_asn'].value_counts()
    print(f"  Unique penultimate ASNs: {len(penult_asns)}")
    print(f"  Top 3 penultimate ASNs:")
    for asn, count in penult_asns.head(3).items():
        print(f"    {asn}: {count} measurements")
    
    return all_sigs

# Analyze each anycast service
anycast_services = ['Cloudflare DNS', 'Google DNS', 'Quad9 DNS', 'Cloudflare CDN']
service_signatures = {}

for service in anycast_services:
    signatures = analyze_penultimate_hops(service)
    if signatures is not None:
        service_signatures[service] = signatures

print()
print("=== METHOD 2: LATENCY-BASED GEOGRAPHIC CLUSTERING ===")

def geographic_latency_clustering(signatures, service_name):
    """Use latency patterns to infer anycast server locations"""
    
    print(f"\n--- {service_name} Geographic Latency Clustering ---")
    
    # Create latency matrix: regions vs measurements
    regional_latencies = signatures.groupby('region')['final_latency'].agg(['mean', 'std', 'count'])
    
    print(f"  Regional latency statistics:")
    for region, stats in regional_latencies.iterrows():
        print(f"    {region}: {stats['mean']:.2f}±{stats['std']:.2f}ms (n={stats['count']})")
    
    # Look for latency clusters that suggest server locations
    # Similar latencies from geographically close regions suggest shared servers
    latency_values = regional_latencies['mean'].values
    
    # Use hierarchical clustering on latency values
    if len(latency_values) > 3:  # Need minimum data points
        try:
            linkage_matrix = linkage(latency_values.reshape(-1, 1), method='ward')
            
            # Try different numbers of clusters (2 to 6 servers)
            possible_clusters = range(2, min(7, len(latency_values)))
            best_cluster_count = None
            best_silhouette = -1
            
            from sklearn.metrics import silhouette_score
            
            for n_clusters in possible_clusters:
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(latency_values.reshape(-1, 1), cluster_labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_cluster_count = n_clusters
            
            if best_cluster_count:
                final_clusters = fcluster(linkage_matrix, best_cluster_count, criterion='maxclust')
                
                print(f"  Optimal cluster count (servers): {best_cluster_count}")
                print(f"  Silhouette score: {best_silhouette:.3f}")
                print(f"  Regional server assignments:")
                
                for i, (region, cluster) in enumerate(zip(regional_latencies.index, final_clusters)):
                    print(f"    {region}: Server {cluster} ({latency_values[i]:.2f}ms)")
                
                return best_cluster_count
            else:
                print(f"  Could not determine optimal clustering")
                return None
                
        except Exception as e:
            print(f"  Clustering failed: {e}")
            return None
    else:
        print(f"  Insufficient data for clustering")
        return None

# Apply geographic clustering to each service
cluster_estimates = {}
for service, signatures in service_signatures.items():
    estimated_servers = geographic_latency_clustering(signatures, service)
    if estimated_servers:
        cluster_estimates[service] = estimated_servers

print()
print("=== METHOD 3: PATH SIMILARITY ANALYSIS ===")

def analyze_path_similarity(signatures, service_name):
    """Analyze AS path similarity to identify server groups"""
    
    print(f"\n--- {service_name} Path Similarity Analysis ---")
    
    # Group by AS path to find common routing patterns
    path_groups = signatures.groupby('as_path').agg({
        'region': lambda x: list(x.unique()),
        'final_latency': 'mean',
        'hop_count': 'mean',
        'measurement_id': 'count'
    }).rename(columns={'measurement_id': 'count'})
    
    # Sort by frequency
    path_groups = path_groups.sort_values('count', ascending=False)
    
    print(f"  Total unique AS paths: {len(path_groups)}")
    print(f"  Most common paths (top 5):")
    
    for i, (path, stats) in enumerate(path_groups.head(5).iterrows()):
        if path:  # Skip empty paths
            regions_served = ', '.join(stats['region'])
            print(f"    Path {i+1}: {stats['count']} measurements")
            print(f"      Serves regions: {regions_served}")
            print(f"      Avg latency: {stats['final_latency']:.2f}ms")
            print(f"      AS path: {path[:100]}{'...' if len(path) > 100 else ''}")
            print()
    
    # Estimate server count based on path diversity
    # More unique paths typically indicate more distributed servers
    major_paths = path_groups[path_groups['count'] >= len(signatures) * 0.05]  # Paths serving >5% of traffic
    
    print(f"  Major routing paths (>5% traffic): {len(major_paths)}")
    
    return len(major_paths)

# Apply path analysis
path_estimates = {}
for service, signatures in service_signatures.items():
    estimated_servers = analyze_path_similarity(signatures, service)
    path_estimates[service] = estimated_servers

print()
print("=== FINAL ANYCAST SERVER ESTIMATES ===")

# Combine all methods for final estimate
final_estimates = {}

print("Service Provider Anycast Server Estimates:")
print("=" * 60)

for service in anycast_services:
    if service in service_signatures:
        signatures = service_signatures[service]
        
        print(f"\n{service}:")
        print(f"  Total measurements analyzed: {len(signatures):,}")
        
        # Method results
        penult_hosts = signatures[signatures['penult_host'] != '']['penult_host'].nunique()
        penult_asns = signatures[signatures['penult_asn'] != '']['penult_asn'].nunique()
        
        print(f"  Unique penultimate hosts: {penult_hosts}")
        print(f"  Unique penultimate ASNs: {penult_asns}")
        
        if service in cluster_estimates:
            print(f"  Latency-based clustering: {cluster_estimates[service]} servers")
        
        if service in path_estimates:
            print(f"  Major routing paths: {path_estimates[service]}")
        
        # Estimate based on multiple indicators
        indicators = []
        if penult_hosts > 0:
            indicators.append(min(penult_hosts, 8))  # Cap at reasonable max
        if service in cluster_estimates:
            indicators.append(cluster_estimates[service])
        if service in path_estimates and path_estimates[service] > 1:
            indicators.append(min(path_estimates[service], 6))
        
        if indicators:
            # Take median of indicators as final estimate
            final_estimate = int(np.median(indicators))
            final_estimates[service] = final_estimate
            print(f"  → ESTIMATED SERVERS: {final_estimate}")
        else:
            print(f"  → INSUFFICIENT DATA for estimate")

print()
print("=== METHODOLOGY VALIDATION ===")
print("Key improvements over previous approach:")
print("1. Uses actual network infrastructure indicators (penultimate hops)")
print("2. Applies geographic latency clustering with validation")
print("3. Analyzes routing path diversity patterns")
print("4. Combines multiple independent methods")
print("5. No longer assumes one server per measurement region")
print()
print("Note: These estimates represent minimum server counts.")
print("Actual deployments may have additional servers not detectable from our vantage points.")