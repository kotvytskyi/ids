import pandas as pd
import numpy as np

def load_data(file_path, label):
    df = pd.read_csv(file_path)
    df['label'] = label
    df['requests_rate'] = 1
    return df

def process_raw_data(dataset):
    dataset['calculated_bwd_avg_segment_size'] = (
        dataset['bwd_total_payload_bytes'] / (dataset['bwd_packets_count'] + 1)
    )

def post_process_aggregations(dataset):
    dataset['requests_rate_src_port'] = dataset['requests_rate'] * dataset['src_ip']
    dataset['requests_rate_dst_port'] = dataset['requests_rate'] * dataset['dst_ip']
    return ['requests_rate_src_port', 'requests_rate_dst_port']

def prepare_aggregated_dataset(df, aggregations, frequency='1s', include_labels=False):
    if include_labels:
        aggregations['label'] = lambda x: 1 if x.eq('benign').all() else -1

    df_agg = df.resample(frequency).agg(aggregations)

    if include_labels:
        aggregations.pop('label', None)

    return df_agg

def engineer_portscan_features(df):
    if 'dst_port' in df.columns and 'requests_rate' in df.columns:
        df['unique_dst_port_ratio'] = df['dst_port'] / (df['requests_rate'] + 1)
    if 'src_port' in df.columns and 'requests_rate' in df.columns:
        df['unique_src_port_ratio'] = df['src_port'] / (df['requests_rate'] + 1)
    return df

def engineer_extra_features(df):
    if 'fwd_packets_count' in df.columns and 'packets_count' in df.columns:
        df['fwd_packet_fraction'] = df['fwd_packets_count'] / (df['packets_count'] + 1)
    else:
        df['fwd_packet_fraction'] = 0
    
    if 'syn_flag_counts' in df.columns and 'packets_count' in df.columns:
        df['frac_syn'] = df['syn_flag_counts'] / (df['packets_count'] + 1)
    else:
        df['frac_syn'] = 0
    
    if 'rst_flag_counts' in df.columns and 'packets_count' in df.columns:
        df['frac_rst'] = df['rst_flag_counts'] / (df['packets_count'] + 1)
    else:
        df['frac_rst'] = 0
    
    if 'total_header_bytes' in df.columns and 'total_payload_bytes' in df.columns:
        df['header_to_payload_ratio'] = (
            df['total_header_bytes'] / (df['total_payload_bytes'] + 1)
        )
    else:
        df['header_to_payload_ratio'] = 0
    
    return df

def engineer_rolling_features(df, window='5s'):
    df.sort_index(inplace=True)
    
    if 'syn_flag_counts' in df.columns:
        df['rolling_syn_5s'] = (
            df['syn_flag_counts']
            .rolling(window=window, min_periods=1)
            .sum()
        )
    else:
        df['rolling_syn_5s'] = 0

    if 'packets_count' in df.columns:
        df['rolling_packets_mean_5s'] = (
            df['packets_count']
            .rolling(window=window, min_periods=1)
            .mean()
        )
    else:
        df['rolling_packets_mean_5s'] = 0

    if 'bytes_rate' in df.columns:
        df['rolling_bytes_rate_std_5s'] = (
            df['bytes_rate']
            .rolling(window=window, min_periods=1)
            .std()
        )
    else:
        df['rolling_bytes_rate_std_5s'] = 0

    return df

def transform_aggregated_dataset(df, window='5s'):
    post_process_aggregations(df)
    engineer_portscan_features(df)

    for col in ['src_port', 'dst_port']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    engineer_extra_features(df)
    engineer_rolling_features(df, window=window)

    return df

def prepare_dataset(
    df,
    aggregations,
    frequency='1s',
    include_labels=False,
    window='5s',
    filter_subnet=False
): 
    aggregations = aggregations.copy()
    
    df['datetime'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    if filter_subnet and 'dst_ip' in df.columns:
        df = df[df['dst_ip'].str.startswith('192.168', na=False)]

    if 'protocol' in df.columns:
        df = pd.get_dummies(df, columns=['protocol'], drop_first=True, dtype=int)
        # Ensure aggregator counts these new protocol columns
        for col in df.columns:
            if col.startswith('protocol_') and (col not in aggregations):
                aggregations[col] = 'count'

    process_raw_data(df)

    df_agg = prepare_aggregated_dataset(
        df,
        aggregations,
        frequency=frequency,
        include_labels=include_labels
    )

    transform_aggregated_dataset(df_agg, window=window)

    return df_agg

def safe_agg(func, default=0):
    def wrapper(series):
        if not series.empty:
            return func(series)
        else:
            return default
    return wrapper

def combine_datasets(s1, s2):
    combined_data = pd.concat([s1, s2])
    combined_data.sort_index(inplace=True)
    return combined_data

def allign_columns(s1, s2):
    all_columns = set(s1.columns).union(set(s2.columns))

    s1a = s1.reindex(columns=all_columns, fill_value=0)
    s2a = s2.reindex(columns=all_columns, fill_value=0)

    return s1a, s2a