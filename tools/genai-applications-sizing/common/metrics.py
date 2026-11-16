# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Metrics calculation and reporting utilities.

This module provides functions for calculating performance metrics,
writing metrics to CSV and JSON files, and generating reports.
"""

import csv
import json
import math
import os
import re
import numpy as np



def calculate_metrics(latencies):
    """
    Calculate statistical metrics for a given dataset.

    Args:
        latencies: List of numerical values.

    Returns:
        tuple: (average, min, max, p99, p90, p75) values, or all None on error.
    """
    try:
        return (
            round(np.mean(latencies), 2),
            round(np.min(latencies), 2),
            round(np.max(latencies), 2),
            round(np.percentile(latencies, 99), 2),
            round(np.percentile(latencies, 90), 2),
            round(np.percentile(latencies, 75), 2),
        )
    except Exception as e:
        print(f"Failed to calculate metrics: {e}")
        return None, None, None, None, None, None


def write_metrics(metrics, report_dir):
    """
    Write metrics to a JSON file.
    
    Args:
        metrics: List of metric dictionaries.
        report_dir: Directory to save the JSON file.
        
    Returns:
        tuple: (latencies, input_tokens, output_tokens, ttfts, itls, tpss) lists.
    """
    latencies, input_tokens, output_tokens, ttfts, itls, tpss = [], [], [], [], [], []
    filename = os.path.join(report_dir, "chat_api_individual_metrics.json")
    try:
        with open(filename, "a") as file:
            for metric in metrics:
                json.dump(metric, file, indent=4)
                latencies.append(metric["LATENCY (ms)"])
                ttfts.append(metric["TTFT (ms)"])
                itls.append(metric["ITL (ms)"])
                tpss.append(metric["TPS"])
                input_tokens.append(metric["INPUT_TOKENS"])
                output_tokens.append(metric["OUTPUT_TOKENS"])
        return latencies, input_tokens, output_tokens, ttfts, itls, tpss
    except Exception as e:
        print(f"Failed to write metrics to file: {e}")
        return [], [], [], [], [], []


def write_chatqna_metrics_to_csv(report_dir, latencies, input_tokens, output_tokens, ttfts, itls, tpss, file_details=None):
    """
    Write metrics summary to CSV files (both detailed and WSF format).

    Args:
        report_dir: Directory to save the CSV files.
        latencies: List of latencies.
        input_tokens: List of input tokens.
        output_tokens: List of output tokens.
        ttfts: List of time-to-first-token values.
        itls: List of inter-token latencies.
        tpss: List of tokens per second.
        file_details: Details of the file including name and size (optional).
    """
    if file_details is None:
        file_details = {}
        
    summary_file = os.path.join(report_dir, "chat_api_summary_metrics.csv")
    wsf_file = os.path.join(report_dir, "chatqna_metrics_wsf.csv")
    
    try:
        throughput = len(latencies) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0
        
        # Calculate detailed metrics for summary file
        detailed_metrics = {
            "Request Latency (ms)": calculate_metrics(latencies),
            "Time to First Token (ms)": calculate_metrics(ttfts),
            "Inter Token Latency (ms)": calculate_metrics(itls),           
            "Tokens Per Second": calculate_metrics(tpss),
            "Input Tokens": calculate_metrics(input_tokens),
            "Output Tokens": calculate_metrics(output_tokens)            
        }
        
        # Write detailed summary metrics CSV
        with open(summary_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Avg', 'Min', 'Max', 'p99', 'p90', 'p75'])
            for metric_name, values in detailed_metrics.items():
                writer.writerow([metric_name, *values])
            writer.writerow(['Throughput', round(throughput, 2), 'NA', 'NA', 'NA', 'NA', 'NA'])
            writer.writerow(['File Name', file_details.get("name", "N/A"), '', '', '', '', ''])
            writer.writerow(['File Size (MB)', file_details.get("size_mb", "N/A"), '', '', '', '', ''])
        
        # Calculate WSF metrics (averages only)
        wsf_metrics = {
            "Request Latency (ms)": round(np.mean(latencies), 2),
            "Time to First Token (ms)": round(np.mean(ttfts), 2),
            "Inter Token Latency (ms)": round(np.mean(itls), 2),           
            "Tokens Per Second": round(np.mean(tpss), 2),
            "Input Tokens": round(np.mean(input_tokens), 2) if input_tokens else 0,
            "Output Tokens": round(np.mean(output_tokens), 2) if output_tokens else 0
        }
        
        # Write WSF metrics CSV
        with open(wsf_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            for metric_name, avg_value in wsf_metrics.items():
                writer.writerow([metric_name, avg_value])
            writer.writerow(['File Name', file_details.get("name", "N/A")])
            writer.writerow(['File Size (MB)', file_details.get("size_mb", "N/A")])
            
    except Exception as e:
        print(f"Failed to write metrics to CSV: {e}")


def write_rest_metrics(report_dir, metrics):
    """
    Write API metrics to a JSON file.
    
    Args:
        report_dir: Directory to save the JSON file.
        metrics: List of latency values.
    """
    json_file = os.path.join(report_dir, "document_api_individual_metrics.json")
    try:
        with open(json_file, "a") as file:
            for metric in metrics:
                json.dump({"LATENCY (ms)": metric}, file, indent=4)
    except Exception as e:
        print(f"Writing rest metrics to file, failed with exception {e}")


def write_rest_metrics_summary_to_csv(report_dir, latency, throughput):
    """
    Write a summary of REST API metrics to a CSV file.
    
    Args:
        report_dir: Directory to save the CSV file.
        latency: List of latency values.
        throughput: Throughput value.
    """
    output_file = os.path.join(report_dir, "document_api_summary_metrics.csv")
    
    try:
        metrics = calculate_metrics(latency)
        if metrics[0] is None:
            print("Failed to calculate metrics. Check latency data.")
            return
        
        avg, min_val, max_val, p99, p90, p75 = metrics
        
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['', '', '', 'Rest API Metrics', '', '', ''])
            writer.writerow(['Statistic', 'Avg', 'Min', 'Max', 'p99', 'p90', 'p75'])
            writer.writerow(['Latency (ms)', avg, min_val, max_val, p99, p90, p75])
            writer.writerow(['Throughput', round(throughput, 4), 'NA', 'NA', 'NA', 'NA', 'NA'])
        
        print(f"REST API metrics summary written to: {output_file}")
        
    except TypeError as e:
        print(f"Invalid data type for latency or throughput: {e}")
    except IOError as e:
        print(f"Failed to write metrics to file {output_file}: {e}")
    except Exception as e:
        print(f"Unexpected error writing REST metrics summary: {e}")


def rest_api_metrics(api_name, report_dir, latencies):
    """
    Collect and write REST API metrics to files.
    
    Args:
        api_name: Name of the API for file naming.
        report_dir: Directory to save the files.
        latencies: List of latency values.
    """
    output_file = os.path.join(report_dir, f"{api_name}_api_summary_metrics.csv")
    json_file = os.path.join(report_dir, f"{api_name}_api_metrics.json")
    
    average_latency, min_latency, max_latency, p99_latency, p90_latency, p75_latency = calculate_metrics(latencies)
    throughput = len(latencies) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0
    
    try:
        with open(json_file, "a") as file:
            for latency in latencies:
                json.dump({"LATENCY (ms)": latency}, file, indent=4)
                
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['', '', '', 'Rest API Metrics', '', '', ''])
            writer.writerow(['Statistic', 'Avg', 'Min', 'Max', 'p99', 'p90', 'p75'])
            writer.writerow(['Latency (ms)', average_latency, min_latency, max_latency, p99_latency, p90_latency, p75_latency])
            writer.writerow(['Throughput', round(throughput, 2), "NA", "NA", "NA", "NA", "NA"])

    except Exception as e:
        print(f"Writing rest metrics to file, failed with exception {e}")
        print(f"{api_name.capitalize()} API metrics collection completed. Reports saved in: {report_dir}")


# ==============================================================================
# Video Summary Metrics
# ==============================================================================

def write_vss_metrics(report_dir, metrics):
    """Write video summary/search metrics to a JSON file."""
    json_file = os.path.join(report_dir, "video_summary_metrics.json")
    try:
        with open(json_file, "a") as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        print(f"Writing rest metrics to file, failed with exception {e}")


def write_video_summary_metrics(report_dir, metrics):
    """
    Write individual video summary API metrics to a JSON file.
    
    Args:
        report_dir: Directory path where the metrics file will be saved.
        metrics: Dictionary containing metrics data to be written.
    """
    json_file = os.path.join(report_dir, "summary_api_individual_metrics.json")
    try:
        with open(json_file, "a") as file:
            json.dump(metrics, file, indent=4)
            file.write('\n')
    except IOError as e:
        print(f"Failed to write video summary metrics to {json_file}: {e}")
    except TypeError as e:
        print(f"Invalid metrics data type (must be JSON serializable): {e}")
    except Exception as e:
        print(f"Unexpected error writing video summary metrics: {e}")


def write_video_search_metrics(report_dir, metrics):
    """Write individual video search API metrics to a JSON file."""
    json_file = os.path.join(report_dir, "search_api_individual_metrics.json")
    try:
        with open(json_file, "a") as file:
            json.dump(metrics, file, indent=4)
            file.write('\n')
    except IOError as e:
        print(f"Failed to write video search metrics to {json_file}: {e}")
    except TypeError as e:
        print(f"Invalid metrics data type (must be JSON serializable): {e}")
    except Exception as e:
        print(f"Unexpected error writing video search metrics: {e}")


def write_video_search_metrics_summary_to_csv(report_dir, search_latencies, throughput, video_file_paths=None):
    """
    Write a comprehensive summary of video search API metrics to a CSV file.
    
    Args:
        report_dir: Directory to save the CSV file.
        search_latencies: List of search latency values.
        throughput: Throughput value.
        video_file_paths: Optional list of video file paths for metadata.
    """
    from common.video import get_video_details
    
    output_file = os.path.join(report_dir, "search_api_metrics_summary.csv")
    
    try:
        latency_metrics = calculate_metrics(search_latencies)
        
        if latency_metrics[0] is None:
            print("Failed to calculate metrics. Check search latency data.")
            return
        
        avg_latency, min_latency, max_latency = latency_metrics[:3]
        
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            if video_file_paths:
                writer.writerow(['', 'Video file details', '', ''])
                
                detail_fields = [
                    ('File Size (MB)', 'File Size (MB)'),
                    ('Length (s)', 'Duration (s)'),
                    ('FPS', 'FPS'),
                    ('File Name', 'Video File Name'),
                    ('Resolution', 'Resolution'),
                    ('Codec', 'Video Codec'),
                    ('Audio Codec', 'Audio Codec')
                ]
                
                for i, video_path in enumerate(video_file_paths, 1):
                    try:
                        video_details = get_video_details(video_path)
                        for label, key in detail_fields:
                            prefix = 'Audio' if 'Audio' in label else 'Video'
                            writer.writerow([f'{prefix}_{i} {label}', video_details.get(key, 'N/A'), '', ''])
                    except Exception as e:
                        print(f"Error getting video details for {video_path}: {e}")
                        writer.writerow([f'Video_{i} Error', str(e), '', ''])
                
                writer.writerow(['', '', '', ''])
            
            writer.writerow(['', 'Video Search API Metrics', '', ''])
            writer.writerow(['Statistic', 'Avg', 'Min', 'Max'])
            writer.writerow(['Query Search Duration (in seconds)', avg_latency, min_latency, max_latency])
            writer.writerow(['Search Throughput', round(throughput, 4), 'NA', 'NA'])
        
        print(f"Video search metrics written to: {output_file}")
        
    except TypeError as e:
        print(f"Invalid data type for metrics: {e}")
    except IOError as e:
        print(f"Failed to write metrics to file {output_file}: {e}")
    except Exception as e:
        print(f"Unexpected error writing video search metrics: {e}")


def write_video_summary_metrics_summary_to_csv(report_dir, latencies, ttft, tps, video_file_path=None, sampling_params=None):
    """
    Write a comprehensive summary of video summary API metrics to a CSV file.
    """
    from common.video import get_video_details
    
    output_file = os.path.join(report_dir, "summary_api_metrics_summary.csv")
    
    try:
        latency_metrics = calculate_metrics(latencies)
        ttft_metrics = calculate_metrics(ttft)
        tps_metrics = calculate_metrics(tps)
        
        if latency_metrics[0] is None or ttft_metrics[0] is None:
            print("Failed to calculate metrics. Check input data.")
            return
        
        avg_latency, min_latency, max_latency = latency_metrics[:3]
        avg_ttft, min_ttft, max_ttft = ttft_metrics[:3]
        avg_tps, min_tps, max_tps = tps_metrics[:3]
        
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            if video_file_path:
                writer.writerow(['', 'Video file details', '', ''])
                try:
                    video_details = get_video_details(video_file_path)
                    
                    detail_fields = [
                        ('Video File Size (MB)', 'File Size (MB)'),
                        ('Video Length (s)', 'Duration (s)'),
                        ('Video FPS', 'FPS'),
                        ('Video File Name', 'Video File Name'),
                        ('Video Resolution', 'Resolution'),
                        ('Video Codec', 'Video Codec'),
                        ('Audio Codec', 'Audio Codec')
                    ]
                    
                    for label, key in detail_fields:
                        writer.writerow([label, video_details.get(key, 'N/A'), '', ''])
                        
                except Exception as e:
                    print(f"Error getting video details for {video_file_path}: {e}")
                    writer.writerow(['Error retrieving video details', str(e), '', ''])
                
                writer.writerow(['', '', '', ''])
            
            if sampling_params:
                writer.writerow(['', 'Sampling Configuration', '', ''])
                sampling_fields = [
                    ('Chunk Duration', 'chunkDuration'),
                    ('Sample Frame Per Chunk', 'samplingFrame'),
                    ('Frames Overlap', 'frameOverlap'),
                    ('MultiFrame', 'multiFrame')
                ]
                
                for label, key in sampling_fields:
                    writer.writerow([label, sampling_params.get(key, 'N/A'), '', ''])
                
                writer.writerow(['', '', '', ''])
            
            writer.writerow(['', 'Video Summary API Metrics', '', ''])
            writer.writerow(['Statistic', 'Avg', 'Min', 'Max'])
            writer.writerow(['Time to First Chunk Summary (in seconds)', avg_ttft, min_ttft, max_ttft])
            writer.writerow(['Video Summarization Duration (in seconds)', avg_latency, min_latency, max_latency])
            writer.writerow(['Token Per Sec', avg_tps, min_tps, max_tps])
        
        print(f"Video summary metrics written to: {output_file}")
        
    except TypeError as e:
        print(f"Invalid data type for metrics: {e}")
    except IOError as e:
        print(f"Failed to write metrics to file {output_file}: {e}")
    except Exception as e:
        print(f"Unexpected error writing video summary metrics: {e}")


def get_video_summary_telemetry_kpis(start_time, end_time, chunk_completion_time, video_properties):
    """
    Extract and calculate video summarization telemetry KPIs from telemetry response data.
    
    Args:
        start_time: Start timestamp of the summarization.
        end_time: End timestamp of the summarization.
        telemetry_json_response: JSON response from telemetry API.
        video_properties: Dictionary to store video metrics.
        
    Returns:
        tuple: (video_properties, telemetry_details)
    """
    from common.video import convert_timestamp_to_float
    
    try:
        # Calculate metrics from telemetry data        
        e2e_summary_latency = end_time - start_time
        rtf = e2e_summary_latency / (video_properties.get('File_Duration (s)', 1))
        complexity = (video_properties.get('File_videoFPS', 0) * video_properties.get('File_Duration (s)', 0)) / e2e_summary_latency

        # Write metrics to video properties
        video_properties['Video Summary Pre ProcessingTime (s)'] = chunk_completion_time - start_time
        video_properties['Video Summary E2E Latency (s)'] = e2e_summary_latency
        video_properties['Video Summarization RTF (latency/duration)'] = rtf
        video_properties['Video Summary Processing Efficiency ((fps*duration)/latency)'] = complexity

        return video_properties

    except Exception as e:
        print(f"Unexpected error in get_video_summary_telemetry_kpis: {e}")
        return video_properties


## As per new implementaion
def get_video_search_telemetry_kpis(start_time, end_time, telemetry_json_response, search_metrics):
    """
    Extract video search telemetry KPIs from the telemetry response within a time window.
    
    Args:
        start_time: Start timestamp of the search.
        end_time: End timestamp of the search.
        telemetry_json_response: JSON response from telemetry API.
        search_metrics: Search metrics to include in output.
        
    Returns:
        tuple: (metrics, telemetry_details)
    """
    from common.video import convert_timestamp_to_float
    
    metrics = {}
    input_videos = []
    telemetry_details = []
    
    metrics["Video_Search_E2E_Latency"] = round(end_time - start_time, 2)
    items = telemetry_json_response.get("items", [])
    
    for item in items:
        try:
            timestamp_str = item.get("timestamps", {}).get("requested_at", "")
            if not timestamp_str:
                continue
            
            timestamp = convert_timestamp_to_float(timestamp_str)            
            if not (start_time <= timestamp):
                continue
            
            telemetry_details.append(item)
            video_file_details = item.get("video", {})
            video_details = {
                "id": video_file_details.get("video_id"),
                "file_name": video_file_details.get("filename", "N/A"),
                "duration_seconds": round(video_file_details.get("video_duration_seconds", 1), 2),
                "fps": round(video_file_details.get("fps", 0), 2),
                "total_frames": video_file_details.get("total_frames", 0),
                "frames_extracted": item.get("counts", {}).get("frames_extracted", 0),
                "embeddings_stored": item.get("counts", {}).get("embeddings_stored", 0)
            }
            
            video_details.update(item.get("stage_duration", {}))            
            video_details["wall_time_seconds"] = item.get("timestamps", {}).get("wall_time_seconds", 0)
            video_details["embedding_per_sec"] = item.get("stage_throughput", {}).get("embeddings_throughput", 0)
            relative_rtf = (
                (video_details.get("wall_time_seconds", 0) / video_details.get("duration_seconds", 1))
                * (30 / video_details.get("fps", 1))
            )
            video_details["Normalized_Embedding_RTF"] = round(relative_rtf, 4)
            input_videos.append(video_details)
            
        except (ValueError, TypeError, KeyError) as e:
            print(f"Warning: Skipping telemetry item due to error: {e}")
            continue
    
    metrics["Input_Videos"] = input_videos
    metrics["Search_Metrics"] = search_metrics
    return metrics, telemetry_details

def save_video_summary_search_telemetry_kpis(report_dir, metrics, telemetry_details=None, query_metrics=None):
    """
    Save video search telemetry KPIs to a JSON file.
    
    Args:
        report_dir: Directory to save the files.
        metrics: Metrics dictionary to save.
        telemetry_details: Optional telemetry details to save.
        query_metrics: Optional query metrics to save.
    Returns:
        str: Path to the output file.
    """
    output_file = os.path.join(report_dir, "video_summary_search_metrics.json")
    telemetry_file = os.path.join(report_dir, "video_summary_search_telemetry_details.json")

    telemetry_embedd_search_details = []
    telemetry_embedd_search_details.append({"Embedding_telemetry": telemetry_details})
    telemetry_embedd_search_details.append({"Search_telemetry": query_metrics})
    
    try:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
        print(f"Video summary and search embedding metrics written to: {output_file}")

        if telemetry_details is None:
            with open(telemetry_file, "w") as t_file:
                json.dump(telemetry_embedd_search_details, t_file, indent=4)
            print(f"Video embedding telemetry details written to: {telemetry_file}")


    except IOError as e:
        print(f"Failed to write video search embedding metrics to {output_file}: {e}")
    except TypeError as e:
        print(f"Invalid data type for embedding metrics: {e}")
    except Exception as e:
        print(f"Unexpected error writing video search embedding metrics: {e}")
    
    return output_file


def convert_summary_metrics_to_wsf_format(report_dir, json_file_path, samples=None):
    """
    Read values from a JSON file and write to CSV in key,value format.
    
    Args:
        report_dir: Directory to save the CSV file.
        json_file_path: Path to the input JSON file.
        samples: Optional sampling parameters to include.
    """
    output_file = os.path.join(report_dir, "video_summary_metrics_wsf.csv")
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        data = data[0]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if samples:
            for key, value in samples.items():
                writer.writerow([f"Sampling-{key}", value])
        
        wsf_keys = [
            'File_Duration (s)', 'File_videoFPS', 'Time To First Token (s)',
            'Throughput (tokens/sec)', 'Video Summary Pre ProcessingTime (s)',
            'Video Summary E2E Latency (s)', 'Video Summarization RTF (latency/duration)',
            'Video Summary Processing Efficiency ((fps*duration)/latency)'
        ]
        
        for key, value in data.items():
            if key in wsf_keys:
                writer.writerow([key, value])

    print(f"WSF formatted output written to: {output_file}")


def convert_search_metrics_to_wsf_format(report_dir, json_file_path):
    """
    Read video metrics from JSON file and write to CSV file.
    
    Args:
        report_dir: Directory to save the CSV file.
        json_file_path: Path to the input JSON file.
        
    Returns:
        str: Path to the output file.
    """
    output_file = os.path.join(report_dir, "video_search_embedding_metrics_wsf.csv")

    with open(json_file_path, 'r') as f:
        data = json.load(f)
       
    videos = data.get('Input_Videos', [])
    embedding_per_sec_values = []
    rows = []
    
    for idx, video in enumerate(videos, start=1):
        video_prefix = f"Video_{idx}"
        
        rows.append([f"{video_prefix}_00_duration (s)", video.get('duration_seconds', 0.0)])
        rows.append([f"{video_prefix}_01_FPS", video.get('fps', 0.0)])
        rows.append([f"{video_prefix}_02_total_frames", video.get('total_frames', 0)])
        rows.append([f"{video_prefix}_03_frames_extracted", video.get('frames_extracted', 0)])
        rows.append([f"{video_prefix}_04_Embedding_Throughput (frames/sec)", video.get('embedding_per_sec', 0.0)])
        
        if 'embedding_per_sec' in video:
            embedding_per_sec_values.append(video['embedding_per_sec'])
    
    embedding_avg, embedding_min, embedding_max = calculate_metrics(embedding_per_sec_values)[:3]
    rows.append(["Embedding_Throughput_min (frames/sec)", embedding_min])
    rows.append(["Embedding_Throughput_avg (frames/sec)", embedding_avg])
    rows.append(["Embedding_Throughput_max (frames/sec)", embedding_max])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"WSF formatted output written to: {output_file}")
    return output_file


# ==============================================================================
# Live Caption Metrics
# ==============================================================================

def get_live_caption_metrics(metadata_list):
    """
    Process collected metadata entries and return aggregated KPIs grouped by runId.
    
    Args:
        metadata_list: List of metadata strings from the live caption API.
        
    Returns:
        dict: Dictionary with runIds as keys and lists of KPI dictionaries as values.
    """
    kpis_by_run_id = {}
    
    for metadata in metadata_list:
        json_string = metadata[6:]
        try:
            parsed_data = json.loads(json_string)
            if parsed_data.get("runId"):
                run_id = parsed_data.get("runId", "unknown")
                metrics = parsed_data.get("data", {}).get("metrics", {})
                resolution = parsed_data.get("data", {}).get("resolution", {})
                kpis = {
                    "InputTokens": metrics.get("num_input_tokens"),
                    "TotalGeneratedTokens": metrics.get("num_generated_tokens"),
                    "TTFT (ms)": metrics.get("ttft_mean"),
                    "TPOT (ms)": metrics.get("tpot_mean"),
                    "Latency (ms)": metrics.get("generate_duration_mean"),
                    "Throughput (tok/s)": metrics.get("throughput_mean"),
                    "Resolution_Width": resolution.get("width"),
                    "Resolution_Height": resolution.get("height")
                }
                if run_id not in kpis_by_run_id:
                    kpis_by_run_id[run_id] = []
                kpis_by_run_id[run_id].append(kpis)
            
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON data: {e}")
            continue
    
    return kpis_by_run_id


def save_live_video_caption_telemetry_kpis(report_dir, kpis_by_run_id, run_configs=None):
    """
    Save live video caption telemetry KPIs grouped by runId.
    
    Args:
        report_dir: Directory to save the metrics files.
        kpis_by_run_id: Dictionary with runIds as keys and lists of KPI dictionaries as values.
        run_configs: Optional dictionary mapping runIds to their config.
        
    Returns:
        str: Path to the summary file.
    """
    os.makedirs(report_dir, exist_ok=True)
    run_configs = run_configs or {}
    
    filename = os.path.join(report_dir, "live_caption_indv_metrics.json")
    with open(filename, "w") as file:
        json.dump(kpis_by_run_id, file, indent=4)
    
    summary_file = os.path.join(report_dir, "live_caption_summary_metrics.json")
    summary = {}
    
    for run_id, kpis_list in kpis_by_run_id.items():
        if not kpis_list:
            continue
        
        config = run_configs.get(run_id, {})
        run_summary = {
            "rtspUrl": config.get("rtspUrl"),
            "modelName": config.get("modelName"),
            "pipelineName": config.get("pipelineName"),
            "Resolution_Width": max(kpi["Resolution_Width"] for kpi in kpis_list if kpi["Resolution_Width"] is not None) if kpis_list else None,
            "Resolution_Height": max(kpi["Resolution_Height"] for kpi in kpis_list if kpi["Resolution_Height"] is not None) if kpis_list else None,
            "Frame Rate": config.get("frameRate"),
            "Chunk Size": config.get("chunkSize"),
            "sample_count": len(kpis_list),
            "Total InputTokens": max(kpi["InputTokens"] for kpi in kpis_list if kpi["InputTokens"] is not None) if kpis_list else None,
            "Total GeneratedTokens": max(kpi["TotalGeneratedTokens"] for kpi in kpis_list if kpi["TotalGeneratedTokens"] is not None) if kpis_list else None,
            "Average TTFT (ms)": sum(kpi["TTFT (ms)"] for kpi in kpis_list if kpi["TTFT (ms)"] is not None) / len(kpis_list),
            "Average TPOT (ms)": sum(kpi["TPOT (ms)"] for kpi in kpis_list if kpi["TPOT (ms)"] is not None) / len(kpis_list),
            "Average Latency (ms)": sum(kpi["Latency (ms)"] for kpi in kpis_list if kpi["Latency (ms)"] is not None) / len(kpis_list),
            "Average Throughput (tok/s)": sum(kpi["Throughput (tok/s)"] for kpi in kpis_list if kpi["Throughput (tok/s)"] is not None) / len(kpis_list),
            
        }
        summary[run_id] = run_summary
    
    with open(summary_file, "w") as file:
        json.dump(summary, file, indent=4)
    
    return summary_file


def save_lvc_metrics_to_wsf_format(report_dir, summary_file, live_caption_duration_seconds):
    """
    Save live caption metrics to WSF CSV format.
    
    Args:
        report_dir: Directory to save the CSV file.
        summary_file: Path to the summary JSON file.
        live_caption_duration_seconds: Duration of caption collection.
    """
    output_file = os.path.join(report_dir, "live_caption_metrics_wsf.csv")
    
    with open(summary_file, "r") as file:
        summary = json.load(file)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for run_id, metrics in summary.items():
            writer.writerow([f"Avg TTFT (ms)", round(metrics.get("Average TTFT (ms)", 0), 2)])
            writer.writerow([f"Avg TPOT (ms)", round(metrics.get("Average TPOT (ms)", 0), 2)])
            writer.writerow([f"Avg Latency (ms)", round(metrics.get("Average Latency (ms)", 0), 2)])
            writer.writerow([f"Avg Throughput (tok/s)", round(metrics.get("Average Throughput (tok/s)", 0), 2)])
            writer.writerow([f"Avg Output Tokens", round((metrics.get("Total GeneratedTokens", 0) / metrics.get("sample_count", 1)), 2)])
            writer.writerow([])
    
    print(f"WSF formatted live caption metrics written to: {output_file}")

def save_lvc_rag_metrics_to_wsf_format(report_dir, summary_file, live_caption_duration_seconds, *chat_metrics):
    """
    Save live caption RAG metrics to WSF CSV format.
    
    Args:
        report_dir: Directory to save the CSV file.
        summary_file: Path to the summary JSON file.
        live_caption_duration_seconds: Duration of caption collection.
    """
    output_file = os.path.join(report_dir, "live_caption_rag_metrics_wsf.csv")

    latencies, input_tokens, output_tokens, ttfts, itls, tpss = chat_metrics

    detailed_metrics = {
            "Request Latency (ms)": calculate_metrics(latencies),
            "Time to First Token (ms)": calculate_metrics(ttfts),        
            "Tokens Per Second": calculate_metrics(tpss),
            "Output Tokens": calculate_metrics(output_tokens)            
        }
    
    with open(summary_file, "r") as file:
        summary = json.load(file)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        for run_id, metrics in summary.items():
            writer.writerow([f"LVC_Metric Avg TTFT (ms)", metrics.get("Average TTFT (ms)", 0)])
            writer.writerow([f"LVC_Metric Avg TPOT (ms)", metrics.get("Average TPOT (ms)", 0)])
            writer.writerow([f"LVC_Metric Avg Latency (ms)", metrics.get("Average Latency (ms)", 0)])
            writer.writerow([f"LVC_Metric Avg Throughput (tok/s)", metrics.get("Average Throughput (tok/s)", 0)])
            writer.writerow([f"LVC_Metric Avg Output Tokens", round((metrics.get("Total GeneratedTokens", 0) / metrics.get("sample_count", 1)), 2)])
            writer.writerow([f"Chat_Metric Latency (ms)", round(np.mean(latencies), 2)])
            writer.writerow([f"Chat_Metric TTFT (ms)", round(np.mean(ttfts), 2)])
            writer.writerow([f"Chat_Metric Throughput (tok/s)", round(np.mean(tpss), 2)])
            writer.writerow([f"Chat_Metric Output Tokens", round(np.mean(output_tokens), 2)])
            writer.writerow([])
        
    print(f"WSF formatted live caption RAG metrics written to: {output_file}")


# ==============================================================================
# Real-Time Scene Understanding Metrics
# ==============================================================================

RTSU_GENAI_KPIS = ["TTFT (ms)", "TPOT (ms)", "Throughput (tok/s)"]
RTSU_ALERT_CHECK_KPIS = RTSU_GENAI_KPIS + ["Latency (ms)", "Cycle Interval (ms)"]
RTSU_DEEP_ANALYSIS_KPIS = RTSU_GENAI_KPIS + [
    "Output Tokens", "Generation Latency (ms)", "E2E Latency (ms)",
    "Trigger Latency (ms)", "Wait (ms)", "Publish Lag (ms)",
]
RTSU_STARTUP_KPIS = ["Time to Publishing (ms)", "Time to First Caption (ms)", "Time to First Deep Analysis (ms)"]

RTSU_ALERT_CHECK_COLUMNS = [
    "Stream ID", "Sample", "Caption TS", "Playback (s)", "Decision",
    *RTSU_ALERT_CHECK_KPIS, "Possibly Stale", "Caption",
]
RTSU_DEEP_ANALYSIS_COLUMNS = [
    "Stream ID", "Frame ID", "Segment", "Confirmed", "Frames Sampled", *RTSU_DEEP_ANALYSIS_KPIS,
    "Trigger Match", "Registry Status", "Frame Created TS", "Trigger Caption TS",
    "Trigger Playback (s)", "Uploaded TS", "First Seen TS",
]
RTSU_STARTUP_COLUMNS = ["Stream ID", *RTSU_STARTUP_KPIS]
RTSU_STAT_KEYS = ("min", "max", "mean", "p90")

# RTSU hands frames to the alert VLM at least ALERT_VLM_INTERVAL (default 2 s) apart.
RTSU_TRIGGER_MATCH_TOLERANCE_S = 1.0
_RTSU_DECISION_RE = re.compile(r"\bdecision\s*:\s*(yes|no)\b", re.IGNORECASE)


def calculate_summary_stats(values):
    """Return min, max, mean and p90 for non-null values (linear percentile)."""
    data = [float(v) for v in values if v is not None]
    if not data:
        return {"min": None, "max": None, "mean": None, "p90": None}
    arr = np.asarray(data)
    return {
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "mean": round(float(arr.mean()), 2),
        "p90": round(float(np.percentile(arr, 90)), 2),
    }


def _rtsu_number(value):
    """Return a finite float, or None for missing, boolean, or non-numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _rtsu_round(value, digits=2):
    return None if value is None else round(value, digits)


def _rtsu_ms(start, end):
    """Milliseconds from ``start`` to ``end`` (epoch seconds), or None if either is missing."""
    if start is None or end is None:
        return None
    return round((end - start) * 1000, 2)


def parse_rtsu_decision(caption):
    """Parse the ``Decision: Yes/No`` line of an RTSU alert caption into True, False, or None."""
    match = _RTSU_DECISION_RE.search(caption if isinstance(caption, str) else "")
    return None if match is None else match.group(1).lower() == "yes"


def extract_rtsu_alert_check_samples(tracker, stream_entry, poll_time):
    """
    Turn one ``GET /api/streams`` entry into new alert-check samples.

    A sample is recorded when ``caption_ts`` changes. ``caption_history`` entries (newest
    first) that appeared since the previous poll but are not the newest are recorded as
    ``missed``: they have text and ``playback_seconds`` but no TTFT or TPOT. The newest
    sample is flagged ``possibly_stale`` when TTFT, TPOT, and throughput equal the previous
    sample, because the server keeps the last non-null values.

    Args:
        tracker: Per-stream dict, updated in place between polls.
        stream_entry: One item of the ``streams`` list.
        poll_time: Client time when the response was received.

    Returns:
        list: New samples, oldest first.
    """
    caption_ts = _rtsu_number(stream_entry.get("caption_ts"))
    if caption_ts is None or caption_ts == tracker.get("last_caption_ts"):
        return []

    history = [entry for entry in (stream_entry.get("caption_history") or []) if isinstance(entry, dict)]
    if not history:
        history = [{"response": stream_entry.get("caption"), "playback_seconds": None}]
    keys = [(entry.get("response"), entry.get("playback_seconds")) for entry in history]

    previous = tracker.get("last_newest")
    history_gap = False
    if previous is not None and previous in keys:
        new_count = max(keys.index(previous), 1)
    else:
        new_count = len(history)
        # The previous newest entry scrolled out of the history: more checks may have been missed.
        history_gap = previous is not None

    metrics = tuple(_rtsu_number(stream_entry.get(key)) for key in ("ttft_ms", "tpot_ms", "throughput_tps"))
    last_metrics = tracker.get("last_metrics")
    possibly_stale = last_metrics is not None and any(m is not None for m in metrics) and metrics == last_metrics

    def make_sample(entry, sample_type, sample_ts, sample_metrics, stale, missed_before):
        caption = entry.get("response")
        return {
            "stream_id": stream_entry.get("stream_id"),
            "sample": sample_type,
            "caption": caption if isinstance(caption, str) else None,
            "decision": parse_rtsu_decision(caption),
            "playback_seconds": _rtsu_number(entry.get("playback_seconds")),
            "caption_ts": sample_ts,
            "ttft_ms": sample_metrics[0],
            "tpot_ms": sample_metrics[1],
            "throughput_tps": sample_metrics[2],
            "possibly_stale": stale,
            "missed_before": missed_before,
            "history_gap": False,
            "poll_time": poll_time,
        }

    samples = [
        make_sample(history[index], "missed", None, (None, None, None), False, 0)
        for index in range(new_count - 1, 0, -1)
    ]
    samples.append(make_sample(history[0], "observed", caption_ts, metrics, possibly_stale, new_count - 1))
    samples[0]["history_gap"] = history_gap

    tracker["last_caption_ts"] = caption_ts
    tracker["last_newest"] = keys[0]
    tracker["last_metrics"] = metrics
    return samples


def rtsu_stop_reason(completed_by_stream, stream_ids, target, min_per_stream, now, deadline):
    """
    Apply the RTSU stop rule.

    Args:
        completed_by_stream: Mapping of stream ID to completed deep analyses.
        stream_ids: Streams created for the run.
        target: Global number of deep analyses to collect.
        min_per_stream: Minimum deep analyses per stream (0 disables the check).
        now: Current time.
        deadline: Time at which the run times out.

    Returns:
        str: ``target_met``, ``timeout``, or None to keep collecting.
    """
    total = sum(completed_by_stream.get(stream_id, 0) for stream_id in stream_ids)
    if total >= target and all(completed_by_stream.get(stream_id, 0) >= min_per_stream for stream_id in stream_ids):
        return "target_met"
    if now >= deadline:
        return "timeout"
    return None


def match_rtsu_trigger(stream_samples, trigger_caption, created_ts, uploaded_at, anchor):
    """
    Find the alert check that triggered a deep analysis.

    When the trigger frame's ``created_ts`` is known, the caption with the same text whose
    frame time (``anchor + playback_seconds``) is closest to ``created_ts`` is used, within
    ``RTSU_TRIGGER_MATCH_TOLERANCE_S``. Without ``created_ts`` (registry record evicted), the
    latest caption with the same text and ``caption_ts <= uploaded_at`` is used.

    Returns:
        tuple: (sample or None, ``frame``, ``text``, or ``none``)
    """
    if not isinstance(trigger_caption, str) or not trigger_caption:
        return None, "none"
    candidates = [sample for sample in stream_samples if sample.get("caption") == trigger_caption]

    if created_ts is not None:
        timed = [sample for sample in candidates if sample.get("playback_seconds") is not None]
        if anchor is not None and timed:
            best = min(timed, key=lambda sample: abs(anchor + sample["playback_seconds"] - created_ts))
            if abs(anchor + best["playback_seconds"] - created_ts) <= RTSU_TRIGGER_MATCH_TOLERANCE_S:
                return best, "frame"
        return None, "none"

    if uploaded_at is not None:
        earlier = [s for s in candidates if s.get("caption_ts") is not None and s["caption_ts"] <= uploaded_at]
        if earlier:
            return max(earlier, key=lambda sample: sample["caption_ts"]), "text"
    return None, "none"


def build_rtsu_deep_analysis_rows(analyses, samples, anchors, clock_offset_s):
    """
    Build one row per completed deep analysis, including trigger matching.

    ``E2E Latency = Trigger Latency + Wait + Generation Latency + Publish Lag``, where
    Trigger Latency is ``caption_ts(trigger) - created_ts``.

    Args:
        analyses: Collected analyses with ``stream_id``, ``frame_id``, ``first_seen_client``,
            ``detail`` (alert detail body), ``created_ts``, and ``registry_status``.
        samples: Alert-check samples from ``extract_rtsu_alert_check_samples``.
        anchors: Stream ID mapped to the stream start time (server clock).
        clock_offset_s: Server-minus-client clock offset in seconds.

    Returns:
        list: Rows keyed by ``RTSU_DEEP_ANALYSIS_COLUMNS``. Timestamps use the server clock.
    """
    from common.video import parse_rtsu_timestamp

    samples_by_stream = {}
    for sample in samples:
        samples_by_stream.setdefault(sample["stream_id"], []).append(sample)

    rows = []
    for analysis in sorted(analyses, key=lambda item: item["first_seen_client"]):
        stream_id = analysis["stream_id"]
        detail = analysis.get("detail") if isinstance(analysis.get("detail"), dict) else {}
        metrics = detail.get("metrics") if isinstance(detail.get("metrics"), dict) else {}
        created_ts = _rtsu_number(analysis.get("created_ts"))
        uploaded_ts = parse_rtsu_timestamp(detail.get("uploaded_at"))
        first_seen_ts = analysis["first_seen_client"] + clock_offset_s
        generation_ms = _rtsu_number(metrics.get("total_duration_ms"))

        trigger, match = match_rtsu_trigger(
            samples_by_stream.get(stream_id, []), detail.get("trigger_caption"),
            created_ts, uploaded_ts, anchors.get(stream_id),
        )
        trigger_ts = trigger["caption_ts"] if trigger else None
        wait_ms = None
        if uploaded_ts is not None and trigger_ts is not None and generation_ms is not None:
            wait_ms = round((uploaded_ts - trigger_ts) * 1000 - generation_ms, 2)

        segment = metrics.get("segment_path")
        frames = _rtsu_number(metrics.get("frames_sampled"))
        tokens = _rtsu_number(metrics.get("total_tokens_generated"))
        confirmed = detail.get("confirmed")
        row = {
            "Stream ID": stream_id,
            "Frame ID": analysis["frame_id"],
            "Segment": os.path.basename(segment) if isinstance(segment, str) else None,
            "Confirmed": confirmed if isinstance(confirmed, bool) else None,
            "Frames Sampled": int(frames) if frames is not None else None,
            "TTFT (ms)": _rtsu_round(_rtsu_number(metrics.get("ttft_ms"))),
            "TPOT (ms)": _rtsu_round(_rtsu_number(metrics.get("tpot_ms"))),
            "Throughput (tok/s)": _rtsu_round(_rtsu_number(metrics.get("throughput_tps"))),
            "Output Tokens": int(tokens) if tokens is not None else None,
            "Generation Latency (ms)": _rtsu_round(generation_ms),
            "E2E Latency (ms)": _rtsu_ms(created_ts, first_seen_ts),
            "Trigger Latency (ms)": _rtsu_ms(created_ts, trigger_ts),
            "Wait (ms)": wait_ms,
            "Publish Lag (ms)": _rtsu_ms(uploaded_ts, first_seen_ts),
            "Trigger Match": match,
            "Registry Status": analysis.get("registry_status"),
            "Frame Created TS": _rtsu_round(created_ts, 6),
            "Trigger Caption TS": _rtsu_round(trigger_ts, 6),
            "Trigger Playback (s)": _rtsu_round(trigger["playback_seconds"], 6) if trigger else None,
            "Uploaded TS": _rtsu_round(uploaded_ts, 6),
            "First Seen TS": round(first_seen_ts, 6),
        }
        if analysis.get("artifacts"):
            row["Artifacts"] = json.dumps(analysis["artifacts"])
        rows.append(row)
    return rows


def refine_rtsu_anchors(initial_anchors, deep_rows):
    """
    Refine each stream start time to ``created_ts - playback_seconds`` of its frame-matched triggers.

    Both values come from the server, so the refined anchor does not depend on the clock offset.
    Streams without a frame-matched trigger keep the initial (POST midpoint) anchor.
    """
    estimates = {}
    for row in deep_rows:
        if row["Trigger Match"] == "frame":
            estimates.setdefault(row["Stream ID"], []).append(row["Frame Created TS"] - row["Trigger Playback (s)"])
    anchors = dict(initial_anchors)
    for stream_id, values in estimates.items():
        anchors[stream_id] = float(np.median(values))
    return anchors


def build_rtsu_alert_check_rows(samples, anchors):
    """
    Build one row per alert check.

    ``Latency (ms)`` is ``caption_ts - (anchor + playback_seconds)``: the age of the frame when
    its verdict was published. ``Cycle Interval (ms)`` is the time since the previous observed
    caption of the same stream; it is left empty when checks were missed in between.

    Args:
        samples: Alert-check samples in collection order.
        anchors: Stream ID mapped to the stream start time (server clock).

    Returns:
        list: Rows keyed by ``RTSU_ALERT_CHECK_COLUMNS``.
    """
    rows, last_observed = [], {}
    for sample in samples:
        stream_id = sample["stream_id"]
        anchor = anchors.get(stream_id)
        playback = sample["playback_seconds"]
        frame_ts = anchor + playback if anchor is not None and playback is not None else None
        cycle_ms = None
        if sample["sample"] == "observed":
            previous = last_observed.get(stream_id)
            if not sample["missed_before"] and not sample["history_gap"]:
                cycle_ms = _rtsu_ms(previous, sample["caption_ts"])
            last_observed[stream_id] = sample["caption_ts"]
        rows.append({
            "Stream ID": stream_id,
            "Sample": sample["sample"],
            "Caption TS": _rtsu_round(sample["caption_ts"], 6),
            "Playback (s)": _rtsu_round(playback, 6),
            "Decision": {True: "yes", False: "no"}.get(sample["decision"]),
            "TTFT (ms)": _rtsu_round(sample["ttft_ms"]),
            "TPOT (ms)": _rtsu_round(sample["tpot_ms"]),
            "Throughput (tok/s)": _rtsu_round(sample["throughput_tps"]),
            "Latency (ms)": _rtsu_ms(frame_ts, sample["caption_ts"]),
            "Cycle Interval (ms)": cycle_ms,
            "Possibly Stale": sample["possibly_stale"],
            "Caption": sample["caption"],
        })
    return rows


def build_rtsu_startup_rows(stream_ids, post_mid_client, publishing_first_seen_client, alert_rows, deep_rows, anchors):
    """
    Build per-stream startup KPIs, reported separately from the steady-state summaries.

    Time to publishing uses client times. Time to first caption and first deep analysis are
    measured from the stream start anchor on the server clock.
    """
    rows = []
    for stream_id in stream_ids:
        anchor = anchors.get(stream_id)
        caption_times = [r["Caption TS"] for r in alert_rows if r["Stream ID"] == stream_id and r["Caption TS"] is not None]
        deep_times = [r["First Seen TS"] for r in deep_rows if r["Stream ID"] == stream_id]
        rows.append({
            "Stream ID": stream_id,
            "Time to Publishing (ms)": _rtsu_ms(post_mid_client.get(stream_id), publishing_first_seen_client.get(stream_id)),
            "Time to First Caption (ms)": _rtsu_ms(anchor, min(caption_times)) if caption_times else None,
            "Time to First Deep Analysis (ms)": _rtsu_ms(anchor, min(deep_times)) if deep_times else None,
        })
    return rows


def summarize_rtsu_kpis(rows, kpi_columns):
    """Return ``{"overall": {kpi: stats}, "<stream_id>": {kpi: stats}}`` using calculate_summary_stats."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["Stream ID"], []).append(row)
    summary = {"overall": {kpi: calculate_summary_stats([row.get(kpi) for row in rows]) for kpi in kpi_columns}}
    for stream_id, stream_rows in grouped.items():
        summary[stream_id] = {kpi: calculate_summary_stats([row.get(kpi) for row in stream_rows]) for kpi in kpi_columns}
    return summary


def compute_rtsu_rates(alert_rows, deep_rows, stream_ids, duration_s):
    """Return system-level rates for the collection window."""
    minutes = duration_s / 60 if duration_s else None

    def per_minute(count):
        return round(count / minutes, 2) if minutes else None

    decisions = [row["Decision"] for row in alert_rows if row["Decision"] is not None]
    confirmed = [row["Confirmed"] for row in deep_rows if row["Confirmed"] is not None]
    timed = [(row["Output Tokens"], row["Generation Latency (ms)"]) for row in deep_rows
             if row["Output Tokens"] is not None and row["Generation Latency (ms)"]]
    generation_s = sum(ms for _, ms in timed) / 1000
    return {
        "Alert Checks per Minute": per_minute(len(alert_rows)),
        "Alert Checks per Minute per Stream": {
            stream_id: per_minute(sum(1 for row in alert_rows if row["Stream ID"] == stream_id))
            for stream_id in stream_ids
        },
        "Alert Check Yes Ratio": round(decisions.count("yes") / len(decisions), 4) if decisions else None,
        "Deep Analyses per Minute": per_minute(len(deep_rows)),
        "Deep Analysis Confirmed Ratio": round(sum(confirmed) / len(confirmed), 4) if confirmed else None,
        "Aggregate Deep Decode Throughput (tok/s)": (
            round(sum(tokens for tokens, _ in timed) / generation_s, 2) if generation_s > 0 else None
        ),
    }


def compute_rtsu_data_quality(samples, deep_rows, counters, reconnects):
    """Return completion and data-quality counters."""
    observed = [sample for sample in samples if sample["sample"] == "observed"]
    matches = [row["Trigger Match"] for row in deep_rows]
    return {
        "alert_checks_observed": len(observed),
        "alert_checks_missed": len(samples) - len(observed),
        "alert_check_history_gaps": sum(1 for sample in samples if sample["history_gap"]),
        "alert_checks_possibly_stale": sum(1 for sample in observed if sample["possibly_stale"]),
        "alert_checks_null_metrics": sum(
            1 for sample in observed
            if sample["ttft_ms"] is None and sample["tpot_ms"] is None and sample["throughput_tps"] is None
        ),
        "yes_verdicts_observed": sum(1 for sample in samples if sample["decision"] is True),
        "deep_analyses_completed": len(deep_rows),
        "triggers_matched_by_frame": matches.count("frame"),
        "triggers_matched_by_text": matches.count("text"),
        "unmatched_triggers": matches.count("none"),
        "registry_not_found": sum(1 for row in deep_rows if row["Registry Status"] == "not_found"),
        "registry_errors": sum(1 for row in deep_rows if row["Registry Status"] == "error"),
        **counters,
        "stream_reconnects": reconnects,
    }


def _rtsu_na(value):
    return "NA" if value is None else value


def _write_rtsu_json(path, data):
    from common.video import redact_rtsu_data

    with open(path, "w", encoding="utf-8") as file:
        json.dump(redact_rtsu_data(data), file, indent=4, ensure_ascii=False)


def _write_rtsu_csv(path, columns, rows):
    from common.video import redact_rtsu_data

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(redact_rtsu_data(rows))


def _write_rtsu_summary_csv(path, summary):
    sections = (
        ("Alert Check", summary["alert_check"]["kpis"], RTSU_ALERT_CHECK_KPIS),
        ("Deep Analysis", summary["deep_analysis"]["kpis"], RTSU_DEEP_ANALYSIS_KPIS),
    )
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Section", "KPI", "Scope", "Min", "Max", "Mean", "p90"])
        for section, kpis_by_scope, kpis in sections:
            for kpi in kpis:
                for scope, stats in kpis_by_scope.items():
                    writer.writerow([section, kpi, scope, *(_rtsu_na(stats[kpi][key]) for key in RTSU_STAT_KEYS)])


def save_rtsu_metrics_to_wsf_format(report_dir, summary_file):
    """
    Save Real-Time Scene Understanding metrics to WSF CSV format (key,value).

    Args:
        report_dir: Directory to save the CSV file.
        summary_file: Path to ``rtsu_summary_metrics.json``.

    Returns:
        str: Path to the output file.
    """
    output_file = os.path.join(report_dir, "rtsu_metrics_wsf.csv")
    with open(summary_file, "r", encoding="utf-8") as file:
        summary = json.load(file)

    run = summary["run"]
    alert, deep, rates = summary["alert_check"], summary["deep_analysis"], summary["rates"]
    alert_overall, deep_overall = alert["kpis"]["overall"], deep["kpis"]["overall"]
    rows = [
        ("Streams", run["streams"]),
        ("Stop Reason", run["stop_reason"]),
        ("Run Duration (s)", run["run_duration_s"]),
        ("Alert Check Count", alert["count"]),
        ("Alert Check Avg TTFT (ms)", alert_overall["TTFT (ms)"]["mean"]),
        ("Alert Check Avg TPOT (ms)", alert_overall["TPOT (ms)"]["mean"]),
        ("Alert Check Avg Throughput (tok/s)", alert_overall["Throughput (tok/s)"]["mean"]),
        ("Alert Check Avg Latency (ms)", alert_overall["Latency (ms)"]["mean"]),
        ("Alert Check p90 Latency (ms)", alert_overall["Latency (ms)"]["p90"]),
        ("Deep Analysis Count", deep["count"]),
        ("Deep Analysis Avg TTFT (ms)", deep_overall["TTFT (ms)"]["mean"]),
        ("Deep Analysis Avg TPOT (ms)", deep_overall["TPOT (ms)"]["mean"]),
        ("Deep Analysis Avg Throughput (tok/s)", deep_overall["Throughput (tok/s)"]["mean"]),
        ("Deep Analysis Avg Output Tokens", deep_overall["Output Tokens"]["mean"]),
        ("Deep Analysis Avg Generation Latency (ms)", deep_overall["Generation Latency (ms)"]["mean"]),
        ("Deep Analysis Avg E2E Latency (ms)", deep_overall["E2E Latency (ms)"]["mean"]),
        ("Deep Analysis p90 E2E Latency (ms)", deep_overall["E2E Latency (ms)"]["p90"]),
        ("Alert Checks per Minute", rates["Alert Checks per Minute"]),
        ("Deep Analyses per Minute", rates["Deep Analyses per Minute"]),
    ]
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for key, value in rows:
            writer.writerow([key, _rtsu_na(value)])
    print(f"WSF formatted scene understanding metrics written to: {output_file}")
    return output_file


def print_rtsu_console_summary(summary, report_dir):
    """Print compact alert-check and deep-analysis tables (overall scope)."""
    run, rates = summary["run"], summary["rates"]
    print(f"\nReal-Time Scene Understanding: stop reason '{run['stop_reason']}', "
          f"{run['streams']} stream(s), {_rtsu_na(run['run_duration_s'])} s")
    tables = (
        ("Alert check", summary["alert_check"], RTSU_ALERT_CHECK_KPIS),
        ("Deep analysis", summary["deep_analysis"], RTSU_DEEP_ANALYSIS_KPIS),
    )
    for title, section, kpis in tables:
        print(f"\n{title} (overall, {section['count']} samples)")
        print(f"{'KPI':<26}" + "".join(f"{label:>12}" for label in ("Min", "Max", "Mean", "p90")))
        for kpi in kpis:
            stats = section["kpis"]["overall"][kpi]
            print(f"{kpi:<26}" + "".join(f"{_rtsu_na(stats[key]):>12}" for key in RTSU_STAT_KEYS))
    print(f"\nAlert checks per minute: {_rtsu_na(rates['Alert Checks per Minute'])}, "
          f"deep analyses per minute: {_rtsu_na(rates['Deep Analyses per Minute'])}")
    print(f"Scene understanding reports saved in: {report_dir}")


def save_rtsu_reports(report_dir, run):
    """
    Compute Real-Time Scene Understanding KPIs and write every report file.

    Args:
        report_dir: Application report directory (``<report>/scene_understanding``).
        run: Collected run data with ``metadata``, ``stream_ids``, ``alert_check_samples``,
            ``deep_analyses``, ``initial_anchors``, ``post_mid_client``,
            ``publishing_first_seen_client``, ``clock_offset_s``, ``run_start_client``,
            ``run_end_client``, ``stop_reason``, ``stream_start_status``,
            ``target_deep_analyses``, ``counters``, and ``reconnects``.

    Returns:
        dict: The summary written to ``rtsu_summary_metrics.json``.
    """
    os.makedirs(report_dir, exist_ok=True)
    stream_ids = run["stream_ids"]
    samples = run["alert_check_samples"]
    initial_anchors = run["initial_anchors"]

    deep_rows = build_rtsu_deep_analysis_rows(run["deep_analyses"], samples, initial_anchors, run["clock_offset_s"])
    anchors = refine_rtsu_anchors(initial_anchors, deep_rows)
    alert_rows = build_rtsu_alert_check_rows(samples, anchors)
    startup_rows = build_rtsu_startup_rows(
        stream_ids, run["post_mid_client"], run["publishing_first_seen_client"], alert_rows, deep_rows, anchors,
    )

    start, end = run["run_start_client"], run["run_end_client"]
    duration_s = round(end - start, 2) if start is not None and end is not None else None

    # Stale GenAI values repeat an earlier caption's metrics: keep them in the rows, not in the statistics.
    stats_rows = [
        dict(row, **{kpi: None for kpi in RTSU_GENAI_KPIS}) if row["Possibly Stale"] else row
        for row in alert_rows
    ]
    observed = sum(1 for row in alert_rows if row["Sample"] == "observed")
    summary = {
        "run": {
            "stop_reason": run["stop_reason"],
            "stream_start_status": run["stream_start_status"],
            "streams": len(stream_ids),
            "stream_ids": stream_ids,
            "target_deep_analyses": run["target_deep_analyses"],
            "run_duration_s": duration_s,
        },
        "alert_check": {
            "count": len(alert_rows),
            "observed": observed,
            "missed": len(alert_rows) - observed,
            "kpis": summarize_rtsu_kpis(stats_rows, RTSU_ALERT_CHECK_KPIS),
        },
        "deep_analysis": {
            "count": len(deep_rows),
            "confirmed": sum(1 for row in deep_rows if row["Confirmed"] is True),
            "kpis": summarize_rtsu_kpis(deep_rows, RTSU_DEEP_ANALYSIS_KPIS),
        },
        "startup": startup_rows,
        "rates": compute_rtsu_rates(alert_rows, deep_rows, stream_ids, duration_s),
        "data_quality": compute_rtsu_data_quality(samples, deep_rows, run["counters"], run["reconnects"]),
    }

    metadata = dict(run["metadata"])
    metadata["run_duration_s"] = duration_s
    metadata["stream_start_anchors"] = {
        stream_id: {"initial": _rtsu_round(initial_anchors.get(stream_id), 6), "refined": _rtsu_round(anchors.get(stream_id), 6)}
        for stream_id in stream_ids
    }

    summary_file = os.path.join(report_dir, "rtsu_summary_metrics.json")
    _write_rtsu_json(os.path.join(report_dir, "rtsu_run_metadata.json"), metadata)
    deep_columns = RTSU_DEEP_ANALYSIS_COLUMNS + (["Artifacts"] if any("Artifacts" in row for row in deep_rows) else [])
    _write_rtsu_csv(os.path.join(report_dir, "rtsu_alert_check_indv_metrics.csv"), RTSU_ALERT_CHECK_COLUMNS, alert_rows)
    _write_rtsu_csv(os.path.join(report_dir, "rtsu_deep_analysis_indv_metrics.csv"), deep_columns, deep_rows)
    _write_rtsu_csv(os.path.join(report_dir, "rtsu_startup_metrics.csv"), RTSU_STARTUP_COLUMNS, startup_rows)
    _write_rtsu_json(summary_file, summary)
    _write_rtsu_summary_csv(os.path.join(report_dir, "rtsu_summary_metrics.csv"), summary)
    save_rtsu_metrics_to_wsf_format(report_dir, summary_file)
    print_rtsu_console_summary(summary, report_dir)
    return summary
