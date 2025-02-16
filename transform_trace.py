import pandas as pd
import os
from datetime import datetime
import sys

def convert_trace(input_csv_path, output_csv_path):
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_csv_path}")
        sys.exit(1)
    
    # GPU 작업 필터링 및 정렬
    # TODO 1) COMPLETED 뿐만 아니라 Failed도 trace에 충분히 넣을 수 있음 
    # TODO 1-a) 그러려면 Duration 계산 관련해서, Failed의 경우 Duration 시간 계산 고려해야 함.  

    # TODO 2) type: Pretrain, Evaluation, SFT 넣을 수 있음. (Debug, Other X)
    # TODO 3) Bandwidth의 경우, Pretrain & Evaluation 각각 숫자 많은 변경 필요함. e.g. Pretrain => 400000 Inference => 1200, 지금 Bandwidth 너무 작음 
    # TODO 4) cpu_per_gpu_worker,cpu_per_ps_worker 이거 어떻게 해야할지? 일단 PS는 없는데 
    # TODO 5) 
    gpu_jobs = df[
        (df['gpu_num'] > 0) & 
        (df['type'].isin(['Pretrain', 'Other', 'SFT'])) &
        (df['state'] == 'COMPLETED')
    ].copy()
    
    gpu_jobs['submit_time'] = pd.to_datetime(gpu_jobs['submit_time'])
    gpu_jobs = gpu_jobs.sort_values('submit_time')
    
    # arrival_time 계산
    base_time = gpu_jobs['submit_time'].min()
    gpu_jobs['arrival_time'] = (gpu_jobs['submit_time'] - base_time).dt.total_seconds()
    
    # GPU 활용도 계산
    gpu_jobs['gpu_utilization'] = gpu_jobs['gpu_time'] / (gpu_jobs['gpu_num'] * gpu_jobs['duration'])
    
    # 기본 iteration 시간 (0.5초)
    avg_iter_time = 0.5  
    
    # 결과 DataFrame 생성
    result = pd.DataFrame({
        'job_id': range(len(gpu_jobs)),
        'arrival_time': gpu_jobs['arrival_time'],
        'num_iteration': (gpu_jobs['duration'] / avg_iter_time).astype(int),
        'iteration_computing_time': gpu_jobs['gpu_time'] / (gpu_jobs['gpu_num'] * gpu_jobs['duration']) * avg_iter_time,
        'iteration_networking_time': gpu_jobs.apply(
            lambda x: avg_iter_time * (1.5 if x['type'] == 'Pretrain' else 0.5), axis=1
        ),
        'gpu_workers': gpu_jobs['node_num'],
        'ps': 0,
        'gpu_per_worker': gpu_jobs['gpu_num'] / gpu_jobs['node_num'],
        'cpu_per_gpu_worker': gpu_jobs['cpu_num'] / gpu_jobs['gpu_num'],
        'cpu_per_ps_worker': gpu_jobs['cpu_num'] / gpu_jobs['gpu_num']
    })
    
    # tensorsize 설정 (mem_per_pod_GB가 있으면 사용, 없으면 기본값)
    if 'mem_per_pod_GB' in gpu_jobs.columns:
        result['tensorsizes'] = gpu_jobs['mem_per_pod_GB'] / 100
    else:
        result['tensorsizes'] = 5.0  # 기본값
    
    # profiled_network 설정
    result['profiled_network'] = gpu_jobs.apply(
        lambda x: 1000 if x['type'] == 'Pretrain' else 500, axis=1
    )
    
    # 결과 저장
    try:
        result.to_csv(output_csv_path, index=False)
        print(f"Successfully converted trace and saved to {output_path}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    
    return result

def main():
    # 현재 작업 디렉토리 기준으로 경로 설정
    base_dir = os.getcwd()
    
    # Input/output paths
    kalos_input = os.path.join(base_dir, 'data', 'job_trace', 'trace_kalos.csv')
    seren_input = os.path.join(base_dir, 'data', 'job_trace', 'trace_seren.csv')
    
    output_dir = os.path.join(base_dir, 'data', 'converted_trace')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert both traces
    for input_path, name in [(kalos_input, 'kalos'), (seren_input, 'seren')]:
        output_path = os.path.join(output_dir, f'converted_{name}_trace.csv')
        convert_trace(input_path, output_path)

if __name__ == "__main__":
    main()