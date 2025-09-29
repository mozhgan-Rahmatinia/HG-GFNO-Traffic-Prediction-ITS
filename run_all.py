import subprocess
import os
import time 



original_config_path = './configurations/PEMS07_mgcn.conf'
temp_config_path = './configurations/PEMS07_temp.conf'
script_to_run = 'run.py'
predict_values = [48, 96]

print("begin experiments for PEMS07..", flush=True)
print("-" * 40, flush=True)
time.sleep(1)

for value in predict_values:
    print(f"current experiment num_for_predict = {value}", flush=True)

    with open(original_config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('num_for_predict'):
            new_lines.append(f'num_for_predict = {value}\n')
        else:
            new_lines.append(line)
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    command = ['python', script_to_run, '--config', temp_config_path]

    try:

        subprocess.run(command, check=True)
        # =================================

        print(f"\nexperiment for {value} finish successfully", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! error for {value} !!!", flush=True)

        print(f"کد خطای بازگشتی: {e.returncode}", flush=True)
        break
    finally:
        print("-" * 40, flush=True)
        time.sleep(1)

if os.path.exists(temp_config_path):
    os.remove(temp_config_path)
    print(f"temp file({temp_config_path}) delete.", flush=True)



original_config_path = './configurations/PEMS08_mgcn.conf'
temp_config_path = './configurations/PEMS08_temp.conf'
script_to_run = 'run.py'
predict_values = [12, 24, 48, 96]

print("begin experiments for PEMS08..", flush=True)
print("-" * 40, flush=True)
time.sleep(1)

for value in predict_values:
    print(f"current experiment num_for_predict = {value}", flush=True)

    with open(original_config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith('num_for_predict'):
            new_lines.append(f'num_for_predict = {value}\n')
        else:
            new_lines.append(line)
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    command = ['python', script_to_run, '--config', temp_config_path]

    try:

        subprocess.run(command, check=True)
        # =================================

        print(f"\nexperiment for {value} finish successfully", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! error for {value} !!!", flush=True)

        print(f"کد خطای بازگشتی: {e.returncode}", flush=True)
        break
    finally:
        print("-" * 40, flush=True)
        time.sleep(1)

if os.path.exists(temp_config_path):
    os.remove(temp_config_path)
    print(f"temp file({temp_config_path}) delete.", flush=True)




print("all experiments complete successfully.", flush=True)