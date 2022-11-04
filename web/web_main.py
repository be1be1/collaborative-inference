from flask import Flask, request, jsonify, flash, make_response, send_from_directory
import os
import sys
import inspect
import time
import json
import tarfile
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from model_representation import ModelStruction
from source_code_rewriting import SourceCodeUpdate
import shutil
import yaml
import random
import logging
import subprocess


app = Flask(__name__)

WORK_DIR = '/data/wangds/collaborative-inference'
UPLOAD_FOLDER = '/data/wangds/collaborative-inference/dnn_models'
REFACTOR_FOLDER = "/data/wangds/collaborative-inference/refactor_dnn_models"
JOB_LIST_PATH = "/data/wangds/collaborative-inference/web/job_list.json"
DATASET_FOLDER = "/data/wangds/collaborative-inference/dataset"
RUN_FOLDER = "/data/wangds/collaborative-inference/runs"
YAML_FOLDER = "/data/wangds/collaborative-inference/yamls"
ALLOWED_EXTENSIONS = ["py"]
app.config['WORK_DIR'] = WORK_DIR
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REFACTOR_FOLDER'] = REFACTOR_FOLDER
app.config['JOB_LIST_PATH'] = JOB_LIST_PATH
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['RUN_FOLDER'] = RUN_FOLDER
app.config['YAML_FOLDER'] = YAML_FOLDER


# 执行分解操作
@app.route("/dnn/refactor", methods=["GET"])
def receive_file():
    url = request.args.get("url")  # 源码下载地址
    job_id = request.args.get("job_id")  # 作业id
    model_name = url.split("-")[-1].split(".")[0]
    file_name = f'{model_name}_{job_id}.py'  # 拼接文件名
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)  # 下载路径
    if os.path.exists(save_path):
        os.remove(save_path)
    flag = os.system(f"wget --no-check-certificate {url} -O {save_path}")
    if flag != 0:
        # 下载失败重试
        for i in range(10):
            retry = os.system(f"wget --no-check-certificate {url} -O {save_path}")
            if retry == 0:
                print("重试成功")
                break
            else:
                time.sleep(1)
        else:
            print("下载失败，超过最大重试次数")
    else:
        print("源码下载成功")

    # 源码路径
    source_path = save_path

    code_str = \
f'''
from dnn_models.{file_name[:-3]} import *
global model
model={model_name}
'''
    c = compile(code_str, "", mode="exec")
    exec(c)
    model = globals()["model"]

    return_code = refactor(model, source_path)

    job_data = {
        "url": url,
        "model_name": model_name,
        "file_name": file_name,
        "path": source_path,
        "re_code": return_code
    }

    if not os.path.exists(app.config['JOB_LIST_PATH']):  # 第一条数据
        json_str = json.dumps({job_id: job_data})
        with open(app.config['JOB_LIST_PATH'], "w") as f:
            f.write(json_str)
    else:  # 不为空 读取json
        with open(app.config['JOB_LIST_PATH'], "r") as f:
            data = json.load(f)
        if job_id not in list(data.keys()):
            data[job_id] = job_data
        json_str = json.dumps(data)
        with open(app.config['JOB_LIST_PATH'], "w") as f:
            f.write(json_str)

    return jsonify({"code": 0, "message": "success", "model": model_name, "job_id": job_id})


# 获取分解结果
@app.route('/dnn/file', methods=['GET'])
def get_file():
    job_id = request.args.get("job_id")  # 作业id
    with open(app.config['JOB_LIST_PATH'], 'r') as f:
        data = f.read()
    if not bool(data):
        return jsonify({"code": 1, "message": "此作业分解结果，请先执行源码分解", "job_id": job_id})

    with open(app.config['JOB_LIST_PATH'], "r") as f:
        data = json.load(f)
    if job_id not in list(data.keys()):
        return jsonify({"code": 1, "message": "此作业分解结果，请先执行源码分解", "job_id": job_id})
    if "re_code" not in list(data[job_id].keys()):
        return jsonify({"code": 1, "message": "此作业分解结果，请先执行源码分解", "job_id": job_id})
    re_code = data[job_id]['re_code']
    with open(f"/tmp/{'refactor_' + data[job_id]['file_name']}", "w") as f:
        f.write(re_code)
    try:
        m_res = make_response(send_from_directory("/tmp", 'refactor_' + data[job_id]['file_name'], as_attachment=True))
        return m_res
    except Exception as e:
        return jsonify({"code": 1, "message": "{}".format(e)})
    # return jsonify({"code": 0, "message": "success", "result": m_res})


# 运行分解后的代码
@app.route("/dnn/run", methods=["GET"], strict_slashes=False)
def run_model():
    data_url = request.args.get("url")  # 数据下载地址
    job_id = request.args.get("job_id")  # 作业id

    # 下载数据并解压到指定目录
    # data_save_path = os.path.join(app.config['DATASET_FOLDER'], job_id + ".tar.gz")
    # flag = os.system(f"wget --no-check-certificate {data_url} -O {data_save_path}")
    # if flag != 0:
    #     # 下载失败重试
    #     for i in range(10):
    #         retry = os.system(f"wget --no-check-certificate {data_url} -O {data_save_path}")
    #         if retry == 0:
    #             print("重试成功")
    #             break
    #         else:
    #             time.sleep(1)
    #     else:
    #         print("下载失败，超过最大重试次数")
    # else:
    #     print("数据集下载成功")
    #
    # data_folder = os.path.join(app.config['DATASET_FOLDER'], job_id)  # 解压到此目录
    # if os.path.exists(data_folder):
    #     shutil.rmtree(data_folder)
    # os.mkdir(data_folder)
    # # 解压操作
    # tf = tarfile.open(data_save_path)
    # tf.extractall(data_folder)
    #
    # all_data = []  # 数据路径列表
    # for p, _, files in os.walk(data_folder):
    #     if files:
    #         for f in files:
    #             all_data.append(os.path.join(p, f))

    with open(app.config['JOB_LIST_PATH'], "r") as f:
        job_list = json.load(f)
    if job_id not in list(job_list.keys()):
        return jsonify({"code": 1, "message": "请先执行源码分解", "job_id": job_id})
    file_name = job_list[job_id]['file_name']
    model_name = job_list[job_id]['model_name']

    run_file_path = os.path.join(app.config['RUN_FOLDER'], file_name)
    source_code_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)

    call_code = generate_run_code(data_url, model_name, source_code_path, job_id)
    if os.path.exists(run_file_path):
        os.remove(run_file_path)
    with open(run_file_path, "w") as f:
        f.write(call_code)

    # 生成当前作业的yaml模板
    template_path = os.path.join(app.config['YAML_FOLDER'], "ulian-cluster.autoscaler.yaml")
    yaml_path = os.path.join(app.config['YAML_FOLDER'], f"job-{job_id}.yaml")
    generate_yaml(template=template_path, path=yaml_path, job_id=job_id)

    # 创建命名空间
    ns = f"ulian-job-{job_id}"
    cmd = f'kubectl get ns | grep {ns}'
    logging.info(cmd)
    out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    infos = out.stdout.read().splitlines()
    if not infos:
        os.system(f"kubectl create namespace {ns}")
        # 创建集群
        os.system(f"kubectl -n {ns} apply -f {yaml_path}")

    time.sleep(5)  # 等待svc创建

    # 干掉之前的转发进程
    cmd = 'ps aux | grep "head-svc 8265:8265" | grep -v grep'
    logging.info(cmd)
    out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    infos = out.stdout.read().splitlines()
    if infos:
        pid = int(infos[0][9:16].strip())
        # 杀死进程
        os.system(f"kill -9 {pid}")

    # 端口转发 非阻塞执行
    os.system(f"kubectl port-forward service/ulian-job-{job_id}-head-svc 8265:8265 -n {ns} > out.file 2>&1 &")
    time.sleep(3)  # 等待同步端口转发

    # 提交作业，获取作业提交id
    submission_id = job_commit(file_name)

    with open(app.config['JOB_LIST_PATH'], "r") as f:
        data = json.load(f)
    data[job_id]['submission_id'] = submission_id
    json_str = json.dumps(data)
    with open(app.config['JOB_LIST_PATH'], "w") as f:
        f.write(json_str)

    return jsonify({"code": 0, "message": "success", "detail": "作业开始运行", "cluster": "zhijiang-gpu5", "namespace": f"{ns}"})


# 获取作业结果
@app.route('/dnn/result', methods=['GET'])
def get_result():
    job_id = request.args.get("job_id")  # 作业id
    with open(app.config['JOB_LIST_PATH'], "r") as f:
        data = json.load(f)
    if job_id not in list(data.keys()):
        return jsonify({"code": 1, "message": "error", "detail": "此作业无执行结果，请先执行源码分解再运行", "job_id": job_id})
    submission_id = data[job_id]['submission_id']

    s = os.popen(f"ray job logs '{submission_id}'").readlines()
    if s[-1].startswith("time"):
        return jsonify({"code": 0, "result": f"{s[-2].strip()}  {s[-1].strip()}", "message": "运行成功"})
    elif s[-1].startswith("fail") or s[-1].startswith("Killed"):
        # 作业执行失败，重新提交
        file_name = data[job_id]['file_name']
        submission_id = job_commit(file_name)
        data[job_id]['submission_id'] = submission_id
        json_str = json.dumps(data)
        with open(app.config['JOB_LIST_PATH'], "w") as f:
            f.write(json_str)
        return jsonify({"code": 1, "result": "", "message": "作业正在运行"})
    else:
        return jsonify({"code": 1, "result": "", "message": "作业正在运行"})


# 生成yaml模板
def generate_yaml(template, path, job_id):
    with open(template) as f:
        content = yaml.load(f, Loader=yaml.FullLoader)
    replicas = random.choice([3, 5, 7])
    content['spec']['workerGroupSpecs'][0]['replicas'] = replicas
    content['spec']['workerGroupSpecs'][0]['minReplicas'] = replicas
    content['spec']['workerGroupSpecs'][0]['maxReplicas'] = replicas
    content['metadata']['name'] = f"ulian-job-{job_id}"
    with open(path, "w") as f:
        yaml.dump(content, f)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def refactor(model, source_path):
    # 以str形式提取model定义源码
    model_source_code = inspect.getsource(model)

    model = model()
    model.eval()

    # 获取网络层信息和自动生成的代码
    print("获取网络层信息...")
    scu = SourceCodeUpdate(source_path, model)
    nodes = scu.get_nodes_and_code()

    # 模型层分块打包
    print("网络层分块打包...")
    model_struction = ModelStruction(nodes)
    blocks = model_struction.get_blocks()

    # 需要插入的源码
    print("重构代码生成...")
    generate_codes = scu.generate_forward(blocks)

    print("注释源码中的forward方法，添加重构代码...")
    new_code = scu.modify_foward(generate_codes, model_source_code)
    return_code = new_code.replace("@ray.remote", "@ulian.collaborate")
    print("替换源码...")
    scu.replace_source_code(new_code)
    print(f"success, modify source code -> {source_path}")
    # 转换UNIX编码
    os.system(f"sed -i 's/^M//g' {source_path}")
    return return_code


def job_commit(file_name):
    # 作业提交代码
    submit_code = \
f'''from ray.job_submission import JobSubmissionClient
client = JobSubmissionClient("http://localhost:8265")
kick_off_benchmark = (
    # Clone ray. If ray is already present, don't clone again.
    #"git clone https://github.com/ray-project/ray || true;"
    # Run the benchmark.
    "python3 {file_name}"
    #" --size 100G --disable-check"
)
global submission_id
submission_id = client.submit_job(
    entrypoint=kick_off_benchmark,
    runtime_env={{
        "working_dir": {app.config['RUN_FOLDER']}
    }}
)
'''
    c = compile(submit_code, "", mode="exec")
    exec(c)
    submission_id = globals()["submission_id"]
    return submission_id


def generate_run_code(data_url, model_name, source_code, job_id):
    with open(source_code, "r") as f:
        sc = f.read()
    body = \
f'''import torch
import time
import requests, warnings
import tarfile
import shutil
import os
warnings.filterwarnings("ignore")
s = requests.Session()
for i in range(10):
    try:
        res = s.get({data_url}, verify=False)
        print("下载成功")
        with open("googlenet.tar.gz", "wb") as f:
            f.write(res.content)
        data_folder = os.path.join("/tmp", {job_id})
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
        os.mkdir(data_folder)
        tf = tarfile.open("googlenet.tar.gz")
        tf.extractall(data_folder)
        break
    except:
        print("下载失败, 正在重试...")
        continue
else:
    print("fail， 超过最大重试次数")
    exit()
all_data = []  # 数据路径列表
for p, _, files in os.walk(data_folder):
    if files:
        for f in files:
            all_data.append(os.path.join(p, f))

model = {model_name}
t1 = time.time()
for p in all_data:
    x = torch.load(p)
    y = model.forward(x)
    print("current frame: %s %s"% (type(y), y.shape))
t2 = time.time() - t1
print(f"inference result: {{ray.get(y).shape}})
print(f"time： {{t2}}s")
'''
    call_code = sc + "\n" + body
    return call_code


if __name__ == '__main__':
    app.run(host="0.0.0.0",
            port=5000,
            debug=False)