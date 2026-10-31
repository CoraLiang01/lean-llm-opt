# Air-NRM 大规模 NP / Flow · gpt-oss:20b 独立服务器运行包

这个包运行 LEAN-LLM-OPT 的 **Air-NRM-NP-LS**，使用 `query_NP_Flow.csv` 的15题，
每题32–115个候选航班、174–621个变量。包含五份输入CSV、Flow示例、分类参考数据、
notebook和日志工具。默认模型和embedding都走Ollama，不需要OpenAI API key。

## 新服务器类型还不清楚时

解压后先执行这个命令；仅用Python标准库，无须先安装依赖，不调用推理：

```bash
python3 check_server.py
# 服务不在当前机器/默认端口时：
python3 check_server.py --url http://实际服务器地址:端口
```

它区分：Ollama可用且模型齐全、模型缺失、地址不可达，以及识别到其他
OpenAI兼容接口（后者可能是vLLM/SGLang，单凭接口不能确定后台软件）。
如果只提供其他接口，当前运行包会明确停止，需要适配后再跑。

## 服务器首次运行

请在连接服务器后，把**整个压缩包**上传、解压，在解压后的文件夹运行。
需要Linux x86_64和Python 3.11；Python代码的位置可以任意移动，不要修改notebook。

```bash
bash setup.sh
```

这条命令先检查服务器；通过后创建独立的 `.venv`、安装固定版本依赖、检查依赖一致性及文件完整性、
离线检查15题，再用最小和最大两题做真实模型试跑。

如果服务器默认 `python3` 不是3.11：

```bash
PYTHON_BIN=python3.11 bash setup.sh
```

如果已有conda，可先 `conda create -n airnrm-np python=3.11 -y` 并
`conda activate airnrm-np`，再运行 `bash setup.sh`。

## Ollama服务器地址

Python任务直接在提供Ollama的服务器上运行时，默认连接 `http://localhost:11434`。
另一台推理服务器已经提供Ollama时，只需要传地址，无须在Python任务所在机器安装Ollama：

```bash
OLLAMA_BASE_URL=http://服务器地址:11434 bash setup.sh
```

也兼容旧脚本使用的 `OLLAMA_URL`；优先级为命令行 `--ollama-url`、
`OLLAMA_BASE_URL`、`OLLAMA_URL`、默认localhost。地址末尾的 `/` 或 `/api/`
会被规范化。模型列表通过JSON解析，返回格式中的空格、缩进和字段顺序不会影响识别。
只要标签准确，`nomic-embed-text` 和 `nomic-embed-text:latest` 都能识别。
服务版本、模型详情或本机GPU信息无法读取时会记录原因，继续试跑。

Ollama服务需要已经启动，并且有以下两个准确的模型标签；缺少时脚本会指出哪个缺少：

```bash
# 以下命令在提供Ollama的机器执行。
ollama serve
# 如果服务已经运行，不需要重复启动。新开的终端中下载模型：
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

此包不会自动安装系统服务、下载十几GB模型或改变服务器监听地址。
服务器如果只有vLLM/SGLang的OpenAI兼容接口，不能把它的地址直接当作Ollama地址；
此包明确使用Ollama的 `/api/*` 协议和本地embedding。

## 正式运行

两题试跑通过后，跑全部15题、重复3轮：

```bash
nohup .venv/bin/python -u run_airnrm_np.py --repeats 3 > console.log 2>&1 &
```

远程Ollama可直接传参：

```bash
nohup .venv/bin/python -u run_airnrm_np.py --repeats 3 \
  --ollama-url http://服务器地址:11434 > console.log 2>&1 &
```

`tail -f console.log` 查看进度。中断后重跑相同命令，按轮次接着已有结果运行，
已经成功的题跳过；失败的题会重新尝试。需要另起一批时指定 `--output-dir runs_batch2`。
试跑写入 `smoke_runs/`，正式运行写入 `runs/`，两者不混用。

每轮都会检查15道题的实际状态、NP分类、完整参数和代码原样提取是否成功。
即使底层notebook进程返回0，只要有题失败，启动器也会报告失败并停止后续轮次，
保留全部原始回答和错误，避免把“进程跑完”误认为“15题成功”。
代码不自动补参数、不修改模型回答、不删失败题。

## 回传

程序自动生成 `results_airnrm_np_时间.tar.gz`；试跑生成 `smoke_airnrm_np_时间.tar.gz`。
把压缩包和 `console.log` 发回即可。压缩包包含原始运行记录、配置、硬件/模型信息和运行日志。
日志保留在 `logs/`；失败细节见每题 `instances.jsonl` 的 `errors`。

硬件报告自动收集Ollama版本、模型详情/量化和Python任务所在机器的GPU。
如果Ollama在另一台机器，GPU报告不会冒充推理服务器GPU；可额外设置
`HARDWARE_NOTE='推理服务器GPU型号、显存、驱动/CUDA信息'`，该值会保存到硬件报告。

## 其他命令

```bash
# 只装依赖和离线检查，不需要Ollama
bash setup.sh --offline

# 已装好环境，只做离线检查
.venv/bin/python run_airnrm_np.py --check

# 检查服务和模型是否存在，不调用推理
.venv/bin/python run_airnrm_np.py --preflight

# 真实两题试跑
.venv/bin/python -u run_airnrm_np.py --smoke

# 只预览计划，不联网、不写结果
python3 run_airnrm_np.py --dry-run
```

生成模型和代码只需能导入 `gurobipy`，不在服务器执行生成的代码。
完整输出检查不等于优化目标正确；要获得求解正确率，需在有Gurobi许可证的环境评分。
原始结果足以回传后统一评分，包中附带 `score_runs.py` 和 `aggregate_runs.py`。

本包保持 `num_ctx=65536`、`num_predict=32768`、`reasoning=true` 的现有OSS配置。
硬件需要能承载模型及该上下文的缓存；真实试跑才会验证显存和当前Ollama版本是否支持。
离线检查只证明包内代码、数据和初始化链路完整，不会伪装真实模型成功。
