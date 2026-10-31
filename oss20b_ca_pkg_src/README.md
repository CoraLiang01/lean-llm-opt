# Air-NRM 大规模 CA · gpt-oss:20b 服务器运行包

包含15道 `query_CA.csv`、五份输入CSV、CA few-shot示例、分类参考文件、独立CA notebook、
终端启动器和评分工具。变量规模112–421。模型和embedding均走Ollama，无需OpenAI key。
适用服务器：Linux x86_64、Python3.11、可访问的Ollama。

## 先检查环境、安装依赖并试跑

把整个压缩包上传到服务器并解压，在解压目录执行：

```bash
bash setup.sh
```

先诊断服务，创建本包的 `.venv` 并安装固定依赖，离线自检后真实试跑最小/最大两题。
若默认Python不是3.11：`PYTHON_BIN=python3.11 bash setup.sh`。
模型服务器不在本机：

```bash
OLLAMA_BASE_URL=http://推理服务器地址:11434 bash setup.sh
```

只诊断服务器、不安装依赖、不调用推理：

```bash
python3 check_server.py --url http://推理服务器地址:11434
```

必须已有 `gpt-oss:20b` 和 `nomic-embed-text`；检查支持JSON空格/缩进、embedding的
`:latest` 标签，以及旧环境变量 `OLLAMA_URL`。可选版本/硬件接口失败不会阻止实验。
如果只有vLLM/SGLang等OpenAI兼容接口，本包会指出接口不匹配；当前运行路线是Ollama。
缺模型时请在提供Ollama的机器执行 `ollama pull gpt-oss:20b`、
`ollama pull nomic-embed-text`；服务本身也需已经启动。

## 后台正式运行：15题 × 3轮

```bash
nohup .venv/bin/python -u run_airnrm_ca.py \
  --repeats 3 \
  --output-dir airnrm_ca_results \
  > airnrm_ca.log 2>&1 &
```

```bash
tail -f airnrm_ca.log
```

远程服务地址可加 `--ollama-url http://推理服务器地址:11434`。
中断后重跑相同命令，续接已有轮次并跳过成功题；另起新批次换一个 `--output-dir`。
试跑默认写 `smoke_runs/`，正式实验默认写 `runs/`；不同模式禁止共用目录。
程序保留原始回答和逐题错误；有题失败会返回非零状态并停止后续轮次。

## 这次修复了什么

协议标记为 `ca-direct-full-v1`。保留LEAN分类、CSVQA工具、检索和CA few-shot；当前题
数据由五个CSV精确连接。用简短的 `(flight_id, product_id)` 键减少大字典抄写长度。
模型必须返回数学说明及完整、自包含的Gurobi代码，十组参数各自独占一行。
严格核对参数是否完整且与当前题一致、代码能否解析，以及有无Agent停止消息。
原始回答在检查之前保存；失败不会被标成成功，不在事后补参数、替换数值或修理代码。

与8月旧版本有明确协议差别：旧CA把数据Agent最终文本拼进固定数学/代码模板；
本版让模型一次输出完整数学说明和程序，检查后原样提取代码。新实验须按此协议标记，
不要与旧版本原始输出/后处理结果混作同一批重复。
CA只用连续sales/no-purchase变量，不加NP的binary、航班数量限制或机场流守恒。
保留原CA few-shot候选航班，不另外更换实验示例。

## 回传与评分

自动生成 `results_airnrm_ca_时间.tar.gz`（试跑为 `smoke_airnrm_ca_时间.tar.gz`）。
将该压缩包和 `airnrm_ca.log` 发回。包里包含本批运行记录、配置、原始模型输出和硬件报告。
Python任务机器与远程推理机器不同时，硬件报告会注明本机GPU来源，不冒充远程GPU。

生成代码不需要求解许可证；实际正确率还要执行代码、比对目标值。若服务器可用Gurobi，
可自行评分（否则将原始结果回传后评分）：

```bash
mkdir -p scoring
.venv/bin/python score_runs.py airnrm_ca_results \
  --dataset Air-NRM-CA-LS --tol 0.0001 --time-limit 120 \
  -o scoring/gold_labels.csv
```

`scoring/scored_all.csv` 是分轮逐题结果，`status=ok` 仅代表完整输出检查通过，
不等于优化目标正确。评分使用相对容差 `1e-4`。

## 其他命令

```bash
bash setup.sh --offline                 # 安装依赖、离线检查，不连接Ollama
.venv/bin/python run_airnrm_ca.py --check
.venv/bin/python run_airnrm_ca.py --preflight
.venv/bin/python run_airnrm_ca.py --smoke
python3 run_airnrm_ca.py --dry-run       # 仅预览，不联网、不写结果
```

离线检查会用假响应验证批处理，并删除临时记录；不会伪装成真实OSS实验。
可用 `verify_package.py --solve-reference` 额外检查参考LP与15题标签是否对齐，需要Gurobi。
上下文仍为65536、最大生成32768、reasoning单独返回；真实两题试跑检查当前服务与显存。
