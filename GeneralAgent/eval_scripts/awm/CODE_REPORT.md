# run_eval.py 代码报告

> 逆向工程式解读：从零开始，为什么每一行要这么写。
> 面向了解 Python 但不熟悉 async 的读者。

---

## 0. 一句话总结

`run_eval.py` 的核心逻辑是：**对每一个测试任务，启动一个独立的 MCP 工具服务器（HTTP + SQLite），让 LLM Agent 通过多轮对话调用服务器上的工具来完成任务，最后对比数据库的前后状态来判定成功/失败。**

可以类比为：给大模型一个「模拟器」（比如模拟了一个电商后台），告诉它「帮我创建一个订单」，看它能不能通过调 API 把事情做对。

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_eval.py (主进程)                      │
│                                                                 │
│  for each scenario:                                             │
│    for each task:                                               │
│      ┌──────────────────────────────────────────────────────┐   │
│      │  1. 创建 SQLite 数据库 (initial.db)                    │   │
│      │  2. 启动 MCP Server 子进程 (uvicorn + FastAPI)         │   │
│      │  3. Agent Loop:                                       │   │
│      │     ┌─────────┐    ┌──────────┐    ┌──────────────┐  │   │
│      │     │ LLM API │◄──►│ run_eval │◄──►│ MCP Server   │  │   │
│      │     │ (Qwen)  │    │ (调度器)  │    │ (localhost:N) │  │   │
│      │     └─────────┘    └──────────┘    └──────────────┘  │   │
│      │  4. 保存 trajectory + final.db                        │   │
│      │  5. 对比 initial.db vs final.db → 判定 complete/incomplete│   │
│      │  6. 杀掉 MCP Server                                   │   │
│      └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 为什么需要 async？

### 问题背景

这个脚本涉及三种 I/O 操作，每一种都可能让程序「等着」：
1. **调 LLM API**（网络请求，等大模型返回，几秒到几十秒）
2. **调 MCP Server**（HTTP 请求到 localhost，毫秒级）
3. **等 Server 启动**（轮询检测，最多 60 秒）

### 同步 vs 异步

```python
# ===== 同步写法（不用 async）=====
# 调 LLM 的时候，CPU 什么都不干，傻等网络响应
response = client.chat.completions.create(...)  # 阻塞 5 秒
tool_result = mcp.call_tool(...)                # 阻塞 0.1 秒

# ===== 异步写法（用 async）=====
# 调 LLM 的时候，event loop 可以去做别的事（比如检查 server 状态）
response = await client.chat.completions.create(...)  # 让出控制权
tool_result = await mcp.call_tool(...)                 # 让出控制权
```

**关键术语**：
- `async def`：声明一个「协程函数」，意思是「这个函数里面有需要等待的操作」
- `await`：遇到 I/O 操作时说「我先让出 CPU，等结果好了再回来」
- `asyncio.run(main())`：创建一个「事件循环」来调度所有 `await`

**在本脚本中**：虽然目前任务是串行的（一个接一个），async 仍然是必要的，因为：
- `AsyncOpenAI` 客户端只提供 async 接口
- `MCPToolExecutor` 的 `list_tools()` 和 `call_tool()` 是 async 的（内部用 mcp_agent 库）
- `async_wait_for_server()` 需要 async sleep 来轮询

如果不用 async，就需要用 `openai.OpenAI`（同步版）和自己写同步的 MCP 客户端。

---

## 3. AWM 的架构决定了代码必须这么写

### 3.1 AWM 是什么

AWM (Agent World Model) 提供了 1000 个「模拟世界」，每个世界是：
- 一个 **SQLite 数据库**（存储业务数据，如用户表、订单表）
- 一套 **FastAPI 路由**（约 30-50 个 REST API 端点）
- 一个 **MCP 适配层**（`fastapi-mcp` 把 REST API 暴露为 MCP 工具）
- 4-10 个 **任务**（如"创建一个名为 John 的客户"）
- 对应的 **验证器**（检查数据库是否被正确修改）

### 3.2 为什么每个 task 要启动独立的 MCP Server？

```
Task 0: "创建客户 John"  →  需要一个干净的空数据库
Task 1: "删除订单 #42"   →  需要一个预填了订单数据的数据库
Task 2: "修改客户邮箱"   →  同一个 scenario，但需要独立的数据库状态
```

每个 task 的**数据库初始状态**是独立的。Agent 执行 tool 调用时会修改数据库（INSERT/UPDATE/DELETE）。
验证器通过对比 `initial.db`（任务前）和 `final.db`（任务后）来判定结果。
所以每个 task 必须有自己的数据库实例 → 必须有自己的 HTTP Server。

### 3.3 为什么用 MCP 而不是直接调函数？

AWM 论文的设计选择：MCP (Model Context Protocol) 是 Anthropic 提出的标准化工具调用协议。
通过 MCP，Agent 可以：
1. **发现工具**：`list_tools` 返回所有可用的 API 列表和参数说明
2. **调用工具**：`call_tool` 通过 HTTP 调用具体的 API

这模拟了真实场景中 Agent 连接外部服务的方式，而不是直接在 Python 里 import 函数。
代价是：每个 task 都要启动/销毁一个 HTTP Server。

---

## 4. 逐段代码解析

### 4.1 环境配置 (L82-99)

```python
def setup_env_vars(args):
    os.environ["OPENAI_API_KEY"] = args.api_key
    os.environ["OPENAI_BASE_URL"] = args.base_url
```

**为什么设置 OpenAI 环境变量？**
AWM 库内部（`awm.core.agent`）也会创建 OpenAI 客户端。它读 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`
环境变量。实际连接的是内部 MaaS API（兼容 OpenAI 接口），不是 OpenAI 本身。

```python
    for var in ["http_proxy", "HTTP_PROXY", ...]:
        if var in os.environ:
            del os.environ[var]
    os.environ["no_proxy"] = "127.0.0.1,localhost"
```

**为什么要删 proxy？**
MCP Server 跑在 localhost 上。如果环境里配了企业代理（http_proxy），
Python 的 HTTP 库会把 `localhost:12345` 的请求也发到代理服务器，导致连不上本地 Server。

```python
    os.environ["HOST"] = "0.0.0.0"
```

**为什么绑定 0.0.0.0？**
在某些容器环境（如 K8s Pod）中，`127.0.0.1` 的 DNS 解析有问题（uvicorn 报 `[Errno -2]`）。
`0.0.0.0` 表示监听所有网络接口，绕过了这个问题。

### 4.2 Scenario 选择 (L102-116)

```python
def load_scenarios(args):
    tasks_data = tools_jsonl_load(TASKS_PATH)  # 加载 1000 个 scenario
    random.seed(args.seed)
    random.shuffle(indices)
    selected = [tasks_data[i] for i in indices[:args.num_scenarios]]
```

**为什么随机打乱后选前 N 个？**
AWM 有 1000 个 scenario，但跑全部太慢。随机选 100 个作为代表性子集。
固定 `seed=42` 保证**可复现**：同一个 seed 总是选出相同的 100 个 scenario。

**数据格式**（`gen_tasks.jsonl`）：
```json
{"scenario": "e_commerce_7", "tasks": ["创建客户 John...", "查询订单...", ...]}
```

### 4.3 核心：run_single_task (L144-343)

这是最关键的函数。分六个阶段。

#### 阶段 1：准备数据库 (L165-172)

```python
server_cfg = ServerConfig(
    scenario=scenario_name,      # e.g. "e_commerce_7"
    envs_load_path=ENVS_PATH,   # 生成的 FastAPI 代码所在文件
    db_schema_path=DB_PATH,      # 表结构定义
    sample_path=SAMPLE_PATH,     # 预填数据
)
db_file_path = _prepare_database(server_cfg, task_output_dir)
```

**做了什么？**
1. 从 `gen_db.jsonl` 读取 `e_commerce_7` 的 DDL（CREATE TABLE 语句）
2. 创建 SQLite 文件并执行 DDL
3. 从 `gen_sample.jsonl` 读取 INSERT 语句，填入初始数据
4. 保存为 `initial.db`（初始快照）和 `final.db`（工作副本，Agent 操作会修改它）

**为什么在主进程做这一步？**
历史原因。子进程（Server）也会重复做一次。可以优化但不影响正确性。

#### 阶段 2：启动 MCP Server (L174-177)

```python
port = get_random_available_port()       # 随机找一个空闲端口
server_proc = start_server_process(...)  # subprocess.Popen 启动子进程
mcp_url = f"http://127.0.0.1:{port}/mcp"
```

**进程树**（这是之前 bug 的来源）：
```
run_eval.py (主进程)
  └─ python -m awm.core.server (Popen 子进程)
       └─ /bin/bash -c "python server_code.py 2>&1 | tee server.log" (os.system)
            ├─ python server_code.py (实际的 uvicorn HTTP Server)
            └─ tee server.log
```

**为什么不直接在主进程里 import FastAPI app？**
因为每个 scenario 的 FastAPI 代码是**动态生成的**（从 `gen_envs.jsonl` 中读取）。
它不是一个固定的 Python 模块，而是一段约 2000 行的、scenario 特定的代码。
AWM 的做法是把它写成临时文件，然后 `python` 执行它。

**`get_random_available_port()` 的实现**：
```python
with socket.socket(AF_INET, SOCK_STREAM) as s:
    s.bind(('', 0))    # 让 OS 分配空闲端口
    s.listen(1)
    port = s.getsockname()[1]  # 拿到端口号
# socket 关闭后端口释放，留给 Server 用
```
注意：端口释放和 Server 绑定之间有微小的竞争窗口，但实践中极少冲突。

#### 阶段 3：等待 Server 就绪 (L190-197)

```python
if not await async_wait_for_server(port, timeout=args.server_timeout):
    result["error"] = "MCP server failed to start."
    return result
```

**为什么要等？**
Popen 只是启动了进程，Server 需要几秒来：
1. 解析 Python 代码（2000 行）
2. 初始化 FastAPI 路由
3. 初始化 FastApiMCP（把 REST API 转成 MCP 工具）
4. uvicorn 开始监听端口

**`async_wait_for_server` 的逻辑**（在 awm/tools.py 中）：
```python
async def async_wait_for_server(port, timeout=60.0):
    while time.time() - start < timeout:
        # 用 MCP 协议连接 Server，尝试列出工具
        running, tools_count, tools, err = await check_mcp_server(url, timeout=10)
        if running and tools_count > 0:
            return True
        await asyncio.sleep(0.5)  # 每 0.5 秒重试
    return False
```

`check_mcp_server` 内部：
1. 创建一个 `mcp_agent` 库的 MCP 客户端
2. 通过 HTTP POST 发送 MCP `initialize` 请求
3. 调用 `tools/list` 获取工具列表
4. 如果返回 > 0 个工具 → 认为 Server 就绪

#### 阶段 4：Agent 循环 (L213-294)

这是 eval 的核心逻辑——**LLM Agent 的多轮交互循环**：

```
┌─── Iteration 1 ────────────────────────────────────────────┐
│  User: "创建客户 John..."                                    │
│  → LLM: "我先看看有什么工具"                                   │
│  → LLM 输出: <tool_call>{"name": "list_tools"}</tool_call>  │
│  → 执行 list_tools → 返回 36 个可用工具的描述                   │
└────────────────────────────────────────────────────────────┘
┌─── Iteration 2 ────────────────────────────────────────────┐
│  User: [工具列表]                                            │
│  → LLM: "我需要调用 create_customer API"                      │
│  → LLM 输出: <tool_call>{"name": "call_tool",               │
│              "arguments": {"tool_name": "mcp_tool_create..." │
│              "arguments": "{\"name\": \"John\"}"}}</tool_call>│
│  → 通过 MCP 调用 Server 的 POST /customers → 返回 {"id": 1}  │
└────────────────────────────────────────────────────────────┘
┌─── Iteration 3 ────────────────────────────────────────────┐
│  User: [工具返回结果]                                         │
│  → LLM: "客户 John 已创建成功，ID 为 1。"（不含 tool_call）     │
│  → 检测到没有 tool_call → Agent 认为任务完成 → 退出循环          │
└────────────────────────────────────────────────────────────┘
```

**关键代码解读**：

```python
for iteration in range(1, args.max_iterations + 1):
```
**最多 25 轮**（防止 LLM 死循环调工具）。

```python
    elif role == "tool":
        api_messages.append({"role": "user", "content": f"Tool response:\n{content}"})
```
**为什么把 tool 消息转成 user 消息？**
OpenAI 的 Chat API 有专门的 `tool` role，但这里用的是内部 MaaS API（兼容 OpenAI 格式）。
AWM 没有用 OpenAI 的 native tool calling（`tools` 参数），而是让 LLM 在文本中输出
`<tool_call>` XML 标签。所以 tool 的返回值被包装成 user 消息送回。
这种方式更通用，不依赖特定 API 的 function calling 功能。

```python
    tool_calls = parse_tool_calls(content)
```
**解析 LLM 输出中的 `<tool_call>` 标签**：
用正则 `<tool_call>\s*(.*?)\s*</tool_call>` 提取 JSON，然后解析出 `name` 和 `arguments`。
如果 LLM 直接输出了 `mcp_tool_xxx` 格式的工具名（跳过了 `call_tool` 的间接层），
`parse_tool_calls` 会自动转换。

```python
    if args.no_thinking:
        content = strip_thinking(content)
```
**为什么要去掉 `<think>` 标签？**
Qwen3 系列模型会输出 `<think>思考过程...</think>` 再输出正式回答。
这些思考内容如果保留在 messages 里，会占用 context window 且可能干扰后续对话。
但 tool call 解析必须在 strip 之前做（因为 `<tool_call>` 可能在 `<think>` 外面）。

```python
    if not tool_calls:
        # 没有 tool_call → Agent 认为任务做完了
        break
```
**退出条件**：LLM 不再调工具 = 任务结束。不管对不对，到验证阶段再判。

```python
    tc = tool_calls[0]  # 只取第一个
```
**为什么只执行第一个 tool call？**
AWM 的 system prompt 规定「每步只能调一个工具」。即使 LLM 输出多个 `<tool_call>`，
也只执行第一个。这简化了错误处理和 trajectory 记录。

```python
    if name == "list_tools":
        response_text = tools_response_text  # 返回预加载的工具列表
    elif name == "call_tool":
        tool_name, tool_args = parse_call_tool_arguments(arguments)
        response_text = await mcp.call_tool(tool_name, tool_args)
```
**两层工具调用**：
- `list_tools`：不走 MCP，直接返回缓存的工具列表文本
- `call_tool`：通过 MCP 协议 → HTTP POST → FastAPI → SQLite 操作

**`MCPToolExecutor.call_tool` 的执行流程**：
```
run_eval.py → MCPToolExecutor.call_tool("create_customer", {"name": "John"})
    → mcp_agent 库 → HTTP POST http://localhost:PORT/mcp
        → JSON-RPC: {"method": "tools/call", "params": {"name": "create_customer", ...}}
    → FastAPI 收到请求 → FastApiMCP 路由到对应的 POST /customers handler
        → handler 执行 SQL: INSERT INTO customers (name) VALUES ('John')
        → 返回 {"id": 1, "name": "John"}
    → MCP 响应 → MCPToolExecutor → 返回文本给 run_eval.py
```

#### 阶段 5：验证 (L346-375)

```python
verify_result = run_verification(task_output_dir, scenario_name, task_id, args.verify_mode)
```

**验证模式 `code`**（确定性验证）：
```python
# gen_verifier.pure_code.jsonl 中的验证器示例：
def verify(initial_db_path, final_db_path):
    conn = sqlite3.connect(final_db_path)
    result = conn.execute("SELECT * FROM customers WHERE name = 'John'").fetchone()
    if result is None:
        return "incomplete"
    if result["email"] != "john@example.com":
        return "incomplete"
    return "complete"
```

直接用 Python 代码检查数据库的最终状态。**没有 LLM 参与**，结果完全确定。

**验证模式 `sql`**（LLM judge）：
让另一个 LLM 比较 SQL 查询结果来判断。更灵活但有随机性，且需要额外 API 调用。

#### 阶段 6：进程清理 (L325-341)

```python
    finally:
        if server_proc and server_proc.poll() is None:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
```

**为什么用 `os.killpg` 而不是 `server_proc.terminate()`？**
这是之前发现的 bug fix。Server 进程树有三层（见阶段 2），`terminate()` 只杀第一层，
uvicorn 和 tee 变成孤儿进程。`os.killpg` 杀掉整个进程组，避免资源泄漏。

### 4.4 增量保存和 Resume (L411-414, L467-469)

```python
# Resume：跳过已完成的任务
completed_tasks = get_completed_tasks(output_dir)
if (scenario_name, task_id) in completed_tasks:
    continue

# 每跑完一个 task，立即 append 到 results.jsonl
with open(results_path, "a") as f:
    f.write(json.dumps(result) + "\n")
```

**为什么要增量保存？**
跑 400 个任务需要 15+ 小时。如果中间断了（OOM、网络断开、手动停止），
不想重头跑。`--resume` 模式扫描 output_dir 里已有的 `trajectory.json`，
跳过已完成的任务。`results.jsonl` 用 append 模式，每跑一个写一行。

---

## 5. 当前性能瓶颈：串行执行

### 现状

```python
for scenario_entry in scenarios:           # 100 个 scenario
    for task_id in range(...):             # 每个 4 个 task
        result = await run_single_task(...)  # 一个一个跑
```

**所有 400 个 task 完全串行**。每个 task 平均 3.3 分钟，总共约 22 小时。

时间花在哪：
- LLM API 调用：~70%（每次等大模型返回 3-15 秒，每个 task 5-10 次）
- MCP Server 启动：~15%（每个 task 启动/销毁一次）
- MCP Tool 调用：~10%（本地 HTTP，快但次数多）
- 验证：~5%

### 为什么可以并行？

每个 task 是**完全独立**的：
- 独立的数据库文件
- 独立的 MCP Server（独立端口）
- 独立的 LLM 对话
- 独立的验证

唯一的共享资源是 LLM API 的并发限制（MAAS 平台的 QPS 限制）。

### 方案对比

| 方案 | 并行度 | 复杂度 | 适合场景 |
|------|--------|--------|---------|
| **asyncio.gather** | 中（10-50） | 低 | 单机，API 并发 |
| **Ray** | 高（100+） | 中 | 多机分布式 |
| **multiprocessing** | 中（CPU核数） | 低 | 单机CPU密集 |

### 方案 A：asyncio.gather（推荐，改动最小）

当前代码已经是 async 的，只需把串行 `await` 改成并发 `gather`：

```python
# 当前：串行
for task in tasks:
    result = await run_single_task(task)

# 改为：并发（比如同时跑 20 个）
import asyncio

CONCURRENCY = 20

async def run_with_semaphore(sem, task):
    async with sem:  # 限制同时运行的数量
        return await run_single_task(task)

sem = asyncio.Semaphore(CONCURRENCY)
all_tasks = [run_with_semaphore(sem, t) for t in tasks]
results = await asyncio.gather(*all_tasks)
```

**优点**：改动 < 20 行，单进程，无序列化开销
**缺点**：受限于单机，GIL 不影响（因为瓶颈是 I/O 等待而非 CPU）
**预期加速**：20x 并发 → 22 小时 → ~1.1 小时

### 方案 B：Ray（适合大规模或多机）

```python
import ray

@ray.remote
def run_task_ray(scenario_name, task_id, task_text, args_dict, output_dir):
    # Ray worker 是独立进程，需要重新 setup
    import asyncio
    args = reconstruct_args(args_dict)
    setup_env_vars(args)
    return asyncio.run(run_single_task(scenario_name, task_id, task_text, args, output_dir))

# 提交所有任务
ray.init()
futures = [
    run_task_ray.remote(name, tid, text, vars(args), output_dir)
    for name, tid, text in all_tasks
]
results = ray.get(futures)  # 等待所有完成
```

**优点**：
- 自动跨多机分布
- 自带失败重试、进度追踪
- 绕过 GIL（每个 worker 是独立进程）

**缺点**：
- 需要安装 Ray（`pip install ray`）
- 需要序列化参数（args 不能有 lambda 等不可序列化对象）
- 每个 worker 进程启动有开销（~2 秒）
- 对于纯 API 调用的任务，Ray 的调度开销可能大于收益

**适合**：如果你有多台机器，或者需要跑 1000+ 个 task。

### 推荐

**对你的场景（100 scenarios × 4 tasks = 400 tasks，单机，API 调用为主），
asyncio.gather + Semaphore 是最优解**。改动最小，加速效果最明显，
且完美匹配你的瓶颈类型（I/O 等待而非 CPU 计算）。

Ray 的优势在于多机分布和容错，但你的 API 并发上限是 MAAS 平台决定的，
单机 asyncio 已经能打满 API QPS。
