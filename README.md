# 🏥 医学教育智能体服务 (Medical Education AI Agent Service)

基于大语言模型的医学教育智能体服务端，**完全兼容 OpenAI API 协议**，可直接被 **Cherry Studio**、**Open WebUI**、**LibreChat** 等平台调用。

---

## 📐 系统架构

```
┌─────────────────────────────────────────────────────────┐
│              前端客户端 (Open WebUI / LibreChat)          │
└───────────────────┬─────────────────────────────────────┘
                    │ OpenAI 兼容 API (HTTP/SSE)
                    ▼
┌─────────────────────────────────────────────────────────┐
│              医学教育智能体服务 (FastAPI)                  │
│                                                         │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ 认证中间件  │  │  速率限制    │  │    CORS 中间件   │  │
│  └────────────┘  └──────────────┘  └────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │               智能体编排器 (Orchestrator)          │  │
│  │                                                  │  │
│  │  ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │  │
│  │  │ 通用助手  │ │ 临床病例  │ │   药理学教学    │  │  │
│  │  ├──────────┤ ├──────────┤ ├─────────────────┤  │  │
│  │  │ 基础医学  │ │ 考试辅导  │ │   辅助诊断教学  │  │  │
│  │  └──────────┘ └──────────┘ └─────────────────┘  │  │
│  │                                                  │  │
│  │  ┌──────────────────────────────────────────┐   │  │
│  │  │         医学工具 (Function Calling)        │   │  │
│  │  │  药物查询 | 临床评分 | 指南检索 | 出题    │   │  │
│  │  └──────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────┬─────────────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        ▼                      ▼
┌──────────────┐      ┌──────────────────┐
│  Anthropic   │      │  OpenAI /        │
│  Claude API  │      │  其他兼容接口    │
│              │      │  (Ollama, vLLM,  │
│              │      │   DeepSeek 等)   │
└──────────────┘      └──────────────────┘
```

---

## 🤖 智能体模式

通过选择不同的**模型名称**来切换智能体模式：

| 模型 ID | 智能体 | 适用场景 |
|---------|--------|----------|
| `med-general` | 通用医学教育助手 | 全科知识问答、学习辅导 |
| `med-clinical` | 临床病例分析 | 病例教学、临床思维培训 |
| `med-pharmacology` | 药理学教学 | 药物机制、用药指导 |
| `med-anatomy` | 基础医学教学 | 解剖学、生理学、病理学 |
| `med-exam` | 考试备考辅导 | 执医/规培/专科考试备考 |
| `med-diagnosis` | 辅助诊断教学 | 诊断推理、辅助检查解读 |

---

## 🛠️ 医学工具 (Function Calling)

智能体内置以下工具，可在对话中自动调用：

- **`get_drug_information`** - 药物信息查询（适应症、剂量、禁忌、相互作用）
- **`calculate_clinical_score`** - 临床评分（Wells、CURB-65、Glasgow、Child-Pugh 等）
- **`search_clinical_guideline`** - 临床指南检索
- **`generate_exam_question`** - 模拟出题
- **`interpret_lab_result`** - 实验室检查结果解读

---

## 🚀 快速开始

### 1. 克隆与安装依赖

```bash
git clone <your-repo>
cd medical-edu-agent

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 API Key 和配置
```

关键配置项：

```env
# 选择 LLM 提供商
LLM_PROVIDER=anthropic          # 或 openai / openai_compatible

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# 或 OpenAI 兼容接口（Ollama 本地部署示例）
# LLM_PROVIDER=openai_compatible
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama
# OPENAI_MODEL=llama3.1:70b

# 服务 API Key（客户端连接时使用）
SERVICE_API_KEY=your-secret-key-here
```

### 3. 启动服务

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 4. Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f medical-edu-agent
```

---

## 🔗 接入 Open WebUI

1. 打开 Open WebUI → **设置 → 连接**
2. 在 **OpenAI API** 部分：
   - **API Base URL**: `http://your-server:8000/v1`
   - **API Key**: 你的 `SERVICE_API_KEY`
3. 保存后，在模型选择器中选择 `med-general`、`med-clinical` 等模型即可使用

---

## 🔗 接入 LibreChat

在 `librechat.yaml` 中添加自定义端点：

```yaml
endpoints:
  custom:
    - name: "医学教育智能体"
      apiKey: "${MEDICAL_EDU_API_KEY}"
      baseURL: "http://your-server:8000/v1"
      models:
        default:
          - "med-general"
          - "med-clinical"
          - "med-pharmacology"
          - "med-anatomy"
          - "med-exam"
          - "med-diagnosis"
        fetch: false
      titleConvo: true
      titleModel: "med-general"
      summarize: false
      forcePrompt: false
      dropParams:
        - "stop"
        - "user"
```

在 `.env` 中添加：
```env
MEDICAL_EDU_API_KEY=your-service-api-key
```

---

## 📡 API 参考

### POST /v1/chat/completions

标准 OpenAI Chat Completions 接口。

```bash
# 非流式请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "med-clinical",
    "messages": [
      {"role": "user", "content": "请分析一个35岁男性，突发胸痛伴大汗3小时的病例"}
    ]
  }'

# 流式请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "med-exam",
    "messages": [
      {"role": "user", "content": "给我出5道内科学单选题"}
    ],
    "stream": true
  }'
```

### GET /v1/models

获取可用模型列表。

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key"
```

### GET /health

健康检查（无需认证）。

---

## ⚙️ 配置说明

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_PROVIDER` | `anthropic` | LLM 后端：`anthropic` / `openai` / `openai_compatible` |
| `SERVICE_API_KEY` | - | 服务访问密钥 |
| `ENABLE_API_KEY_AUTH` | `true` | 是否启用 API Key 验证 |
| `DEFAULT_AGENT_MODE` | `general` | 默认智能体模式 |
| `MAX_OUTPUT_TOKENS` | `4096` | 最大输出 Token 数 |
| `ENABLE_STREAMING` | `true` | 是否支持流式响应 |
| `ENABLE_FUNCTION_CALLING` | `true` | 是否启用工具调用 |
| `RATE_LIMIT_PER_MINUTE` | `60` | 每分钟最大请求数（0=不限） |

---

## ⚠️ 免责声明

本系统仅供**医学教育**目的使用，不构成具体患者的诊疗建议。
所有临床决策须由执业医师根据患者具体情况作出判断。
