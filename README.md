# Novel Constellation｜小说人物关系 3D 星图

基于大型语言模型（LLM）的小说人物关系提取与交互式 3D 星云可视化工具。

上传 TXT 或 Markdown 小说文本，系统会自动识别人物、别名、关系与原文证据，并生成可搜索、可筛选、可交互的 3D 人物关系图。

---

## 核心功能

- **人物关系提取**：识别人物、别名、关系类型、强度、可信度与原文证据。
- **长文本分析**：自动分块处理长篇小说，并归并跨章节人物与关系。
- **3D 星云图谱**：支持旋转、缩放、人物聚焦和关系高亮。
- **人物搜索**：通过姓名、昵称、尊称或代号快速定位人物。
- **剧情阶段筛选**：按开篇、发展、转折、高潮和结局查看关系变化。
- **结果恢复与导出**：在浏览器中恢复最近一次分析，或导出完整 JSON 数据。

---

## 🚀 快速开始

### 1. 准备环境

安装 [Python 3.10 或更高版本](https://www.python.org/downloads/)。Windows 安装 Python 时请勾选 `Add Python to PATH`。

### 2. 启动项目（Windows 推荐）

双击项目根目录中的 `start.bat`。

首次运行会自动：

1. 创建 `.venv` 虚拟环境；
2. 安装 Python 依赖；
3. 根据 `.env.example` 生成本地 `.env`；
4. 打开记事本，请你填写自己的 `LLM_API_KEY`。

保存 `.env` 后，再次双击 `start.bat`。服务启动成功后，浏览器会自动打开：

```text
http://127.0.0.1:8080
```

> 不要直接双击 `index.html`，浏览器会阻止部分模块加载。

<details>
<summary>手动启动方式</summary>

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
# 编辑 .env 并填写 LLM_API_KEY
.venv\Scripts\python.exe start.py
```

</details>

---

## 使用方法

1. 导入 TXT、Markdown 文件，或直接粘贴小说文本；
2. 点击「开始分析人物关系」；
3. 等待模型完成提取并生成 3D 星图；
4. 双击人物聚焦关系，悬停连线查看故事与证据；
5. 使用搜索、剧情阶段和导出功能继续探索。

---

## 模型配置

项目支持兼容 OpenAI API 格式的大模型服务。在 `.env` 中填写：

```env
LLM_API_KEY=你的_API_KEY
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL_NAME=deepseek-chat
```

更多可调参数见 [`.env.example`](.env.example)。

---

## 常见问题

**启动后提示 `LLM_API_KEY 未配置`？**

检查项目根目录的 `.env`，确认 `LLM_API_KEY=` 后填写的不是示例占位符。

**分析速度比较慢？**

速度取决于文本长度、模型响应速度和 API 限流。长篇小说通常需要等待数分钟。

**音乐按钮没有声音？**

公开版本不内置第三方音乐。可将有权使用的 `.ogg` 文件放入 `music/`，并在 `assets/config.js` 中配置文件名。

---

## 技术栈

- **前端**：Three.js、d3-force-3d、原生 JavaScript
- **后端**：Python、Flask、OpenAI SDK
- **模型**：兼容 OpenAI API 格式的大语言模型服务
