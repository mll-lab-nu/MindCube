# OpenAI Multiprocess Processing Tool

这是一个多进程OpenAI API调用工具，能够高效处理大量图像+文本数据，支持断点续传和错误恢复。

## 功能特点

- ✅ **多进程并行处理**：使用64个进程同时处理，大幅提升速度
- ✅ **断点续传**：自动检测已完成的任务，重启后从中断处继续
- ✅ **错误处理**：单个任务失败不影响其他任务，保存所有成功结果
- ✅ **实时监控**：提供进度条和详细日志
- ✅ **结果验证**：检查输出文件完整性和格式正确性
- ✅ **安全写入**：每个结果立即写入文件，避免数据丢失

## 文件说明

- `openai_other_models.py`: 核心处理脚本（多进程版本）
- `run_multiprocess.py`: 启动管理脚本
- `openai_processing.log`: 运行日志文件

## 快速开始

### 1. 环境准备

确保你的 `.env` 文件包含OpenAI API密钥：
```
OPENAI_API_KEY=your_api_key_here
```

### 2. 基本使用

**启动处理任务：**
```bash
python run_multiprocess.py
```

**查看当前状态：**
```bash
python run_multiprocess.py --status
```

**验证结果文件：**
```bash
python run_multiprocess.py --validate
```

**自定义进程数：**
```bash
python run_multiprocess.py --processes 32
```

### 3. 断点续传

脚本会自动检测以下情况：

1. **首次运行**：检查输出文件是否存在
   - 不存在：从头开始处理
   - 存在：读取已完成的ID列表

2. **文件可读性检查**：
   - 可读：解析已完成任务
   - 不可读：自动备份损坏文件，重新开始

3. **已完成任务识别**：
   - 解析输出文件中每一行
   - 提取已处理的ID
   - 只处理剩余未完成的任务

## 详细使用说明

### 命令行参数

```bash
python run_multiprocess.py [OPTIONS]

OPTIONS:
  -h, --help                显示帮助信息
  -p, --processes NUM       设置进程数量 (默认: 64)
  -v, --validate           验证结果文件完整性
  -s, --status             显示当前处理状态
```

### 监控和日志

**实时进度监控：**
- 进度条显示处理进度
- 控制台输出关键信息
- 详细日志保存到文件

**日志文件位置：**
```
openai_processing.log
```

**日志包含内容：**
- 处理开始/结束时间
- 每个任务的处理状态
- 错误详情和堆栈信息
- 性能统计信息

### 错误处理

**常见错误类型：**

1. **API调用失败**
   - 网络超时
   - API密钥无效
   - 速率限制

2. **图像处理错误**
   - 文件不存在
   - 格式不支持
   - 文件损坏

3. **文件I/O错误**
   - 磁盘空间不足
   - 权限问题
   - 文件锁定

**错误恢复机制：**
- 单个任务失败不影响其他任务
- 所有成功结果立即保存
- 重启后自动跳过已完成任务
- 损坏文件自动备份

### 性能优化

**调整进程数：**
```bash
# 对于API密钥有严格速率限制的情况
python run_multiprocess.py --processes 16

# 对于高性能服务器
python run_multiprocess.py --processes 128
```

**监控系统资源：**
- CPU使用率
- 内存占用
- 网络带宽
- API配额

## 故障排除

### 1. 环境检查

**检查API密钥：**
```bash
echo $OPENAI_API_KEY
```

**检查输入文件：**
```bash
head -1 /path/to/input/file.jsonl | python -m json.tool
```

### 2. 常见问题

**Q: 进程数量设置多少合适？**
A: 建议从32开始测试，根据API速率限制和系统性能调整。

**Q: 如何处理中断后的恢复？**
A: 直接重新运行脚本，会自动检测并跳过已完成任务。

**Q: 如何验证结果完整性？**
A: 使用 `--validate` 参数检查结果文件格式和完整性。

**Q: 磁盘空间不足怎么办？**
A: 结果会持续写入，已完成部分不会丢失。清理空间后重新运行。

### 3. 调试技巧

**查看详细日志：**
```bash
tail -f openai_processing.log
```

**检查进程状态：**
```bash
ps aux | grep python
```

**监控资源使用：**
```bash
top -p $(pgrep -f "python.*run_multiprocess")
```

## 输出格式

输出文件为JSONL格式，每行包含：

```json
{
  "id": "unique_task_id",
  "input_prompt": "original_question",
  "images": ["list", "of", "image", "paths"],
  "answer": "openai_generated_answer",
  "processed_at": 1234567890.123
}
```

## 安全注意事项

1. **API密钥保护**：确保 `.env` 文件权限设置为 600
2. **数据备份**：重要数据处理前先备份
3. **速率限制**：遵守OpenAI API使用限制
4. **资源监控**：避免过度消耗系统资源

## 版本历史

- v2.0: 多进程版本，支持断点续传和错误恢复
- v1.0: 单线程原始版本

## 支持

如遇问题请检查：
1. 日志文件内容
2. 系统资源使用情况
3. API配额状态
4. 网络连接质量 