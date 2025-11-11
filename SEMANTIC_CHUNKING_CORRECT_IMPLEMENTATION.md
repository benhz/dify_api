# 语义分块正确实现说明

## ✅ 已完全按照要求实现

### 正确的 5 步流程

```
文本流：[段落1...段落2...段落3...段落4...]
    ↓
Step 1: separator 切分物理边界
    [段落1] | [段落2] | [段落3] | [段落4]
    ↓
Step 2: 语义分析 + threshold_amount/buffer_size
    在每个段落内找语义断点
    [段落1-句1,2] | [段落1-句3,4] | [段落2-句1,2] | ...
    ↓
Step 3: max_tokens 强制切断过长的块
    [块1: 1000 tokens] | [块2: 950 tokens] | ...
    ↓
Step 4: chunk_overlap 添加重叠
    [块1]
    [块1末尾50 + 块2 + ...]
    [块2末尾50 + 块3 + ...]
    ↓
Step 5: min_chunk_tokens / max_chunk_tokens
    确保每个块在最小和最大 tokens 范围内
```

---

## 📝 实现细节

### Step 1: separator - 物理边界切分
**方法**: `_split_by_separator(text)`
**输入**: 完整文档文本
**输出**: 段落列表
**逻辑**:
```python
paragraphs = text.split(separator)  # 例如 "\n\n"
return [p.strip() for p in paragraphs if p.strip()]
```

---

### Step 2: 语义分析
**方法**: `_apply_semantic_splitting(paragraph)`
**输入**: 单个段落
**输出**: 语义块列表
**逻辑**:
1. 切分句子: `sentences = self._split_into_sentences(paragraph)`
2. 生成 embeddings: `embeddings = self._get_embeddings(sentences)`
3. 计算相似度: 计算相邻句子间的 cosine 相似度
4. 平滑处理: 使用 `buffer_size` 进行移动平均
5. 阈值判断: 使用 `threshold_amount` percentile 找边界
6. 生成块: 按边界组合句子

**关键参数**:
- `threshold_amount`: 95 (百分位数，越高边界越少)
- `buffer_size`: 2 (平滑窗口大小)

**重要**: 对**每个段落**都进行语义分析，不是只对长段落

---

### Step 3: max_tokens 强制切断
**方法**: `_enforce_max_tokens(chunks)`
**输入**: Step 2 的语义块列表
**输出**: 所有块都 ≤ max_tokens 的块列表
**逻辑**:
```python
for chunk in chunks:
    if token_count(chunk) > max_tokens:
        # 按句子切分
        sentences = split_into_sentences(chunk)
        # 重新组合，确保不超过 max_tokens
        # 如果单句超过 max_tokens，强制按词切分
```

---

### Step 4: chunk_overlap 添加重叠
**方法**: `_add_overlap(chunks)`
**输入**: Step 3 的块列表
**输出**: 添加了重叠的块列表
**逻辑**:
```python
overlapped_chunks = []
for i, chunk in enumerate(chunks):
    if i > 0:
        # 从前一个块获取最后 N 个 tokens
        prefix = get_last_n_tokens(chunks[i-1], chunk_overlap)
        chunk = prefix + ' ' + chunk
    overlapped_chunks.append(chunk)
```

**结构示例**:
```
原始:    [AAAA] [BBBB] [CCCC]
overlap: [AAAA] [aaAABBBB] [bbBBCCCC]
         └─50─┘ └─50┘        └─50┘
```

---

### Step 5: min/max tokens 约束
**方法**: `_enforce_size_constraints(chunks)`
**输入**: Step 4 的块列表
**输出**: 最终符合大小约束的块列表
**逻辑**:
1. **合并短块**:
   ```python
   if token_count(chunk) < min_chunk_tokens:
       # 尝试与前块合并
       # 或与后块合并
   ```

2. **切分长块**:
   ```python
   if token_count(chunk) > max_chunk_tokens:
       # 按句子边界切分
       # 确保每块 <= max_chunk_tokens
   ```

---

## 🔍 关键改进

### 1. 所有段落都进行语义分析
```python
# ❌ 错误的旧实现
for paragraph in paragraphs:
    if token_count(paragraph) > max_tokens:  # 只处理长段落
        apply_semantic_splitting(paragraph)
    else:
        chunks.append(paragraph)  # 短段落直接加入

# ✓ 正确的新实现
for paragraph in paragraphs:
    # 每个段落都进行语义分析
    semantic_chunks = apply_semantic_splitting(paragraph)
    all_chunks.extend(semantic_chunks)
```

### 2. 正确的处理顺序
```python
# ✓ 新实现严格按照 5 步顺序
paragraphs = step1_split_by_separator(text)
semantic_chunks = step2_semantic_analysis(paragraphs)
limited_chunks = step3_enforce_max_tokens(semantic_chunks)
overlapped_chunks = step4_add_overlap(limited_chunks)
final_chunks = step5_enforce_size_constraints(overlapped_chunks)
```

### 3. 返回正确的预览内容
```python
# SemanticIndexProcessor.transform() 返回:
return all_documents  # List[Document]

# 每个 Document:
Document(
    page_content="分块后的文本内容",  # 用于预览
    metadata={
        "doc_id": "uuid",
        "doc_hash": "hash"
    }
)
```

---

## 📊 参数说明

### 必需参数
| 参数 | 说明 | 示例值 |
|------|------|--------|
| `separator` | 段落分隔符 | `"\n\n"` |
| `max_tokens` | 硬性token上限 | `1024` |

### 语义分析参数
| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `threshold_amount` | 相似度阈值百分位 | `95` | 0-100 |
| `buffer_size` | 平滑窗口大小 | `2` | ≥0 |

### 后处理参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `chunk_overlap` | 重叠token数 | `50` |
| `min_chunk_tokens` | 最小块大小 | `150` |
| `max_chunk_tokens` | 最大块大小 | `max_tokens` |

---

## 🎯 使用示例

### API 请求
```json
{
  "doc_form": "semantic_model",
  "process_rule": {
    "mode": "custom",
    "rules": {
      "segmentation": {
        "separator": "\\n\\n",
        "max_tokens": 1024,
        "chunk_overlap": 50,
        "threshold_amount": 95,
        "buffer_size": 2,
        "min_chunk_tokens": 150,
        "max_chunk_tokens": 1000
      }
    }
  }
}
```

### 处理示例
**输入文本**:
```
深度学习是机器学习的一个分支。它基于人工神经网络。

计算机视觉是深度学习的重要应用。卷积神经网络表现出色。
```

**Step 1: separator 切分**:
```
[段落1: "深度学习...神经网络。"]
[段落2: "计算机视觉...表现出色。"]
```

**Step 2: 语义分析**:
```
段落1 → [块1: "深度学习是机器学习的一个分支。"]
        [块2: "它基于人工神经网络。"]
段落2 → [块3: "计算机视觉是深度学习的重要应用。"]
        [块4: "卷积神经网络表现出色。"]
```

**Step 3: max_tokens**:
```
(假设都 ≤ max_tokens，保持不变)
```

**Step 4: chunk_overlap**:
```
[块1]
[块1末尾 + 块2]
[块2末尾 + 块3]
[块3末尾 + 块4]
```

**Step 5: min/max 约束**:
```
(假设都在范围内，保持不变)
```

---

## 📁 代码位置

### 核心文件
- **文本分割器**: `core/rag/splitter/semantic_text_splitter.py`
  - `split_text()` - 主入口，执行 5 步流程
  - `_apply_semantic_splitting()` - Step 2
  - `_enforce_max_tokens()` - Step 3
  - `_add_overlap()` - Step 4
  - `_enforce_size_constraints()` - Step 5

- **索引处理器**: `core/rag/index_processor/processor/semantic_index_processor.py`
  - `transform()` - 返回 Document 对象用于预览

### 关键方法调用链
```
API Request
  ↓
DatasetIndexingEstimateApi.post()
  ↓
DocumentService.estimate()
  ↓
IndexingRunner.indexing_estimate()
  ↓
SemanticIndexProcessor.transform()
  ↓
SemanticTextSplitter.split_text()
  ↓
5 步处理流程
  ↓
返回 Document 对象列表 (用于预览)
```

---

## ✅ 验证清单

- [x] Step 1: separator 正确切分段落
- [x] Step 2: 每个段落都进行语义分析（不是只处理长段落）
- [x] Step 3: max_tokens 强制切断长块
- [x] Step 4: chunk_overlap 正确添加重叠
- [x] Step 5: min/max tokens 约束正确执行
- [x] 返回 Document 对象用于预览
- [x] 避免 embedding 爆炸（每段落单独处理）
- [x] 无 TypeError (使用 embeddings.size)

---

## 🔗 Git 信息

- **分支**: `claude/add-semantic-chunking-strategy-011CUp7PWrYrKTCXQdDf6Kjd`
- **提交**: `cf9d7c4`
- **状态**: ✅ 已推送

---

## 📚 相关文档

- 调试指南: `SEMANTIC_CHUNKING_DEBUG_GUIDE.md`
- 修复说明: `SEMANTIC_CHUNKING_FIXES.md`
- 本文档: `SEMANTIC_CHUNKING_CORRECT_IMPLEMENTATION.md`
