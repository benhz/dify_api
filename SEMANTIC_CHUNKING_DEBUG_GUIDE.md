# 语义分块调试指南

## 📍 入口点

### API 端点
```
POST /console/api/datasets/indexing-estimate
```

**Controller 文件**: `controllers/console/datasets/datasets.py`
**类**: `DatasetIndexingEstimateApi`
**行数**: 540-647

### 请求示例
```json
{
  "info_list": {
    "data_source_type": "upload_file",
    "file_info_list": {
      "file_ids": ["0c3e9cc1-e7a0-4069-8d4c-eba4d169147e"]
    }
  },
  "indexing_technique": "high_quality",
  "process_rule": {
    "mode": "custom",
    "rules": {
      "pre_processing_rules": [
        { "id": "remove_extra_spaces", "enabled": true },
        { "id": "remove_urls_emails", "enabled": false }
      ],
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
  },
  "doc_form": "semantic_model",
  "doc_language": "Chinese Simplified",
  "embedding_model": "emb",
  "embedding_model_provider": "langgenius/openai_api_compatible/openai_api_compatible"
}
```

---

## 🔄 完整调用流程

### 阶段 1: API 请求处理
**文件**: `controllers/console/datasets/datasets.py:540-647`

```python
def post(self):
    # 1. 解析参数
    parser = reqparse.RequestParser()
    args = parser.parse_args()

    # 2. 参数验证
    DocumentService.estimate_args_validate(args)
    # 位置: services/dataset_service.py:2372-2452
    # 验证内容: info_list, process_rule, segmentation 参数
    # 包括语义分块参数: threshold_amount, buffer_size, min_chunk_tokens, max_chunk_tokens

    # 3. 调用 indexing_estimate
    response = DocumentService.estimate(args)
```

**调试点 1.1**: 在 `datasets.py:560` 打印接收到的 args
```python
print("=== DEBUG 1.1: Received args ===")
print(f"doc_form: {args.get('doc_form')}")
print(f"segmentation: {args.get('process_rule', {}).get('rules', {}).get('segmentation')}")
```

---

### 阶段 2: 估算服务入口
**文件**: `services/dataset_service.py`
**方法**: `DocumentService.estimate()`
**行数**: 约 2100-2200

```python
@staticmethod
def estimate(args: dict) -> dict:
    # 1. 构建 extract_setting
    extract_setting = ExtractSetting(...)

    # 2. 调用 IndexingRunner.indexing_estimate
    indexing_estimate = IndexingRunner.indexing_estimate(
        tenant_id=current_user.current_tenant_id,
        extract_settings=[extract_setting],
        tmp_processing_rule=args["process_rule"],
        doc_form=args["doc_form"],  # "semantic_model"
        doc_language=args.get("doc_language", "English"),
        indexing_technique=args["indexing_technique"],
    )
```

**调试点 2.1**: 在 `DocumentService.estimate()` 方法开始处
```python
print("=== DEBUG 2.1: DocumentService.estimate ===")
print(f"doc_form: {args['doc_form']}")
print(f"indexing_technique: {args['indexing_technique']}")
print(f"process_rule mode: {args['process_rule']['mode']}")
```

---

### 阶段 3: 索引运行器估算
**文件**: `core/indexing_runner.py`
**方法**: `IndexingRunner.indexing_estimate()`
**行数**: 245-343

```python
@classmethod
def indexing_estimate(
    cls,
    tenant_id: str,
    extract_settings: list[ExtractSetting],
    tmp_processing_rule: dict,
    doc_form: str = "text_model",  # 这里会传入 "semantic_model"
    doc_language: str = "English",
    dataset_id: Optional[str] = None,
    indexing_technique: str = "economy",
) -> IndexingEstimate:

    # 1. 创建 IndexProcessor
    index_processor = IndexProcessorFactory(doc_form).init_index_processor()
    # 当 doc_form = "semantic_model" 时，创建 SemanticIndexProcessor

    # 2. Extract 阶段 - 提取文档内容
    documents = index_processor.extract(extract_setting, ...)

    # 3. Transform 阶段 - 语义分块
    documents = index_processor.transform(
        documents=documents,
        process_rule=tmp_processing_rule,
        embedding_model_instance=embedding_model_instance,
        ...
    )

    # 4. 返回结果
    return IndexingEstimate(
        total_segments=len(documents),
        preview=documents[:10],
        ...
    )
```

**调试点 3.1**: 在创建 processor 后
```python
print("=== DEBUG 3.1: IndexProcessor Created ===")
print(f"Processor type: {type(index_processor).__name__}")
print(f"doc_form: {doc_form}")
```

**调试点 3.2**: Extract 阶段后
```python
print("=== DEBUG 3.2: After Extract ===")
print(f"Number of documents: {len(documents)}")
for i, doc in enumerate(documents[:3]):
    print(f"Doc {i} length: {len(doc.page_content)} chars")
    print(f"Doc {i} preview: {doc.page_content[:100]}...")
```

**调试点 3.3**: Transform 阶段后
```python
print("=== DEBUG 3.3: After Transform (Semantic Chunking) ===")
print(f"Number of chunks: {len(documents)}")
for i, doc in enumerate(documents[:5]):
    print(f"Chunk {i} length: {len(doc.page_content)} chars")
    print(f"Chunk {i} preview: {doc.page_content[:80]}...")
```

---

### 阶段 4: 语义索引处理器
**文件**: `core/rag/index_processor/processor/semantic_index_processor.py`

#### 4.1 Extract 方法
**行数**: 45-59

```python
def extract(self, extract_setting: ExtractSetting, **kwargs) -> list[Document]:
    text_docs = ExtractProcessor.extract(
        extract_setting=extract_setting,
        is_automatic=(kwargs.get("process_rule_mode") == "automatic" or ...),
    )
    return text_docs
```

**调试点 4.1**: Extract 开始和结束
```python
print("=== DEBUG 4.1: SemanticIndexProcessor.extract START ===")
print(f"extract_setting: {extract_setting}")

# ... extraction logic ...

print("=== DEBUG 4.1: SemanticIndexProcessor.extract END ===")
print(f"Extracted {len(text_docs)} documents")
for i, doc in enumerate(text_docs):
    print(f"  Doc {i}: {len(doc.page_content)} chars")
```

#### 4.2 Transform 方法 (核心语义分块)
**行数**: 61-127

```python
def transform(self, documents: list[Document], **kwargs) -> list[Document]:
    # 1. 获取 process_rule
    process_rule = kwargs.get("process_rule")
    rules = Rule.model_validate(process_rule.get("rules"))

    # 2. 创建 SemanticTextSplitter
    splitter = SemanticTextSplitter(
        separator=rules.segmentation.separator,
        max_tokens=rules.segmentation.max_tokens,
        chunk_overlap=rules.segmentation.chunk_overlap,
        threshold_amount=rules.segmentation.threshold_amount or 95,
        buffer_size=rules.segmentation.buffer_size or 2,
        min_chunk_tokens=rules.segmentation.min_chunk_tokens or 150,
        max_chunk_tokens=rules.segmentation.max_chunk_tokens or rules.segmentation.max_tokens,
        embedding_model_instance=embedding_model_instance,
    )

    # 3. 对每个文档进行处理
    for document in documents:
        # 3.1 清理文档
        document_text = CleanProcessor.clean(document.page_content, ...)

        # 3.2 语义分块
        document_nodes = splitter.split_documents([document])

        # 3.3 后处理
        for document_node in document_nodes:
            # 添加 metadata, hash 等
            ...

    return all_documents
```

**调试点 4.2**: Transform 各阶段
```python
print("=== DEBUG 4.2: SemanticIndexProcessor.transform START ===")
print(f"Input documents: {len(documents)}")
print(f"Segmentation config:")
print(f"  separator: {rules.segmentation.separator}")
print(f"  max_tokens: {rules.segmentation.max_tokens}")
print(f"  chunk_overlap: {rules.segmentation.chunk_overlap}")
print(f"  threshold_amount: {rules.segmentation.threshold_amount}")
print(f"  buffer_size: {rules.segmentation.buffer_size}")
print(f"  min_chunk_tokens: {rules.segmentation.min_chunk_tokens}")
print(f"  max_chunk_tokens: {rules.segmentation.max_chunk_tokens}")

# 在循环中
for idx, document in enumerate(documents):
    print(f"\n--- Processing document {idx} ---")
    print(f"Original length: {len(document.page_content)} chars")

    # 清理后
    print(f"After cleaning: {len(document_text)} chars")

    # 分块后
    print(f"Generated {len(document_nodes)} chunks")
    for i, node in enumerate(document_nodes[:3]):
        print(f"  Chunk {i}: {len(node.page_content)} chars")

print("=== DEBUG 4.2: SemanticIndexProcessor.transform END ===")
print(f"Total output chunks: {len(all_documents)}")
```

---

### 阶段 5: 语义文本分割器 (核心算法)
**文件**: `core/rag/splitter/semantic_text_splitter.py`

#### 5.1 主入口: split_text
**行数**: 69-96

```python
def split_text(self, text: str) -> list[str]:
    # Step 1: 按 separator 切分物理边界
    paragraphs = self._split_by_separator(text)

    # Step 2: 切分成句子
    all_sentences = []
    for paragraph in paragraphs:
        sentences = self._split_into_sentences(paragraph)
        all_sentences.extend(sentences)

    # Step 3-4: 生成 embeddings 并找语义边界
    semantic_boundaries = self._find_semantic_boundaries(all_sentences)

    # Step 5: 生成语义块
    semantic_chunks = self._generate_semantic_chunks(all_sentences, semantic_boundaries)

    # Step 6: 后处理 (合并短块、切分长块、添加重叠)
    final_chunks = self._post_process_chunks(semantic_chunks)

    return final_chunks
```

**调试点 5.1**: split_text 主流程
```python
print("=== DEBUG 5.1: SemanticTextSplitter.split_text START ===")
print(f"Input text length: {len(text)} chars")

# Step 1
print(f"\nStep 1: Split by separator")
print(f"Paragraphs: {len(paragraphs)}")
for i, para in enumerate(paragraphs[:3]):
    print(f"  Para {i}: {len(para)} chars")

# Step 2
print(f"\nStep 2: Split into sentences")
print(f"Total sentences: {len(all_sentences)}")
for i, sent in enumerate(all_sentences[:5]):
    print(f"  Sent {i}: {sent[:60]}...")

# Step 3-4
print(f"\nStep 3-4: Find semantic boundaries")
print(f"Boundaries found: {semantic_boundaries}")

# Step 5
print(f"\nStep 5: Generate semantic chunks")
print(f"Semantic chunks: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks[:3]):
    print(f"  Chunk {i}: {len(chunk)} chars")

# Step 6
print(f"\nStep 6: Post-process chunks")
print(f"Final chunks: {len(final_chunks)}")
for i, chunk in enumerate(final_chunks[:3]):
    print(f"  Chunk {i}: {len(chunk)} chars, {self._get_token_count(chunk)} tokens")

print("=== DEBUG 5.1: SemanticTextSplitter.split_text END ===")
```

#### 5.2 按分隔符切分
**行数**: 98-103

```python
def _split_by_separator(self, text: str) -> list[str]:
    if self._separator:
        parts = text.split(self._separator)
        return [p.strip() for p in parts if p.strip()]
    return [text]
```

**调试点 5.2**:
```python
print(f"=== DEBUG 5.2: _split_by_separator ===")
print(f"Separator: {repr(self._separator)}")
print(f"Input length: {len(text)}")
parts = text.split(self._separator) if self._separator else [text]
print(f"Raw parts: {len(parts)}")
result = [p.strip() for p in parts if p.strip()]
print(f"Cleaned parts: {len(result)}")
return result
```

#### 5.3 切分句子
**行数**: 105-158

```python
def _split_into_sentences(self, text: str) -> list[str]:
    # 使用正则表达式按句子边界切分
    # 支持中文 (。！？) 和英文 (.!?\s+)
    combined_pattern = '|'.join(f'({p})' for p in self._sentence_patterns)
    parts = re.split(combined_pattern, text)

    # 重新组装句子（包含分隔符）
    sentences = []
    current_sentence = ""
    for part in parts:
        # 判断是否为分隔符
        # 拼接句子
        ...

    return sentences
```

**调试点 5.3**:
```python
print(f"=== DEBUG 5.3: _split_into_sentences ===")
print(f"Input: {text[:100]}...")
print(f"Patterns: {self._sentence_patterns}")

# 在 split 后
print(f"Raw parts: {len(parts)}")

# 在循环中
for i, part in enumerate(parts[:10]):
    print(f"  Part {i}: {repr(part[:30])}")

# 结果
print(f"Sentences found: {len(sentences)}")
for i, sent in enumerate(sentences[:5]):
    print(f"  Sent {i}: {sent[:60]}...")
```

#### 5.4 查找语义边界 (核心算法)
**行数**: 178-220

```python
def _find_semantic_boundaries(self, sentences: list[str]) -> list[int]:
    # 1. 生成 embeddings
    embeddings = self._get_embeddings(sentences)

    # 2. 计算相邻句子的余弦相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # 3. 应用平滑 (buffer_size)
    smoothed_similarities = self._apply_smoothing(similarities, self._buffer_size)

    # 4. 使用百分位数计算阈值
    threshold = np.percentile(smoothed_similarities, self._threshold_amount)

    # 5. 找出低于阈值的位置作为边界
    boundaries = []
    for i, sim in enumerate(smoothed_similarities):
        if sim < threshold:
            boundaries.append(i + 1)

    return boundaries
```

**调试点 5.4**: 语义边界检测详细过程
```python
print(f"=== DEBUG 5.4: _find_semantic_boundaries ===")
print(f"Input sentences: {len(sentences)}")

# 1. Embeddings
print(f"\nStep 1: Generate embeddings")
embeddings = self._get_embeddings(sentences)
print(f"Embeddings shape: {embeddings.shape}")

# 2. Similarities
print(f"\nStep 2: Calculate similarities")
similarities = []
for i in range(len(embeddings) - 1):
    sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
    similarities.append(sim)
    if i < 5:
        print(f"  Sim[{i}→{i+1}]: {sim:.4f}")
print(f"Similarities: min={min(similarities):.4f}, max={max(similarities):.4f}, mean={np.mean(similarities):.4f}")

# 3. Smoothing
print(f"\nStep 3: Apply smoothing (buffer_size={self._buffer_size})")
smoothed_similarities = self._apply_smoothing(similarities, self._buffer_size)
print(f"Smoothed: min={min(smoothed_similarities):.4f}, max={max(smoothed_similarities):.4f}")

# 4. Threshold
print(f"\nStep 4: Calculate threshold (percentile={self._threshold_amount})")
threshold = np.percentile(smoothed_similarities, self._threshold_amount)
print(f"Threshold: {threshold:.4f}")

# 5. Boundaries
print(f"\nStep 5: Find boundaries")
boundaries = []
for i, sim in enumerate(smoothed_similarities):
    if sim < threshold:
        boundaries.append(i + 1)
        if len(boundaries) <= 5:
            print(f"  Boundary at position {i+1}, sim={sim:.4f}")
print(f"Total boundaries: {len(boundaries)}")
print(f"Boundaries: {boundaries[:10]}...")

return boundaries
```

#### 5.5 生成 Embeddings
**行数**: 172-176

```python
def _get_embeddings(self, texts: list[str]) -> np.ndarray:
    if self._embedding_model_instance:
        # 使用 embedding 模型
        embeddings = self._embedding_model_instance.invoke_text_embedding(texts=texts)
        return np.array(embeddings)
    else:
        # 使用后备方案
        return self._fallback_embeddings(texts)
```

**调试点 5.5**:
```python
print(f"=== DEBUG 5.5: _get_embeddings ===")
print(f"Texts to embed: {len(texts)}")
print(f"Has embedding model: {self._embedding_model_instance is not None}")

if self._embedding_model_instance:
    print(f"Using embedding model")
    embeddings = self._embedding_model_instance.invoke_text_embedding(texts=texts)
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
else:
    print(f"Using fallback embeddings")
    embeddings = self._fallback_embeddings(texts)
    print(f"Fallback embeddings shape: {embeddings.shape}")

return embeddings
```

#### 5.6 平滑处理
**行数**: 255-272

```python
def _apply_smoothing(self, similarities: list[float], buffer_size: int) -> list[float]:
    smoothed = []
    for i in range(len(similarities)):
        start = max(0, i - buffer_size)
        end = min(len(similarities), i + buffer_size + 1)
        window = similarities[start:end]
        smoothed.append(sum(window) / len(window))
    return smoothed
```

**调试点 5.6**:
```python
print(f"=== DEBUG 5.6: _apply_smoothing ===")
print(f"Input similarities: {len(similarities)}")
print(f"Buffer size: {buffer_size}")

smoothed = []
for i in range(len(similarities)):
    start = max(0, i - buffer_size)
    end = min(len(similarities), i + buffer_size + 1)
    window = similarities[start:end]
    avg = sum(window) / len(window)
    smoothed.append(avg)
    if i < 5:
        print(f"  Position {i}: window[{start}:{end}], original={similarities[i]:.4f}, smoothed={avg:.4f}")

print(f"Smoothed: {len(smoothed)} values")
return smoothed
```

#### 5.7 生成语义块
**行数**: 274-297

```python
def _generate_semantic_chunks(self, sentences: list[str], boundaries: list[int]) -> list[str]:
    chunks = []
    start_idx = 0

    for boundary_idx in boundaries:
        if boundary_idx > start_idx:
            chunk_sentences = sentences[start_idx:boundary_idx]
            chunk_text = ' '.join(chunk_sentences)
            chunks.append(chunk_text)
            start_idx = boundary_idx

    # 添加剩余句子
    if start_idx < len(sentences):
        chunk_sentences = sentences[start_idx:]
        chunk_text = ' '.join(chunk_sentences)
        chunks.append(chunk_text)

    return chunks
```

**调试点 5.7**:
```python
print(f"=== DEBUG 5.7: _generate_semantic_chunks ===")
print(f"Total sentences: {len(sentences)}")
print(f"Boundaries: {boundaries}")

chunks = []
start_idx = 0

for idx, boundary_idx in enumerate(boundaries):
    if boundary_idx > start_idx:
        chunk_sentences = sentences[start_idx:boundary_idx]
        chunk_text = ' '.join(chunk_sentences)
        chunks.append(chunk_text)
        print(f"  Chunk {idx}: sentences[{start_idx}:{boundary_idx}] = {len(chunk_sentences)} sentences, {len(chunk_text)} chars")
        start_idx = boundary_idx

if start_idx < len(sentences):
    chunk_sentences = sentences[start_idx:]
    chunk_text = ' '.join(chunk_sentences)
    chunks.append(chunk_text)
    print(f"  Final chunk: sentences[{start_idx}:] = {len(chunk_sentences)} sentences, {len(chunk_text)} chars")

print(f"Total chunks generated: {len(chunks)}")
return chunks
```

#### 5.8 后处理 (合并、切分、重叠)
**行数**: 299-319

```python
def _post_process_chunks(self, chunks: list[str]) -> list[str]:
    # Step 1: 合并短块 (< min_chunk_tokens)
    merged_chunks = self._merge_short_chunks(chunks)

    # Step 2: 切分长块 (> max_chunk_tokens)
    split_chunks = self._split_long_chunks(merged_chunks)

    # Step 3: 添加重叠
    final_chunks = self._add_overlap(split_chunks)

    return final_chunks
```

**调试点 5.8**:
```python
print(f"=== DEBUG 5.8: _post_process_chunks ===")
print(f"Input chunks: {len(chunks)}")

# Step 1
merged_chunks = self._merge_short_chunks(chunks)
print(f"\nStep 1: Merge short chunks (< {self._min_chunk_tokens} tokens)")
print(f"After merging: {len(merged_chunks)} chunks")
for i, chunk in enumerate(merged_chunks[:3]):
    tokens = self._get_token_count(chunk)
    print(f"  Chunk {i}: {len(chunk)} chars, {tokens} tokens")

# Step 2
split_chunks = self._split_long_chunks(merged_chunks)
print(f"\nStep 2: Split long chunks (> {self._max_chunk_tokens} tokens)")
print(f"After splitting: {len(split_chunks)} chunks")
for i, chunk in enumerate(split_chunks[:3]):
    tokens = self._get_token_count(chunk)
    print(f"  Chunk {i}: {len(chunk)} chars, {tokens} tokens")

# Step 3
final_chunks = self._add_overlap(split_chunks)
print(f"\nStep 3: Add overlap ({self._chunk_overlap} tokens)")
print(f"Final chunks: {len(final_chunks)} chunks")
for i, chunk in enumerate(final_chunks[:3]):
    tokens = self._get_token_count(chunk)
    print(f"  Chunk {i}: {len(chunk)} chars, {tokens} tokens")

return final_chunks
```

---

## 🔍 快速调试方案

### 方案 1: 在 API 层添加详细日志

在 `controllers/console/datasets/datasets.py` 的 `DatasetIndexingEstimateApi.post()` 方法中：

```python
def post(self):
    import json

    parser = reqparse.RequestParser()
    # ... parse args ...

    # 调试点: 打印完整请求
    print("\n" + "="*80)
    print("SEMANTIC CHUNKING DEBUG - API Entry")
    print("="*80)
    print(json.dumps(args, indent=2, ensure_ascii=False))
    print("="*80 + "\n")

    # ... rest of method ...

    # 调试点: 打印结果
    print("\n" + "="*80)
    print("SEMANTIC CHUNKING DEBUG - API Result")
    print("="*80)
    print(f"Total segments: {response['total_segments']}")
    print(f"Preview count: {len(response.get('preview', []))}")
    for i, preview in enumerate(response.get('preview', [])[:3]):
        print(f"\nChunk {i}:")
        print(f"  Length: {len(preview.get('content', ''))} chars")
        print(f"  Content: {preview.get('content', '')[:100]}...")
    print("="*80 + "\n")

    return response
```

### 方案 2: 在 SemanticTextSplitter 添加日志

在 `core/rag/splitter/semantic_text_splitter.py` 的 `split_text()` 方法中：

```python
def split_text(self, text: str) -> list[str]:
    print("\n" + "="*80)
    print("SEMANTIC TEXT SPLITTER - START")
    print("="*80)
    print(f"Input: {len(text)} chars")
    print(f"Config:")
    print(f"  separator: {repr(self._separator)}")
    print(f"  max_tokens: {self._max_tokens}")
    print(f"  chunk_overlap: {self._chunk_overlap}")
    print(f"  threshold_amount: {self._threshold_amount}")
    print(f"  buffer_size: {self._buffer_size}")
    print(f"  min_chunk_tokens: {self._min_chunk_tokens}")
    print(f"  max_chunk_tokens: {self._max_chunk_tokens}")

    # Step 1
    paragraphs = self._split_by_separator(text)
    print(f"\n[Step 1] Paragraphs: {len(paragraphs)}")

    # Step 2
    all_sentences = []
    for para in paragraphs:
        sentences = self._split_into_sentences(para)
        all_sentences.extend(sentences)
    print(f"[Step 2] Sentences: {len(all_sentences)}")

    # Step 3-4
    semantic_boundaries = self._find_semantic_boundaries(all_sentences)
    print(f"[Step 3-4] Boundaries: {len(semantic_boundaries)}")
    print(f"  Positions: {semantic_boundaries[:10]}...")

    # Step 5
    semantic_chunks = self._generate_semantic_chunks(all_sentences, semantic_boundaries)
    print(f"[Step 5] Semantic chunks: {len(semantic_chunks)}")

    # Step 6
    final_chunks = self._post_process_chunks(semantic_chunks)
    print(f"[Step 6] Final chunks: {len(final_chunks)}")

    print("\n" + "="*80)
    print("SEMANTIC TEXT SPLITTER - END")
    print(f"Result: {len(final_chunks)} chunks")
    for i, chunk in enumerate(final_chunks[:3]):
        tokens = self._get_token_count(chunk)
        print(f"\nChunk {i}: {len(chunk)} chars, {tokens} tokens")
        print(f"Preview: {chunk[:80]}...")
    print("="*80 + "\n")

    return final_chunks
```

### 方案 3: 创建测试脚本

创建文件 `test_semantic_chunking.py`:

```python
import sys
sys.path.insert(0, '/home/user/dify_api')

from core.rag.splitter.semantic_text_splitter import SemanticTextSplitter

# 测试文本
test_text = """
深度学习是机器学习的一个分支。它基于人工神经网络的研究。

深度学习模型通常包含多个隐藏层。每一层都可以学习数据的不同特征。这使得模型能够处理复杂的任务。

计算机视觉是深度学习的重要应用领域。卷积神经网络在图像识别中表现出色。它们可以自动学习图像的特征。

自然语言处理也受益于深度学习。循环神经网络和Transformer模型在文本处理中非常有效。它们能够理解语言的上下文关系。
"""

# 创建分割器
splitter = SemanticTextSplitter(
    separator="\n\n",
    max_tokens=1024,
    chunk_overlap=50,
    threshold_amount=95,
    buffer_size=2,
    min_chunk_tokens=150,
    max_chunk_tokens=1000,
    embedding_model_instance=None  # 使用后备方案
)

# 执行分块
chunks = splitter.split_text(test_text)

# 打印结果
print(f"\n生成了 {len(chunks)} 个块:\n")
for i, chunk in enumerate(chunks):
    print(f"--- 块 {i+1} ---")
    print(f"长度: {len(chunk)} 字符")
    print(f"内容: {chunk}")
    print()
```

运行测试:
```bash
cd /home/user/dify_api
python3 test_semantic_chunking.py
```

---

## 📊 关键指标监控

在调试过程中，关注以下指标：

### 输入阶段
- 原始文档长度
- 分隔符类型
- 语义参数配置

### 处理阶段
- 段落数量
- 句子数量
- 相似度分布 (min, max, mean)
- 阈值大小
- 边界位置

### 输出阶段
- 块数量
- 每块的 token 数
- 是否有过短/过长的块
- 重叠是否正确应用

---

## 🐛 常见问题排查

### 问题 1: 生成的块太多或太少
**检查点**:
- `threshold_amount` 值 (建议 90-98)
- `buffer_size` 值 (建议 1-5)
- 句子切分是否正确

### 问题 2: 块大小不符合预期
**检查点**:
- `min_chunk_tokens` 和 `max_chunk_tokens` 设置
- token 计数函数是否正确
- `_merge_short_chunks` 和 `_split_long_chunks` 逻辑

### 问题 3: 语义边界不准确
**检查点**:
- Embedding 模型是否可用
- 相似度计算是否正确
- 平滑处理是否合理

### 问题 4: 重叠不正确
**检查点**:
- `chunk_overlap` 值
- `_add_overlap` 实现逻辑

---

## 📝 日志输出示例

启用所有调试点后，你会看到类似这样的输出：

```
=== DEBUG 1.1: Received args ===
doc_form: semantic_model
segmentation: {'separator': '\\n\\n', 'max_tokens': 1024, ...}

=== DEBUG 2.1: DocumentService.estimate ===
doc_form: semantic_model
indexing_technique: high_quality

=== DEBUG 3.1: IndexProcessor Created ===
Processor type: SemanticIndexProcessor
doc_form: semantic_model

=== DEBUG 3.2: After Extract ===
Number of documents: 1
Doc 0 length: 5234 chars

=== DEBUG 4.2: SemanticIndexProcessor.transform START ===
Input documents: 1
Segmentation config:
  separator: \n\n
  max_tokens: 1024
  threshold_amount: 95
  ...

=== DEBUG 5.1: SemanticTextSplitter.split_text START ===
Step 1: Split by separator
Paragraphs: 4

Step 2: Split into sentences
Total sentences: 12

Step 3-4: Find semantic boundaries
Boundaries found: [3, 7, 10]

Step 5: Generate semantic chunks
Semantic chunks: 4

Step 6: Post-process chunks
Final chunks: 5

=== DEBUG 5.4: _find_semantic_boundaries ===
Similarities: min=0.3421, max=0.9876, mean=0.7234
Threshold: 0.8234
Boundaries: [3, 7, 10]

=== DEBUG 3.3: After Transform ===
Number of chunks: 5
Chunk 0: 234 chars
Chunk 1: 456 chars
...
```

---

## 🎯 建议的调试顺序

1. **先测试 API 入口** - 确认请求能正确到达
2. **检查参数验证** - 确保参数格式正确
3. **验证 Processor 创建** - 确认使用了 SemanticIndexProcessor
4. **追踪 Extract** - 确认文档正确提取
5. **深入 Transform** - 这是核心，重点调试
6. **详查 split_text** - 逐步验证每个阶段
7. **分析边界检测** - 检查相似度和阈值
8. **验证后处理** - 确保合并/切分/重叠正确

---

## 📞 需要帮助？

如果遇到问题，请提供以下信息：
- 完整的请求 JSON
- 关键调试点的输出
- 期望的结果 vs 实际结果
- 错误信息（如有）
