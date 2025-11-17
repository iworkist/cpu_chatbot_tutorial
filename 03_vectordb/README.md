# Vector Database 예제

이 디렉토리는 벡터 데이터베이스(Vector Database) 사용 예제를 포함합니다.

<https://docs.trychroma.com/docs/collections/manage-collections>

## 벡터 데이터베이스란?

벡터 데이터베이스는 고차원 벡터 데이터를 효율적으로 저장하고 검색할 수 있는 특수한 데이터베이스입니다. 주로 다음과 같은 용도로 사용됩니다:

- **의미 기반 검색**: 텍스트의 의미를 이해하여 유사한 내용 검색
- **추천 시스템**: 사용자의 선호도와 유사한 항목 추천
- **RAG (Retrieval Augmented Generation)**: LLM에 관련 문서를 제공하여 답변 품질 향상
- **이미지/음성 검색**: 멀티모달 데이터의 유사도 검색

## 파일 설명

### `eg_chroma_add.py`

Chroma DB의 `.add()` 메서드 사용 예제입니다.

**주요 내용:**

- 방법 1: documents만 제공 (Chroma가 자동 임베딩)
- 방법 2: embeddings + documents 모두 제공
- 방법 3: embeddings + metadata만 (documents 외부 저장)
- 중복 ID 처리

**실행 방법:**

```bash
python 03_vectordb/eg_chroma_add.py
```

### `eg_chroma_bge-m3.py`

Chroma DB와 BGE-M3 모델을 사용한 한국어 벡터 데이터베이스 예제입니다.

**주요 기능:**

- BGE-M3 모델을 사용한 한국어 텍스트 임베딩
- Chroma DB에 문서와 메타데이터 저장
- 의미 기반 유사도 검색
- 메타데이터 필터링을 통한 검색
- 특정 ID로 문서 조회

**실행 방법:**

```bash
python 03_vectordb/eg_chroma_bge-m3.py
```

### `eg_chroma_update.py`

Chroma DB의 `.update()`와 `.upsert()` 메서드 간단 예제입니다.

**주요 내용:**

- `update()`: 기존 ID만 수정 (존재하지 않으면 무시)
- `upsert()`: 있으면 업데이트, 없으면 추가
- documents 업데이트 시 자동 임베딩 재계산

**실행 방법:**

```bash
python 03_vectordb/eg_chroma_update.py
```

### `eg_chroma_delete.py`

Chroma DB의 `.delete()` 메서드 간단 예제입니다.

**주요 내용:**

- ID로 삭제
- where 필터로 삭제
- ID와 where 조합으로 삭제

**실행 방법:**

```bash
python 03_vectordb/eg_chroma_delete.py
```

### `eg_cross_encoder.py`

BGE Reranker를 사용한 검색 결과 재순위화 예제입니다.

**주요 내용:**

- 1차 검색: 임베딩 벡터 유사도 기반 (빠름)
- 2차 검색: Cross Encoder로 재순위화 (정확함)
- Two-stage retrieval 패턴

**실행 방법:**

```bash
python 03_vectordb/eg_cross_encoder.py
```

## BGE-M3 모델

[BGE-M3](https://huggingface.co/BAAI/bge-m3)는 BAAI(Beijing Academy of Artificial Intelligence)에서 개발한 다국어 임베딩 모델입니다.

**특징:**

- 다국어 지원 (한국어 포함)
- 1024차원 임베딩 벡터
- 높은 검색 성능
- Multi-Functionality: dense retrieval, multi-vector retrieval, sparse retrieval 지원
- Multi-Linguality: 100개 이상의 언어 지원
- Multi-Granularity: 다양한 길이의 텍스트 처리 (최대 8192 토큰)

## BGE Reranker

[BGE Reranker v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)는 검색 결과의 순위를 재조정하는 Cross Encoder 모델입니다.

**특징:**

- Query와 Document의 관계를 직접 평가
- 임베딩 검색보다 높은 정확도
- Two-stage retrieval에 적합 (1차: 빠른 검색, 2차: 정밀한 재순위화)
- 다국어 지원

**사용 시나리오:**

1. 임베딩으로 상위 N개 후보 검색 (빠름)
2. Reranker로 후보들의 순위 재조정 (정확함)
3. 최종 상위 K개 결과 반환

## Chroma DB

[Chroma](https://www.trychroma.com/)는 AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스입니다.

**특징:**

- 간단한 Python API
- 메모리 모드와 영구 저장 모드 지원
- 메타데이터 필터링
- 다양한 임베딩 함수 지원
- 빠른 유사도 검색

## CRUD 작업 예제

### Create (추가)

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(name="my_collection")

collection.add(
    ids=["id1", "id2"],
    documents=["첫 번째 문서", "두 번째 문서"],
    metadatas=[{"category": "A"}, {"category": "B"}]
)
```

### Read (조회)

```python
# ID로 조회
result = collection.get(ids=["id1"])

# 검색 (유사도 기반)
results = collection.query(
    query_texts=["검색어"],
    n_results=5
)

# 메타데이터 필터링
results = collection.query(
    query_texts=["검색어"],
    where={"category": "A"}
)
```

### Update (수정)

```python
# 기존 ID만 수정
collection.update(
    ids=["id1"],
    documents=["수정된 문서"]
)

# 있으면 수정, 없으면 추가
collection.upsert(
    ids=["id1", "id3"],
    documents=["수정된 문서", "새 문서"]
)
```

### Delete (삭제)

```python
# ID로 삭제
collection.delete(ids=["id1"])

# 조건으로 삭제
collection.delete(where={"category": "A"})
```

## 필요한 패키지

```bash
# 기본 패키지
uv add chromadb

# BGE-M3 사용 시
uv add chromadb sentence-transformers
```

## 참고 자료

- [Chroma 공식 문서](https://docs.trychroma.com/)
- [BGE-M3 모델 페이지](https://huggingface.co/BAAI/bge-m3)
- [BGE Reranker v2-m3 모델 페이지](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Sentence Transformers 문서](https://www.sbert.net/)
