"""
BGE Reranker를 사용한 검색 결과 재순위화 예제

Cross Encoder는 검색 결과의 순위를 재조정하여 정확도를 높입니다.

필요한 패키지:
  uv add chromadb sentence-transformers
"""

from chromadb.api.types import ID, Document


import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. 임베딩 모델 및 Reranker 로드
print("모델 로드 중...")
embedding_model = SentenceTransformer('BAAI/bge-m3')
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
print("모델 로드 완료!\n")

# 2. Chroma DB 설정 및 문서 추가
client = chromadb.Client()
collection = client.create_collection(name="rerank_example")

documents = [
    "파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
    "자바스크립트는 웹 개발에 널리 사용됩니다.",
    "파이썬은 데이터 과학과 머신러닝에 인기가 많습니다.",
    "Go는 구글이 만든 효율적인 언어입니다.",
    "파이썬의 간결한 문법은 초보자에게 적합합니다.",
]

embeddings = embedding_model.encode(documents, normalize_embeddings=True)

collection.add(
    ids=[f"doc{i}" for i in range(len(documents))],
    embeddings=embeddings.tolist(),
    documents=documents
)

print(f"총 {len(documents)}개 문서 추가\n")

# 3. 검색 쿼리
query = "초보자에게 좋은 프로그래밍 언어는?"
print(f"질문: {query}\n")

# 4. 벡터 검색 (1차 검색)
query_embedding = embedding_model.encode([query], normalize_embeddings=True)
search_results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=5
)

print("=" * 60)
print("1차 검색 결과 (벡터 유사도)")
print("=" * 60)
for i, (doc_id, doc, distance) in enumerate(zip[tuple[ID, Document, float]](
    search_results['ids'][0],
    search_results['documents'][0],
    search_results['distances'][0]
), 1):
    print(f"{i}. [{doc_id}] {doc}")
    print(f"   거리: {distance:.4f}\n")

# 5. Reranker로 재순위화
print("=" * 60)
print("2차 검색 결과 (Reranker 적용)")
print("=" * 60)

# query와 각 문서 쌍의 relevance score 계산
pairs = [[query, doc] for doc in search_results['documents'][0]]
rerank_scores = reranker.predict(pairs)

# 점수로 재정렬
reranked_results = sorted(
    zip(search_results['ids'][0], search_results['documents'][0], rerank_scores),
    key=lambda x: x[2],
    reverse=True
)

for i, (doc_id, doc, score) in enumerate(reranked_results, 1):
    print(f"{i}. [{doc_id}] {doc}")
    print(f"   점수: {score:.4f}\n")

print("=" * 60)
print("요약")
print("=" * 60)
print("✓ 1차 검색: 임베딩 벡터 유사도 기반 (빠름)")
print("✓ 2차 검색: Cross Encoder로 재순위화 (정확함)")
print("✓ Reranker는 query와 document의 관계를 더 정밀하게 평가")
print("\n완료!")

