"""
Chroma DB와 BGE-M3 모델을 사용한 한국어 벡터 데이터베이스 예제

필요한 패키지:
  uv add chromadb sentence-transformers
"""

from chromadb.api.types import Document


from chromadb.base_types import Metadata


from chromadb.api.types import Document


from chromadb.base_types import Metadata


import chromadb
from sentence_transformers import SentenceTransformer
import json

# BGE-M3 모델 로드 (한국어 임베딩에 강력한 다국어 모델)
print("BGE-M3 모델을 로드하는 중...")
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로드 완료!")

# Chroma DB 클라이언트 생성 (메모리 모드)
client = chromadb.Client()

# 컬렉션 생성
collection = client.create_collection(
    name="korean_documents",
    metadata={"description": "한국어 문서 임베딩 컬렉션"}
)

# 한국어 샘플 문서들
documents = [
    "인공지능은 인간의 학습능력, 추론능력, 지각능력을 컴퓨터 프로그램으로 실현한 기술입니다.",
    "머신러닝은 데이터를 기반으로 컴퓨터가 스스로 학습하는 인공지능의 한 분야입니다.",
    "딥러닝은 인공신경망을 기반으로 한 머신러닝의 한 종류로, 다층 구조를 가집니다.",
    "자연어 처리는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 기술입니다.",
    "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색할 수 있는 데이터베이스입니다.",
    "임베딩은 텍스트나 이미지 같은 데이터를 고차원 벡터 공간으로 변환하는 과정입니다.",
    "서울은 대한민국의 수도이자 최대 도시입니다.",
    "김치는 한국의 대표적인 발효 음식으로 세계적으로 유명합니다.",
    "한글은 1443년 세종대왕이 창제한 한국의 고유 문자입니다.",
    "K-pop은 한국 대중음악을 지칭하는 말로 전 세계적인 인기를 얻고 있습니다.",
]

# 문서에 대한 메타데이터
metadatas = [
    {"category": "AI", "topic": "인공지능 개론"},
    {"category": "AI", "topic": "머신러닝"},
    {"category": "AI", "topic": "딥러닝"},
    {"category": "AI", "topic": "자연어 처리"},
    {"category": "AI", "topic": "벡터 데이터베이스"},
    {"category": "AI", "topic": "임베딩"},
    {"category": "한국", "topic": "도시"},
    {"category": "한국", "topic": "음식"},
    {"category": "한국", "topic": "언어"},
    {"category": "한국", "topic": "문화"},
]

# 문서 ID
ids = [f"doc_{i}" for i in range(len(documents))]

print("\n문서를 임베딩하는 중...")
# BGE-M3로 문서 임베딩 생성
embeddings = model.encode(documents, normalize_embeddings=True)
print(f"임베딩 완료! 임베딩 차원: {embeddings.shape}")

# Chroma DB에 문서와 임베딩 저장
print("\nChroma DB에 문서를 저장하는 중...")
collection.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print(f"총 {len(documents)}개의 문서가 저장되었습니다.")

# 저장된 문서 수 확인
print(f"\n컬렉션에 저장된 문서 수: {collection.count()}")

# ============================================================
# 유사도 검색 예제
# ============================================================

print("\n" + "="*60)
print("검색 예제 1: AI 관련 질문")
print("="*60)

query1 = "컴퓨터가 스스로 학습하는 기술은 무엇인가요?"
print(f"질문: {query1}")

# 질문을 임베딩
query_embedding1 = model.encode([query1], normalize_embeddings=True)

# 유사한 문서 검색 (상위 3개)
results1 = collection.query(
    query_embeddings=query_embedding1.tolist(),
    n_results=3
)

print("\n가장 유사한 문서들:")
for i, (doc, metadata, distance) in enumerate[tuple[Document, Metadata, float]](zip[tuple[Document, Metadata, float]](
    results1['documents'][0],
    results1['metadatas'][0],
    results1['distances'][0]
), 1):
    print(f"\n{i}. 문서: {doc}")
    print(f"   카테고리: {metadata['category']}, 주제: {metadata['topic']}")
    print(f"   거리: {distance:.4f}")

# ============================================================
print("\n" + "="*60)
print("검색 예제 2: 한국 문화 관련 질문")
print("="*60)

query2 = "한국의 전통 문자에 대해 알려주세요"
print(f"질문: {query2}")

query_embedding2 = model.encode([query2], normalize_embeddings=True)

results2 = collection.query(
    query_embeddings=query_embedding2.tolist(),
    n_results=3
)

print("\n가장 유사한 문서들:")
for i, (doc, metadata, distance) in enumerate(zip(
    results2['documents'][0],
    results2['metadatas'][0],
    results2['distances'][0]
), 1):
    print(f"\n{i}. 문서: {doc}")
    print(f"   카테고리: {metadata['category']}, 주제: {metadata['topic']}")
    print(f"   거리: {distance:.4f}")

# ============================================================
print("\n" + "="*60)
print("검색 예제 3: 메타데이터 필터링과 함께 검색")
print("="*60)

query3 = "데이터를 벡터로 변환하는 방법"
print(f"질문: {query3}")
print("필터: category='AI'인 문서만 검색")

query_embedding3 = model.encode([query3], normalize_embeddings=True)

results3 = collection.query(
    query_embeddings=query_embedding3.tolist(),
    n_results=3,
    where={"category": "AI"}  # AI 카테고리만 검색
)

print("\n가장 유사한 문서들 (AI 카테고리만):")
for i, (doc, metadata, distance) in enumerate(zip(
    results3['documents'][0],
    results3['metadatas'][0],
    results3['distances'][0]
), 1):
    print(f"\n{i}. 문서: {doc}")
    print(f"   카테고리: {metadata['category']}, 주제: {metadata['topic']}")
    print(f"   거리: {distance:.4f}")

# ============================================================
print("\n" + "="*60)
print("특정 ID로 문서 조회")
print("="*60)

# 특정 문서 ID로 조회
doc_ids = ["doc_0", "doc_7"]
get_results = collection.get(
    ids=doc_ids,
    include=["documents", "metadatas"]
)

print(f"\n조회한 문서 ID: {doc_ids}")
for doc_id, doc, metadata in zip(
    get_results['ids'],
    get_results['documents'],
    get_results['metadatas']
):
    print(f"\nID: {doc_id}")
    print(f"문서: {doc}")
    print(f"메타데이터: {metadata}")

# ============================================================
print("\n" + "="*60)
print("요약")
print("="*60)
print(f"✓ BGE-M3 모델을 사용한 한국어 임베딩")
print(f"✓ Chroma DB에 {collection.count()}개 문서 저장")
print(f"✓ 임베딩 차원: {embeddings.shape[1]}차원")
print(f"✓ 유사도 검색 및 메타데이터 필터링 기능 시연")
print("\n완료!")

