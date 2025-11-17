"""
Chroma DB의 .add() 메서드 사용 예제

Chroma에 데이터를 추가하는 세 가지 방법을 보여줍니다:
1. documents만 제공 - Chroma가 자동으로 임베딩 생성
2. embeddings + documents - 수동으로 생성한 임베딩 제공
3. embeddings + metadata만 - documents는 외부에 저장

필요한 패키지:
  uv add chromadb sentence-transformers
"""

import chromadb
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("방법 1: documents만 제공 (Chroma가 자동 임베딩)")
print("=" * 60)

# Chroma 클라이언트 생성 (메모리 모드)
client = chromadb.Client()

# 기본 임베딩 함수를 사용하는 컬렉션 생성
collection1 = client.create_collection(
    name="auto_embedding_collection",
    metadata={"description": "Chroma가 자동으로 임베딩을 생성하는 컬렉션"}
)

# documents만 제공하면 Chroma가 자동으로 임베딩을 생성합니다
collection1.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "인공지능은 컴퓨터가 인간처럼 생각하고 학습하는 기술입니다.",
        "머신러닝은 데이터로부터 패턴을 학습하는 방법입니다.",
        "딥러닝은 인공신경망을 사용하는 머신러닝의 한 분야입니다."
    ],
    metadatas=[
        {"topic": "AI", "level": "basic"},
        {"topic": "ML", "level": "basic"},
        {"topic": "DL", "level": "intermediate"}
    ]
)

print(f"추가된 문서 수: {collection1.count()}")
print("\n저장된 문서 확인:")
results1 = collection1.get(ids=["id1", "id2"])
for i, (doc_id, doc) in enumerate[_T_co](zip(results1['ids'], results1['documents']), 1):
    print(f"{i}. ID: {doc_id}")
    print(f"   문서: {doc}")

# ============================================================
print("\n" + "=" * 60)
print("방법 2: embeddings + documents 모두 제공")
print("=" * 60)

# BGE-M3 모델 로드
print("\nBGE-M3 모델 로드 중...")
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로드 완료!")

# 새 컬렉션 생성
collection2 = client.create_collection(
    name="manual_embedding_collection",
    metadata={"description": "수동으로 임베딩을 제공하는 컬렉션"}
)

# 문서 준비
documents = [
    "서울은 대한민국의 수도입니다.",
    "부산은 한국의 제2의 도시입니다.",
    "제주도는 아름다운 관광지입니다."
]

# 직접 임베딩 생성
embeddings = model.encode(documents, normalize_embeddings=True)
print(f"\n생성된 임베딩 차원: {embeddings.shape}")

# embeddings와 documents를 모두 제공
collection2.add(
    ids=["city1", "city2", "city3"],
    embeddings=embeddings.tolist(),
    documents=documents,
    metadatas=[
        {"city": "서울", "type": "수도"},
        {"city": "부산", "type": "광역시"},
        {"city": "제주", "type": "특별자치도"}
    ]
)

print(f"\n추가된 문서 수: {collection2.count()}")

# 유사도 검색 테스트
query = "한국의 주요 도시는?"
query_embedding = model.encode([query], normalize_embeddings=True)

search_results = collection2.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

print(f"\n검색 질문: {query}")
print("검색 결과:")
for i, (doc, metadata) in enumerate(zip(
    search_results['documents'][0],
    search_results['metadatas'][0]
), 1):
    print(f"{i}. {doc}")
    print(f"   메타데이터: {metadata}")

# ============================================================
print("\n" + "=" * 60)
print("방법 3: embeddings + metadata만 (documents 외부 저장)")
print("=" * 60)

# 새 컬렉션 생성
collection3 = client.create_collection(
    name="embedding_only_collection",
    metadata={"description": "임베딩만 저장하고 문서는 외부에 저장"}
)

# 실제로는 documents를 데이터베이스나 파일 시스템에 저장하고
# Chroma에는 임베딩과 메타데이터만 저장합니다
external_docs = {
    "doc_001": "파이썬은 배우기 쉬운 프로그래밍 언어입니다.",
    "doc_002": "자바스크립트는 웹 개발에 필수적인 언어입니다.",
    "doc_003": "Go는 구글이 만든 효율적인 언어입니다."
}

# 임베딩 생성 (실제 문서 사용)
doc_texts = list(external_docs.values())
embeddings_only = model.encode(doc_texts, normalize_embeddings=True)

# embeddings와 metadata만 Chroma에 저장
# documents는 제공하지 않음 (외부에 저장되어 있다고 가정)
collection3.add(
    ids=list(external_docs.keys()),
    embeddings=embeddings_only.tolist(),
    metadatas=[
        {"language": "Python", "difficulty": "easy", "doc_location": "db_table_1"},
        {"language": "JavaScript", "difficulty": "medium", "doc_location": "db_table_2"},
        {"language": "Go", "difficulty": "medium", "doc_location": "db_table_3"}
    ]
)

print(f"\n추가된 항목 수: {collection3.count()}")

# 검색 후 외부 저장소에서 문서 가져오기
query2 = "초보자에게 좋은 프로그래밍 언어"
query_embedding2 = model.encode([query2], normalize_embeddings=True)

search_results2 = collection3.query(
    query_embeddings=query_embedding2.tolist(),
    n_results=2
)

print(f"\n검색 질문: {query2}")
print("검색 결과 (ID와 메타데이터만 Chroma에서 가져옴):")
for i, (doc_id, metadata) in enumerate(zip(
    search_results2['ids'][0],
    search_results2['metadatas'][0]
), 1):
    # 외부 저장소에서 실제 문서 가져오기
    actual_doc = external_docs[doc_id]
    print(f"{i}. ID: {doc_id}")
    print(f"   메타데이터: {metadata}")
    print(f"   실제 문서 (외부 저장소에서 가져옴): {actual_doc}")

# ============================================================
print("\n" + "=" * 60)
print("중복 ID 처리 테스트")
print("=" * 60)

# 이미 존재하는 ID로 다시 추가 시도 (무시됨, 에러 없음)
print("\n같은 ID로 다시 추가 시도...")
collection1.add(
    ids=["id1"],  # 이미 존재하는 ID
    documents=["이 문서는 추가되지 않습니다."],
    metadatas=[{"status": "ignored"}]
)

# 원본 확인 (변경되지 않음)
original = collection1.get(ids=["id1"])
print(f"ID 'id1'의 문서 (원본 유지): {original['documents'][0]}")

# ============================================================
print("\n" + "=" * 60)
print("요약")
print("=" * 60)
print("\n✓ 방법 1: Chroma가 자동으로 임베딩 생성 (간편함)")
print("✓ 방법 2: 수동 임베딩 제공 (모델 선택 자유로움, 성능 최적화)")
print("✓ 방법 3: 임베딩만 저장 (대용량 문서 처리, 외부 저장소 활용)")
print("✓ 중복 ID는 자동으로 무시됨 (재시도 안전)")
print("\n완료!")

