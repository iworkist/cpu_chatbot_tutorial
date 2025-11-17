"""
Chroma DB의 .update()와 .upsert() 메서드 간단 예제

필요한 패키지:
  uv add chromadb
"""

import chromadb

# Chroma 클라이언트 생성 (메모리 모드)
client = chromadb.Client()

# 컬렉션 생성
collection = client.create_collection(name="update_example")

# 초기 데이터 추가
collection.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "파이썬은 배우기 쉬운 언어입니다.",
        "자바스크립트는 웹 개발에 사용됩니다.",
        "Go는 효율적인 언어입니다."
    ],
    metadatas=[
        {"language": "Python", "level": 1},
        {"language": "JavaScript", "level": 2},
        {"language": "Go", "level": 2}
    ]
)

print(f"초기 문서 수: {collection.count()}")
print("\n=== 초기 데이터 ===")
result = collection.get(ids=["id1", "id2"])
for id, doc in zip(result['ids'], result['documents']):
    print(f"{id}: {doc}")

# ============================================================
# update() - 기존 ID만 수정 (존재하지 않으면 무시)
print("\n=== update() 테스트 ===")

collection.update(
    ids=["id1"],
    documents=["파이썬은 초보자에게 인기 있는 언어입니다."],
    metadatas=[{"language": "Python", "level": 1, "popular": True}]
)

result = collection.get(ids=["id1"])
print(f"id1 업데이트 후: {result['documents'][0]}")

# ============================================================
# upsert() - 있으면 업데이트, 없으면 추가
print("\n=== upsert() 테스트 ===")

collection.upsert(
    ids=["id2", "id4"],  # id2는 업데이트, id4는 새로 추가
    documents=[
        "자바스크립트는 프론트엔드와 백엔드에서 사용됩니다.",
        "Rust는 안전한 시스템 프로그래밍 언어입니다."
    ],
    metadatas=[
        {"language": "JavaScript", "level": 2, "fullstack": True},
        {"language": "Rust", "level": 3}
    ]
)

print(f"upsert 후 문서 수: {collection.count()}")
print("\n=== 최종 데이터 ===")
result = collection.get()
for id, doc in zip(result['ids'], result['documents']):
    print(f"{id}: {doc}")

print("\n완료!")
