"""
Chroma DB의 .delete() 메서드 간단 예제

필요한 패키지:
  uv add chromadb
"""

import chromadb

# Chroma 클라이언트 생성 (메모리 모드)
client = chromadb.Client()

# 컬렉션 생성
collection = client.create_collection(name="delete_example")

# 초기 데이터 추가
collection.add(
    ids=["id1", "id2", "id3", "id4", "id5"],
    documents=[
        "파이썬 기초 강의",
        "파이썬 중급 강의",
        "자바스크립트 기초 강의",
        "자바스크립트 중급 강의",
        "Go 기초 강의"
    ],
    metadatas=[
        {"language": "Python", "level": "basic"},
        {"language": "Python", "level": "intermediate"},
        {"language": "JavaScript", "level": "basic"},
        {"language": "JavaScript", "level": "intermediate"},
        {"language": "Go", "level": "basic"}
    ]
)

print(f"초기 문서 수: {collection.count()}")
print("\n=== 초기 데이터 ===")
result = collection.get()
for id, doc, meta in zip(result['ids'], result['documents'], result['metadatas']):
    print(f"{id}: {doc} - {meta}")

# ============================================================
# 1. ID로 삭제
print("\n=== ID로 삭제 ===")

collection.delete(ids=["id1"])

print(f"id1 삭제 후 문서 수: {collection.count()}")

# ============================================================
# 2. where 필터로 삭제
print("\n=== where 필터로 삭제 ===")

collection.delete(
    where={"language": "JavaScript"}  # JavaScript 문서 모두 삭제
)

print(f"JavaScript 삭제 후 문서 수: {collection.count()}")

# ============================================================
# 3. ID와 where 조합으로 삭제
print("\n=== ID와 where 조합 ===")

# 추가 데이터
collection.add(
    ids=["id6", "id7"],
    documents=[
        "Rust 기초 강의",
        "Rust 중급 강의"
    ],
    metadatas=[
        {"language": "Rust", "level": "basic"},
        {"language": "Rust", "level": "intermediate"}
    ]
)

print(f"데이터 추가 후: {collection.count()}개")

# id6, id7 중에서 level이 basic인 것만 삭제
collection.delete(
    ids=["id6", "id7"],
    where={"level": "basic"}
)

print(f"조건부 삭제 후: {collection.count()}개")

# ============================================================
print("\n=== 최종 데이터 ===")
result = collection.get()
for id, doc, meta in zip(result['ids'], result['documents'], result['metadatas']):
    print(f"{id}: {doc} - {meta}")

print("\n완료!")

