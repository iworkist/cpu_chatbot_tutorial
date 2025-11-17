"""
BGE-M3 임베딩 기본 예제

BGE-M3는 다국어를 지원하는 강력한 임베딩 모델입니다.
한국어, 영어 등 100개 이상의 언어를 지원합니다.

필요한 패키지:
  uv add sentence-transformers
"""

from sentence_transformers import SentenceTransformer

# BGE-M3 모델 로드
print("BGE-M3 모델을 로드하는 중...")
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로드 완료!")

# 임베딩할 텍스트
texts = [
    "안녕하세요, 반갑습니다.",
    "인공지능은 매우 흥미로운 분야입니다.",
    "Hello, nice to meet you.",
]

# 텍스트를 벡터로 변환
print("\n텍스트를 임베딩하는 중...")
embeddings = model.encode(texts, normalize_embeddings=True)

# 결과 출력
print(f"\n임베딩 완료!")
print(f"임베딩 차원: {embeddings.shape[1]}차원")
print(f"임베딩 개수: {embeddings.shape[0]}개\n")

for i, (text, embedding) in enumerate(zip(texts, embeddings)):
    print(f"{i+1}. 텍스트: {text}")
    print(f"   임베딩 벡터 (처음 10개): {embedding[:10]}")
    print(f"   벡터 크기: {len(embedding)}\n")

