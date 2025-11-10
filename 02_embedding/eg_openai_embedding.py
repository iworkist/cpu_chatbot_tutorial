from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import os
# OpenAI 클라이언트 초기화
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

def get_embeddings(texts, model="text-embedding-3-small"):
    """
    OpenAI API를 사용하여 텍스트 리스트의 임베딩 생성
    
    Args:
        texts: 임베딩할 텍스트 리스트
        model: 사용할 임베딩 모델
        
    Returns:
        임베딩 벡터 리스트
    """
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"임베딩 생성 오류: {e}")
        raise

def cosine_similarity(vec1, vec2):
    """두 벡터 간의 코사인 유사도 계산"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def test_get_embeddings():
    """임베딩 모델 테스트"""
    print("=" * 60)
    print("OpenAI 임베딩 모델 테스트")
    print("=" * 60)
    
    # 테스트 텍스트들
    texts = [
        "Hello, world!",
        "This is a test",
        "OpenAI is amazing",
        "안녕하세요, 세계!",
        "이것은 테스트입니다",
        "OpenAI는 놀라워요"
    ]
    
    print(f"\n임베딩할 텍스트 개수: {len(texts)}")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    # 임베딩 생성
    print("\n임베딩 생성 중...")
    embeddings = get_embeddings(texts, model="text-embedding-3-large")
    
    # 결과 출력
    print(f"\n생성된 임베딩 개수: {len(embeddings)}")
    print(f"임베딩 벡터 차원: {len(embeddings[0])}")
    
    # 각 임베딩의 통계 정보
    print("\n임베딩 벡터 통계:")
    for i, emb in enumerate(embeddings, 1):
        emb_array = np.array(emb)
        print(f"  {i}. {texts[i-1]}")
        print(f"     - 벡터 크기: {np.linalg.norm(emb_array):.4f}")
        print(f"     - 최소값: {emb_array.min():.4f}, 최대값: {emb_array.max():.4f}")
        print(f"     - 평균: {emb_array.mean():.4f}")
    
    # 유사도 계산 예제
    print("\n" + "=" * 60)
    print("텍스트 간 유사도 분석")
    print("=" * 60)
    
    # 유사한 의미의 텍스트들 비교
    pairs = [
        (0, 3, "Hello, world! vs 안녕하세요, 세계!"),
        (1, 4, "This is a test vs 이것은 테스트입니다"),
        (2, 5, "OpenAI is amazing vs OpenAI는 놀라워요"),
        (0, 1, "Hello, world! vs This is a test"),
        (3, 4, "안녕하세요, 세계! vs 이것은 테스트입니다")
    ]
    
    for idx1, idx2, description in pairs:
        similarity = cosine_similarity(embeddings[idx1], embeddings[idx2])
        print(f"{description}: {similarity:.4f}")
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    test_get_embeddings()