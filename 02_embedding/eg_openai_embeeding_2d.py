from openai import OpenAI
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 출력 디렉토리 생성
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 파일 경로 설정
embeddings_file = output_dir / "embeddings_2d.json"
image_file = output_dir / "embeddings_2d_visualization.png"

# 예시 데이터: 비슷한 의미끼리 클러스터링되는 모습을 보여주기 위한 예시
examples = {
    "positive_weather": [
        "The weather is beautiful today",
        "It's sunny and bright outside",
        "What a perfect day with clear skies"
    ],
    "negative_weather": [
        "It's raining heavily outside",
        "The weather is gloomy and cloudy",
        "Strong winds and storms are coming"
    ],
    "positive_food": [
        "This food is absolutely delicious",
        "The taste is amazing and perfect",
        "I'm impressed by how good this tastes",
        "SM. I love Indian food. Naan and curry are my favorites."
    ],
    "negative_food": [
        "This food doesn't taste good",
        "The flavor is strange and unpleasant",
        "I don't want to eat this again",
        "SM. I dislike cucumbers. They have a bitter taste."
    ]
}

# 모든 텍스트와 카테고리 정보 수집
texts = []
categories = []

for category, items in examples.items():
    for text in items:
        texts.append(text)
        categories.append(category)

# 임베딩 로딩 또는 생성
if embeddings_file.exists():
    print(f"저장된 임베딩 파일을 로딩 중... ({embeddings_file})")
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        embeddings = data['embeddings']
        saved_texts = data['texts']
        saved_categories = data['categories']
        
        # 데이터가 일치하는지 확인
        if saved_texts == texts and saved_categories == categories:
            print("임베딩 로딩 완료!")
        else:
            print("데이터가 변경되어 새로 임베딩을 생성합니다.")
            embeddings = None
else:
    embeddings = None

# 임베딩이 없으면 새로 생성
if embeddings is None:
    print(f"총 {len(texts)}개의 예시 텍스트를 임베딩 생성 중...")
    
    # OpenAI API를 사용하여 임베딩 생성 (배치 처리로 효율화)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    
    # 임베딩 저장
    print(f"임베딩을 파일에 저장 중... ({embeddings_file})")
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump({
            'texts': texts,
            'categories': categories,
            'embeddings': embeddings
        }, f, ensure_ascii=False, indent=2)
    
    print("임베딩 생성 및 저장 완료!")

# t-SNE 모델 생성 및 변환 (비슷한 의미끼리 클러스터링되는 모습을 보여줌)
print("t-SNE로 2D 시각화 중...")
embeddings_array = np.array(embeddings)  # 리스트를 numpy 배열로 변환
tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(embeddings_array)

# 카테고리별 색상 지정
category_colors = {
    "positive_weather": "gold",
    "negative_weather": "blue",
    "positive_food": "red",
    "negative_food": "green"
}

# 시각화
plt.figure(figsize=(12, 8))
for category in category_colors.keys():
    indices = [i for i, cat in enumerate(categories) if cat == category]
    x_coords = [vis_dims[i][0] for i in indices]
    y_coords = [vis_dims[i][1] for i in indices]
    plt.scatter(x_coords, y_coords, c=category_colors[category], 
                label=category, alpha=0.6, s=100)
    # 각 점에 텍스트 라벨 추가 (간단하게 표시)
    for i, idx in enumerate(indices):
        # 텍스트가 너무 길면 앞부분만 표시
        label = texts[idx][:25] + "..." if len(texts[idx]) > 25 else texts[idx]
        plt.annotate(label, (x_coords[i], y_coords[i]), 
                    fontsize=9, alpha=0.7, ha='left')

plt.title("Semantic Clustering with t-SNE", fontsize=14, fontweight='bold')
plt.xlabel("t-SNE Dimension 1", fontsize=12)
plt.ylabel("t-SNE Dimension 2", fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 이미지 파일로 저장
print(f"시각화 이미지 저장 중... ({image_file})")
plt.savefig(image_file, dpi=300, bbox_inches='tight')
print(f"이미지 저장 완료: {image_file}")

plt.show()

print("시각화 완료!")
print("\n=== 결과 설명 ===")
print("비슷한 의미를 가진 문장들이 2D 공간에서 가까이 모여 클러스터를 형성합니다.")
print("- positive_weather (금색): 긍정적인 날씨 표현")
print("- negative_weather (파랑): 부정적인 날씨 표현")
print("- positive_food (빨강): 긍정적인 음식 표현")
print("- negative_food (초록): 부정적인 음식 표현")