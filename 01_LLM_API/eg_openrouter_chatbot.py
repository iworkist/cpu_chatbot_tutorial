import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# OpenRouter는 OpenAI 호환 API를 제공합니다
# 사용 가능한 모델 예시:
# openai/gpt-4, anthropic/claude-3-opus, meta-llama/llama-3.1-70b-instruct 등
# 모델 목록 확인: https://openrouter.ai/models

# OpenRouter API 클라이언트 초기화
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


# 대화 루프 예제
messages = [
    {"role": "system", "content": "역할:너는 공감을 잘해주는 친구. 사용자의 말을 잘 들어주고 기분 파악도 잘하고, 조언도 잘해줘. 대답은 한국어로 해."}
]

print("채팅을 시작합니다. 'quit'를 입력하면 종료됩니다.\n")

while True:
    # 사용자 입력 받기
    user_input = input("사용자: ")
    
    # 'quit' 입력 시 루프 종료
    if user_input.lower() == 'quit':
        print("대화를 종료합니다.")
        break
    
    # 사용자 메시지 추가
    messages.append({"role": "user", "content": user_input})
    
    # 스트리밍 응답 받기
    response = client.chat.completions.create(
        # model="openai/gpt-4o-mini",  # OpenRouter 모델 사용 (원하는 모델로 변경 가능)
        model="deepseek/deepseek-chat-v3.1:free",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        stream=True
    )
    
    # 응답 스트리밍 출력
    print("공감AI: ", end="", flush=True)
    full_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content
    
    print("\n")  # 줄바꿈
    
    # 어시스턴트 응답을 메시지 히스토리에 추가
    messages.append({"role": "assistant", "content": full_response})  
