from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Ollama는 OpenAI 호환 API를 제공합니다
# Ollama 서버가 localhost:11434에서 실행 중이어야 합니다
# 사용 가능한 모델 예시:
# llama3.2, llama3.1, llama3, mistral, qwen2.5, gemma2, phi3 등
# 설치된 모델 확인: ollama list

# Ollama API 클라이언트 초기화
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama는 API 키가 필요 없지만, OpenAI 클라이언트는 필수이므로 임의 값 사용
)


# 대화 루프 예제
messages = [
    {"role": "system", "content": "역할:너는 공감을 잘해주는 친구. 사용자의 말을 잘 들어주고 기분 파악도 잘하고, 조언도 잘해줘. 대답은 한국어로 해. /no_think"}
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
        model="gpt-oss:20b",  # Ollama 모델 사용 (설치된 모델에 맞게 변경)
        messages=messages,
        temperature=0.7,
        max_tokens=1000,  # Ollama는 max_completion_tokens 대신 max_tokens 사용
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
