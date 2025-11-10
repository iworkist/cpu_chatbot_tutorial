"""
가장 단순한 백프로파게이션 (주석 버전)

목표: 입력 x를 w배 해서 정답 y를 만들기
예: x=0.5, y=1.0 이면 w=2를 학습해야 함
"""

# ============================================
# 초기 설정
# ============================================
x = 0.5      # 입력값
y = 1.0      # 정답 (목표값)
w = 0.1      # 가중치 (랜덤하게 시작)
lr = 0.5     # 학습률 (learning rate) - 한 번에 얼마나 수정할지

print("=" * 60)
print("백프로파게이션 학습 과정")
print("=" * 60)
print(f"목표: 입력 {x}를 {y}로 만들기")
print(f"초기 가중치: {w}\n")

# ============================================
# 학습 시작
# ============================================
for epoch in range(20):
    # ----------------------------------------
    # 1단계: 순전파 (Forward Propagation)
    # ----------------------------------------
    # 현재 가중치로 예측값 계산
    pred = x * w
    
    # ----------------------------------------
    # 2단계: 오차 계산 (Loss Calculation)
    # ----------------------------------------
    # 정답과 예측의 차이
    error = y - pred
    
    # 손실 함수 (Loss Function) = 오차의 제곱
    loss = error ** 2
    
    # ----------------------------------------
    # 3단계: 역전파 (Backward Propagation)
    # ----------------------------------------
    # gradient = loss를 w로 미분한 값
    # loss = (y - x*w)²를 w로 미분하면:
    # d(loss)/dw = 2*(y - x*w)*(-x) = -2*error*x
    gradient = -2 * error * x
    
    # ----------------------------------------
    # 4단계: 가중치 업데이트 (경사하강법)
    # ----------------------------------------
    # gradient가 양수면 w를 줄이고
    # gradient가 음수면 w를 늘림
    w = w - lr * gradient
    
    # ----------------------------------------
    # 결과 출력
    # ----------------------------------------
    print(f"에포크 {epoch+1:2d}: "
          f"예측={pred:.3f}, "
          f"오차={error:.3f}, "
          f"손실={loss:.3f}, "
          f"gradient={gradient:.3f}, "
          f"가중치={w:.3f}")

print(f"\n최종 결과: 입력 {x} → 출력 {x*w:.3f} (목표: {y})")
print(f"학습된 가중치: {w:.3f}")


# ============================================
# 상세 설명: 한 스텝만 자세히
# ============================================
print("\n" + "=" * 60)
print("첫 번째 스텝 상세 분석")
print("=" * 60)

# 초기화
x = 0.5
y = 1.0
w = 0.1
lr = 0.5

print(f"\n[초기 상태]")
print(f"  입력(x) = {x}")
print(f"  정답(y) = {y}")
print(f"  가중치(w) = {w}")
print(f"  학습률(lr) = {lr}")

print(f"\n[1단계: 순전파]")
pred = x * w
print(f"  예측값 = x × w")
print(f"         = {x} × {w}")
print(f"         = {pred}")

print(f"\n[2단계: 오차 계산]")
error = y - pred
print(f"  오차 = y - 예측값")
print(f"       = {y} - {pred}")
print(f"       = {error}")

loss = error ** 2
print(f"  손실 = 오차²")
print(f"       = {error}²")
print(f"       = {loss}")

print(f"\n[3단계: 역전파 - gradient 계산]")
print(f"  loss = (y - x×w)²")
print(f"  d(loss)/dw = -2 × (y - x×w) × x")
print(f"             = -2 × error × x")
gradient = -2 * error * x
print(f"             = -2 × {error} × {x}")
print(f"             = {gradient}")

print(f"\n[4단계: 가중치 업데이트]")
print(f"  새 가중치 = 현재 가중치 - 학습률 × gradient")
new_w = w - lr * gradient
print(f"           = {w} - {lr} × {gradient}")
print(f"           = {w} - {lr * gradient}")
print(f"           = {new_w}")

print(f"\n[결과]")
print(f"  가중치가 {w} → {new_w}로 변경됨")
print(f"  gradient가 {gradient}(음수)라서 가중치가 증가함!")
print(f"  왜? 가중치를 늘려야 예측값이 커져서 정답에 가까워지니까!")


# ============================================
# 핵심 개념 정리
# ============================================
print("\n" + "=" * 60)
print("핵심 개념 정리")
print("=" * 60)

print("""
1. 순전파 (Forward)
   - 현재 가중치로 예측값 계산
   - pred = x × w

2. 오차 계산
   - 정답과 예측의 차이
   - error = y - pred
   - loss = error²

3. 역전파 (Backward)
   - loss를 w로 미분 → gradient
   - gradient = -2 × error × x
   - "어느 방향으로 가중치를 바꿔야 loss가 줄어드는가?"

4. 경사하강법
   - gradient 방향의 반대로 가중치 이동
   - w = w - lr × gradient
   - loss가 줄어드는 방향으로 한 걸음씩 이동

5. 반복
   - 이 과정을 여러 번 반복하면
   - 가중치가 최적값으로 수렴
   - 예측값이 정답에 가까워짐
""")