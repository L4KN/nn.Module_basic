import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터 임시 설정
x_train = torch.FloatTensor([[1], [2], [3]]) # 3x1 2D Tensor , 입력 데이터
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델 선언 > 선형회귀 모델
model = nn.Linear(1,1) #입력 1차원, 출력 1차원

print(list(model.parameters())) #model.parameters() 선언시 W(가중치)와 b(편향)이 랜덤으로 설정됨.

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 최적화 알고리즘 사용

# 전체 훈련 데이터에 대해 경사하강법을 2,000번 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train) #H(x) = Wx + b를 구현

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분

    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item())
      )

# 예측검증

# 임의의 스칼라 4를 선언
new_var =  torch.FloatTensor([[4.0]]) 
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) # forward 연산 : H(x) = Wx + b 훈련된 것에 x를 대입하여 y를 구함.
# y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 


print("{0:-^30}".format("단순 선형 회귀 클래스 구현 단계"))
#클래스 구현 과정

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegressionModel(nn.Module): # 선형회귀 모델을 클래스로 구현
    def __init__(self): #
        super().__init__()
        self.linear = nn.Linear(1, 1) # 단순 선형 회귀이므로 input_dim=1, output_dim=1.

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
