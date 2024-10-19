import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train(data):
    # 데이터 준비
    data = {
        '특징': [
            '명확한 불만 사항 제기', '반복적인 확인 요구', '다른 카드사와의 비교', 
            '약관 및 정책에 대한 불신', '민원 접수 의사', '상담원의 대응에 대한 불만', 
            '문서 및 증빙 요구', '감정적인 표현'
        ],
        '통화_A': [5, 5, 5, 5, 5, 4, 5, 4],
        '통화_B': [4, 4, 3, 4, 5, 3, 4, 3],
        '통화_C': [4, 4, 3, 4, 5, 4, 5, 4],
        '통화_D': [2, 3, 1, 2, 1, 2, 3, 2],
        '통화_E': [2, 3, 1, 2, 1, 2, 2, 2],
        '통화_F': [2, 2, 1, 2, 1, 2, 1, 2]
    }

    df = pd.DataFrame(data)
    print(df)
    print("-"*50)
    
    # 데이터를 모델에 맞게 준비
    X = df[['통화_A', '통화_B', '통화_C', '통화_D', '통화_E', '통화_F']].values.T  # 특징 행렬
    y = [1, 1, 1, 0, 0, 0]  # 금융감독원 신고 가능성을 나타내는 타겟 변수 (1: 신고 가능, 0: 신고 불가능)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 학습 데이터와 테스트 데이터 분리
    # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(X_train)
    print("-"*50)
    print(X_test)
    print("-"*50)
    print(y_train)
    print("-"*50)
    print(y_test)

    # Logistic Regression 모델 생성 및 학습
    model = LogisticRegression()
    model.fit(X_train, y_train)

    model_filename = 'logistic_regression_model.joblib'
    joblib.dump(model, model_filename)

    # 모델 예측
    y_pred = model.predict(X_test)

    # 모델 평가
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return model_filename, scaler

def inference(data, scaler):
    # 새로운 데이터 예측
    new_data = [[3, 4, 1, 2, 1, 2, 4, 2]]  # 예시 데이터
    new_data_scaled = scaler.transform(new_data)
    loaded_model = joblib.load(model_filename)
    prediction = loaded_model.predict(new_data_scaled)

    print(f"신고 가능성 예측: {prediction[0]}")

if __name__ == "__main__":
    data = "temp"
    model_filename, scaler = train(data)
    inference(data, scaler)