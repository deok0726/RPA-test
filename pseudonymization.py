import os
import time
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate


def pseudonymizer(file):
    current_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(current_path, f"data/{file}")

    if os.path.isfile(file_path):
        print('File exists')

    llm = Ollama(model="llama3:70b")

    string_prompt = (
        PromptTemplate.from_template("""입력된 {text} 데이터 보호를 위해 데이터 가명화 도우미 역할을 수행합니다. 
                                    {text}는 민감한 개인정보 데이터를 포함한 텍스트이며, 해당 데이터를 가명화된 형태로 변환해 주어야 합니다. 
                                    가명화된 데이터는 원본 데이터와 유사한 형태를 유지해야 하지만, 실제 민감한 정보는 포함하지 않아야 합니다. 
                                    예를 들어, 이름, 주소, 생년월일, 전화번호, 카드번호 등의 민감한 정보를 가명화해 주세요. 
                                    첫 번째 텍스트는 "홍길동은 서울특별시 강남구 테헤란로 123에 거주하며, 그의 전화번호는 010-1234-5678입니다."입니다.""")
    )

    with open(file_path, 'r') as f:
        input = f.read()
        # print(input)

        string_prompt_value = string_prompt.format_prompt(text=input)
        result = llm.invoke(string_prompt_value)

        print("-"*30)
        print(result)
    
    return result

if __name__ == "__main__":
    result = pseudonymizer("pseudo.txt")
    time_stamp = int(time.time())

    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), f"output/pseudonymized_{time_stamp}.txt"), 'w', encoding='utf-8') as f:
        f.write(result)
