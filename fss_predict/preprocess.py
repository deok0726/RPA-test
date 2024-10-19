import os
import json
import shutil
import random
import pandas as pd

def xlsx_to_json(xlsx):
    # 엑셀 파일 로드
    file_path = xlsx
    file_name = os.path.basename(file_path).split('.')[0]

    xls = pd.ExcelFile(file_path)
    whole = []

    for sheet_name in xls.sheet_names:
        # 시트 읽기
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        whole.append({sheet_name: df.to_dict(orient='records')})
        
    with open(f'{file_name}.json', 'w', encoding='utf-8') as f:
        json.dump(whole, f, ensure_ascii=False, indent=4)
        print(f"Saved all sheets to {file_name}.json")

def xlsx_to_txt(xlsx):
    # 엑셀 파일 읽기
    file_path = xlsx
    file_name = os.path.basename(file_path).split('.')[0]
    
    xls = pd.ExcelFile(file_path)
    
    with open(f'{file_name}.txt', 'w', encoding='utf-8') as f:
        # 모든 시트를 순회하며 텍스트 파일로 저장
        for sheet_name in xls.sheet_names:
            # 시트 읽기
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # 시트 이름과 구분자 쓰기
            f.write(f"Sheet: {sheet_name}\n")
            
            # DataFrame을 텍스트 파일로 저장
            df.to_csv(f, sep='\t', index=False)
            
            # 시트 사이에 구분자 추가
            f.write("\n")

def sheet_to_txt(xlsx):
    # 엑셀 파일 읽기
    file_path = xlsx
    xls = pd.ExcelFile(file_path)
    
    for sheet_name in xls.sheet_names:
        with open(f'/svc/project/genaipilot/fss_predict/temp/{sheet_name}.txt', 'w', encoding='utf-8') as f:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # DataFrame을 텍스트 파일로 저장
            df.to_csv(f, sep='\t', index=False)

def random_choice(number, folder):
    files = os.listdir(folder)
    samples = random.sample(files, number)
    print(samples)

    for sample in samples:
        shutil.copy(folder + sample, "/svc/project/genaipilot/fss_predict/temp/" + sample)


if __name__ == "__main__":
    for file in os.listdir("/svc/project/genaipilot/fss_predict/data/"):
        sheet_to_txt(f"/svc/project/genaipilot/fss_predict/data/{file}")

    # number = 38
    # random_choice(number, "/svc/project/genaipilot/fss_predict/false/")