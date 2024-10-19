import math
import heapq
import copy
from pydub import AudioSegment

# 시간 문자열을 초로 변환하는 함수
def time_to_seconds_pyannote(time_str):
    h, m, s = map(float, time_str.split(':'))
    return (h * 3600 + m * 60 + s)
def time_to_seconds_whisper(time_str):
    m,s = map(float, time_str.split(':'))
    return (m * 60 + s)
def parse_time_data(time_str):
    time_str = time_str.strip("[]").split(" --> ")
    start_time = time_str[0]
    end_time = time_str[1]
    return start_time, end_time

def pyannote_diarization(file):
    ListA = [] #whisper
    for i in whisper_intervals:
        start, end = parse_time_data(i)
        ListA.append([time_to_seconds_whisper(start), time_to_seconds_whisper(end)])
    ListB = []
    for j in speaker_00_intervals:
        start, end = parse_time_data(j)
        ListB.append([time_to_seconds_pyannote(start), time_to_seconds_pyannote(end)])
    ListC = []
    for k in speaker_01_intervals:
        start, end = parse_time_data(k)
        ListC.append([time_to_seconds_pyannote(start), time_to_seconds_pyannote(end)])

    #ListB와 ListC를 우선순위큐로
    heapq.heapify(ListB)
    heapq.heapify(ListC)

    #영역을 저장할 area_B, area_C init
    area_B = []
    area_C = []

    area_B=[[0,0] for _ in range(math.ceil(ListA[-1][1])+1)]
    area_C=[[0,0] for _ in range(math.ceil(ListA[-1][1])+1)]

    #result = 저장해둘 곳 ListA와 크기 같아야 함
    result = ['A or B' for _ in range(len(ListA))]
    index_B=0

    while ListB :
        start, end = heapq.heappop(ListB)
        area_B[math.floor(start)+1:min(math.floor(end), math.floor(ListA[-1][0]))] = [[0.5,0.5] for _ in range(min(math.floor(end), math.floor(ListA[-1][0]))-math.floor(start)-1)]
        if (start-int(start))>0.5:
            area_B[math.floor(start)][1] = max(0.5-(start-int(start)), area_B[math.floor(start)][1])
        else:
            area_B[math.floor(start)][1] = 0.5
            area_B[math.floor(start)][0] = max((start-int(start)), area_B[math.floor(start)][0])
        if (end-int(end)) < 0.5:
            area_B[math.floor(end)][0] = max((end-int(end)), area_B[math.floor(end)][1])
        else:
            area_B[math.floor(end)][0] = 0.5
            area_B[math.floor(end)][1] = max((end-int(end))-0.5, area_B[math.floor(end)][1])


    while ListC :
        start, end = heapq.heappop(ListC)
        area_C[math.floor(start)+1:min(math.floor(end), math.floor(ListA[-1][0]))] = [[0.5,0.5] for _ in range(min(math.floor(end), math.floor(ListA[-1][0]))-math.floor(start)-1)]
        if (start-int(start))>0.5:
            area_C[math.floor(start)][1] = max(0.5-(start-int(start)), area_C[math.floor(start)][1])
        else:
            area_C[math.floor(start)][1] = 0.5
            area_C[math.floor(start)][0] = max((start-int(start)), area_C[math.floor(start)][0])
        if (end-int(end)) < 0.5:
            area_C[math.floor(end)][0] = max((end-int(end)), area_C[math.floor(end)][1])
        else:
            area_C[math.floor(end)][0] = 0.5
            area_C[math.floor(end)][1] = max((end-int(end))-0.5, area_C[math.floor(end)][1])

    for j in range(len(ListA)):
        start=ListA[j][0]; end=ListA[j][1]
        # 밀리세컨드까지의 확률이 B가 더 높을 경우
        temp_B=0; temp_C=0;
        for i in range(math.floor(start),math.ceil(end)):
            temp_B += sum(area_B[i])
            temp_C += sum(area_C[i])
        print(temp_B, temp_C)
        if temp_C*0.9 < temp_B : result[j]='A'
        else: result[j]='B'

    # 한 번 더 가공
    for i in range(1,len(ListA)-1):
        if ListA[i][1]-ListA[i][0]<=1.0 :
            standard = result[i]
            if result[i-1]==result[i] and result[i+1]==result[i]:
            if result[i]=='A': result[i]='B'
            else: result[i]='A'

    result_data = copy.deepcopy(text_data)
    temp=0
    for i in range(len(result_data)):
        result_data[i]=list(result_data[i])
        if result[i]==temp: print(result_data[i][1], end = ' ')
        else: print('\n\n',result[i], result_data[i][1], end = '')
        temp = result[i]

if __name__ == "__main__":
    pyannote_diarization("shopping/쇼핑_3863.m4a")