# -*- coding: utf-8 -*-
import sys
import csv
import random
import time
import pandas as pd
from tqdm import tqdm
from itertools import product

nicknames = [
    "스카이러너", "음악마술사", "신비의섬", "루나틱스톰", "플래시메모리", "에픽워리어", "드림캐처", "섀도우퀸", "코스믹플레이어", "미스터리메이져",
    "토네이도소울", "스타더스트드림", "플로라마스터", "아이언픽셀", "솔라플레어", "버블블리코", "라이트닝스트라이크", "페어리테일", "올드마법사", "네오네모",
    "블리치드윙", "픽셀퓨전", "네오스톰", "실버서퍼", "매직워치", "세레니티", "크로노스플레이어", "퀀텀퀸", "선셋라이더", "라스트메모리",
    "홀로그린", "레트로라이트", "마법의소녀", "프로스트파이어", "스피릿하트", "에어로메이저", "프로토니움플레이", "미라클메이커", "엔더마스터", "드래곤소울",
    "황금시티", "블랙옵스", "익스플로라", "실버스트림", "로맨틱헤일", "트와일라잇", "화이트매직", "에코스타", "글로우픽셀", "티탄플레이어",
    "스타더스트마스터", "퍼플파워", "아쿠아마린", "엔더나이트", "레전드워리어", "스노우퀸", "아우라메이저", "마법의미러", "레드스카이", "크리스탈프린세스",
    "블루불릿", "섀도우블레이드", "화이트위치", "크로마틱퀸", "플래시블레이즈", "플루토니움플레이어", "스카이폭스", "아크틱윈드", "플레임워치", "샤도우스카우터",
    "에메랄드아이", "오로라스파크", "메테오라이트", "크림슨블레이드", "세븐서피", "네온퀸", "블레이즈마법사", "크립톤나이트", "에어리얼마스터", "샌드스톰",
    "블루플레임", "실버아로우", "썬더스트라이더", "글로벌위치", "피닉스블레이드", "알파스파크", "레이저메이저", "프로피시", "스카이브레이즈", "미스티코블러",
    "아스트랄블리츠", "시그마플레이어", "레볼루션스카이", "에어리얼블레이즈", "플레임서피", "퓨처메이저", "네오스카이", "토네이도블레이드", "썬더퀸", "로열메이지"
]

def save_name(nicknames, csv_name, label):
# CSV 파일에 쓰기
    with open(csv_name, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 헤더 작성
        csvwriter.writerow(['summonerName', 'label'])

        # 닉네임 데이터 작성
        for nickname in nicknames:
            csvwriter.writerow([nickname, label])

    print('CSV 파일이 생성되었습니다. 파일명: ' + csv_name)

def get_csv(csv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile)

        # 헤더 읽기
        header = next(csvreader)
        summoner_name_index = header.index('summonerName')  # 'summonerName' 열의 인덱스 찾기

        # 리스트 초기화
        names_list = []

        # 소환사 이름 추가
        for row in csvreader:
            if len(row) > summoner_name_index:  # summoner_name_index 이상의 열이 있는 경우에만 추가
                names_list.append(row[summoner_name_index])

    return names_list

def add_f_words(list1, list2):
    new_list = []

    for item1 in list1:
        item2 = random.choice(list2)

        combined_forward = item1 + item2

        if 3 <= len(combined_forward) <= 17:
            new_list.append(combined_forward)

    return new_list

def add_f_words_all(list1, list2):
    new_list = []

    for item1 in list1:
        for item2 in list2:
            combined_forward = item1 + item2
            combined_reverse = item2 + item1

            if 3 <= len(combined_forward) < 17:
                new_list.append(combined_forward)

            if combined_forward != combined_reverse and 3 <= len(combined_reverse) <= 17:
                new_list.append(combined_reverse)

    return new_list

def f_words_only(input_list):
    new_list = []

    # 1차원 리스트 중 하나 또는 그 이상의 요소를 결합하여 새로운 요소 생성 및 추가
    for i in range(len(input_list)):
        for j in range(i + 1, len(input_list)):
            combined = input_list[i] + input_list[j]
            if 3 <= len(combined) <= 17:
                new_list.append(combined)

    return new_list

def main():
    names = get_csv('raw_normal_names.csv')

    raw_f_words = get_csv('raw_f_words.csv')

    f_names_1 = add_f_words(names, raw_f_words)
    save_name(f_names_1, "names_add_f.csv", 1)

    only_f_names = f_words_only(raw_f_words)
    save_name(only_f_names, "names_f_only.csv", 1)


main()