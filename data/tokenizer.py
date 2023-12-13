import numpy as np

# NSMC 데이터 로드
file_path = 'preprocessed.npz'
data = np.load(file_path)

#  문장 및 라벨 데이터 추출
train_data  = data['names']
train_label = data['labels']
print('data loading done!')
print('문장: %s' %(train_data[:3]))
print('라벨: %s' %(train_label[:3]))

# subword 학습을 위해 문장만 따로 저장
with open('train_tokenizer.txt', 'w', encoding='utf-8') as f:
    for line in train_data:
        f.write(line+'\n')

# subword 학습을 위해 문장만 따로 저장
with open('train_tokenizer.txt', 'r', encoding='utf-8') as f:
    test_tokenizer = f.read().split('\n')
print(test_tokenizer[:3])

num_word_list = [len(sentence.split()) for sentence in test_tokenizer]
print('\n코퍼스 평균/총 단어 갯수 : %.1f / %d' % (sum(num_word_list)/len(num_word_list), sum(num_word_list)))