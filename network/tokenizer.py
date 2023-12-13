import os
from tokenizers.implementations import BertWordPieceTokenizer, SentencePieceBPETokenizer, CharBPETokenizer, ByteLevelBPETokenizer

def do_mecab():
    # load korean corpus for tokenizer training
    with open('../data/train_tokenizer.txt', 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
    print(data[:3])

    # mecab for window는 아래 코드 사용
    from konlpy.tag import Mecab  # install mecab for window: https://hong-yp-ml-records.tistory.com/91

    mecab_tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic").morphs
    print('mecab check :', mecab_tokenizer('어릴때보고 지금다시봐도 재밌어요ㅋㅋ'))

    for_generation = False  # or normal

    if for_generation:
        # 1: '어릴때' -> '어릴, ##때' for generation model
        total_morph = []
        for sentence in data:
            # 문장단위 mecab 적용
            morph_sentence = []
            count = 0
            for token_mecab in mecab_tokenizer(sentence):
                token_mecab_save = token_mecab
                if count > 0:
                    token_mecab_save = "##" + token_mecab_save  # 앞에 ##를 부친다
                    morph_sentence.append(token_mecab_save)
                else:
                    morph_sentence.append(token_mecab_save)
                    count += 1
            # 문장단위 저장
            total_morph.append(morph_sentence)

    else:
        # 2: '어릴때' -> '어릴, 때'   for normal case
        total_morph = []
        for sentence in data:
            # 문장단위 mecab 적용
            morph_sentence = mecab_tokenizer(sentence)
            # 문장단위 저장
            total_morph.append(morph_sentence)

    print(total_morph[:3])
    print(len(total_morph))

    # mecab 적용한 데이터 저장
    # ex) 1 line: '어릴 때 보 고 지금 다시 봐도 재밌 어요 ㅋㅋ'
    with open('../data/after_mecab.txt', 'w', encoding='utf-8') as f:
        for line in total_morph:
            f.write(' '.join(line) + '\n')

if __name__ == '__main__':
    do_mecab()

    # 4가지중 tokenizer 선택
    how_to_tokenize = BertWordPieceTokenizer  # The famous Bert tokenizer, using WordPiece
    # how_to_tokenize = SentencePieceBPETokenizer  # A BPE implementation compatible with the one used by SentencePiece
    # how_to_tokenize = CharBPETokenizer  # The original BPE
    # how_to_tokenize = ByteLevelBPETokenizer  # The byte level version of the BPE

    # Initialize a tokenizer
    if str(how_to_tokenize) == str(BertWordPieceTokenizer):
        print('BertWordPieceTokenizer')
        ## 주의!! 한국어는 strip_accents를 False로 해줘야 한다
        # 만약 True일 시 나는 -> 'ㄴ','ㅏ','ㄴ','ㅡ','ㄴ' 로 쪼개져서 처리된다
        # 학습시 False했으므로 load할 때도 False를 꼭 확인해야 한다
        tokenizer = BertWordPieceTokenizer(strip_accents=False,  # Must be False if cased model
                                           lowercase=False)
    elif str(how_to_tokenize) == str(SentencePieceBPETokenizer):
        print('SentencePieceBPETokenizer')
        tokenizer = SentencePieceBPETokenizer()

    elif str(how_to_tokenize) == str(CharBPETokenizer):
        print('CharBPETokenizer')
        tokenizer = CharBPETokenizer()

    elif str(how_to_tokenize) == str(ByteLevelBPETokenizer):
        print('ByteLevelBPETokenizer')
        tokenizer = ByteLevelBPETokenizer()

    else:
        assert ('select right tokenizer')

    corpus_file = ['../data/after_mecab.txt']  # data path
    vocab_size = 32000
    limit_alphabet = 6000
    output_path = 'hugging_%d' % (vocab_size)
    min_frequency = 5

    # Then train it!
    tokenizer.train(files=corpus_file,
                    vocab_size=vocab_size,
                    min_frequency=min_frequency,  # 단어의 최소 발생 빈도, 5
                    limit_alphabet=limit_alphabet,  # ByteLevelBPETokenizer 학습시엔 주석처리 필요
                    show_progress=True)
    print('train complete')

    sentence = '나는 오늘 아침밥을 먹었다.'
    output = tokenizer.encode(sentence)
    print(sentence)
    print('=>idx   : %s' % output.ids)
    print('=>tokens: %s' % output.tokens)
    print('=>offset: %s' % output.offsets)
    print('=>decode: %s\n' % tokenizer.decode(output.ids))

    sentence = 'I want to go my hometown'
    output = tokenizer.encode(sentence)
    print(sentence)
    print('=>idx   : %s' % output.ids)
    print('=>tokens: %s' % output.tokens)
    print('=>offset: %s' % output.offsets)
    print('=>decode: %s\n' % tokenizer.decode(output.ids))

    # save tokenizer
    hf_model_path = 'tokenizer_model'
    if not os.path.isdir(hf_model_path):
        os.mkdir(hf_model_path)
    tokenizer.save_model(hf_model_path)  # vocab.txt 파일 한개가 만들어진다

    from transformers import BertTokenizerFast

    tokenizer_for_load = BertTokenizerFast.from_pretrained(hf_model_path,
                                                           strip_accents=False,  # Must be False if cased model
                                                           lowercase=False)  # 로드

    print('vocab size : %d' % tokenizer_for_load.vocab_size)
    # tokenized_input_for_pytorch = tokenizer_for_load("i am very hungry", return_tensors="pt")
    tokenized_input_for_pytorch = tokenizer_for_load("나는 오늘 아침밥을 먹었다.", return_tensors="pt")

    print("Tokens (str)      : {}".format(
        [tokenizer_for_load.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
    print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
    print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))

    ## 1) define special tokens
    user_defined_symbols = ['[BOS]', '[EOS]', '[UNK0]', '[UNK1]', '[UNK2]', '[UNK3]', '[UNK4]', '[UNK5]', '[UNK6]',
                            '[UNK7]', '[UNK8]', '[UNK9]']
    unused_token_num = 200
    unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
    user_defined_symbols = user_defined_symbols + unused_list

    print(user_defined_symbols)

    # tokenizer에 special token 추가
    special_tokens_dict = {'additional_special_tokens': user_defined_symbols}
    tokenizer_for_load.add_special_tokens(special_tokens_dict)

    # check tokenizer vocab with special tokens
    print('check special tokens : %s' % tokenizer_for_load.all_special_tokens[:20])

    # save tokenizer model with special tokens
    tokenizer_for_load.save_pretrained(hf_model_path + '_special')

    # check special tokens
    from transformers import BertTokenizerFast

    tokenizer_check = BertTokenizerFast.from_pretrained(hf_model_path + '_special')

    print('check special tokens : %s' % tokenizer_check.all_special_tokens[:20])

    print('vocab size : %d' % tokenizer_check.vocab_size)
    tokenized_input_for_pytorch = tokenizer_check("나는 오늘 아침밥을 먹었다.", return_tensors="pt")

    print("Tokens (str)      : {}".format(
        [tokenizer_check.convert_ids_to_tokens(s) for s in tokenized_input_for_pytorch['input_ids'].tolist()[0]]))
    print("Tokens (int)      : {}".format(tokenized_input_for_pytorch['input_ids'].tolist()[0]))
    print("Tokens (attn_mask): {}\n".format(tokenized_input_for_pytorch['attention_mask'].tolist()[0]))

    # test to tf&pytorch bert model
    from transformers import TFBertModel, BertModel

    # load a BERT model for TensorFlow and PyTorch
    model_pt = BertModel.from_pretrained('bert-base-cased')

    ## tf vs torch bert output
    # transformers generates a ready to use dictionary with all the required parameters for the specific framework.
    input_pt = tokenizer_check("나는 오늘 아침밥을 먹었다.", return_tensors="pt")

    # Let's compare the outputs
    output_pt = model_pt(**input_pt)

    print('final layer output shape : %s' % (output_pt['last_hidden_state'].shape,))
    print(output_pt)