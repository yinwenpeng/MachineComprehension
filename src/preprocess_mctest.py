from nltk.tokenize import TreebankWordTokenizer
import nltk.data
import collections
import numpy


path='/mounts/data/proj/wenpeng/Dataset/MCTest/'

def tokenize(str):
    listt=TreebankWordTokenizer().tokenize(str)
    if listt[-1]=='.' or listt[-1]=='?':
        listt=listt[:-1]
    return ' '.join(listt)

def tokenize_answer(str):
    return ' '.join(TreebankWordTokenizer().tokenize(str))

def text2sents(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    text=text.replace('\\newline',' ').replace('\\tab',' ')
    sents=tokenizer.tokenize(text)
    new_text=''
    for sent in sents:
        tokenized_sent=tokenize(sent)
#         if tokenized_sent.find('Jimmy found more and more insects to add to his jar')>=0:
#             print tokenized_sent
#             print sent
#             print tokenized_sent.find('\\')
#             print sent.find('noise')
#             print sent[0], sent[1], sent[2], sent[3]
#             exit(0)
#         if tokenized_sent.find('\\newline')>=0:
#             print tokenized_sent
#             print tokenized_sent.replace('\newline','')
#             exit(0)
#         refined_sent=[]
#         for word in tokenized_sent.split():
#             if word=='?':
#                 continue
#             posi=word.find('.')
#             if posi>=0:
#                 if word[posi+1:posi+2].isupper() or (posi==len(word)-1 and word[0:1].islower()):
#                     word.replace('.','\t')
#             refined_sent.append(word)
#         tokenized_sent=' '.join(refined_sent)                
                
        new_text+='\t'+tokenized_sent
    return new_text.strip()
    
def answer2sents(text):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    text=text.replace('\\newline',' ').replace('\\tab',' ')
    sents=tokenizer.tokenize(text)
    new_text=''
    for sent in sents:
        tokenized_sent=tokenize_answer(sent)
#         if tokenized_sent.find('Jimmy found more and more insects to add to his jar')>=0:
#             print tokenized_sent
#             print sent
#             print tokenized_sent.find('\\')
#             print sent.find('noise')
#             print sent[0], sent[1], sent[2], sent[3]
#             exit(0)
#         if tokenized_sent.find('\\newline')>=0:
#             print tokenized_sent
#             print tokenized_sent.replace('\newline','')
#             exit(0)
#         refined_sent=[]
#         for word in tokenized_sent.split():
#             if word=='?':
#                 continue
#             posi=word.find('.')
#             if posi>=0:
#                 if word[posi+1:posi+2].isupper() or (posi==len(word)-1 and word[0:1].islower()):
#                     word.replace('.','\t')
#             refined_sent.append(word)
#         tokenized_sent=' '.join(refined_sent)                
                
        new_text+=' '+tokenized_sent
    words=new_text.strip().split()
    if words[-1]=='.' or words[-1]=='?':
        words=words[:-1]    
    return ' '.join(words)
def standardlize(answerfile,inputfile):
    readfile=open(path+answerfile, 'r')
    answers=[]
    for line in readfile:
        answer=line.strip().split()
        int_answer=[]
        for ans in answer:
            if ans is 'A':
                int_answer.append(0)
            elif ans is 'B':
                int_answer.append(1)
            elif ans is 'C':
                int_answer.append(2)
            elif ans is 'D':
                int_answer.append(3)
        if len(int_answer)!=4:
            print 'len(int_answer)!=4'
            exit(0)
        answers.append(int_answer)
    readfile.close()
    readfile=open(path+inputfile, 'r')
    writefile=open(path+inputfile+'_standardlized.txt','w')
    line_no=0
    for line in readfile:
        parts=line.strip().split('\t')
        story=parts[2]
        QA1=parts[3:3+5]
        QA2=parts[8:8+5]
        QA3=parts[13:13+5]
        QA4=parts[18:18+5]
        corrent_answers=answers[line_no]
        for QA_ind, QA in enumerate([QA1, QA2, QA3, QA4]):
            colon=QA[0].index(':')
            label=QA[0][:colon]
            Q=QA[0][colon+1:].strip()
            label_int=1
            if label=='multiple':
                label_int=2
            for ans_ind, ans in enumerate(QA[1:]):
                if ans_ind==corrent_answers[QA_ind]:
                    writefile.write('1\t'+str(label_int)+'\t'+text2sents(story)+'\t'+answer2sents(Q)+'\t'+answer2sents(ans)+'\n')
                else:
                    writefile.write('0\t'+str(label_int)+'\t'+text2sents(story)+'\t'+answer2sents(Q)+'\t'+answer2sents(ans)+'\n')
        line_no+=1
    writefile.close()
    readfile.close()
    print 'reform over'
                
def length_sent_text():
    #max_sent_length 57 max_text_length 59
#     files=['mc500.train.tsv_standardlized.txt', 'mc500.dev.tsv_standardlized.txt','mc500.test.tsv_standardlized.txt','mc160.train.tsv_standardlized.txt', 'mc160.dev.tsv_standardlized.txt','mc160.test.tsv_standardlized.txt']                
    #max_sent_length 57 max_text_length 59
    files=['mc500.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt', 'mc500.dev.tsv_standardlized.txt_with_state.txt_DSSSS.txt','mc500.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt','mc160.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt', 'mc160.dev.tsv_standardlized.txt_with_state.txt_DSSSS.txt','mc160.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt']                

    max_sent_length=0
    max_text_length=0
    sent_l2count=collections.defaultdict(int)
    for file in files:
        readfile=open(path+file,'r')
        for line in readfile:
            parts=line.strip().split('\t')
            text_l=len(parts)-5
            if text_l>max_text_length:
                max_text_length=text_l
            for sent in parts[1:]:
                sent_l=len(sent.strip().split())
                sent_l2count[sent_l]+=1
                if sent_l>max_sent_length:
                    max_sent_length=sent_l
        readfile.close()
    print 'max_sent_length',max_sent_length, 'max_text_length', max_text_length
#     print sent_l2count
def Extract_Vocab():
#    files=['mc500.train.tsv_standardlized.txt', 'mc500.dev.tsv_standardlized.txt','mc500.test.tsv_standardlized.txt','mc160.train.tsv_standardlized.txt', 'mc160.dev.tsv_standardlized.txt','mc160.test.tsv_standardlized.txt'] 
    files=['mc500.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt', 'mc500.dev.tsv_standardlized.txt_with_state.txt_DSSSS.txt','mc500.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt','mc160.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt', 'mc160.dev.tsv_standardlized.txt_with_state.txt_DSSSS.txt','mc160.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt']                

    writeFile=open(path+'vocab_DSSSS.txt', 'w')
    vocab={}
    count=0
    for file in files:
        readFile=open(path+file, 'r')
        for line in readFile:
            tokens=line.strip().split('\t')
            sent_size=len(tokens)-1
            for i in range(sent_size):
                words=tokens[i+1].strip().split()
                for word in words:
                    key=vocab.get(word)
                    if key is None:
                        count+=1
                        vocab[word]=count
                        writeFile.write(str(count)+'\t'+word+'\n')
                        
        readFile.close()
    writeFile.close()
    print 'total words: ', count

def transcate_word2vec():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/word2vec_words_300d.txt', 'r')
    dim=300
    word2vec={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            word2vec[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'word2vec loaded over...'
    readFile=open(path+'vocab_DSSSS.txt', 'r')
    writeFile=open(path+'vocab_embs_300d_DSSSS.txt', 'w')
    random_emb=list(numpy.random.uniform(-0.01,0.01,dim))
    unk=0
    for line in readFile:
        tokens=line.strip().split()
        emb=word2vec.get(tokens[1])
        if emb is None:
            emb=word2vec.get(tokens[1].lower())
            if emb is None:
                emb=random_emb
                unk+=1
        writeFile.write(tokens[1]+'\t'+' '.join(map(str, emb))+'\n')
    writeFile.close()
    readFile.close()
    print 'word2vec trancate over, unk:', unk     

def transcate_glove():
    readFile=open('/mounts/data/proj/wenpeng/Dataset/glove.6B.50d.txt', 'r')
    dim=50
    glove={}
    for line in readFile:
        tokens=line.strip().split()
        if len(tokens)<dim+1:
            continue
        else:
            glove[tokens[0]]=map(float, tokens[1:])
    readFile.close()
    print 'glove loaded over...'
    readFile=open(path+'vocab_DSSSS.txt', 'r')
    writeFile=open(path+'vocab_glove_50d.txt', 'w')
    #random_emb=list(numpy.random.uniform(-0.01,0.01,dim))
    unk=0
    for line in readFile:
        tokens=line.strip().split()
        emb=glove.get(tokens[1])
        if emb is None:
            emb=glove.get(tokens[1].lower())
            if emb is None:
                emb=list(numpy.random.uniform(-0.01,0.01,dim))
                unk+=1
        writeFile.write(tokens[1]+'\t'+' '.join(map(str, emb))+'\n')
    writeFile.close()
    readFile.close()
    print 'glove trancate over, unk:', unk        

def change_DQA_into_DQAAAA():
    files=['mc500.train.tsv_standardlized.txt', 'mc500.dev.tsv_standardlized.txt', 'mc500.test.tsv_standardlized.txt',
           'mc160.train.tsv_standardlized.txt', 'mc160.dev.tsv_standardlized.txt', 'mc160.test.tsv_standardlized.txt']
    
    for filee in files:
        readfile=open(path+filee, 'r')
        writefile=open(path+filee+'_DQAAAA.txt', 'w')
        line_no=1
        batch=4
        posi=-1
        answers=[]
        for line in readfile:
            parts=line.strip().split('\t')
            y=parts[0]
            label=parts[1]
            D=parts[2:-2]
            Q=parts[-2]
            A=parts[-1]
            answers.append(A)
            if y=='1':
                posi=len(answers)-1
            if line_no%batch==1:
                writefile.write(label+'\t'+'\t'.join(D)+'\t'+Q+'\t')
            elif line_no%batch==0:#4 lines
                writefile.write(answers[posi])
                for index, answer in enumerate(answers):
                    if index!=posi:
                        writefile.write('\t'+answer)
                writefile.write('\n')
                del answers[:]
                posi=-1
            line_no+=1
        writefile.close()
        readfile.close()
    print 'over'


def change_DQAS_into_DSSSS():
    files=['mc500.train.tsv_standardlized.txt_with_state.txt', 'mc500.dev.tsv_standardlized.txt_with_state.txt', 'mc500.test.tsv_standardlized.txt_with_state.txt',
           'mc160.train.tsv_standardlized.txt_with_state.txt', 'mc160.dev.tsv_standardlized.txt_with_state.txt', 'mc160.test.tsv_standardlized.txt_with_state.txt']
    
    for filee in files:
        readfile=open(path+filee, 'r')
        writefile=open(path+filee+'_DSSSS.txt', 'w')
        line_no=1
        batch=4
        posi=-1
        answers=[]
        for line in readfile:
            parts=line.strip().split('\t')
            y=parts[0]
            label=parts[1]
            D=parts[2:-3]
#             Q=parts[-3]
            A=parts[-1]
            answers.append(A)
            if y=='1':
                posi=len(answers)-1
            if line_no%batch==1:
                writefile.write(label+'\t'+'\t'.join(D)+'\t')
            elif line_no%batch==0:#4 lines
                writefile.write(answers[posi])
                for index, answer in enumerate(answers):
                    if index!=posi:
                        writefile.write('\t'+answer)
                writefile.write('\n')
                del answers[:]
                posi=-1
            line_no+=1
        writefile.close()
        readfile.close()
    print 'over'      
def combine_standardlize_statement(standfile, statefile):
    readstand=open(path+standfile, 'r')
    readstate=open(path+statefile, 'r')
    lines_stand=[]
    for line in readstand:
        lines_stand.append(line.strip())
    readstand.close()
    

    states=[]
    for line in readstate:
        parts=line.strip().split('\t')
        qa_part=parts[-20:]
        states_part=qa_part[1:1+4]+qa_part[6:6+4]+qa_part[11:11+4]+qa_part[16:16+4]
        states+=states_part
    readstate.close()
    writefile=open(path+standfile+'_with_state.txt', 'w')    
    if len(lines_stand)!=len(states):
        print 'size not equal'
        exit(0)
    else:
        for i in range(len(lines_stand)):
            writefile.write(lines_stand[i]+'\t'+answer2sents(states[i])+'\n')
    writefile.close()
    print 'finished'
                    
def change_DSSSS_to_DPN(inputfile):
    readfile=open(path+inputfile, 'r')
    writefile=open(path+inputfile+'_DPN.txt', 'w')
    for line in readfile:
        parts=line.strip().split('\t')
        head=parts[:-3]
        tail=parts[-3:]
        for i in range(3):
            writefile.write('\t'.join(head)+'\t'+tail[i]+'\n')
    writefile.close()
    readfile.close()
    print 'over'
                  
if __name__ == '__main__':
#     standardlize('mc500.train.ans', 'mc500.train.tsv')
#     standardlize('mc500.dev.ans', 'mc500.dev.tsv')
#     standardlize('mc500.test.ans', 'mc500.test.tsv')
#     standardlize('mc160.train.ans', 'mc160.train.tsv')
#     standardlize('mc160.dev.ans', 'mc160.dev.tsv')
#     standardlize('mc160.test.ans', 'mc160.test.tsv')
#     length_sent_text()
#     Extract_Vocab()
#     transcate_word2vec()
#     transcate_glove()
#     change_DQA_into_DQAAAA()

#     combine_standardlize_statement('mc500.train.tsv_standardlized.txt', 'Statements/mc500.train.statements.tsv')
#     combine_standardlize_statement('mc500.dev.tsv_standardlized.txt', 'Statements/mc500.dev.statements.tsv')
#     combine_standardlize_statement('mc500.test.tsv_standardlized.txt', 'Statements/mc500.test.statements.tsv')    
#     combine_standardlize_statement('mc160.train.tsv_standardlized.txt', 'Statements/mc160.train.statements.tsv')
#     combine_standardlize_statement('mc160.dev.tsv_standardlized.txt', 'Statements/mc160.dev.statements.tsv')
#     combine_standardlize_statement('mc160.test.tsv_standardlized.txt', 'Statements/mc160.test.statements.tsv')
#     change_DQAS_into_DSSSS()
    change_DSSSS_to_DPN('mc500.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt')
    change_DSSSS_to_DPN('mc500.dev.tsv_standardlized.txt_with_state.txt_DSSSS.txt')
    change_DSSSS_to_DPN('mc500.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt')
    change_DSSSS_to_DPN('mc160.train.tsv_standardlized.txt_with_state.txt_DSSSS.txt')
    change_DSSSS_to_DPN('mc160.dev.tsv_standardlized.txt_with_state.txt_DSSSS.txt')
    change_DSSSS_to_DPN('mc160.test.tsv_standardlized.txt_with_state.txt_DSSSS.txt')
       
        