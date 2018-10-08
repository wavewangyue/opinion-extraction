from xml.dom.minidom import parse
import xml.dom.minidom
import json

def xml_to_txt(path1, path2, name):
    DOMTree = xml.dom.minidom.parse(path1)
    collection = DOMTree.documentElement
    sentences = collection.getElementsByTagName("sentence")
    opinion_lines = open(path2).read().split('\n')
    opinions = []
    for line in opinion_lines:
        ops = line.split(',')
        opinion_words = []
        for op in ops:
            if op == 'NIL':
                continue
            op = op.replace('+1','')
            op = op.replace('-1','')
            op = op.strip()
            opinion_words.append(op.lower())
        opinions.append(opinion_words)

    texts = []
    labels_a = []
    labels_p = []
    for x,sentence in enumerate(sentences):
        text = sentence.getElementsByTagName('text')[0]
        text = text.childNodes[0].data
        text = text.lower()
        words = text.split(' ')
        label_a = ['2']*len(words)
        label_p = ['2']*len(words)
        aspects = sentence.getElementsByTagName('aspectTerms')
        if len(aspects) > 0:
            aspects = aspects[0].getElementsByTagName('aspectTerm')
            for aspect in aspects:
                s = aspect.getAttribute("from")
                s = int(s)
                e = aspect.getAttribute("to")
                e = int(e)
                l = 0
                for i,word in enumerate(words):
                    if l <= s:
                        if l + len(word) > s:
                            label_a[i] = '0'
                    elif (l > s) and (l < e):
                        label_a[i] = '1'
                    l += len(word)
                    l += 1
        print x
        print text
        print opinions[x]
        if len(opinions[x]) > 0:
            for op_word in opinions[x]:
                s = text.index(op_word)
                e = s + len(op_word)
                l = 0
                for i,word in enumerate(words):
                    if l <= s:
                        if l + len(word) > s:
                            label_p[i] = '0'
                    elif (l > s) and (l < e):
                        label_p[i] = '1'
                    l += len(word)
                    l += 1
        texts.append(text)
        labels_a.append(' '.join(label_a))
        labels_p.append(' '.join(label_p))
        print label_p
    
    output = open(name+'_docs.txt','w')
    output.write('\n'.join(texts))
    output.close()
    output = open(name+'_labels_a.txt','w')
    output.write('\n'.join(labels_a))
    output.close()
    output = open(name+'_labels_p.txt','w')
    output.write('\n'.join(labels_p))
    output.close()
            
  
xml_to_txt('Restaurants_Train_v2.xml', 'train_restaurant', 'train')
xml_to_txt('Restaurants_Test_Data_phaseB.xml', 'test_restaurant', 'test')