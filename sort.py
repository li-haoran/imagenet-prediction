

with open('synset.txt', 'r') as f:
    synsets = [l.rstrip() for l in f]

b=dict()
for text in synsets:
    content=text.split(' ')
    vl=''
    for ve in range(1,len(content)):
            vl+=content[ve]+' '
    b[content[0]]=vl

with open('train.lst', 'r') as f:
    train = [l.rstrip() for l in f]
a=dict()
for text in train:
    content = text.split('\t')
    a[content[1]]=content[2].split('/')[0]
print len(a),len(b)

new_dict=dict()
for key,value in a.iteritems():
    new_dict[key]=value+'\t'+b[value]

f=open('nsynset.txt','w')
for i in range(1000):
    f.write(str(i)+'\t'+new_dict[str(i)]+'\n')
f.close()

