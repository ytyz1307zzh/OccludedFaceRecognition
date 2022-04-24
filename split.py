

def save_file(file_name, x):
    with open(file_name, 'w') as f:
        for item in x:
            f.write("%s\n" % item)

f = open("image.txt")

dic = {}

for line in f:
    subject = line.split("/")[-2]
    
    if subject not in dic:
        dic[subject] = []
    
    dic[subject].append(line.strip())


train = []
val = []

for _, x in dic.items():
    if len(x) == 1:
        print("alert")
    l = int(len(x) * 0.9)
    train.extend(x[:l])
    val.extend(x[l:])
    if len(x) - l == 0 or l == 0:
        print("alerrt")


save_file("train.txt", train)
save_file("validate.txt", val)







