# Function to read and parse a text file into a dictionary
def read_text_file(filename):
    labels = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) == 2:
                labels[parts[0]] = parts[1]
    return labels

# Define the filenames for your text files
file1 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/vgg_seq2seq_dataraw_500k_iter_cer.txt'
file2 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/resnet50_seq2seq_dataraw_300k_iter_cer.txt'
file3 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/resnet50_seq2seq_dataraw_400k_iter_cer.txt'
file4 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/vgg_seq2seq_custom.txt'
file5 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/resnet50_seq2seq_custom.txt'
file6 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/vgg_transfomrer_dataraw_300k_iter.txt'
file7 = '/work/hpc/firedogs/nnt/BK-Challenge/vietocr/predictions/resnet50_transformer_dataraw_300k_iter_cer.txt'

files = [file1, file2, file3, file4, file5, file6, file7]

def cmp(label1, label2):
    count = 0
    labels1 = list(read_text_file(label1).values())
    labels2 = list(read_text_file(label2).values())
    N = len(labels1)
    for i in range(N):
        if (labels1[i] != labels2[i]):
            count = count+1
    # print(count/N)
    return count/N

import numpy as np

res = np.zeros((8,8))
for i in range(8):
    for j in range(8):
        res[i][j] = cmp(files[i], files[j])

with open('./VietOCR/compare.txt', 'w') as file:
    file.truncate(0)
    file.write(str(res))

print(res)