# print(word_lists[0])
# data['annotation'].iloc[0]

find_sublist(['dad', 'with', 'recent', 'heart', 'attcak\r\nAll:'], list(word_lists[0]))
print(word_lists[0][111])

def find_sublist(sl, l):
    print(sl)
    print(l)
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:(ind + sll)] == sl:
            results.append((ind, ind + sll - 1))
    return results




data['annotation'].iloc[0]
word_lists[0]
for word in word_lists[0]:
    if word in data['annotation'].iloc[0]:
        print(word)
        
print(all(data['annotation'].iloc[0] in word_lists[0])