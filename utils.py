import numpy as np
from stse import bytes


def str_vectorizer(word_list, rubric):
    out = []
    for word in word_list:
        if word in rubric:
            encoded_word = bytes.bit_vect(
                len(rubric) + 1,  # Add 1 or NONE option
                rubric.index(word)
            )
        else:
            # Set to NONE (hot at last index)
            encoded_word = bytes.bit_vect(len(rubric) + 1, len(rubric))
        out.append(encoded_word)
    
    return np.array(out)
    

def test_str_vectorizer():
    test_case = ['a', 'b', 'bg', 'g']
    test_rubric = ['bg', 'a']
    
    str_vectorizer(test_case, test_rubric)


if __name__ == '__main__':
    test_str_vectorizer()