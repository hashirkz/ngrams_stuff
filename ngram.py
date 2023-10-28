import string
import math as m
import numpy as np
import json
import os
import glob

'''
# returns a freq dictionary k: ch, v: freq dictionary of ngrams order n

the v is a dictionary ngram: freq wich contains all possible n length substrings which follow the ch
and their frequency *if probability is true returns normalized probability from 0-1 isntead of frequency

e.x 
text = ' hello there \n ok'
n = 2
probability=False
* truncated pretty printed result
{
    " ": {
        "he": 1,
        "th": 1,
        "\n ": 1,
        "ok": 1
    },
    "h": {
        "el": 1,
        "er": 1
    },
    "e": {
        "ll": 1,
        "re": 1,
        " \n": 1
    }...
}
'''
class ngram_model:

    def __init__(self):
        self._files = []
        self.data = None
        self.ngrams = {}
        self.num_parms = 0

    # helper function for train
    # n is ngram length
    def ngram_dict(self, text: str, n: int=7) -> dict:

        print(f'txt: {text[:n]}... n: {n}')
        for i, ch in enumerate(text[:-n]):
            
            # never seen ch so make a new dictionary for its ngram freqs
            if ch not in self.ngrams:
                self.ngrams[ch] = {}
            
            # update ngram freqs to include ngram from text
            if ch in self.ngrams and text[i+1:i+n+1] not in self.ngrams[ch]:
                self.ngrams[ch][text[i+1:i+n+1]] = 1
            
            # self.ngrams already contains self.ngrams[ch][text[i+1:i+n+1]] so ++ 
            else:
                self.ngrams[ch][text[i+1:i+n+1]] += 1

            # update number of parameters
            self.num_parms += 1
        
        return self.ngrams

    # helper function for train
    def ngram_prob(self):
        for ch, ngrams_freq in self.ngrams.items():
            summ = sum(ngrams_freq.values())
            for ngram in ngrams_freq:
                self.ngrams[ch][ngram] /= summ

    # n is a hyperparameter -> lengthof 1 ngram parameter
    def train(self, _dir: str, n: int=7, recursive: bool=True):
        # read data into self.data
        self.data = self.read_data(_dir, recursive=recursive)
        
        print(f'training using {_dir}')
        # run ngram_dict on every .txt file in data and update self.ngram_dict
        np.vectorize(lambda txt : self.ngram_dict(txt, n=n))(self.data)

        print(f'updating probabilites for ngrams')
        # chaning ngram_dict from freq to probability at the end
        self.ngram_prob()

    # markov text chain generator
    # returns text of length k using the learned probabilities from the ngrams_dict 
    def generate(self, seed_text: str, k: int=100) -> str:

        if not seed_text: 
            print(f'the seed_text was empty')
            return

        seed_i = len(seed_text) - 1
        i = 0
        while i < k:
            ch = seed_text[seed_i+i]

            if ch not in self.ngrams:
                print(f'bad sample seed_text missings keys in last n characters')
                return

            chs = np.random.choice(np.array(list(self.ngrams[ch].keys())), 1, p=np.array(list(self.ngrams[ch].values())))[0]

            # updates
            seed_text += chs
            i += len(chs)

        return seed_text

    def read_data(self, _dir: str, recursive: bool=True) -> np.ndarray:
        data = []
        for name in glob.glob(f'{_dir}/*.txt', recursive=recursive):
            with open(name, 'r') as file:
                _repr = file.read()
                data.append(_repr)

            self._files.append(os.path.basename(name))
            print(f'reading {os.path.basename(name)}')
        
        self.data = np.array(data)

        return self.data

    def save_json(self, name: str) -> None:
        print(f'saving ngrams as json to ./{name}')
        with open(name, 'w') as file:
            json.dump(self.ngrams, file)

        
    def __str__(self):
        _repr = "FILES:\n"
        for fname in self._files:
            _repr += fname + '\n'

        _repr += f'\nNUM_PARAMETERS: {self.num_parms}\n\nNGRAMS: {self.ngrams}\n\n'

        return _repr
        
if __name__ == '__main__':
    # text = read_txt('./texts/tiny_light.txt')
    # # print(text)


    # ngrams = ngram_dict(text, n=10)
    # song = markov_text("kokoro", ngrams, 200)
    # print(song)

    model = ngram_model()
    model.train("./text_data/jp_songs")
    # model.save_json('./models/jp_songs.json')
    # print(model)
    print(model.generate('zuuto kokoro', 256))