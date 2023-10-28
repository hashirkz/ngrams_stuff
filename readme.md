# about  
ml markov / ngram models for nlp and text generation  
ngram_model class is trained on text dataset and a json parameter dictionary is updated   
see e.x `./models/jp_songs.json`  
this is then used to predict/autocomplete small snippets of text called seed_texts using the **weights**  
here weights refers to the json parameters akin to neural net weights  


## usage *nix / wsl  
```bash
>> git clone "https://github.com/hashirkz/ngrams_stuff"
>> cd ngrams_stuff
>> python3 -m venv .venv
>> python3 -m pip install -r 'requirements.txt'
>> 
>> python3 -m   
```

### example generated text  
- .txt data                                         ->  `./text_data/jp_songs`  
- n **hyperparameter length of ngrams**             ->  7 
- k **length of extra text to be generated**        ->  256  
- seed text                                         -> "zuuto kokoro"  


the lovely output haha    
```
reading tiny_light.txt
reading tuyu_bus_to_the_other_world.txt
training using ./text_data/jp_songs
txt: Yasashi... n: 7
txt: Yasashi... n: 7
txt: ano yo ... n: 7
updating probabilites for ngrams

output:
zuuto kokoromoi o
Kimi ga wa aruno ato
Omokage pattoshi hajime wa arenu you moroi mo
Kimi saretai hito no nakaki dashi wo sa mama nochi wo yo iki wo sase mukourete
Nozashitara
Nijimi ga ito wa ka
Imi ou ni
Zu you ni no naki no ba ga ata daaku kara ha kono me ni ara
Kodok

```

#### glimpse into model parameters   
see jp_songs.json for full  
```json
"h": {
    "isa ni ": 0.021505376344086023,
    "ii to i": 0.021505376344086023,
    "ita tob": 0.021505376344086023,
    "iisana ": 0.053763440860215055,
    "ibi no ": 0.043010752688172046,
    "ime tet": 0.043010752688172046,
    "isou ni": 0.043010752688172046,
    "iranaka": 0.043010752688172046,
    "i teta ": 0.021505376344086023,
    "ibi ni\n": 0.021505376344086023,
    "i hajim": 0.021505376344086023,
    "ajime t": 0.021505376344086023,
    "ita\nMam": 0.021505376344086023,
    "odo ito": 0.021505376344086023,
    "ii kon'": 0.021505376344086023,
    "imau so": 0.021505376344086023,
    "ito wa ": 0.021505376344086023,
    "i wa ka": 0.010752688172043012,
    "i wo ho": 0.010752688172043012,
    "oumutte": 0.010752688172043012,
    "eizento": 0.010752688172043012,
    "inu no ": 0.010752688172043012,
    "i de yo": 0.010752688172043012,
    "ita maw": 0.010752688172043012,
    "i wo ub": 0.010752688172043012,
    "o\n\nseim": 0.010752688172043012,
    "iwa da ": 0.010752688172043012,
    "oumonai": 0.010752688172043012,
    "akisute": 0.010752688172043012,
    "imaitai": 0.053763440860215055,
    "ougai n": 0.053763440860215055,
    "uushint": 0.010752688172043012,
    "inteki ": 0.010752688172043012,
    "iranai ": 0.010752688172043012,
    "i wo sa": 0.03225806451612903,
    "ite kor": 0.03225806451612903,
    "ite uba": 0.03225806451612903,
    "anarete": 0.010752688172043012,
    "amukaeb": 0.010752688172043012,
    "ita hit": 0.010752688172043012,
    "itogomi": 0.010752688172043012,
    "atte\nso": 0.010752688172043012,
    "i wo sh": 0.010752688172043012,
    "imesu n": 0.010752688172043012,
    "inai sh": 0.010752688172043012,
    "ousouka": 0.010752688172043012,
    "aete\nme": 0.010752688172043012,
    "ikosu d": 0.010752688172043012,
    "i ni de": 0.010752688172043012,
    "o de sa": 0.010752688172043012,
    "i wa ku": 0.010752688172043012
  },
  "i": {
    "sa ni f": 0.005420054200542005,
    " furete": 0.005420054200542005,
    "enai ma": 0.005420054200542005,
    " mama\nI": 0.005420054200542005,
    "i to ie": 0.005420054200542005,
    " to iet": 0.005420054200542005,
    "etara\nK": 0.005420054200542005,
    "ta tobi": 0.005420054200542005,
    "ra no m": 0.005420054200542005,
    "koe ter": 0.005420054200542005,
    "dasu ko": 0.005420054200542005,
    "naikara": 0.005420054200542005,
    "kara\nKo": 0.005420054200542005,
    " yoriso": 0.005420054200542005,
    "sotteru": 0.005420054200542005,
    "kidzuit": 0.01084010840108401,
    "dzuita ": 0.01084010840108401,
    "ta mama": 0.01084010840108401,
    "isana t": 0.01084010840108401,
    "sana to": 0.01084010840108401,
    "bi no y": 0.01084010840108401,
    " no you": 0.01084010840108401,
    " o\nKaze": 0.01084010840108401,
    " uta re": 0.01084010840108401,
    " ame ni": 0.01084010840108401,
    " nurena": 0.01084010840108401,
    " you ni": 0.005420054200542005,
    "\nZutto ": 0.01084010840108401,
    "shime t": 0.01084010840108401,
    "me teta": 0.01084010840108401,
    " dashis": 0.01084010840108401,
    "sou ni ": 0.01084010840108401,
    " naru m": 0.01084010840108401,
    " jibun ": 0.01084010840108401,
    "bun mo\n": 0.01084010840108401,
    "mi ga i": 0.02168021680216802,
    " ga ina": 0.01084010840108401,
    "nakya S": 0.005420054200542005,
    "ranakat": 0.01084010840108401,
    " o mits": 0.01084010840108401,
    "tsuketa": 0.01084010840108401,
    "tomi to": 0.005420054200542005,
    " tojiru": 0.005420054200542005,
    "ru tabi": 0.005420054200542005,
    "\nKioku ": 0.005420054200542005,
    "oku no ": 0.005420054200542005,
    " tadayo": 0.005420054200542005,
    " yume n": 0.005420054200542005,
    " teta n": 0.005420054200542005,
    " monoku": 0.005420054200542005,
    "bi ni\nK": 0.005420054200542005,
    " ni\nKim": 0.005420054200542005,
    "\nKimi g": 0.005420054200542005,
    " ga iro": 0.005420054200542005,
    "ro o so": 0.005420054200542005,
    ...
  } ...

```
