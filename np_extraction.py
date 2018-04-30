# coding=UTF-8
import nltk
from nltk.corpus import brown
#nltk.download('punkt')

# Modify from Shlomi Babluki code:

# This is a fast and simple noun phrase extractor (based on NLTK)
# Feel free to use it, just keep a link back to this post
# http://thetokenizer.com/2013/05/09/efficient-way-to-extract-the-main-topics-of-a-sentence/
# Create by Shlomi Babluki
# May, 2013


# This is our fast Part of Speech tagger
#############################################################################
brown_train = brown.tagged_sents(categories='reviews')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)
#############################################################################


# This is our semi-CFG; Extend it according to your own needs
#############################################################################
cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"
#############################################################################


class NPExtractor(object):

    def __init__(self, sentence):
        self.sentence = sentence

    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged

    # Extract the main topics from the sentence
    def extract(self):

        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))

        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break

        matches = []
        for t in tags:
            if t[1] == "NNP" or t[1] == "NN" or t[1] == "NNI":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
        return matches

# Use all word extracted from the sentence to implement word2vec most_similar model
# Now only work on single word, phrases will encouter some errors
# Add by myself
# April 2018
import gensim
from gensim.models import Word2Vec
model1 = gensim.models.Word2Vec.load("/Users/sabrinalulu/Documents/Python/text8.model")
#所有單字(only)，build_vocab = gensim.models.Word2Vec.load("/Users/sabrinalulu/Documents/Python/text8.model").wv.vocab
print(model1)

# Main method, just run "python np_extractor.py"
def main():
    
    sentence = "My kid has a fever."
    #kid, fever
    #sentence1 = "My son is at high degrees. He slept unwell last night. What pills can he take to feel comfortable?"
    #son, high degrees, slept unwell, night
    np_extractor = NPExtractor(sentence)
    result = np_extractor.extract()
    print "This sentence is about: %s" % ", ".join(result)
    #match陣列放到result了，joiner永遠都是接字串，str(v)是印出字串，片語暫時無法，考慮用doc2vec或phrase2vec或乾脆只偵測單詞
    values=""
    for i in range(len(result)):
      page = model1.wv.most_similar(result[i])
    #用item[0]抓取sublist的第一位
      lst = ",".join(item[0] for item in page)
      values += "".join(str(v) for v in page)+"\n\n"
    f=open('/Users/sabrinalulu/Documents/Python/test.txt','w')
    f.write(values)

if __name__ == '__main__':
    main()
