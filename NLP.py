#!/usr/bin/env python
# coding: utf-8

# In[107]:


import os 
import nltk
nltk.download()


# In[108]:


# you can create your own words 
nlp='''Natural language processing (NLP) refers to the branch of 
computer science—and more specifically, the branch of artificial 
intelligence or AI—concerned with giving computers the ability to 
understand text and spoken words in much the same way human beings can.

NLP combines computational linguistics—rule-based modeling of human 
language—with statistical, machine learning, and deep learning models. 
Together, these technologies enable computers to process human language 
in the form of text or voice data and to ‘understand’ its full meaning, 
complete with the speaker or writer’s intent and sentiment.'''


# In[109]:


nlp


# In[110]:


type(nlp)


# # word_tokenize

# In[111]:


from nltk.tokenize import word_tokenize 


# In[112]:


nlp_tokens=word_tokenize(nlp)


# In[113]:


nlp_tokens


# In[114]:


len(nlp_tokens)


# # sent_tokenize

# In[115]:


from nltk.tokenize import sent_tokenize


# In[116]:


nlp_sen=sent_tokenize(nlp)


# In[117]:


nlp_sen


# In[118]:


len(nlp_sen)


# # blankline_tokenize

# In[119]:


from nltk.tokenize import blankline_tokenize


# In[120]:


nlp_blank=blankline_tokenize(nlp)


# In[121]:


nlp_blank


# In[122]:


len(nlp_blank)


# # WhitespaceTokenizer

# In[123]:


from nltk.tokenize import WhitespaceTokenizer


# In[124]:


nlp_ws2=WhitespaceTokenizer().tokenize(nlp)
nlp_ws2


# In[125]:


len(ws_1)


# # wordpunct_tokenize

# In[169]:


from nltk.tokenize import wordpunct_tokenize


# In[178]:


sen2='the best and most beautifull thing in the world  $30.49rs'


# In[179]:


wpt=wordpunct_tokenize(sen2)


# In[180]:


wpt


# # NEXT WE WILL SEE HOW WE WILL USE UNI-GRAM,BI-GRAM,TRI-GRAM USING NLTK

# In[126]:


from nltk.util import bigrams,trigrams,ngrams 


# In[127]:


sentence = 'NO MATTER HOW HARD OR IMPOSSIBLE IT IS, NEVER LOSE SIGHT OF YOUR GOAL.'


# In[128]:


sentence


# In[129]:


quotes_tokens=word_tokenize(sentence)


# In[130]:


quotes_tokens


# In[131]:


len(quotes_tokens)


# In[132]:


quotes_big=list(bigrams(quotes_tokens))


# In[133]:


quotes_big


# In[134]:


len(quotes_big)


# In[135]:


quotes_tri=list(trigrams(quotes_tokens))


# In[136]:


quotes_tri


# In[137]:


len(quotes_tri)


# In[138]:


quotes_ngr=list(ngrams(quotes_tokens))


# In[139]:


quotes_ngr=list(ngrams(quotes_tokens,4))


# In[140]:


quotes_ngr


# In[141]:


quotes_ngr1=list(ngrams(quotes_tokens,6))


# In[106]:


quotes_ngr1


# # porter-stemmer

# In[142]:


from nltk.stem import PorterStemmer


# In[145]:


PorterStemmer().stem("affection")


# In[146]:


por=PorterStemmer()


# In[147]:


por.stem("working")


# In[148]:


por.stem("playing")


# In[149]:


por.stem('give') 


# In[150]:


words_arr=['give','giving','given','gave']


# In[154]:


for words in words_arr:
    print(words,":",por.stem(words))


# In[155]:


words_arr2=['give','giving','given','gave','thinking', 'loving', 'final', 'finalized', 'finally']


# In[156]:


for words in words_arr2:
    print(words,":",por.stem(words))


# # LancasterStemmer

# In[157]:


from nltk.stem import LancasterStemmer
# lancasterstemmer is more aggresive then the porterstemmer


# In[158]:


las=LancasterStemmer()


# In[160]:


for words in words_arr2:
    print(words,":",las.stem(words))


# # SnowballStemmer

# In[162]:


from nltk.stem import SnowballStemmer
#snowball stemmer is same as portstemmer


# In[164]:


sns=SnowballStemmer('english')


# In[165]:


for words in words_arr2:
    print(words,":",sns.stem(words))


# # WordNetLemmatizer

# In[181]:


from nltk.stem import WordNetLemmatizer


# In[182]:


wnt=WordNetLemmatizer()


# In[183]:


for words in words_arr2:
    print(words,":",wnt.lemmatize(words))


# In[185]:


por.stem('final')


# In[186]:


las.stem('finally')


# In[187]:


sns.stem('finalized')


# In[189]:


las.stem('final')


# In[190]:


las.stem('finalized')


# # stopwords
# #there is other concept called POS (part of speech) which deals with subject, noun, pronoun but before of this lets go with other concept called STOPWORDS
#     #STOPWORDS = i, is, as,at, on, about & nltk has their own list of stopewords 

# In[192]:


from nltk.corpus import stopwords


# In[195]:


stopwords.words("english")


# In[196]:


len(stopwords.words("english"))


# In[197]:


stopwords.words("spanish")


# In[198]:


len(stopwords.words("spanish"))


# In[200]:


stopwords.words("chinese")


# In[201]:


len(stopwords.words("chinese"))


# In[ ]:


stopwords.words('hindi') # research phase 


# # pos_tag([part of sppech])

# In[214]:


from nltk import pos_tag


# In[215]:


sent = 'janu is a natural when it comes to drawing'


# In[216]:


s_token=word_tokenize(sent)


# In[217]:


s_token


# In[222]:


for token in s_token:
    print(pos_tag([token]))


# In[225]:


sent2 = 'jadu is eating a delicious cake'
s_token2=word_tokenize(sent2)
for token in s_token2:
    print(pos_tag([token]))


# # NER (NAMED ENTITIY RECOGNITION)

# In[226]:


from nltk import ne_chunk


# In[227]:


NE_sent = 'The US president stays in the WHITEHOUSE '


# In[228]:


NE_tokens = word_tokenize(NE_sent)


# In[229]:


NE_tokens


# In[230]:


NE_tags =pos_tag(NE_tokens)
NE_tags


# In[233]:


NE_NER = ne_chunk(NE_tags)
print(NE_NER)


# In[245]:


new = 'the big cat ate the little mouse who was after fresh cheese'
NE_tokens = word_tokenize(new)
print(NE_tokens)
NE_NER1=pos_tag(NE_tokens)
print(NE_NER1)


# In[246]:


NE_NER2 = ne_chunk(NE_NER1)
print(NE_NER2)


# # wordcloud

# In[ ]:


pip install wordcloud


# In[251]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[252]:


text=("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")


# In[273]:


wordcloud = WordCloud(width=480, height=480, margin=0).generate(text) 


# In[274]:


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()


# In[269]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)


# In[270]:


wordcloud


# In[272]:


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)


# In[ ]:




