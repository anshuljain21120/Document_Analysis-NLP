import nltk
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import plotly
from IPython.display import Image
stopwords = nltk.corpus.stopwords.words('english')

T = open("Adventures Of SH.txt", 'r').read()

print("...\n" + T[680:999] + "\n...")
T1 = re.sub(r"(^ *[IXV]+\..*$)","\n", T, flags = re.MULTILINE)

T2 = re.sub(r"\s\s+", "\n", T1)
print("...\n" + T2[672:998] + "\n...")

T3 = re.findall("Contents([\s\S]*)End of the Project", T2)[0]
print("...\n" + T3[:304] + "\n...")

T = T3.lower()
print("...\n" + T[:304] + "\n...")

tokens = re.sub(r"[^A-Za-z0-9]", " ", T).split()
print(tokens[100:110])

tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
print(tokens[100:110])

dicti = dict()
for tok in tokens:
    dicti[tok] = dicti.get(tok, 0) + 1
dicttup = sorted(dicti.items(), key=lambda x: x[1], reverse=True)
tokdf = pd.DataFrame.from_records(dicttup, columns=['Token', 'Frequency'])

fig = go.Figure(data=[go.Histogram(x=tokdf[tokdf["Frequency"]<50]["Frequency"])])
fig.update_layout(
    title=go.layout.Title(
        text="For frequencies < 50",
        xref="paper",
        x = 0.5,
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Frequency of the word",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Count of word's frequency",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.show()

fig = go.Figure(data=[go.Histogram(x=tokdf[tokdf["Frequency"]>=50]["Frequency"])])
fig.update_layout(
    title=go.layout.Title(
        text="For frequencies ≥ 50",
        xref="paper",
        x = 0.5,
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Frequency of the word",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Count of word's frequency",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.show()

wc = WordCloud(width = 1600, height = 900, background_color ='white').generate_from_frequencies(frequencies=dicti)
plt.figure(figsize = (16, 9), facecolor = None)
plt.box(False)
plt.axis('off')
plt.imshow(wc)

dictj = dict()
for tok in tokens:
    if tok not in stopwords:
        dictj[tok] = dictj.get(tok, 0) + 1
dicttup = sorted(dictj.items(), key=lambda x: x[1], reverse=True)
tokdf_ws = pd.DataFrame.from_records(dicttup, columns=['Token', 'Frequency'])

wc = WordCloud(width = 1600, height = 900, background_color ='white').generate_from_frequencies(frequencies=dictj)
plt.figure(figsize = (16, 9), facecolor = None)
plt.box(False)
plt.axis('off')
plt.imshow(wc)

tokdf["TokenLength"] = tokdf["Token"].apply(lambda x:len(x))
lenvsfreq = tokdf.groupby("TokenLength")["Frequency"].agg('sum')
fig = go.Figure(data=go.Scatter(x=lenvsfreq.index, y=lenvsfreq))
fig.update_layout(
    title=go.layout.Title(
        text="Word Length Vs Frequency",
        xref="paper",
        x = 0.5,
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Length of the word",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Frequency of the word",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.show()

crp = nltk.corpus.treebank
print(crp.tagged_sents()[:2])

tagdict = dict()
for line in nltk.corpus.treebank.tagged_sents():
    for wordtup in line:
        tagdict[wordtup[1]] = tagdict.get(wordtup[1], 0) + 1
posdf = pd.DataFrame.from_records(list(tagdict.items()), columns=['POSTag', 'Frequency'])

fig = go.Figure(data=go.Bar(x=posdf["POSTag"], y=posdf["Frequency"]))
fig.update_layout(
    title=go.layout.Title(
        text="Tags Vs Frequency",
        xref="paper",
        x = 0.5,
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Tags according to Treebank",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Frequency in WSJ Corpus",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)
fig.show()