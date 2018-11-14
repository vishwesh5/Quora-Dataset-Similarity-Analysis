import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

csv_fname = "q_quora.csv"

question1 = {}
question2 = {}

print("Loading data from {}".format(csv_fname))

numLines = int(input("Enter number of lines to read: "))

with open(csv_fname,'r') as f:
    if numLines == -1:
        totalLines = f.readlines()[1:]
    else:
        totalLines = f.readlines()[1:numLines]
    print(len(totalLines))
    for line in totalLines:
        try:
            qid1, qid2, q1, q2 = line.strip().split(',')[1:5]
            question1[qid1] = q1
            question2[qid2] = q2
        except:
            continue

print("Data loaded successfully")

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"

print("Loading model from {}".format(module_url))
embed = hub.Module(module_url)
print("Model loaded successfully")

def plot_similarity(labels1, labels2, features1, features2, rotation):
    corr = np.inner(features1, features2)
    sns.set(font_scale=0.6)
    plt.figure(figsize=(100,100))
    g = sns.heatmap(corr,\
        xticklabels=labels1,\
        yticklabels=labels2,\
        vmin=0,\
        vmax=1,\
        cmap="YlOrRd")
    g.set_xticklabels(labels1, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    plt.tight_layout()
    plt.savefig("Quora.png")
    plt.show()

def run_and_plot(session_, input_tensor_, messages1_, messages2_, labels1_,labels2_, encoding_tensor):
    print("Embeddings questions 1")
    message_embeddings1_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages1_})
    print("Embeddings questions 2")
    message_embeddings2_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages2_})
    plot_similarity(labels1_,labels2_, \
            message_embeddings1_,\
            message_embeddings2_, 90)

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    run_and_plot(session, similarity_input_placeholder,\
            list(question1.values()),\
            list(question2.values()),\
            list(question1.keys()),\
            list(question2.keys()),\
            similarity_message_encodings)

