import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
from collections import Counter
from sklearn.metrics import accuracy_score , f1_score , ConfusionMatrixDisplay , classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import pickle
df = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\data-science-projects\TimeLine Viz Project\train.tsv',sep='\t',header=None,names=['text','labels','id'])
df['labels'] = df['labels'].apply(lambda x: int(str(x).split(',')[0]))
print(df.isnull().sum())
stop_words = set(stopwords.words('english'))
def clean_text(txt):
    txt = txt.lower()
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = ''.join([i for i in txt if i.isascii()])
    words = word_tokenize(txt)
    cleaned_words = [w for w in words if w not in stop_words]
    return cleaned_words
df['tokenized'] = df['text'].apply(clean_text)

all_words = [word for tokens in df['tokenized']for word in tokens]
word_counts = Counter(all_words)
vocab = {word: i+2 for i,(word,_) in enumerate(word_counts.items())}
vocab['PAD'] = 0
vocab['UNK'] = 1
MAX_LEN = 50
def encode_and_pad(tokens):
    encoded = [vocab.get(w,1) for w in tokens]
    if len(encoded)<MAX_LEN:
        encoded+=[0]*(MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return np.array(encoded)
df['encoded'] = df['tokenized'].apply(encode_and_pad)
class TimeLineVizModel(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim,output_dim):
        super(TimeLineVizModel,self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.conv = nn.Conv1d(embed_dim,128,kernel_size=3)
        self.lstm = nn.LSTM(128,hidden_dim,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2,output_dim)
        self.dropout = nn.Dropout(0.3)
    def forward(self,x):
        x = self.embedding(x).permute(0,2,1)
        x = torch.relu(self.conv(x)).permute(0,2,1)
        lstm_out,(hidden,_) = self.lstm(x)
        cat = torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1)
        return self.fc(self.dropout(cat))
print('data is prepared , Model architechture is ready')
print(df[['text','encoded']].head())
class EmotionDataset(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(np.stack(X),dtype=torch.long)
        self.y = torch.tensor(y.values,dtype = torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
dataset = EmotionDataset(df['encoded'],df['labels'])
x = df['encoded']
y = df['labels']
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=42)
train_data = EmotionDataset(x_train,y_train)
test_data = EmotionDataset(x_test,y_test)

train_loader = DataLoader(train_data,batch_size=64,shuffle = True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)
model = TimeLineVizModel(vocab_size=len(vocab),embed_dim=100,hidden_dim=128,output_dim=len(df['labels'].unique()))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epoch = 10
print('Starting Training : ')

for epoch in range(epoch):
    model.train()
    total_loss = 0
    for text , labels in train_loader:
        optimizer.zero_grad()
        outputs = model(text)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f'Epoch : {epoch+1}/10|| Loss : {total_loss/len(train_loader):.4f}')
print('Training Completed')
def model_evaluation(model , loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for texts , labels in loader:
            outputs = model(texts)
            _,predicted = torch.max(outputs,1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    accuracy =  accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred,average='weighted')
    print(f' --- Model Performance ---')
    print(f'Accuracy : {accuracy*100:.3f}%')
    print(f'F1 Score : {f1:.3f}')
    print('Classification Report :',classification_report(y_true,y_pred))
    return y_true , y_pred
y_true , y_pred = model_evaluation(model , test_loader)
cmd = ConfusionMatrixDisplay.from_predictions(y_true , y_pred , display_labels=df['labels'].unique(),cmap='magma',xticks_rotation='vertical')
fig=plt.gcf()
plt.title('TimeLineViz - Confusion Matrix')
plt.savefig('timelineviz_confusion_matrix.png',dpi = 300 , bbox_inches = 'tight')
plt.show()
def get_timeline(story):
    model.eval()
    sentences = nltk.sent_tokenize(story)
    timeline = []
    with torch.no_grad():
        for sent in sentences:
            tokens = clean_text(sent)
            encoded = encode_and_pad(tokens)
            input_tensor = torch.tensor([encoded] ,dtype =  torch.long)
            output = model(input_tensor)
            _,predicted =   torch.max(output,1)
            timeline.append(predicted.item())
    return sentences , timeline
test_story = '''I am so happy that I finally started this project. It was abit confusing but now that i am working on in
it i feel like i can do it!!wait is the server crashing again ? this is so fuckin frustating. Its so annoying when you work
on something and shit keeps happening i hate it. whatever i will fix it and succeed anyways, I am calm now huh
'''
sentences , emotional_journey = get_timeline(test_story)
print('Sentence is analysed : ',len(sentences))
print('Emotional Pulse : ',emotional_journey)
def plot_interactive_timeline(sentences , journey):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(journey))),
        y=journey,
        mode = 'lines+markers',
        text = sentences,
        marker=dict(size=12,color=journey,colorscale='Viridis'),
        line=dict(width=3,color='royalblue')
    ))
    fig.update_layout(
        title='Emotional Arc pf the Story',
        xaxis_title='Sentences Sequence',
        yaxis_title='Emotion Category (num)',
        template='plotly_dark'
        
    )
    fig.write_image('timeline.png')
    fig.show()
    return
torch.save(model.state_dict(), 'timeline_model.pth')
artifacts = {
    'vocab': vocab,
    'MAX_LEN': MAX_LEN
}
with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print("Model and artifacts saved!")
