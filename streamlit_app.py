import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import nltk
import pickle
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

@st.cache_resource
def load_resources():
    with open('model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['vocab'], artifacts['MAX_LEN']

vocab, MAX_LEN = load_resources()

class TimeLineVizModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TimeLineVizModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x)).permute(0, 2, 1)
        lstm_out, (hidden, _) = self.lstm(x)
        cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(cat))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TimeLineVizModel(len(vocab), 100, 128, 28)
model.load_state_dict(torch.load('timeline_model.pth', map_location=device))
model.eval()

def clean_and_encode(text):
    tokens = word_tokenize(text.lower())
    encoded = [vocab.get(w, 1) for w in tokens if w.isalpha()]
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return torch.tensor([encoded], dtype=torch.long)

st.set_page_config(page_title="TimeLineViz", layout="wide")
st.title("🎭 TimeLineViz: Emotional Journey Tracker")
st.markdown("Enter a story or a series of thoughts to visualize the emotional arc.")

user_input = st.text_area("Enter your story here:", height=200, placeholder="I was so happy today... but then things went wrong.")

if st.button("Analyze Emotions"):
    if user_input:
        sentences = nltk.sent_tokenize(user_input)
        journey = []
        for sent in sentences:
            input_tensor = clean_and_encode(sent)
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                journey.append(predicted.item())
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(journey))),
            y=journey,
            mode='lines+markers',
            hovertext=sentences,
            marker=dict(size=12, color=journey, colorscale='Viridis', showscale=True),
            line=dict(width=3, color='royalblue')
        ))
        
        fig.update_layout(
            title="Emotional Pulse Over Time",
            xaxis_title="Timeline (Sentence Index)",
            yaxis_title="Emotion Class Index",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("See Sentence Details"):
            for i, sent in enumerate(sentences):
                st.write(f"**Sentence {i+1}:** {sent} (Class: {journey[i]})")
    else:
        st.warning("Please enter some text first!")
