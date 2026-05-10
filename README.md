TimelineViz: Emotion Tracking over Time
This project is a tool to analyze how emotions shift throughout a piece of text. Instead of a single label for the entire input, it breaks the text into sentences and predicts an emotion for each step, creating a visual timeline.

Live Demo
Test the model here: TimelineViz App

Core Logic
Dataset: Trained on the GoEmotions dataset which consists of 28 distinct emotion categories.

Architecture: A hybrid model using a CNN layer for local feature extraction and a Bidirectional LSTM (Bi-LSTM) to capture the sequence and context of the narrative.

Visualization: Built with Plotly to generate an interactive graph showing the emotional arc across the sentence sequence.

Interface: Deployed via Streamlit for real-time inference.

Technical Stack
Language: Python

Deep Learning: PyTorch

Data Processing: NLTK, Pandas, NumPy

Performance
The model achieves 44% accuracy across the 28 emotion classes. In the context of high-cardinality emotion detection, this represents a functional baseline where the model successfully identifies nuanced emotional shifts.
