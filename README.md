## TimeLineViz — Understanding Emotions Across Text

Live Demo: [Analyze text emotions here](https://timelineviz-using-cnn-and-bilstm-wcg7jujqmruxxylqzc2azy.streamlit.app/)

### What This Does

Most emotion detection treats text as a single unit - you get one label for the whole thing. This project is different. It breaks text into sentences and predicts emotions for each one, showing how sentiment changes as you read.

Useful for analyzing speeches, stories, customer reviews - anything where the emotional arc matters.

### The Problem I Was Solving

Standard emotion detection is too coarse. If you ask "what's the emotion of this paragraph?" you lose the nuance. A paragraph might start angry, become sad, then end hopeful. You want to see that journey, not just the final score.

### How It Works

The system uses a hybrid approach:

**CNN Layer**: Captures local patterns in text. Words and phrases that signal emotion often appear in short sequences. The CNN finds these.

**BiLSTM Layer**: Takes the CNN output and processes it sequentially in both directions. This helps the model understand context - whether "sick" means unwell or awesome depends on what comes before and after.

**Dataset**: Trained on GoEmotions, which has 28 emotion categories. It's not just happy/sad/angry - it includes contempt, amusement, disappointment, etc.

The model outputs an emotion label for each sentence, and Plotly visualizes how emotions change across the text.

### Performance

44% accuracy on 28 classes. That sounds low until you realize this is a hard problem. With 28 different emotions, random guessing is 3.5%. The model is doing 12x better than random.

The real metric is whether the emotional arc makes sense - does it capture the flow of the text? It does.

### Technical Stack

- Deep Learning: PyTorch
- Data Processing: NLTK, Pandas, NumPy
- Visualization: Plotly
- Frontend: Streamlit

### Why CNN + LSTM?

You might ask - why not just use a Transformer? Two reasons:

First, this runs entirely on free Streamlit cloud. A full Transformer would timeout.

Second, for this specific task, the hybrid approach works well. CNNs are efficient at extracting features, LSTMs handle sequences, and together they're fast enough for real-time inference.

### Challenges I Faced

**Challenge 1: Class Imbalance**
Some emotions in GoEmotions are rare. The model learned to predict common emotions (sad, happy) and ignore rare ones (amusement, approval). Fixed this with weighted loss functions.

**Challenge 2: Sentence Tokenization**
Not all punctuation means end-of-sentence. URLs, abbreviations, numbers mess this up. Used NLTK's sentence tokenizer which handles edge cases.

**Challenge 3: Inference Speed**
With Streamlit, you have seconds to return results. Optimized by batching sentences instead of processing one at a time. Went from 3 seconds to 500ms.

### What I Learned

Hybrid architectures can be practical. You don't always need the latest transformer - sometimes combining two simpler architectures works better for production constraints.

Also learned that high accuracy isn't the only metric. For this use case, the important thing is whether the emotional arc is coherent, not whether every single label is correct.

### Limitations

- Only works well with English text
- Performs worse on very short sentences (less context)
- Sometimes struggles with sarcasm (requires understanding context that maybe should be outside the text)
- 28 emotion classes is granular - sometimes too granular for practical use

### Future Improvements

- Fine-tune on domain-specific data (movies, customer reviews, etc.)
- Add confidence scores alongside predictions
- Support for multiple languages
- Option to reduce to 7-8 basic emotions for simpler output
- Batch processing for long documents
