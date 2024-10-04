# Attention_is_all_you_need

## Generative AI Pipeline


### Data Aquisition
### Data Augmentation

    1. Replace with Synonyms etc.
    2. Biagram flip 
        (My name is Amad -> Amad is my name)
    3. Back translate
        Translate to other language a couple of times and back to english.
    4. Add additional Data/Noise
        I am a Data Scientist -> I am a Data Scientist and I live this job.

### Data Preparation

    1. Clean Up: 
        Rempve HTML Tags, Emojis, Spelling Correction
    2. Basic Preprocessing:
        Tokenization -> Sentence, Word
    3. Optional Preprocessing:
        Stop Word Removal
        Stemming (Less Used)
            play, played, playing -> play
        Lamatization
            Similar to stemming but improved*
        Punctation Removal
        Lower case
        Language Detection
    4. Advanced Preprocessing:
        Parts of Speech Tagging
        Parsing
        Coreference Resolution

![screenshot](images/image.png)


### Feature Engineering
### Modeling
### Evaluation
### Deployment
### Monitoring and Model Updating

