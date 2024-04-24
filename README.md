# Emo-Track

Emo-Track is a project aimed at analyzing speech sentiment using machine learning techniques. The project utilizes five datasets: RAVDESS, TESS, SAVEE, EmoDB, and CREMA-D, each containing emotional speech recordings. The emotions considered in this project are Angry, Disgust, Fearful, Happy, and Sad.

## Datasets Used

* RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song): This dataset contains recordings of actors speaking and singing in various emotional states, including anger, disgust, fear, happiness, sadness, and neutrality. The dataset also includes demographic information about the actors.
  Link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
* TESS (Toronto emotional speech set): TESS comprises recordings of North American English speakers producing sentences with specific emotional content. It includes seven basic emotions: angry, disgusted, fearful, happy, sad, surprised, and neutral.
  Link: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
* SAVEE (Surrey Audio-Visual Expressed Emotion): SAVEE consists of British English speakers enunciating different emotions, including neutral, anger, happiness, sadness, fear, disgust, and surprise.
  Link: https://www.kaggle.com/datasets/barelydedicated/savee-database
* EmoDB (Berlin Database of Emotional Speech): EmoDB contains emotional speech recordings of German actors, including seven emotions: anger, boredom, disgust, fear, happiness, sadness, and neutral.
  Link: https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb
* CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset): CREMA-D features American English speakers performing sentences with various emotional expressions, including anger, disgust, fear, happiness, sadness, and neutral.
  Link: https://www.kaggle.com/datasets/ejlok1/cremad

## Technology Stack

- MATLAB: Used for feature extraction from audio files.
- Python: Utilized for data preprocessing, feature selection, and model training.
  - Pandas: Used for data manipulation and analysis.
  - NumPy: Utilized for numerical computations and array operations.
  - Matplotlib: Employed for data visualization and plotting.
  - Seaborn: Used for statistical data visualization, enhancing Matplotlib's capabilities.
  - Scikit-learn: Employed for machine learning tasks such as feature selection, model training, and evaluation.
  - Support Vector Machine (SVM): Chosen as the classification model for sentiment analysis due to its effectiveness with high-dimensional data and nonlinear relationships.
  - ExtraTreesClassifier: Utilized for feature selection to reduce dimensionality and focus on the most relevant features.
  - StandardScaler: Applied for standardizing the selected features, ensuring uniformity in their scales.
  - Joblib: Used for saving and loading trained models.

## Features Extracted

From the audio files, 17 features were extracted using MATLAB with a window size of 2048 and overlap of 1024. These features include:

1. **Mel-Frequency Cepstral Coefficients (MFCC)**: Captures the spectral envelope of a signal.
2. **Delta MFCC**: First-order derivatives of MFCC coefficients which captures the rate of change of MFCCs over time.
3. **Delta Delta MFCC**: Second-order derivatives of MFCC coefficients which provides information about the acceleration or curvature of MFCCs over time.
4. **Mel Spectrogram**: Represents the power spectral density of a signal on a mel scale.
5. **Spectral Flux**: Measures the change in spectral shape over time.
6. **Spectral Skewness**: Describes the asymmetry of the spectral distribution.
7. **Spectral Slope**: Indicates the rate of change of spectral magnitude.
8. **Spectral Entropy**: Measures the amount of information in the spectral distribution.
9. **Spectral Rolloff**: Represents the frequency below which a certain percentage of the total spectral energy lies.
10. **Chromagram**: Captures the energy distribution of musical notes.
11. **Linear Predictive Coding (LPC)**: Models the spectral envelope of a signal using linear prediction.
12. **Zero Crossing rate (ZCR)**: Represents the rate at which the signal changes its sign.
13. **Energy**: Represents the signal's overall energy content.
14. **Pitch**: Indicates the perceived frequency of the signal.
15. **Intensity**: Represents the loudness of the signal.
16. **Harmonic-to-Noise Ratio (HNR)**: Measures the ratio of harmonic components to noise in the signal.
17. **Root Mean Square (RMS)**: Represents the average power of the signal over time.

These features represent various characteristics extracted from audio signals to capture relevant information for sentiment analysis.

## Model Training

**Feature Selection and Preprocessing:**
- Feature extraction from audio files resulted in a large number of features.
- To reduce dimensionality and focus on the most relevant features, an ExtraTreesClassifier was employed for feature selection.
- The best 30 features were chosen based on their importance scores.
- StandardScaler was applied to standardize the selected features, ensuring uniformity in their scales.

**Model Training:**
- Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel was chosen for its effectiveness in handling high-dimensional data and nonlinear relationships.
- The model was trained using an iterative process, exploring various combinations of C and gamma values.
- C represents the regularization parameter, controlling the trade-off between maximizing the margin and minimizing the classification error.
- Gamma defines the influence of a single training example, affecting the flexibility of the decision boundary.
- The model's accuracy was evaluated on a validation set, and the combination of C and gamma yielding the highest accuracy was selected.
- The chosen model was then saved for future use, allowing for quick and efficient sentiment analysis on new audio samples.

## Results

|               Dataset              |    Best Accuracy    |    C Value   |    Gamma Value   |
|------------------------------------|---------------------|--------------|------------------|
| RAVDESS                            | 0.854               | 4.60         | 0.28             |
| TESS                               | 1.00                | 1.00         | 0.03             |
| SAVEE                              | 0.933               | 4.70         | 0.14             |
| EmoDB                              | 0.957               | 11.70        | 0.01             |
| CREMA-D                            | 0.603               | 14.3         | 0.04             |
| RAVDESS + SAVEE + EmoDB (Combined) | 0.815               | 16.70        | 0.09             |

This table provides a concise overview of the best accuracy achieved, along with the corresponding C and gamma values for each dataset and the combined dataset.

## Class-wise Accuracy

**Test Dataset:**

|               Dataset              | Anger | Disgust | Fear | Happy | Sad  |
|------------------------------------|-------|---------|------|-------|------|
| RAVDESS                            | 0.90  | 0.84    | 0.84 | 0.77  | 0.92 |
| TESS                               | 1.00  | 1.00    | 1.00 | 1.00  | 1.00 |
| SAVEE                              | 0.92  | 0.92    | 1.00 | 0.83  | 1.00 |
| EmoDB                              | 0.90  | 0.89    | 1.00 | 1.00  | 1.00 |
| CREMA-D                            | 0.67  | 0.60    | 0.39 | 0.53  | 0.82 |
| RAVDESS + SAVEE + EmoDB (Combined) | 0.82  | 0.82    | 0.81 | 0.76  | 0.87 |

**Whole Dataset:**

|               Dataset              | Anger | Disgust | Fear | Happy | Sad  |
|------------------------------------|-------|---------|------|-------|------|
| RAVDESS                            | 0.98  | 0.97    | 0.97 | 0.95  | 0.98 |
| TESS                               | 1.00  | 1.00    | 1.00 | 1.00  | 1.00 |
| SAVEE                              | 0.97  | 0.97    | 0.98 | 0.93  | 1.00 |
| EmoDB                              | 0.87  | 0.93    | 0.96 | 0.93  | 0.96 |
| CREMA-D                            | 0.77  | 0.60    | 0.50 | 0.67  | 0.81 |
| RAVDESS + SAVEE + EmoDB (Combined) | 0.95  | 0.96    | 0.96 | 0.93  | 0.96 |

These tables summarize the accuracy of each emotion category for both the test dataset and the entire dataset across different datasets used in the project.

## Conclusion

Emo-Track demonstrates the effectiveness of machine learning in analyzing emotional speech, achieving high accuracies across diverse datasets. By combining feature extraction techniques and Support Vector Machine classification, Emo-Track captures nuanced patterns in speech emotions. Its modular design allows easy integration with new datasets, fostering collaboration and innovation in sentiment analysis. Emo-Track's success highlights the potential of AI in understanding human emotions and advancing affective computing technologies.

## Contributing

Contributions are welcome! Please feel free to submit bug reports, feature requests, or pull requests to help improve this project.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/deepgoenka/Emo-Track/blob/main/LICENSE) file for details.
