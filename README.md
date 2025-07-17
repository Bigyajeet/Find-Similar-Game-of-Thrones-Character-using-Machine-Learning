# Game of Thrones Character Similarity Analysis

This project leverages machine learning techniques to identify and visualize similarities among characters from HBO's Game of Thrones series. By processing high-dimensional textual data (character dialogues and descriptions), we aim to gain insights into character relationships and archetypes within the GoT universe.

## Project Overview

The core idea behind this project is to represent each Game of Thrones character as a numerical vector based on their textual attributes (e.g., words they speak, descriptions). By applying various machine learning algorithms, we can then quantify the "distance" or "similarity" between these character vectors, ultimately identifying groups of characters who share common traits, roles, or narrative functions. A significant focus is placed on visualizing these high-dimensional similarities in an intuitive way.

## Dataset

The primary dataset used for this analysis is sourced from Kaggle:
- **Game of Thrones Data:** [https://www.kaggle.com/datasets/mathurinache/game-of-thrones-data](https://www.kaggle.com/datasets/mathurinache/game-of-thrones-data)
    - Specifically, the `script-bag-of-words.json` file, which provides a bag-of-words representation of character dialogues/scripts.

*(Optional: You can mention the MNIST dataset if it was solely for `tesne` practice and not directly used in the final GoT analysis.)*
- **MNIST Digit Recognizer (for t-SNE practice):** [https://www.kaggle.com/c/digit-recognizer/data](https://www.kaggle.com/c/digit-recognizer/data)
    - This dataset was used for practicing and validating the t-SNE dimensionality reduction algorithm.



### Data Preprocessing

The `script-bag-of-words.json` dataset likely provides pre-processed text data. However, the initial steps for generating such data typically involve:
1.  **Text Extraction:** Collecting character-specific dialogues or descriptions.
2.  **Tokenization:** Breaking text into individual words or phrases.
3.  **Stop Word Removal:** Eliminating common words (e.g., "the", "is", "a") that carry little semantic meaning.
4.  **Stemming/Lemmatization:** Reducing words to their root form (e.g., "running" -> "run").
5.  **Bag-of-Words (BoW) Representation:** Creating vectors where each dimension corresponds to a unique word in the vocabulary, and the value represents its frequency or presence in a character's text.

### Feature Engineering

The bag-of-words approach itself is a form of feature engineering. Depending on the complexity, further steps might include:
-   **TF-IDF (Term Frequency-Inverse Document Frequency):** Weighting words based on their importance in a character's text relative to the entire corpus. This helps in highlighting unique character traits.

### Dimensionality Reduction & Visualization

To effectively visualize the similarities among high-dimensional character vectors, dimensionality reduction techniques are crucial.
-   **t-Distributed Stochastic Neighbor Embedding (t-SNE):** Applied to project the high-dimensional character vectors into a 2D or 3D space, preserving local similarities. This allows for intuitive scatter plots where closer points represent more similar characters.


### Similarity Measurement

Once characters are represented as vectors, their similarity is quantified using metrics such as:
-   **Cosine Similarity:** Measures the cosine of the angle between two vectors. It is particularly effective for text data as it focuses on the orientation of the vectors rather than their magnitude, making it robust to differences in text length.
-
## Web Application (Character Matcher)

The project includes a web application, likely built with a Python web framework (e.g., Flask or Django), that allows users to:

1.  **Search for a character:** Users can input a Game of Thrones character's name.
2.  **Find similar characters:** The application leverages the `scikit-learn` trained similarity model (using the processed character vectors and cosine similarity) to identify and display the most similar characters to the queried one.
3.  **Interactive Display:** Results are presented in an easy-to-understand format, potentially including character names, similarity scores, and links to character profiles or relevant visualizations.

The core of this matching functionality relies heavily on `scikit-learn` for:
-   Loading and transforming character data (e.g., using `TfidfVectorizer` if not pre-computed).
-   Calculating similarity scores between a queried character's vector and all other character vectors in the dataset using `cosine_similarity`.
-   Potentially, if a machine learning model is involved in ranking or filtering, `scikit-learn` estimators would be used.

## Results & Insights

* **Character Clusters:** [Describe observations from your t-SNE plots. Do characters from the same house cluster together? Are villains and heroes distinct? Are there unexpected similarities?]
* **Key Contributing Features:** [If possible, discuss which words or terms heavily contribute to certain character similarities or distinctness.]
* **Visualizations:** Include screenshots of your t-SNE plots or any other relevant visualizations here.

* ## Future Work

* Explore other similarity metrics and clustering algorithms.
* Incorporate character metadata (e.g., house, allegiances, death status) to see its impact on similarity.
* Experiment with more advanced NLP techniques like Word Embeddings (Word2Vec, GloVe, BERT) for richer character representations.
* Develop a more interactive visualization tool.

├── .gitattributes
├── README.md
├── app.py                     # (Optional) Web application for interactive exploration
├── create_dummy_data.py       # (Optional) Script for generating dummy data
├── data.pkl                   # Processed data pickle file
├── go.ipynb                   # (Optional) Another Jupyter notebook, perhaps for exploration
├── got.ipynb                  # Main Jupyter notebook for GoT character analysis
├── script-bag-of-words.json   # Raw Game of Thrones dataset
├── requirements.txt           # Python dependencies (create this!)
└── visualizations/            # Directory to save generated plots (create this!)


## How to Run

To replicate this project:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link-here>
    cd <your-repo-name>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file based on your `go.ipynb` or `got.ipynb` dependencies, e.g., `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `json`, `tsne` (if separate package), etc.)*
3.  **Download the dataset:**
    Download `script-bag-of-words.json` from [Kaggle](https://www.kaggle.com/datasets/mathurinache/game-of-thrones-data) and place it in the `data/` directory (or wherever your `got.ipynb` expects it).
4.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook got.ipynb
    ```
    Follow the steps in the notebook to execute the analysis and generate visualizations.
5.  *(If you have an `app.py` for an interactive demo):*
    ```bash
    python app.py
    ```
    Then navigate to `http://localhost:5000` (or whatever port your `app.py` runs on).

## File Structure

