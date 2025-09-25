# Intelligent Patent Clustering and Recommendation Platform

## I. Project Overview

This is an **Intelligent Patent Clustering and Recommendation Platform** based on **Graph Neural Networks (GNN)** and **Machine Learning (ML)**. The platform utilizes a **microservices architecture** to separate business logic and machine learning services, providing **intelligent analysis, clustering, and recommendation functionalities** for patent documents.

---

## II. Project Structure

```txt
patent/
├── data/
│   ├── processed/                    # Processed data
│   ├── cache/                        # Cached data
│   └── build.py                      # Script for building graph data
│   └── make_embedding.py.            # Script for generating patent embedding vectors
├── src/
│   ├── configs/                      # Configuration files
│   │   ├── config_clustering.yaml   # Clustering model configuration
│   │   └── config_rgcn.yaml         # RGCN model configuration
│   └── models/                       # Model code
│       ├── clustering.py             # Clustering model
│       └── rgcn.py                   # RGCN model
├── frontend/
│   ├── index.html                    # Frontend main page
│   ├── example.html                  # Example page
│   ├── css/style.css
│   └── js/script.js
├── backend/
│   └── app.py                        # Backend API
├── scripts/
│   └── pretrain.py                   # Model pre-training script
└── models/                            # Trained models
```
---

## III. Environment Setup

### 1. Install Dependencies

Install the required Python libraries using the following command in the **project root directory**:

```bash
pip install -r requirements.txt
```

### 2. Configuration File Descriptions

* **`src/configs/config_clustering.yaml`**: Configuration file for the **clustering model**, including the number of clusters, and parameters for **KMeans** and **t-SNE**.
* **`src/configs/config_rgcn.yaml`**: Configuration file for the **RGCN model**, including hidden layer dimensions, number of basis matrices, training epochs, and other parameters.

---

## IV. Data Preparation

### 1. Data Processing

Run the **`data/build.py`** script to process the raw patent data into **graph format**, saving the output to the **`data/cache`** directory:

```bash
python data/build\.py
```

### 2. Generate Embedding Vectors

Run the **`data/make_embedding.py`** script to use the trained RGCN model to generate **embedding vectors** for each patent, saving them as a **CSV file**:

```bash
python data/make_embedding.py
```

---

## V. Model Training

Run the **`scripts/pretrain.py`** script to perform pre-training for both the **RGCN model** and the **clustering model**:

```bash
python scripts/pretrain.py
```

After training is complete, the models will be saved to the **`models`** directory.

---

## VI. Starting the Service

### 1. Start Backend Service

Navigate to the **`backend`** directory and run the **`app.py`** script to start the **Flask backend service**:

```bash
python backend/app.py
```

### 2. Open Frontend Page

Open the **`frontend/index.html`** file in your browser to access the patent recommendation system's frontend interface.

---

## VII. Usage Instructions

1. **Search Similar Patents**: Enter author, applicant, or title keywords into the search box. Click the **"Search Similar Patents"** button to get a list of similar patents.
2. **View Clustering Overview**: Click the **"View Clustering Overview"** button to see the **count of patents** contained within each cluster.
3. **View Patent List for a Specific Cluster**: Enter the cluster number into the input box. Click the **"View Patents in This Cluster"** button to retrieve the list of all patents in that cluster.
4. **View Example Patents**: Click **"Example"** in the top navigation bar. Select a patent ID from the 100 samples and enter it into the input box to find its similar patents.

---

## VIII. Important Notes

* Please ensure that the input patent data file path and format are correct, and the file encoding is set to **`gbk`**.
* During model training, you may need to **adjust the parameters in the configuration files** to achieve better performance.
* If errors occur during runtime, check the **log files** for error messages.

---

## License

This project is licensed under the **MIT License**. See the **`LICENSE`** file for details.
