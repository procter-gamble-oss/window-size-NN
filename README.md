## Scientific paper infos
### Authors

Guillaume Levasseur & Hugues Bersini

IRIDIA

Université libre de Bruxelles

Brussels, Belgium

### Title
*Time Series Representation for Real-World Applications of Deep Neural Networks*

### Abstract
Neural networks have a proven usefulness at predicting, denoising or classifying
time series.
However, the performance of deep learning models is bound to the size of the
input window.
Yet, no common method has emerged to determine the optimal window size.
In this paper, we compare two heuristics and three event detection algorithms to find the
best time representation for three different tasks, using one simulated and two
real-world datasets.
The two real-world applications are the electricity disaggregation for energy
efficiency in buildings and the detection of fibrillation for diagnosis in cardiology.
We compare the obtained window sizes with the experimental values from previous
research and we experimentally validate the relevance of the results using
both convolutional and recurrent deep neural networks.
Results confirm the impact of the sequence length on model performance and show
that window sizes cannot be simply transferred to another dataset, even for the
same problem.
We also find that the false nearest neighbors method can reliably estimate the
window size and can help with the tedious work of finding the right time
representation.


## Installation
1. Create a virtual environment and activate it.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the required packages.
```bash
pip install -r requirements.txt
```
3. Install NILMTK from the source.
```bash
git clone https://github.com/nilmtk/nilm_metadata.git
cd nilm_metadata
python3 setup.py develop # will symlink the repo instead of installing
cd ..
git clone https://github.com/nilmtk/nilmtk.git
cd nilmtk
python3 setup.py develop --no-deps
cd ..
```

4. Download the datasets for NILM and PAF use cases.
   See instructions in `datasets/README.md`.


## Usage
1. Run `1_window_sizes.py` to compute the window sizes for each test case.
2. Depending on the window sizes from the step 1., you use the function in
   `2_neural_performance.py` and change the `seqlens` argument.
3. `paper_figures.py` is used to generate the figures for the paper.
    To generate figures for different results, replace the file
    `output/results-summary.csv`.
    This file has been generated manually from the output logs, printed in the
    console.

NB: Running steps 1 and 2 takes a significant time (days).
Starting the computation on a server under a `nohup` or `screen` session is
recommended.

