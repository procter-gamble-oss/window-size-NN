### Authors

Guillaume Levasseur & Hugues Bersini

IRIDIA

Université libre de Bruxelles

Brussels, Belgium

### Title
*Determining the sequence length of neural network regressors for supervised non-intrusive load monitoring*

### Abstract
Recent non-intrusive load monitoring (NILM) uses neural networks to tackle regression tasks.
These neural networks use a sliding window to process the electrical power time series.
The size of the window is impacting the model performance, but the NILM research is lacking a method to compute it.
In this paper, we introduce an event-based heuristic to determine the window size from the training data.
We compare it to existing methods from the physics literature: the false nearest neighbors and the smoothness of the mapping.
To evaluate the methods, state-of-the-art neural network models are applied to public datasets.
Our results show that the automated methods help narrowing down the search space for the window size with much less manual work and runtime.

