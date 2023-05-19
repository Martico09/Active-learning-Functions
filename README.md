# Active-learning-Functions
Active learning has been shown to improve the accuracy of image classification models, especially when dealing with limited labeled data. In active learning for image classification, the model is trained on a small initial set of labeled images, and then the active learning algorithm selects the most informative images for annotation. 

**Uncertainty Sampling:**<br>
Uncertainty sampling is a popular active learning method that selects samples for annotation based on the model's prediction uncertainty. The idea is that the model is most likely to make errors on samples that it is uncertain about, and annotating these samples can help improve its accuracy. Uncertainty sampling selects samples that have the highest entropy, or the least confidence, in their predicted class probabilities. By selecting these uncertain samples, the active learning algorithm can provide the model with additional training data that is most informative for improving its accuracy.
<br>
<br>
**Diversity Sampling:**<br>
Diversity sampling is another active learning method that selects samples for annotation based on their dissimilarity to the samples already selected. The idea is that selecting diverse samples can help the model learn more generalizable features and reduce overfitting to specific training examples. Diversity sampling selects samples that have the lowest similarity to the samples already selected, usually measured using a distance metric. By selecting diverse samples, the active learning algorithm can provide the model with a more representative set of training data that captures a wider range of the underlying data distribution.
<br>
<br>
**Density-Weighted Sampling:**<br>
Density-weighted sampling is a more recent active learning method that selects samples for annotation based on their density in the input space. The idea is that regions with high data density are more informative than regions with low data density, as they contain more information about the underlying data distribution. Density-weighted sampling assigns a weight to each sample based on its local density, usually measured using a kernel density estimator. Samples with higher weights are more likely to be selected for annotation, as they are deemed more informative for improving the model's accuracy. By selecting samples from high-density regions, the active learning algorithm can provide the model with a more informative set of training data that captures the underlying data distribution more accurately.
<br>






