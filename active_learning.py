'''
Uncertainty Sampling:
Uncertainty sampling is a popular active learning method that selects samples 
for annotation based on the model's prediction uncertainty. 
The idea is that the model is most likely to make errors on samples that it is uncertain about,
and annotating these samples can help improve its accuracy. Uncertainty sampling selects samples 
that have the highest entropy, or the least confidence, in their predicted class probabilities.
By selecting these uncertain samples, the active learning algorithm can provide the model 
with additional training data that is most informative for improving its accuracy.


Diversity Sampling:
Diversity sampling is another active learning method that selects samples 
for annotation based on their dissimilarity to the samples already selected. 
The idea is that selecting diverse samples can help the model learn more generalizable features 
and reduce overfitting to specific training examples. Diversity sampling selects samples that 
have the lowest similarity to the samples already selected, usually measured using a distance metric. 
By selecting diverse samples, the active learning algorithm can provide the model with a more
representative set of training data that captures a wider range of the underlying data distribution.

Density-Weighted Sampling:
Density-weighted sampling is a more recent active learning method that selects samples
for annotation based on their density in the input space. The idea is that regions 
with high data density are more informative than regions with low data density,
as they contain more information about the underlying data distribution. 
Density-weighted sampling assigns a weight to each sample based on its local density, 
usually measured using a kernel density estimator. 
Samples with higher weights are more likely to be selected for annotation, 
as they are deemed more informative for improving the model's accuracy.
By selecting samples from high-density regions, the active learning algorithm 
can provide the model with a more informative set of training data that captures 
the underlying data distribution more accurately.
'''

__version__ = 1






#avoid warnings (optional)
import warnings
warnings.filterwarnings("ignore")

#prerequisites for the functions
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from tensorflow.keras.optimizers import Adam


def uncertainty_sampling(model, images,req): #req required number of samples ,images is unlabelled data
    if len(images) == 0:
        return []
    y_pool = model.predict(images)
    uncertainty = np.max(y_pool, axis=1)
    query_indices = np.argsort(uncertainty)[::-1]
    return query_indices[:req]

def diversity_sampling(model, images,req): #req required number of samples ,images is unlabelled data
    features = model.predict(images)
    n_clusters = int(np.ceil(len(images) / 10))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    distances = kmeans.transform(features)
    min_distances = np.min(distances, axis=1)
    sorted_indices = np.argsort(min_distances)[::-1]
    selected_indices = []
    for i in sorted_indices:
        if i not in selected_indices:
            selected_indices.append(i)
    return selected_indices[:req]

def density_weighted_sampling(model, images,req): #req required number of samples ,images is unlabelled data
    features = model.predict(images)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(features)
    log_density = kde.score_samples(features)
    density = np.exp(log_density - np.max(log_density))
    density /= np.sum(density)
    indices = np.random.choice(range(len(images)), size=len(images), replace=False, p=density)
    return indices[:req]
