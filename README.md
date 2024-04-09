# Epilepsy-Detection
A Study on Seizure Detection Performance in an Automated Process by Extracting Entropy Features
Abstract—Millions of people of all ages have been diagnosed
with epilepsy all over the world. Electroencephalography (EEG),
a quantitative component, is vital in identifying and analyzing
epileptic seizures. Manual EEG detection takes a long time
and has serious consequences. To prevent this circumstance, the
world needs alternative detection technologies. For more than
a decade, several strategies and procedures have been used to
assist medical doctors in detecting technological improvement.
Numerous automated seizure identification frameworks that use
machine learning approaches have recently been presented to
replace traditional methodologies. However, because multichannel EEG data is unpredictable, selecting optimal channels as
well as characteristics and classifying them remain unsolved
issues. During automatic signaling, the gadget also emits a noisy
signal, making detection and prediction challenging. In this paper,
we have tested various entropy features and selected the best
features to classify seizure patients. First, we preprocessed the
data using IMF of the raw signal, and then entropy features were
extracted to classify the epileptic patients. The proposed channel
selection method, SVM-GA, works well with our framework and
successfully removes redundant channels from the data.
Index Terms—Fuzzy Entropy, Distribution Entropy, Sample
Entropy, EEG, SVM, Random Forest
I. INTRODUCTION
Epilepsy is a neurophysiological disorder in the CNS due
to which abnormal activities are induced, causing behavioural
issues, loss of awareness, and sometimes sensations [1], [2].
Treatment with surgery or medications will probably control
seizures for the majority. It takes a lifetime for some people
to get treatment. The seizures eventually go away. Some
instruments that aid in the diagnosis of epilepsy are known
as electroencephalograms (EEGs) [2], [3]. Because EEG data
has excellent temporal resolution but limited spatial resolution
[4]–[6], it has been considered as one of the finest medical
tests for assisting in epilepsy diagnosis. Automatic EEG
seizure identification, characterization, and classification have
been topics of interest and research in the medical, science,
and engineering disciplines since the mid-1960s. In clinics,
patients with medically intractable partial epilepsy usually
require time-consuming video EEG monitoring of spontaneous
seizures.
The rationale for utilizing longer signal recordings may be
in part because the techniques used in most previously reported
work, such as EMD, Signal Wavelet Transform, and DFA
analysis protocol, a huge bulk of data points are required in
order to obtain accurate classification results [7], [8]. Although
the authors reported that their statistical characteristics were
derived from the two fifty-six pointer segments (= 1.5 sec) of
the IMF (intrinsic mode derived using the EMD algorithm),
their preprocessing (filtration using a Butter-worth band-pass
filter) the EEG decomposition of IMF’s were carried out on
full EEG data. Nonetheless, the findings have not yet been
recorded precisely in this way to the best of our understanding
(including existing research, and see for review). Although
various models were developed that combined several classification issues, most of them were found to cover only some
of the problems. Despite the fact that significant epileptic
EEG feature-extraction [9] research has been published, few
publications on EEG channel selection have been documented
in recent decades. Furthermore, research on machine learning performance comparisons between outcomes with chosen
channels and results with all channels is scarce. The utilization
of epilepsy seizure in this paper [10] The detection was
carried out, and the ictal and variance values were calculated.
Non-ictal periods were estimated for use in channel analysis.
selection. The channels with the greatest value were then
selected. chosen by subtracting the variance value from the
computed for the ictal and calculated for the variance value
for the interictal period. Ibrahim et al. [11] also showed that
the seizure prediction probability by the chosen channel and
characteristic was greater than 70 percent, whereas the falsealarm probability was less than 30 percent. A classifier was
used to categorize the channels. SVM combines Recursive
Feature Elimination with Zero-Norm in reference [12]. Optimization was employed as a feature selection filter. The study
shows that the number of channels may be lowered without
affecting classification error [12]. Right-left hand control was
demonstrated in the paper [13]. Data from an EEG. Using the
top ten channels chosen by GNMM has reached 80 percent
classification accuracy. Then, by picking 6 channels from a
2022 5th International Conference on Signal Processing and Information Security (ICSPIS)
978-1-6654-9265-2/22/$31.00 ©2022 IEEE 86
2022 5th International Conference on Signal Processing and Information Security (ICSPIS) | 978-1-6654-9265-2/22/$31.00 ©2022 IEEE | DOI: 10.1109/ICSPIS57063.2022.10002385
Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR. Downloaded on April 09,2024 at 06:11:49 UTC from IEEE Xplore. Restrictions apply.
Fig. 1: Overview of the work
possible 32, 82 percent of hand control was acquired with
precision.
In fact, a few experiments that addressed the diagnosis of
epileptic seizures often set ”S” as a single class and one
or a mixture (e.g. ZOF, ONZ, or rather rarely FZNO) of
the sets F, Z, O and N. At the ictal level, several studies
centered on distinguishing the EEG from those in the interictal
phase. However, along with element S, the models were built
using only one member of the interictal class (either N or F).
While these three-class models theoretically meet traditional
therapeutic criteria, their applicability will need to be tested
more while taking into account ongoing monitoring, say for
online tracking and prompting epileptic behavior, as all of
them have been established based on long records of 4,097
sampling points. So the objectives of this work are:
1) Proposing a method for channel selection to improve the
seizure detection process and reduce the time complexity.
2) Analyzing the trends and training a predictive classification model to distinguish seizure sleep patterns using
available data sets.
3) Studying the feature extraction process of EEG Signals
using metrics such as Fuzzy Entropy and Sample Entropy on short-term EEGs.
II. METHODOLOGY
A. Data Pre-processing
The first step we have taken after downloading the data set is
preprocessing the data to remove noise and artifacts that have
been generated during the collection of the data from subjects.
Due to the unusual symptoms of this particular disease, lots of
muscle movements are common, and that is why the muscle
artifacts most likely will be imposed in the data set. A method
called Canonical Empirical Mode Decomposition has been
applied to remove those muscle artifacts, as well as a bandpass
filter, has been used to get noise-free data. Intrinsic mode
functions have been generated for each channel and to separate
the data, the Canonical Variate Analysis method has been
applied.
B. Channel Selection Algorithm
Identifying essential channels for this sort of experiment
is critical since the purpose of this study is to identify the
optimal answer in the shortest amount of time. Reducing
the number of channels simplifies the procedure and reduces
the temporal complexity. We suggested an SVM-GA model
pick the appropriate channels for this experiment. SVM is a
promising machine learning classifier, and GA mimics and
follows Charles Darwin’s idea of natural selection. There
are three phases to this procedure. Cross-over, mutation, and
selection: this approach may be used for feature selection.
In this study, the total SVM-GA channel selection procedure
is as follows.
1) Step 1: The permutation entropy was calculated for each
channel used in this experiment. PE is a resilient time
series approach that provides a quantifiable assessment
of the complexity of a dynamic system by capturing the
2022 5th International Conference on Signal Processing and Information Security (ICSPIS)
87
Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR. Downloaded on April 09,2024 at 06:11:49 UTC from IEEE Xplore. Restrictions apply.
Fig. 2: Channel selection process by SVM-GA method
TABLE I: Comparison of accuracy with a different type of features with different numbers of channels
Serial
No Entropy Features Classifier Number of Channels
1 2 3 4 5 6 7 8 9 10
1
Fuzzy Entropy
Sample Entropy
Distribution Entropy
SVM (Linear Kernel)
0.766
0.792
0.713
0.723
0.759
0.736
0.756
0.723
0.729
0.723
0.795
0.723
0.756
0.721
0.713
0.742
0.761
0.695
0.623
0.712
0.693
0.621
0.712
0.65
0.615
0.623
0.635
0.611
0.621
0.62
2
Fuzzy Entropy
Sample Entropy
Distribution Entropy
SVM (RBF Kernel)
0.51
0.500
0.513
0.556
0.512
0.636
0.562
0.552
0.642
0.523
0.541
0.643
0.521
0.653
0.613
0.512
0.541
0.619
0.511
0.523
0.578
0.552
0.552
0.563
0.545
0.518
0.542
0.512
0.513
0.513
3
Fuzzy Entropy
Sample Entropy
Distribution Entropy
Random Forest
0.552
0.524
0.523
0.586
0.553
0.636
0.572
0.576
0.642
0.561
0.635
0.643
0.521
0.523
0.60
0.511
0.521
0.618
0.510
0.600
0.617
0.596
0.574
0.615
0.542
0.572
0.602
0.522
0.513
0.601
order linkages between time series data and providing a
probability distribution of the ordinal patterns [14].
2) Step 2:After calculating PE SVM-GA takes a set of
subject’s data and represents one subject as one gene.
To make a chromosome various combination of genes
is possible. By the combination of various subject’s data
a string has been generated where one gene is selected
out of a population.
3) Step 3:An fitness function by SVM has been incorporated to evaluate each chromosome in the given population. This step will help to understand how well the
model predicted the pre- ictal time stamp. So for each
subject a fitness score has been calculated.
4) Step 4:Now in the reproduction phase the operator has
selected some values based on the probability distribution value. We have selected 0.95 for the probability
initially. For example if fn(x) is the fitness function,
then the probability of Chx chromosome for reproduction will be
probChx =
fn(x)
PNumpop
i=1 fn(Chi)
(1)
Where the Numpop is the number of total chromosome.
5) Step 5:Next the chromosoms are mixed up to make the
crossover. The genes are selected in a random order from
the pre selected chromosome which is called as parent
chromosome.
6) Step 6:Then the random mutation has been applied and
in this step a small amount of probability error has been
allowed for each gene.
7) Step 7:We need to repeat the process from step 2 untill
and unless the population converge in some point. In this
process the genetic algorithm actually provided a set of
possible solutions to our channel selection problem.
2022 5th International Conference on Signal Processing and Information Security (ICSPIS)
88
Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR. Downloaded on April 09,2024 at 06:11:49 UTC from IEEE Xplore. Restrictions apply.
C. Feature Engineering
In terms of features used to describe the EEG, non-linear
parameters attract growing interest nowadays because nonlinearity is assumed to be intrinsic in physiological processes. Different entropy parameters , i.e. approximate entropy
(AppEnt), sample entropy (Sam.Ent.), combination entropy,
entropy dependent functional dynamics, and fuzzy entropy
(FuzzyEn), were chosen since they were capable of supplying complexity figures, a nonlinear dynamic biomarker for
controlled physiology that relies on restricted tests. We have
to target high accuracy for extremely short-term database
collection. We have recently discovered a new method centered
on distribution. This distributed entropy, which functions as a
current component of the overall entropy measuring device,
has demonstrated remarkably remarkable efficiency in various
sectors when compared to older techniques.
D. Algorithm (Feature Definition)
Entropy increases the ability to distinguish between the
ordinary and the unexpected. Signals from the EEG. Entropy
is the measure of a system’s unpredictability. Entropy may
be used to quantify the unpredictability and complexity of a
signal. Various Entropy has been discovered as features for
classifying the EEG signal. We will discuss one by one:
E. Sample Entropy
Sometimes we need to know the randomness of a data which
is represented in series and Approximate Entropy and Sample
Entropy [15] helps to that. It estimates the value without
having any prior knowledge of the source. It can be said that
Sample Entropy is a kind of approximation entropy that is used
to analyse the complexities of biological time-series signals in
order to diagnose sick states.
1) Fuzzy Entropy: The fuzzy entropy [16] is used to
calculate the subjective value of information in the context
of uncertainty in the seal impression problem. Fuzzy Entropy
is a sophisticated Sample Entropy algorithm focused on the
fuzzy principle. By extension it does not depend on the total
probability of identical vectors according to the criteria of
hard thresholding as implemented in Sample Entropy. Instead,
Fuzzy Entropy calculates the chance, based on the fuzzy
membership function, of two vectors being identical.
2) Distribution Entropy: Initially, Distribution Entropy [17]
was proposed for the mitigation of parametric dependency and
robustness of Approximate Entropy Sample Entropy especially when extended to limited data sets. Through quantifying
the propagation properties of the inter-vector intervals, it takes
complete advantage of the state space equivalent of the underanalysed time sequence.
F. Classification
We have used various popular algorithm to test the features
and we select the the best performing classifiers and noted
their results in this paper.
Fig. 3: Accuracy-based performance study of SVM and RF
classifiers
Fig. 4: Specificity-based performance study of SVM and RF
classifiers
G. Experiment and Results
1) Data Description: In this work, an EEG database, currently accessible online, is used [18]. The database contains
five subsets of the EEG recordings (noted as Z, O, N, F, and S),
each with 100 singular-channel EEG recordings, with a duration of 23.6 s. After visual inspection for artifacts, these signals
were selected from continuous multichannel EEG recording.
Among all the subsets Z and O were recorded extracranially,
while subsets N, F, and S were recorded intracranially.
2022 5th International Conference on Signal Processing and Information Security (ICSPIS)
89
Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR. Downloaded on April 09,2024 at 06:11:49 UTC from IEEE Xplore. Restrictions apply.
(a) ROC curve for SVM (Linear Kernel)
(b) ROC Curve for SVM (RBF Kernel)
(c) ROC curve of Random Forest
Fig. 5: ROC curve for all three classifiers
On the dataset from the Bonn site, we used Fuzzy Entropy,
Distribution Entropy, and Sample Entropy to analyze 5-s EEG.
We found that it performed well at differentiating between
interictal, ictal, and ictal from interictal EEG, but Sample
Entropy had trouble with one of the three classification issues
Fig. 6: Sensitivity-based performance study of SVM and RF
classifiers
and we have mentioned that Distribution Entropy may still
use a technique for switching analytical frames if it uses
a 1-s EEG segment. Create an entropy-focused short-term
Electroencephalogram classification model based on the preceding findings that are appropriate for clinical settings and
may be used to start evaluating epileptic diseases or epileptic
therapies for epileptic patients. Which entropy works best
actually depends on the classifier. For example from Table
1, we can see that for Linear Kernel SVM Fuzz Entropy is
giving the best results on the other side with Random Forest
Distribution Entropy giving the best results among the three
entropies. Also, the classifiers’ performance depends on the
number of channels. In Table 1 the performance with the
different number of channels has been shown.
2) Experiment 1- Entropy Features Survey : In a contrast
to the Sample Entropy in this study, we will apply the Fuzzy
Entropy and Distribution Entropy techniques. Additionally,
although Approximate Entropy is defined for short data times,
it wasn’t included owing to the measure’s bias (particularly
for short-term data). As far as we know there is no systematic
study is available for the application of various types of
entropy or dynamic entropy to the biological signal which is
highly constrained. Finding the ideal entropy and minimum
features to get maximum performance is still a challenge. Two
protocols must be established in order to do this: E entropy a
single window with lengths ranging from 1 to 23 s and ii) a
multi-window entropy based on protocol measurements E with
overlap (l = 1 s) with lengths spanning from 1s to a certain
range x. The capacity to tell apart between: will be used to
determine whether the procedure is more successful. Regular
EEGs are created from interictal and ictal EEGs, while ictal
EEGs are created from interictal EEGs.
2022 5th International Conference on Signal Processing and Information Security (ICSPIS)
90
Authorized licensed use limited to: INDIAN INSTITUTE OF TECHNOLOGY KHARAGPUR. Downloaded on April 09,2024 at 06:11:49 UTC from IEEE Xplore. Restrictions apply.
3) Experiment 2-Classification Process between seizure and
non-seizure : From the following datasets in which, for each
window size, we obtain entropy values for all the 500 signals
classified as Z, F, N, O, and S in different tabs. Now, we apply
various classification algorithms like Least Square Support
Vector Machine using different kernels and Random Forest In
all of these, the ‘S’ data values are classified as that of seizure
patients whereas all the other four data values are classified
as being from non-seizure patients who are healthy.
4) Results: The ROC curve for SVM and RF has been
shown in Figure 3. (a), (b) and (c). It has been noticed that
the linear kernel is performing better than the RBF kernel.
When we tested the dataset with the Random Forest classifier
it gave better results than the RBF kernel but the linear kernel
outperformed both of the classifiers in most of the cases. We
are saying most of the cases because we have noticed for
some number of channels RF is performing better. So we can
say that RF and SVM (Linear Kernel) are good classifiers
for this kind of experiment. Figures 3, 4, and 5 have shown
the performance analysis of both of the classifiers based on
accuracy, sensitivity, and specificity.
CONCLUSION AND FUTURE SCOPE
The proposed artifact removal method works well with our
framework and successfully removes noise from the data.
Nonlinear characteristics have been shown to be critical for
this type of research. Furthermore, the inclusion of several
week-long classifiers distinguishes and verifies the study. The
cognitive framework that has been used in this paper can be
improved in such a way that it can be utilized in real-time,
also for medical purposes. An alarm system or sensor can
be installed with the machine and it will help the medical
practitioners understand the information quickly. More clinical
datasets can be tested with this algorithm to understand
whether it is good for all or not. Designing new gadgets like
smartphones or robots to assist epileptic patients can also be
possible if a new algorithm or advanced method has been
incorporated. Here we have considered the machine learning
method only, while the deep learning method can be used
to extract features for the seizure and it may help to make
the process much faster. A more hierarchical structure can
be used using several entropies and deep learning methods.
It may minimize the time and computational cost as well as
the uncertainty or overfitting of the model. Subject variability
issues or intensity measurement can be addressed in the future
for further improvement.
