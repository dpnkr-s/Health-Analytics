### Hidden Markov model for vowel recognition

Hidden Markov Models(HMMs) are trained in MATLAB to identify a vowel sound from recording. Firstly, HMMs for each vowel sound ‘a’, ‘e’, ‘i’, etc. are trained using 5 samples for each vowel. Then, the obtained models analyze a new recording and identify which vowel is it by evaluating the maximum likelihood probability of new sample over each HMM.

A new sound sample for each vowel is recorded and then probability of sample over each model is evaluated and presented as a probability matrix below. In the probability matrix, HMM models for each vowel represent columns and new recording for each vowel represents rows. Probability of each vowel for each HMM is,

P = exp(logpseqxx), which is not actually given in table below but only the values logpseqxx (e.g. Paa = exp(logpseqaa), where logpseqaa = -8.735e+3). Maximum values in each row are highlighted.

<img width="700" alt="Screenshot 2023-02-23 at 12 58 10 AM" src="https://user-images.githubusercontent.com/25234772/220738017-f113b3f5-6900-46b3-acac-4a48f96f94e7.png">

### Comments

It can be observed from the probability matrix that recorded sample of each vowel has maximum probability for its corresponding HMM as expected. Also, it is noted that the probability values over each HMM are very low and lie close to each other, this could be because only 5 samples per vowel are used to train HMMs resulting in basic models.
