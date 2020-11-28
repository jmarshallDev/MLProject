# ML Project

#### PreProcessing.py
I've finished writing PreProcessing.py; it handles the preprocessing from the 'all.txt' file. A couple of things you have to do first to get it to work:
<ul>
  <li>Install nltk with whatever of these works for you:<ul><li>pip install nltk</li><li>pip3 install nltk</li></ul></li>
  <li>Once you've installed nltk, import it and run the following (you only have to do this once):
    <ul>
      <li>nltk.download('stopwords')</li>
      <li>nltk.download('wordnet')</li>
      <li>nltk.download('punkt')</li>
      <li>nltk.download('averaged_perceptron_tagger')</li>
    </ul>
  </li>
  <li>from preprocessing import gimme</li>
</ul>
Once you've done the above, just call 'gimme()' to get the processed data and labels, respectfully. The data comes back as a 2d sparse matrix, but from what I can see of some examples, we should be able to plug those into ML algorithms just fine. The lables return as a numpy array of integers, where '0' is a negative sentiment and '1' is a positive sentiment. A note: the data is not randomized; it's in the same order as all.txt, so we should probably randomize what data is used for training / testing / verification when the time comes.
<br />
If you have any problems/questions, let me know.
<br />
Happy belated Thanksgiving!
<br />
-N
