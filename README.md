# DEDL-Kcr
<hr>
DEDL-Kcr is an explainable method for identifying lysine crotonylation (Kcr) sites based on dynamic ensemble deep learning framework.

# Dependencies<hr>
DEDL-Kcr requires python=3.9
Below are the core Python packages required by DEDL-Kcr:<br>
<ul>
    <li>tensorflow(2.12.0)</li>
    <li>scikit-learn(1.3.0)</li>
    <li>numpy(1.24.3)</li>
    <li>pandas(2.0.3)</li>
    <li>matplotlib(3.7.1)</li>
    <li>seaborn(0.12.2)</li>
    <li>statsmodels(0.14.0)</li>
</ul>

# Using DEDL-Kcr
* Using DEDL-Kcr for identifying Kcr Sites is easy. You can see the examples in this <a href='https://github.com/ghws1/DEDL-Kcr/blob/master/code/demo/demo.ipynb'>notebook</a>.<br>
* If you want to repeat the 20-fold cross-validation experiment, you can execute the file <a href='https://github.com/ghws1/DEDL-Kcr/blob/master/code/experiment/cross_validation.py'>cross_validation.py</a>.
* If you want to repeat the independent test experiment, you can execute the file <a href='https://github.com/ghws1/DEDL-Kcr/blob/master/code/experiment/Independent_test.ipynb'>Independent.ipynb</a>.
* If you want to use your own datasets to train a predictor, you can see the file <a href='https://github.com/ghws1/DEDL-Kcr/blob/master/code/experiment/model_training.py'>model_training.py</a>.


