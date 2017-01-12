# LSTM with residual connection
A neural network model to generate sanskrit text and learn reprenstations of sankrit aksharas(unit of graphemic symbols in the Indian writing system). 

### Hypothesis: 
Sanskrit is a finely structured language. Minor variations in aksharas change the meaning of the words. Composition and translation of Sanskrit texts is still a challenging  task due to the semantic perplexity in its beautiful and rich verses. Refer to the following examples-
#### Pronouns-
| विभक्ति – Cases | एकवचन – Singular | द्विवचन – Dual | बहुवचन – Plular |
|:--------------:|:----------------:|:-------------:|:--------------:|
|       1.       |       रामः       |      रामौ     |      रामाः     |
|       2.       |        रामं       |      रामौ     |      रामान्     |
|       3.       |       रामेण       |    रामाभ्यां    |      रामैः      |

Here look at rama – अत्र  ***रामं***  पश्य 
&nbsp;
Here look at two ramas – अत्र  ***रामौ***  पश्य 
&nbsp; 
in a rama - ***रामे*** 

#### Sandhi-
***सः पश्यति → स पश्यति 
&nbsp;
< saḥ paśyati → sa paśyati >
&nbsp;
He sees.***


Residual connection[[1]](#resnet) in the model will aid in learning simple aksharas mappings, whereas LSTM[[2]](#lstm) cells will learn complex relationships. 

*Note: This model was trained on Nvidia GeForce GT 730 using Chapter-7 of Mahabharata ~ 677972 characters. If you've more computation resources, you can use bigger training corpus.*

<p align="center">
<img src="model.png">
</p>

### Requirements:
* Keras =< 0.3.3 ([Download](https://github.com/fchollet/keras/releases?after=1.0.3))
* Theano

#### Installation Keras:
```
cd to Keras dir

$ sh
$ sudo python setup.py install
```


### Usage:
`python reslstm.py`

#### To use pretrained weights:
* Remove layers weight intializations.
* Uncomment line 139 in reslstm.py and add your weights.
* Fine-tuning: load given weights and add parameter *trainable = False* for layers which doesn't require fine-tuning.

#### Future Work:
* The residual connection in this model, merges input vectors (binary  vectors) and dense layer features. One can implement element wise addition to check for improvemnts in feature representation with more data. 
* Compare Character level CNN and ResLSTM for Sanskrit language modeling tasks.
* Apply stateful LSTM and compare.



####Citations:
<a id="resnet">
[1]He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. arXiv preprint arXiv:1603.05027.

<a id="lstm">
[2]Graves, A., & Schmidhuber, J. (2005). Framewise phoneme classification with bidirectional LSTM and other neural network architectures. Neural Networks, 18(5), 602-610.


*Please feel free to contact me at ishank269[at]gmail.com to report any issues.*
