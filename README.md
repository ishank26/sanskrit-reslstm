#LSTM with residual connections
A model to generate sanskrit text and learn reprenstations of sankrit aksharas. 

Residual connection will aid in learning simple mappings, whereas LSTM cells learn complex relationships.

*Note: The model was trained using Chapter-7 of Mahabharata on Nvidia GeForce GT 730.*

<p align="center">
<img src="reslstm.png">
</p>

###Requirements:
* Thenao
* Keras

###Usage:
`python reslstm.py`
#####Pretrained weights:
* Remove layers weight intializations.
* Uncomment line 139 and add the weights.
* Fine-tuning: load weights and add *trainable = False* for layers which doesn't require fine-tuning.


