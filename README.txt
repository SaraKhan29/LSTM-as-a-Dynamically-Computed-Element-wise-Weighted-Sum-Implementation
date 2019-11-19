***** Instructions for running the code********

'''Information about files:'''
ptb folder: contains original Penn Tree Bank dataset
ptb_word_lm.py : maine file
lstm_variants.py: contains various LSTM variations for ablations
results_sample: Contains the sample output of a particular run 
Documentation: Information about LSTM and preliminary results

'''Required libraries''':
python 3.6.4
tensorflow
numpy


'''In order to run it on terminal:'''
run: python ptb_word_lm.py  --data_path=ptb/  
"optional arguments:"
for configuration size: add --small | --medium| --large 
for architecture type: add --arch_type=LSTM | LSTM_SRNN  | LSTM_GATES | LSTM_SRNN_HIDDEN | LSTM_SRNN_OUT
example: python ptb_word_lm.py  --data_path=ptb/  --small --arch_type=LSTM


NOTE: Please ignore the warnings as they arise because of tensorflow version issues
NOTE: The running time is quite slow because of large dataset.  A sample output can be seen in results_sample
