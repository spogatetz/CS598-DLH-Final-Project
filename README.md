# CS598-DLH-Final-Project
## Team
Scott Pogatetz  
Yiao Ding
## Dependencies
This project contains a dev container which means if you are using VS Code, you can build this container and all of the dependencies should be automatically installed for you. If you are curious, you can see the exact dependencies in ./devcontainer/devcontainer.json.

## Preprocessing Data
You can use the trunk_builder.ipynb file to generate the trunk data. To do so, you will need to start the Stanford parser by running 

``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000``

in the stanford-corenlp-full-2018-02-27 directory. You can play around with the input and output files to generate the desired data. The data cleaner notebook can then be used to remove missing values.

## Training and Evaluation
You can simply run the model.ipynb to both train and evaluate the model on the data we have provided for you. Feel free to modify the file paths to use your own data.

The model is very compact, so you should have no issue training locally.