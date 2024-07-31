# Introduction

BioML is an all-in-one library capable of performing the complete machine-learning pipeline consisting of 4 blocks: feature extraction, selection, outlier detection, and model training. You can use the blocks independetly as long as you provide the necessary inputs so if you already has the features you could jump directly to the model training block.

The feature extraction block accepts various biomolecules such as DNA, proteins sequences or structures, RNA and small molecules.

Here is an explanation of the different blocks and the necessary inputs and outputs

# Feature extraction

BioML can extract 3 types of features in the case of protein sequences (in FASTA format):

* Physicochemical features using [Ifeature](https://github.com/Superzchen/iFeature)
* Evolutionary features using [Possum](https://possum.erc.monash.edu/)
* Embeddings from Protein Large Languages from [Hugging Face](https://huggingface.co/models)

The features for the other biomolecules are extracted using [iFeatureOmega](https://github.com/Superzchen/iFeatureOmega-CLI)

In the [examples/test_end_to_end.ipynb](https://github.com/etiur/BioML/blob/main/examples/test_end_to_end.ipynb) I show all the steps necessary to run BioML from end to end look there for updates
