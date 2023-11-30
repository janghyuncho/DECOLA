# DECOLA Abstract

<p align="center"> <img src='../figs/pipeline_light.png' align="center" width="70%"> </p>

## Abstract
We present a new open-vocabulary detection framework. 
Our framework uses both image-level labels and detailed detection annotations when available.
Our framework proceeds in three steps. 
We first train a language-conditioned object detector on fully-supervised detection data. 
This detector gets to see the presence or absence of ground truth classes during training, and conditions prediction on the set of present classes. 
We use this detector to pseudo-label images with image-level labels. 
Our detector provides much more accurate pseudo-labels than prior approaches with its conditioning mechanism. 
Finally, we train an unconditioned open-vocabulary detector on the pseudo-annotated images. 
The resulting detector, named \ours, shows strong zero-shot performance in open-vocabulary LVIS benchmark as well as direct zero-shot transfer benchmarks on LVIS, COCO, Object365, and OpenImages. 
\ours outperforms the prior arts by \textbf{17.1} AP$_\text{rare}$ and \textbf{9.4} mAP on zero-shot LVIS benchmark.
\ours achieves state-of-the-art results in various model sizes, architectures, and datasets by only training on open-sourced data and academic-scale computing. 
