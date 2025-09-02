# DiConStruct: Causal Concept-based Explanations through Black-Box Distillation

This repository contains the code to reproduce experiments from the paper "DiConStruct: Causal Concept-based Explanations through Black-Box Distillation"


Bibtex citation:

@InProceedings{pmlr-v236-moreira24a,\
  title = 	 {DiConStruct: Causal Concept-based Explanations through Black-Box Distillation},\
  author =       {Moreira, Ricardo Miguel de Oliveira and Bono, Jacopo and Cardoso, M\'ario and Saleiro, Pedro and Figueiredo, M\'ario A. T. and Bizarro, Pedro},\
  booktitle = 	 {Proceedings of the Third Conference on Causal Learning and Reasoning},\
  pages = 	 {740--768},
  year = 	 {2024},
  editor = 	 {Locatello, Francesco and Didelez, Vanessa},
  volume = 	 {236},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {01--03 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v236/moreira24a/moreira24a.pdf},
  url = 	 {https://proceedings.mlr.press/v236/moreira24a.html},
  abstract = 	 {Model interpretability plays a central role in human-AI decision-making systems. Ideally, explanations should be expressed using human-interpretable semantic concepts. Moreover, the causal relations between these concepts should be captured by the explainer to allow for reasoning about the explanations. Lastly, explanation methods should be efficient and not compromise the predictive task performance. Despite the recent rapid advances in AI explainability, as far as we know, no method yet fulfills these three desiderata. Indeed, mainstream methods for local concept explainability do not yield causal explanations and incur a trade-off between explainability and prediction accuracy. We present DiConStruct, an explanation method that is both concept-based and causal, which produces more interpretable local explanations in the form of structural causal models and concept attributions. Our explainer works as a distillation model to any black-box machine learning model by approximating its predictions while producing the respective explanations. Consequently, DiConStruct generates explanations efficiently while not impacting the black-box prediction task. We validate our method on an image dataset and a tabular dataset, showing that DiConStruct approximates the black-box models with higher fidelity than other concept explainability baselines, while providing explanations that include the causal relations between the concepts.}
