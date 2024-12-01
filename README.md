# POVa Project
Authors of the project:
* Malika Baratova
* Thaina Helena de Oliveira Alves
* Ian Caruana 

## Organisational information
* cv-project.ipynb : Kaggle notebook where the final model was trained and OCR tests were conducted to compare the accuracy of PaddleOCR, EasyOCR and Tesseract.
* best.pt : Best weights of final model.
* dataset.yaml : Definition of dataset configuration.
* results.csv : results of training of the final model.

* /hyperparameter_tuning : directory containing all files relating to the attempts made at tuning hyperparameters via Ultralytics' tune.model function.
  * model_hyperparameter_tuning : Kaggle notebook where first tuning was conducted (6 iterations, 25 epochs, no optimiser explicitly defined).
  * results_25epochs.png: Visual results summary of the best weights found during tuning (in iteration 5).
  * tune_results.csv: Summary of results of tuning for each iteration.

  * model_hyperparameter_tuning_2 : Kaggle notebook where second tuning was conducted (6 iterations, 25 epochs, optimiser set to AdamW).
  * results_25epochs_2.png: Visual results summary of the best weights found during tuning (in iteration 2).
  * tune_results_2.csv: Summary of results of tuning for each iteration.
