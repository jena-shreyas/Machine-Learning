# Brief Description

- The directory contains the following two sub-directories for Ques 1 and Ques 2 and the requirements.txt and the report:
- Grp_5_Assn_2_Q1:  contains the following files:

    - A2_Q1.py: python code for PCA, k-means and NMI
    - A2_Q1_output.txt: contains the sample output for code
    - wine.data: input dataset file
    - output_plots:
    	- NMI_vs_k.png: shows the variation of NMI for k = 2 to 8
    	- pca.png: shows the scatter plot for PCA in 2D
    	- pca_var_vs_PC.png: shows the cumulative explained variance vs number of PCs
    	
- Grp_5_Assn_2_Q2: contains the following files:
    - A2_Q2.py : python code for Binary SVM and MLP Classifier
    - A2_Q2_output.txt: : contains the model results
    - wine.csv : input dataset file
    - output_plots:
    	- MLP_final_accuracy_vs_learn_rate.png: shows final MLP accuracy vs learning rate
    	- MLP_loss_curve_learn_rate_0.1.png: shows loss curve for best MLP model (256, 16) with learning rate = 0.1
    	- MLP_loss_curve_learn_rate_0.01.png: shows loss curve for best MLP model (256, 16) with learning rate = 0.01
    	- MLP_loss_curve_learn_rate_0.001.png: shows loss curve for best MLP model (256, 16) with learning rate = 0.001    	
    	- MLP_loss_curve_learn_rate_0.0001.png: shows loss curve for best MLP model (256, 16) with learning rate = 0.0001    	
    	- MLP_loss_curve_learn_rate_1e-05.png: shows loss curve for best MLP model (256, 16) with learning rate = 1e-05    	

- requirements.txt
- Grp5_Assn2_Report.pdf

## Install the dependencies

```
pip3 install -r requirements.txt
```

## Running Question 1

```
cd Grp_5_Assn_2_Q1
python3 A2_Q1.py
```

>The corresponding output plots will be generated after execution of the code and saved to ./output_plots .

## Running Question 2

```
cd Grp_5_Assn_2_Q2
python3 A2_Q2.py
```

>The corresponding output plots will be generated after execution of the code and saved to ./output_plots .


