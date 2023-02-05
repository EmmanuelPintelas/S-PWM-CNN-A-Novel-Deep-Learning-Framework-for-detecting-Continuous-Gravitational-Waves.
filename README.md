# S-PWM-CNN-A-Novel-Deep-Learning-Framework-for-detecting-Continuous-Gravitational-Waves.


More specifically, the main objective of this work, is to propose a novel C-GW prediction framework called:             
S-PWM-CNN which will be powerful enough to efficiently detect possible signs of C-GWs. In brief, our proposed framework is based on three main components: 

•	The PWM-CNN (Power spectrum Window Mean - CNN) which aims to create a more robust and clear representation form, comparing to the initial one (baseline approach; initial representation form of LIGO laboratory), in order to feed a CNN model by also incorporating a sophisticated augmentation method called, SpecAugm [20] (was initially proposed and applied on automatic Speech and Audio Recognition). 

•	The S-ML (Statistical - ML) which aims to extract statistical features, in order to exploit additional information lied into the initial form, feeding them into a machine learning model.  

•	The Ensemble component, which aims to efficiently combine these two models in order to extract the final predictions.  Our proposed framework significantly outperformed the baseline approaches, for every applied experimental configuration (e.g., CNN model selection) revealing the efficiency of the proposed approach.
