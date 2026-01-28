# RefineSegFormer
An improved model of SegFormer

Quantitative evaluation results on the WHU building dataset
Method	P (%)	R (%)	F1 (%)	IoU (%)
RefineSegFormer-B0	96.16	95.97	96.06	92.43
RefineSegFormer-B2	96.94	96.16	96.55	93.33
<img width="350" height="354" alt="image" src="https://github.com/user-attachments/assets/a4aee6a2-ca2f-4078-9d04-cf4274de6100" />


Quantitative evaluation results on the Mandalay in Myanmar
Method	Pixels	P (%)	R (%)	F1 (%)	IoU (%)
	256×256
U-Net    90.75	91.36	91.05	83.57
DeeplabV3+		84.80	91.14	87.85	78.34
PyramidMamba		88.34	89.08	88.71	79.71
SegFormer-B2		91.14	92.56	91.85	84.92
RefineSegFormer-B0		91.42	92.82	92.11	85.38
RefineSegFormer-B2		92.27	93.44	92.85	86.65
	512×512	
U-Net    91.67	88.64	90.13	82.03
DeeplabV3+    84.02	91.61	87.65	78.02
PyramidMamba		86.40	88.35	87.36	77.56
SegFormer-B2		90.59	92.39	91.48	84.30
RefineSegFormer-B0		91.60	93.11	92.35	85.78
RefineSegFormer-B2		92.35	94.05	93.19	87.25
<img width="501" height="309" alt="image" src="https://github.com/user-attachments/assets/3ae1935f-cd60-4e1a-ad93-ced5f0f72b73" />


Quantitative evaluation results on the Dingri County, Tibet, China
Model	P (%)	R (%)	F1 (%)	IoU (%)
U-Net	81.91	87.76	84.73	73.51
DeeplabV3+	67.72	79.09	72.97	57.44
PyramidMamba	63.05	79.66	70.39	54.31
SegFormer-B2	81.52	86.25	83.82	72.15
RefineSegFormer-B0	83.99	87.57	85.69	74.97
RefineSegFormer-B2	89.51	91.57	90.53	82.70
<img width="489" height="303" alt="image" src="https://github.com/user-attachments/assets/637ca43c-101a-4381-aa1d-8ff1e5c50eb8" />
