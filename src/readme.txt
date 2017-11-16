To train or predict run: 
python train.py
python predict.py

Our project explores multiple models, hence to change which model to use,
you will need to modify last line of train.py or predict.py:
eg: 
For training:	
	train("fusion", "train.json", "glove/glove.6B.300d.txt","fusion_report.h5","fusion.h5",10,1)
		Need to provide model name, json data location, glove 300D location, 
						previous weight location (if doesn't have, provide None),
						weight to be saved,
						batch size,
						epoch
For predicting:
	predict("fusion", "fusion.h5", "test.json", "glove/glove.6B.300d.txt", "prediction_fusion.csv")
		Need to provide model name, model weights location, json data location,
						glove 300D location, prediction output file

To prevent resource exhausted when training, we use the following batch size (max number possible) 
on GTX-1080 (8gb VMem):
	Fusion: 85 
	DrQA: 256
	CoA-HMN: 50
	CoA-Res: 128
