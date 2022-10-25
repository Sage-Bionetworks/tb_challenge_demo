### Write inference code here
import onnxruntime as rt
import pandas as pd
import numpy as np

##Dummy example
def prediction(input_path, model_path):
	
	input_df = pd.read_csv(input_path)
	model_inference = rt.InferenceSession(model_path)

	onnx_pred = model_inference.run(['output'], {"input": input_df})
	pred = np.argmax(onnx_pred[0], axis=1)

	df_pred = pd.DataFrame(input_df['StudyID'], columns=['participant']) #Paitent id
	df_pred['probability'] = pred
	df_pred.to_csv('/output/predictions.csv', index=False)

if __name__ == '__main__':
    
    input_path = '/input/*' #holdout sample file path
    model_path = 'model/*.onnx' #model path

	prediction(input_path, model_path)
