#In[0] Libraries
'''import libraries'''
from scripts import model as mdl


#In[1] ML execution
'''
run or fit ML model
'''

print('Which process would you like to execute? please choose "training" or "deploy":')
phase = input()
print(phase)

if phase not in ['training', 'deploy']:
    raise Exception('Only "training" or "deploy" is allowed, please try again.')
else:
    model = mdl.Model(modeling_phase=phase)

    if phase == 'training':
        print('Enter a model name if you want to fit/refit a model, otherwise input No:')
        model_name = input()
        print(model_name)
        if model_name == 'no' or model_name == 'No':
            pass
        else:
            model.model_tuning(model_name)
    
    model.get_trained_model()


#In[2] Make predictions
'''generate predictions'''
model_predictions = model.model_prediction()


#In[9] Command line execution

'''
import argparse
parser = argparse.ArgumentParser('fit new or get existing model')
parser.add_argument('phase', type=str, help="""Which phase would you like to execute? please choose "training" or "deploy".""", choices=['training', 'deploy'])
args = parser.parse_args()

model = mdl.Model(modeling_phase=phase)

if phase == 'training':
    print('Enter a model name that you want to fit:')
    model_name = input()
    model.model_tuning(model_name)

model.get_trained_model()
'''