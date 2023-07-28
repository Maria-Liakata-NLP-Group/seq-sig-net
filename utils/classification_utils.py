from os import listdir
from sklearn import metrics
import torch
import numpy as np
import pandas as pd
import random
import pickle

class Splits:

    def __init__(self, num_folds=1):
        self.NUM_folds = num_folds


    def get_labels(self, df):
        #dictionary of labels - 3-class classification
        y_dict3 = {}
        y_dict3['0'] = 0
        y_dict3['IE'] = 1
        y_dict3['IEP'] = 1
        y_dict3['IS'] = 2
        y_dict3['ISB'] = 2

        #GET THE FLAT y LABELS
        y_data = df['label'].values
        y_data = np.array([y_dict3[xi] for xi in y_data])
        y_data = torch.from_numpy(y_data.astype(int))
        
        return y_data


    def get_reddit_splits(df, x_data, y_data):

        # Just getting the train/test data: timelines_ids, posts_id, texts, labels
        test_tl_ids = df[(df.train_or_test =='test')].timeline_id.unique()
        train_tl_ids = df[(df.train_or_test !='test') & (df.fold !=0)].timeline_id.unique()
        valid_tl_ids = df[(df.fold ==0)].timeline_id.unique()

        timeline_test = np.unique(test_tl_ids)
        timeline_train = np.unique(train_tl_ids)
        timeline_valid = np.unique(valid_tl_ids)

        x_test = x_data[(df.timeline_id.isin(timeline_test)) , :]
        y_test = y_data[df.timeline_id.isin(timeline_test)]
        x_valid = x_data[df.timeline_id.isin(timeline_valid), :]
        y_valid = y_data[df.timeline_id.isin(timeline_valid)]
        x_train = x_data[df.timeline_id.isin(timeline_train), :]
        y_train = y_data[df.timeline_id.isin(timeline_train)]

        test_pids_ = np.array(df[(df.timeline_id.isin(timeline_test))]['postid'].tolist())
        test_pids_ = test_pids_.reshape(test_pids_.shape[0],1)

        #print('The size of train/valid/test timelines are: ', timeline_train.shape[0], timeline_valid.shape[0], timeline_test.shape[0])
        #print('Samples in test set: ', x_test.shape[0])

        return x_test, y_test, x_valid, y_valid, x_train , y_train, test_tl_ids, test_pids_


def set_seed(seed):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation(model, valid_loader, criterion, loss):

    model.eval()
    loss_total = 0

     # Calculate Metrics         
    correct = 0
    total = 0
    labels_all = torch.empty((0))
    predicted_all = torch.empty((0))

    # Validation data
    with torch.no_grad():     
      # Iterate through validation dataset
        for emb_v, labels_v in valid_loader:

            # Forward pass only to get logits/output
            outputs = model(emb_v)

            # Get predictions from the maximum value
            if loss == 'cross_entropy':
                outputs_softmax = torch.log_softmax(outputs, dim = 1)
                _, predicted_v = torch.max(outputs_softmax, dim = 1) 
            else:
                _, predicted_v = torch.max(outputs.data, 1)
                
            loss_v = criterion(outputs, labels_v)
            loss_total += loss_v.item()

            # Total number of labels
            total += labels_v.size(0)

            # Total correct predictions
            correct += (predicted_v == labels_v).sum()
            labels_all = torch.cat([labels_all, labels_v])
            predicted_all = torch.cat([predicted_all, predicted_v])

        accuracy = 100 * correct / total
        f1_v = 100 * metrics.f1_score(labels_all, predicted_all, average = 'macro')

        # Print Loss
        #print('Iteration: {}. Loss: {}. Accuracy: {}. Macro-Precision: {}'.format(iter, loss.item(), accuracy, precision))
        return loss_total / len(valid_loader), f1_v, labels_all, predicted_all


def training(model, train_loader, criterion, optimizer, epoch, num_epochs):
    model.train()
    
    for i, (emb, labels) in enumerate(train_loader):
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(emb)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        # Show training progress
        if (i % 100 == 0):
            print('[{}/{}, {}/{}] loss: {:.8}'.format(epoch, num_epochs, i, len(train_loader), loss.item()))


def testing(model, test_loader, loss):
      model.eval()

      labels_all = torch.empty((0))
      predicted_all = torch.empty((0))

      #Test data
      with torch.no_grad():     
            # Iterate through test dataset
            for emb_t, labels_t in test_loader:

                  # Forward pass only to get logits/output
                  outputs_t = model(emb_t)
                  
                  # Get predictions from the maximum value
                  if (loss == 'cross_entropy'):
                    outputs_t_softmax = torch.log_softmax(outputs_t, dim = 1)
                    _, predicted_t = torch.max(outputs_t_softmax, dim = 1) 
                  else:
                    _, predicted_t = torch.max(outputs_t.data, 1)

                  # Total correct predictions
                  labels_all = torch.cat([labels_all, labels_t])
                  predicted_all = torch.cat([predicted_all, predicted_t])
      
      return predicted_all, labels_all


def process_model_results(model_code_name, FOLDER_results,type='Talklife'):
    if type=='Talklife':
        per_model_files = [f for f in listdir(FOLDER_results) if model_code_name in f if 'tuning' not in f if 'Reddit' not in f]
    else:
        per_model_files = [f for f in listdir(FOLDER_results) if model_code_name in f if 'tuning' not in f]

    print('There are ', len(per_model_files), ' files')
    metrics_overall = pd.DataFrame(0, index = ['O', 'IE', 'IS', 'accuracy', 'macro avg', 'weighted avg'], columns = ['precision', 'recall', 'f1-score', 'support'])
    with open(FOLDER_results+per_model_files[0], 'rb') as fin:
        results0 = pickle.load(fin)


    for my_ran_seed in results0['classifier_params']['RANDOM_SEED_list']:
        labels_final = torch.empty((0))
        predicted_final = torch.empty((0))

        seed_files = [f for f in per_model_files if (str(my_ran_seed)+'seed') in f]
        for sf in seed_files :
            with open(FOLDER_results+sf, 'rb') as fin:
                results = pickle.load(fin)
                labels_results = results['labels']
                predictions_results = results['predictions']
        
            #for each seed combine fold results
            labels_final = torch.cat([labels_final, labels_results])
            predicted_final = torch.cat([predicted_final, predictions_results])

        #calculate metrics for each seed
        metrics_tab = metrics.classification_report(labels_final, predicted_final, target_names = ['O','IE','IS'], output_dict=True)
        metrics_tab = pd.DataFrame(metrics_tab).transpose()
        #combine the metrics with the rest of the seeds in order to take average at the end
        metrics_overall += metrics_tab

    return metrics_overall /len(results0['classifier_params']['RANDOM_SEED_list'])


def process_model_val_results(model_code_name, FOLDER_results, type='Talklife'):
    if type=='Talklife':
        per_model_files = [f for f in listdir(FOLDER_results) if model_code_name in f if 'tuning' not in f if 'Reddit' not in f]
    else:        
        per_model_files = [f for f in listdir(FOLDER_results) if model_code_name in f if 'tuning' not in f]

    print('There are ', len(per_model_files), ' files')
    metrics_overall = pd.DataFrame(0, index = ['O', 'IE', 'IS', 'accuracy', 'macro avg', 'weighted avg'], columns = ['precision', 'recall', 'f1-score', 'support'])
    with open(FOLDER_results+per_model_files[0], 'rb') as fin:
        results0 = pickle.load(fin)


    for my_ran_seed in results0['classifier_params']['RANDOM_SEED_list']:
        labels_final = torch.empty((0))
        predicted_final = torch.empty((0))

        seed_files = [f for f in per_model_files if (str(my_ran_seed)+'seed') in f]
        for sf in seed_files :
            with open(FOLDER_results+sf, 'rb') as fin:
                results = pickle.load(fin)
                labels_results = results['labels_val']
                predictions_results = results['predicted_val']
        
            #for each seed combine fold results
            labels_final = torch.cat([labels_final, labels_results])
            predicted_final = torch.cat([predicted_final, predictions_results])

        #calculate metrics for each seed
        metrics_tab = metrics.classification_report(labels_final, predicted_final, target_names = ['O','IE','IS'], output_dict=True)
        metrics_tab = pd.DataFrame(metrics_tab).transpose()
        #combine the metrics with the rest of the seeds in order to take average at the end
        metrics_overall += metrics_tab

    return metrics_overall /len(results0['classifier_params']['RANDOM_SEED_list'])