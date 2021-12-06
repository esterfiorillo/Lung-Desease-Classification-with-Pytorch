from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
from torchvision.utils import save_image
from sklearn.metrics import confusion_matrix
import metrics


def train_model(model, dataloaders, criterion, optimizer, device, num_classes, scheduler, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_balanced_acc = 0.0
    output_dir = "./outputs/"
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=open(str(output_dir + "output.txt"), "a"))
        print('-' * 10, file=open(str(output_dir + "output.txt"), "a"))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            incorrect_path = []
            cm_total = np.zeros((num_classes, num_classes))
            num_iter_per_epoch = len(dataloaders[phase])
            scheduler.step()

            # Iterate over data.
            for iteration, (inputs, labels, inputs_path) in enumerate(dataloaders[phase]):  
                iteration_step = num_iter_per_epoch*epoch + iteration 
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #save_image(inputs, 'output1/' + str(iteration)+ '.png')

                # zero the parameter gradients
                optimizer.zero_grad()

  
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                        
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                    #get the wrong classification samples
                    elif phase == 'val':
                        mask_incorrect = preds != labels.data
                        for idx, k in enumerate(list(mask_incorrect)):
                            if k:
                                incorrect_path.append(inputs_path[idx])

              
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                cm_train = confusion_matrix(labels.data.cpu(), preds.cpu(), labels=range(num_classes))
                cm_total += cm_train
                
                mean_acc = metrics.balanced_accuracy_score_from_cm(cm_train)
                f1_score = metrics.f1_score_from_cm(cm_train)

                if phase == 'train':

                    print('{} Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Acc: {:.4f} CM: \n{}'.format(phase,
                            epoch, num_epochs, 
                            iteration, len(dataloaders[phase]),
                            loss.item(), mean_acc, cm_train), file=open(str(output_dir + "output.txt"), "a"))


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            balanced_epoch_acc = metrics.balanced_accuracy_score_from_cm(cm_total)
            epoch_f1, epoch_precision, epoch_recall = metrics.f1_score_from_cm(cm_total)

            print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Balanced_Acc:{:.4f} CM: \n{}'.format(phase, epoch_loss, epoch_acc, epoch_f1, balanced_epoch_acc, cm_total), file=open(str(output_dir + "output.txt"), "a"))
            print('Precision: ', epoch_precision, file=open(str(output_dir + "output.txt"), "a"))
            print ('Recall: ',  epoch_recall, file=open(str(output_dir + "output.txt"), "a"))


            # deep copy the model
            if phase == 'val' and balanced_epoch_acc > best_balanced_acc:
                best_acc = epoch_acc
                best_balanced_acc = balanced_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_cm = cm_total
                best_f1 = epoch_f1
                best_precision = epoch_precision
                best_recall = epoch_recall
                best_incorrect = incorrect_path
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=open(str(output_dir + "output.txt"), "a"))
    print('Best val Acc: {:4f}'.format(best_acc), file=open(str(output_dir + "output.txt"), "a"))
    print('Best val Balanced Acc: {:4f}'.format(best_balanced_acc), file=open(str(output_dir + "output.txt"), "a"))
    print('Best val F1: {:4f}'.format(best_f1), file=open(str(output_dir + "output.txt"), "a"))
    print('Best val precision: {}'.format(best_precision), file=open(str(output_dir + "output.txt"), "a"))
    print('Best val recall: {}'.format(best_recall), file=open(str(output_dir + "output.txt"), "a"))
    print('Best CM:\n{0}'.format(best_cm), file=open(str(output_dir + "output.txt"), "a"))

    print(best_incorrect, file=open(str(output_dir + "incorrect_results_30.txt"), "a"))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

    