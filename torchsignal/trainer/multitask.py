import torch
from torch import nn, optim
import numpy as np
import time
import copy
from sklearn.metrics import accuracy_score, f1_score


class Multitask_Trainer(object):

    def __init__(self, model, model_name='model', optimizer=None, criterion=None, scheduler=None, learning_rate=0.001, device='cpu', num_classes=4, multitask_learning=True, patience=10, verbose=False, print_training_metric=False):
        
        SEED = 12
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        self.device = device
        self.verbose = verbose
        self.print_training_metric = print_training_metric
        self.num_classes = num_classes
        self.model_path = '_tmp_models/'+str(model_name)+'.pth'
        self.learning_rate = learning_rate

        self.multitask_learning = multitask_learning

        self.model = model.to(self.device)

        params_to_update = self.get_parameters()

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(params_to_update, lr=learning_rate, weight_decay=0.05)

        if criterion:
            self.criterion = criterion
        else:
            if multitask_learning:
                self.criterion = []

                for i in range(self.num_classes):
                    class_weights = torch.FloatTensor([1, self.num_classes]).to(device)
                    self.criterion.append(nn.CrossEntropyLoss(weight=class_weights))

            else:
                self.criterion = nn.CrossEntropyLoss()

        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience)

    def computes_accuracy(self, outputs, targets, k=1):
        _, preds = outputs.topk(k, 1, True, True)
        preds = preds.t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))
        correct_k = correct[:k].view(-1).float()
        return correct_k

    def get_preds(self, outputs, k=1):
        _, preds = outputs.topk(k, 1, True, True)
        preds = preds.t()
        return preds[0]

    def train(self, data_loader, topk_accuracy):
        self.model.train()
        return self._loop(data_loader, train_mode=True, topk_accuracy=topk_accuracy)

    def validate(self, data_loader, topk_accuracy):
        self.model.eval()
        return self._loop(data_loader, train_mode=False, topk_accuracy=topk_accuracy)

    def get_parameters(self):
        if self.verbose:
            print("Layers with params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if self.verbose:
                    print("\t",name)
        if self.verbose:
            print('\t', len(params_to_update), 'layers')
        return params_to_update

    def fit(self, dataloaders_dict, num_epochs=10, early_stopping=5, topk_accuracy=1, min_num_epoch=10, save_model=False):
        if self.verbose:
            print("-------")
            print("Starting training, on device:", self.device)

        time_fit_start = time.time()
        train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
        early_stopping_counter = early_stopping

        best_epoch_info = {
            'model_wts':copy.deepcopy(self.model.state_dict()),
            'loss':1e10
        }

        for epoch in range(num_epochs):
            time_epoch_start = time.time()

            train_loss, train_acc, train_classification_f1 = self.train(dataloaders_dict['train'], topk_accuracy)
            val_loss, val_acc, val_classification_f1 = self.validate(dataloaders_dict['val'], topk_accuracy)

            train_losses.append(train_loss)
            test_losses.append(val_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(val_acc)

            improvement = False
            if val_loss < best_epoch_info['loss']:
                improvement = True
                best_epoch_info = {
                    'model_wts':copy.deepcopy(self.model.state_dict()),
                    'loss':val_loss,
                    'epoch':epoch,
                    'metrics':{
                        'train_loss':train_loss,
                        'val_loss':val_loss,
                        'train_acc':train_acc,
                        'val_acc':val_acc,
                        'train_classification_f1':train_classification_f1,
                        'val_classification_f1':val_classification_f1
                    }
                }

            if early_stopping and epoch > min_num_epoch:
                if improvement:
                    early_stopping_counter = early_stopping
                else:
                    early_stopping_counter -= 1

                if early_stopping_counter <= 0:
                    if self.verbose:
                        print("Early Stop")
                    break
            if val_loss < 0:
                print('val loss negative')
                break
            
            if self.verbose:
                print("Epoch {:2} in {:.0f}s || Train loss={:.3f}, acc={:.3f}, f1={:.3f} | Val loss={:.3f}, acc={:.3f}, f1={:.3f} | LR={:.1e} | best={} | improvement={}-{}".format(
                    epoch+1,
                    time.time() - time_epoch_start,
                    train_loss,
                    train_acc,
                    train_classification_f1,
                    val_loss,
                    val_acc,
                    val_classification_f1,
                    self.optimizer.param_groups[0]['lr'],
                    int(best_epoch_info['epoch'])+1,
                    improvement,
                    early_stopping_counter)
                )

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
        self.model.load_state_dict(best_epoch_info['model_wts'])

        if self.print_training_metric:
            print()
            time_elapsed = time.time() - time_fit_start
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            print('Epoch with lowest val loss:', best_epoch_info['epoch'])
            for m in best_epoch_info['metrics']:
                print('{}: {:.5f}'.format(m, best_epoch_info['metrics'][m]))
            print()

        if save_model:
            torch.save(self.model.state_dict(), self.model_path)

    def _loop(self, data_loader, train_mode=True, topk_accuracy=1):
        running_loss = 0.0
        running_corrects = 0
        total_data_count = 0
        y_true = []
        y_pred = []

        for X, Y in data_loader:
            inputs = X.to(self.device)

            if self.multitask_learning:
                labels = []
                for i in range(self.num_classes):
                    labels.append( Y.T[i].long().to(self.device) )
            else:
                labels = Y.long().to(self.device)
            
            if train_mode:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            if self.multitask_learning:
                loss = 0
                
                for i in range(self.num_classes):
                    output = outputs[:, i, :]
                    label = labels[i]

                    loss += self.criterion[i](output, label) * output.size(0)

                    label = label.data.cpu().numpy()
                    out = self.get_preds(output, topk_accuracy).cpu().numpy()

                    index_label_1 = np.where(label == 1)

                    if len(index_label_1) > 0:
                        label_1s = label[index_label_1]
                        output_1s = out[index_label_1]

                        y_true.extend(label_1s)
                        y_pred.extend(output_1s)

            else:
                loss = self.criterion(outputs, labels) * outputs.size(0)
                y_true.extend(labels.data.cpu().numpy())
                y_pred.extend(self.get_preds(outputs, topk_accuracy).cpu().numpy())

            running_loss += loss.item() * self.num_classes

            if train_mode:
                loss.backward()
                self.optimizer.step()

        epoch_loss = running_loss / len(y_true)
        epoch_acc = accuracy_score(y_true, y_pred)
        classification_f1 = 0
        if self.multitask_learning:
            classification_f1 = np.round(f1_score(y_true, y_pred), 3)

        return epoch_loss, np.round(epoch_acc.item(), 3), classification_f1
