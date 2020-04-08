from mnmt.inputter import ArgsFeeder
from mnmt.inputter import generate_batch_iterators
from mnmt.translator import Seq2SeqTranslator
from mnmt.alternating_character_table import AlternatingCharacterTable
from mnmt.alternating_character_table import dict_act_path
from mnmt.trainer.utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import pandas as pd


class Trainer:
    data_container: DataContainer

    def __init__(self, args_feeder: ArgsFeeder, model):
        """
        Args:
            args_feeder (ArgsFeeder):
            model: the NMT model
        """
        self.args_feeder = args_feeder
        # init train.log
        self.train_log_path = "experiments/exp{}/train.log".format(self.args_feeder.exp_num)
        # init model
        self.model = model

        log_print(self.train_log_path, model.apply(init_weights))
        self.num_params = count_parameters(self.model)

        self.optimizer = getattr(optim, args_feeder.optim_choice)(model.parameters(), lr=args_feeder.learning_rate)

        # learning rate scheduler
        if args_feeder.valid_criterion == 'ACC':
            self.decay_mode = 'max'  # decay when less than maximum
        elif args_feeder.valid_criterion == 'LOSS':
            self.decay_mode = 'min'  # decay when more than minimum
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode=self.decay_mode, factor=args_feeder.lr_decay_factor,  # 0.9 in paper
            patience=args_feeder.decay_patience)

        # evaluation memory bank
        class EvalMemoryBank:
            def __init__(self, best_valid_loss=float('inf'), acc_valid_loss=float('inf'),
                         best_valid_acc=float(-1), best_valid_epoch=float(-1), best_train_step=float(-1),
                         early_stopping_patience=args_feeder.early_stopping_patience,
                         best_valid_loss_aux=float('inf'), best_valid_acc_aux=float(-1)):
                self.best_valid_loss = best_valid_loss
                self.acc_valid_loss = acc_valid_loss
                self.best_valid_acc = best_valid_acc
                self.best_valid_epoch = best_valid_epoch
                self.best_train_step = best_train_step
                self.early_stopping_patience = early_stopping_patience
                self.best_valid_loss_aux = best_valid_loss_aux
                self.best_valid_acc_aux = best_valid_acc_aux

        self.eval_memory_bank = EvalMemoryBank()
        # to recover full patience when improving
        self.early_stopping_patience = args_feeder.early_stopping_patience

        # training memory bank
        class TrainMemoryBank:
            def __init__(self, exp_num=args_feeder.exp_num,
                         total_epochs=args_feeder.total_epochs,
                         n_epoch=0, n_steps=0,
                         report_interval=args_feeder.report_interval):
                self.exp_num = exp_num
                self.total_epochs = total_epochs
                self.n_epoch = n_epoch
                self.n_steps = n_steps
                self.report_interval = report_interval

        self.train_memory_bank = TrainMemoryBank()

        # single or multi task
        self.multi_task_ratio = args_feeder.multi_task_ratio
        if self.multi_task_ratio == 1:
            log_print(self.train_log_path, "Running single-main-task experiment...")
            self.task = "Single-Main"
            self.FLAG = "main-task (single)"
        elif self.multi_task_ratio == 0:
            log_print(self.train_log_path, "Running single-auxiliary-task experiment...")
            self.task = "Single-Auxiliary"
            self.FLAG = "aux-task (single)"
        else:
            log_print(self.train_log_path, "Running multi-task experiment...")
            self.task = "Multi"
            self.FLAG = "main-task (multi)"

        # data
        self.data_container = args_feeder.data_container
        self.train_iter, self.valid_iter, self.test_iter = generate_batch_iterators(self.data_container,
                                                                                    self.args_feeder.batch_size,
                                                                                    self.args_feeder.device,
                                                                                    src_lang=self.args_feeder.src_lang)
        for (name, field) in self.data_container.fields:
            if name == self.args_feeder.src_lang:
                self.src_field = field
            elif name == self.args_feeder.trg_lang:
                self.trg_field = field
            elif name == self.args_feeder.auxiliary_name:
                self.auxiliary_field = field

        # teacher forcing
        self.tfr = 0.8

        # loss function
        self.loss_function = self.construct_loss_function()

        # translator
        self.translator = Seq2SeqTranslator(self.args_feeder.quiet_translate)

    def run(self, burning_epoch):
        try:
            for epoch in range(self.train_memory_bank.total_epochs):
                self.train_memory_bank.n_epoch = epoch
                # apply nothing during the burning phase, recall Bayesian Modelling
                if epoch <= burning_epoch:
                    log_print(self.train_log_path, "Renew Evaluation Records in the Burning Phase...")
                    # abandon the best checkpoint in early stage
                    self.eval_memory_bank.best_valid_loss = float('inf')
                    self.eval_memory_bank.best_valid_acc = 0
                    self.eval_memory_bank.early_stopping_patience = self.early_stopping_patience

                if self.eval_memory_bank.early_stopping_patience == 0:
                    log_print(self.train_log_path, "Early Stopping!")
                    break

                start_time = time.time()

                self.tfr = max(1 - (float(10 + epoch * 1.5) / 50), 0.2)
                train_loss = self.train()
                valid_loss, valid_acc, valid_acc_aux = self.evaluate(is_test=False)

                end_time = time.time()

                epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

                self.update(valid_loss, valid_acc)
                if self.task == "Multi":
                    self.update_aux(valid_acc_aux)

                self.scheduler.step(valid_acc)  # update learning rate

                log_print(self.train_log_path,
                          f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
                log_print(self.train_log_path,
                          f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
                log_print(self.train_log_path, f'\t Val. Loss: {valid_loss:.3f} | '
                                               f'Val. Acc: {valid_acc:.3f} | '
                                               f'Val. PPL: {math.exp(valid_loss):7.3f}')
        except KeyboardInterrupt:
            log_print(self.train_log_path, "Exiting loop")

    @staticmethod
    def epoch_time(start_time, end_time):
        """
        Args:
            start_time:
            end_time:
        """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def update(self, valid_loss, valid_acc):
        """
        Args:
            valid_loss: current validation loss
            valid_acc: current validation accuracy
        """
        valid_criterion = self.args_feeder.valid_criterion
        assert valid_criterion in ['LOSS', 'ACC']
        log_print(self.train_log_path, "\n---------------------------------------")
        log_print(self.train_log_path, "[Epoch: {}][Validatiing...]".format(self.train_memory_bank.n_epoch))

        # For Validation Loss
        if valid_loss <= self.eval_memory_bank.best_valid_loss:
            log_print(self.train_log_path, '\t\t Better Valid Loss! (at least equal)')
            self.eval_memory_bank.best_valid_loss = valid_loss
            if valid_criterion == 'LOSS':
                torch.save(self.model.state_dict(),
                           'experiments/exp' + str(self.train_memory_bank.exp_num) + '/loss-model-seq2seq.pt')
            # restore full patience if obtain new minimum of the loss
            self.eval_memory_bank.early_stopping_patience = self.early_stopping_patience
        else:
            self.eval_memory_bank.early_stopping_patience = \
                max(self.eval_memory_bank.early_stopping_patience - 1, 0)  # cannot be lower than 0
        # For Validation Accuracy
        if valid_acc >= self.eval_memory_bank.best_valid_acc:
            log_print(self.train_log_path, '\t\t Better Valid Acc! (at least equal)')
            self.eval_memory_bank.best_valid_acc = valid_acc
            self.eval_memory_bank.acc_valid_loss = valid_loss
            self.eval_memory_bank.best_valid_epoch = self.train_memory_bank.n_epoch
            self.eval_memory_bank.best_train_step = self.train_memory_bank.n_steps
            if valid_criterion == 'ACC':
                torch.save(self.model.state_dict(),
                           'experiments/exp' + str(self.train_memory_bank.exp_num) + '/acc-model-seq2seq.pt')
        log_print(self.train_log_path,
                  f'\t Early Stopping Patience: '
                  f'{self.eval_memory_bank.early_stopping_patience}/{self.early_stopping_patience}')
        log_print(self.train_log_path,
                  f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')
        log_print(self.train_log_path,
                  f'\t BEST. Val. Loss: {self.eval_memory_bank.best_valid_loss:.3f} | '
                  f'BEST. Val. Acc: {self.eval_memory_bank.best_valid_acc:.3f} | '
                  f'Val. Loss: {self.eval_memory_bank.acc_valid_loss:.3f} | '
                  f'BEST. Val. Epoch: {self.eval_memory_bank.best_valid_epoch} | '
                  f'BEST. Val. Step: {self.eval_memory_bank.best_train_step}')
        log_print(self.train_log_path, "---------------------------------------\n")

    def update_aux(self, valid_acc_aux):
        if valid_acc_aux >= self.eval_memory_bank.best_valid_acc_aux:
            self.eval_memory_bank.best_valid_acc_aux = valid_acc_aux
            log_print(self.train_log_path, '\t\t Better Valid Acc on Auxiliary Task! (at least equal)')
        log_print(self.train_log_path, f'\tBEST. Val. Acc Aux: {self.eval_memory_bank.best_valid_acc_aux}')
        log_print(self.train_log_path, "---------------------------------------\n")

    @staticmethod
    def fix_output_n_trg(output, trg):
        """Remove first column because they are <sos> symbols
        Args:
            output: [trg len, batch size, output dim]
            trg: [trg len, batch size]
        """
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)  # [(trg len - 1) * batch size, output dim]
        trg = trg[1:].view(-1)  # [(trg len - 1) * batch size]
        return output, trg

    def construct_loss_function(self):
        loss_criterion = nn.CrossEntropyLoss(ignore_index=self.args_feeder.trg_pad_idx)
        if self.task == "Multi":
            return lambda output, output_aux, trg, trg_aux: \
                (self.multi_task_ratio * loss_criterion(output, trg)) + \
                ((1 - self.multi_task_ratio) * loss_criterion(output_aux, trg_aux))
        else:
            return loss_criterion

    def compute_loss(self, output, trg):
        if isinstance(output, tuple) and isinstance(trg, tuple):
            assert self.task == "Multi"
            output, output_aux, trg, trg_aux = output[0], output[1], trg[0], trg[1]
            output, trg = self.fix_output_n_trg(output, trg)
            output_aux, trg_aux = self.fix_output_n_trg(output_aux, trg_aux)
            return self.loss_function(output, output_aux, trg, trg_aux)
        else:
            output, trg = self.fix_output_n_trg(output, trg)
            return self.loss_function(output, trg)

    def train(self):

        self.model.train()
        self.model.teacher_forcing_ratio = self.tfr
        log_print(self.train_log_path,
                  "[Train]: Current Teacher Forcing Ratio: {:.3f}".format(self.model.teacher_forcing_ratio))

        epoch_loss = 0

        for i, batch in enumerate(self.train_iter):

            src, src_lens = getattr(batch, self.args_feeder.src_lang)
            trg, trg_lens = getattr(batch, self.args_feeder.trg_lang)

            self.optimizer.zero_grad()

            if self.task == 'Multi':
                trg_aux, trg_lens_aux = getattr(batch, self.args_feeder.auxiliary_name)
                output, output_aux = self.model(src, src_lens, trg, trg_aux)
                loss = self.compute_loss((output, output_aux), (trg, trg_aux))
            else:
                output = self.model(src, src_lens, trg)
                loss = self.compute_loss(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)  # clip = 1
            self.optimizer.step()

            epoch_loss += loss.item()
            running_loss = epoch_loss / (i + 1)

            self.train_memory_bank.n_steps += 1

            # print every ${report_interval} batches (${report_interval} steps)
            if self.train_memory_bank.n_steps % self.train_memory_bank.report_interval == 0:

                lr = -1
                for param_group in self.optimizer.param_groups:
                    lr = param_group['lr']
                n_examples = len(self.data_container.dataset['train'].examples)
                log_print(self.train_log_path, '[Epoch: {}][#examples: {}/{}][#steps: {}]'.format(
                    self.train_memory_bank.n_epoch,
                    (i + 1) * self.args_feeder.batch_size,
                    n_examples,
                    self.train_memory_bank.n_steps))
                log_print(self.train_log_path, f'\tTrain Loss: {running_loss:.3f} | '
                                               f'Train PPL: {math.exp(running_loss):7.3f} '
                                               f'| lr: {lr:.3e}')

                # eval the validation set for every * steps
                if (self.train_memory_bank.n_steps % (10 * self.train_memory_bank.report_interval)) == 0:
                    log_print(self.train_log_path, '-----Val------')
                    valid_loss, valid_acc, valid_acc_aux = self.evaluate(is_test=False)
                    log_print(self.train_log_path, '-----Tst------')
                    self.evaluate(is_test=True)

                    self.update(valid_loss, valid_acc)
                    if self.task == 'Multi':
                        self.update_aux(valid_acc_aux)
                    self.scheduler.step(valid_acc)  # scheduled on validation acc
                    self.model.train()

        return epoch_loss / len(self.train_iter)

    def evaluate(self, is_test=False, beam_size=1, output_file=None):

        self.model.eval()
        self.model.teacher_forcing_ratio = 0  # turn off teacher forcing

        epoch_loss = 0
        correct = 0
        correct_aux = 0
        iterator = self.valid_iter if not is_test else self.test_iter

        with torch.no_grad():

            for i, batch in enumerate(iterator):

                src, src_lens = getattr(batch, self.args_feeder.src_lang)
                trg, trg_lens = getattr(batch, self.args_feeder.trg_lang)

                if self.task == 'Multi':
                    trg_aux, trg_lens_aux = getattr(batch, self.args_feeder.auxiliary_name)
                    output, output_aux = self.model(src, src_lens, trg, trg_aux)
                    loss = self.compute_loss((output, output_aux), (trg, trg_aux))
                    correct_aux += self.translator.translate(output_aux, trg_aux, trg_field=self.auxiliary_field,
                                                             beam_size=beam_size)
                else:
                    output = self.model(src, src_lens, trg)
                    loss = self.compute_loss(output, trg)
                epoch_loss += loss.item()

                # compute acc through seq2seq translation
                correct += self.translator.translate(output, trg, trg_field=self.trg_field,
                                                     beam_size=beam_size, output_file=output_file)

            epoch_loss = epoch_loss / len(iterator)

            n_examples = len(self.data_container.dataset['valid'].examples) if not is_test \
                else len(self.data_container.dataset['test'].examples)

            flag = "TEST" if is_test else "VAL"
            log_print(self.train_log_path, '[{}]: The number of correct predictions ({}): {}/{}'
                      .format(flag, correct, self.FLAG, n_examples))
            if self.task == 'Multi':
                log_print(self.train_log_path, '[{}]: The number of correct predictions (aux-task (multi)): {}/{}'
                          .format(flag, correct_aux, n_examples))

            acc = correct / n_examples
            acc_aux = correct_aux / n_examples  # if single-task, then just zero

            self.model.teacher_forcing_ratio = self.tfr  # restore teacher-forcing ratio

        return epoch_loss, acc, acc_aux

    def load_best_model(self):
        self.model.load_state_dict(torch.load('experiments/exp' +
                                              str(self.args_feeder.exp_num) + '/acc-model-seq2seq.pt'))

    def best_model_output(self, enable_acc_act=True, enable_acc_multi=False):

        self.load_best_model()

        # evaluate val set
        f = open(self.args_feeder.valid_out_path, 'w')
        f.write("PRED\tREF\n")
        valid_loss, valid_acc, valid_acc_aux = self.evaluate(is_test=False, output_file=f)
        f.close()

        # evaluate tst set
        f = open(self.args_feeder.test_out_path, 'w')
        f.write("PRED\tREF\n")
        test_loss, test_acc, test_acc_aux = self.evaluate(is_test=True, output_file=f)
        f.close()

        # save model settings
        with open("experiments/exp{}/settings".format(self.args_feeder.exp_num), "w+") as f:
            f.write("Task\t{}\n".format(self.task))
            f.write("MTR\t{}\n".format(self.multi_task_ratio))
            f.write("#Params\t{}\n".format(self.num_params))

        # save evaluation results
        eval_results = pd.DataFrame(columns=["Loss", "ACC"], index=["Valid", "Test"])
        eval_results["Loss"] = [valid_loss, test_loss]
        eval_results["ACC"] = [valid_acc, test_acc]
        if self.task == 'Multi':
            eval_results["ACC-aux"] = [valid_acc_aux, test_acc_aux]
        if enable_acc_act:
            act = AlternatingCharacterTable(act_path=dict_act_path)
            valid_out = act.tsv_to_df(self.args_feeder.valid_out_path)
            test_out = act.tsv_to_df(self.args_feeder.test_out_path)
            results_valid = act.compute_ACC_ACT(valid_out)
            results_test = act.compute_ACC_ACT(test_out)
            eval_results["ACC-ACT"] = [results_valid["acc-act"], results_test["acc-act"]]
            eval_results["Replaced"] = [results_valid["replaced"], results_test["replaced"]]
        if enable_acc_multi:
            ...
        print(eval_results)
        eval_results.to_csv("experiments/exp" + str(self.args_feeder.exp_num) + "/eval.results", sep="\t")
