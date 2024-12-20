import numpy as np
import pandas as pd
import math
import time
#import jax.numpy as np
from scipy.stats import ks_2samp, ttest_ind
import time
import multiprocessing

class mia(object):

    def __init__(self, shadow_train_performance, shadow_train_performance_y, shadow_test_performance, shadow_test_performance_y, query_output, query_output_y, num_classes=10,
                  x_target=None, y_target=None,x_shadow=None, y_shadow=None, target_train_performance=None, target_test_performance=None,):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        '''

        self.num_classes = num_classes
        self.x_target = x_target
        self.y_target = y_target
        self.x_shadow = x_shadow
        self.y_shadow = y_shadow

        # confidence vector
        start = time.time()
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance, shadow_train_performance_y
        self.s_te_outputs, self.s_te_labels = shadow_test_performance, shadow_test_performance_y
        self.q_outputs, self.q_labels =  query_output, query_output_y
        # self.t_tr_outputs, self.t_tr_labels = target_train_performance
        # self.t_te_outputs, self.t_te_labels = target_test_performance
        end = time.time()
        #print(f'>> time cost: {end-start}')

        # label matching (label-only)
        start = time.time()
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.q_corr = (np.argmax(self.q_outputs, axis=1)==self.q_labels).astype(int)
        # self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        # self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        end = time.time()
        #print(f'>> time cost: {end - start}')

        start = time.time()
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.q_conf = np.array([self.q_outputs[i, self.q_labels[i]] for i in range(len(self.q_labels))])
        # self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        # self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        end = time.time()
        #print(f'>> time cost: {end - start}')

        # entropy
        start = time.time()
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.q_entr = self._entr_comp(self.q_outputs)
        # print(self.q_entr)
        # print(self.q_outputs)
        # self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        # self.t_te_entr = self._entr_comp(self.t_te_outputs)
        end = time.time()
        #print(f'>> time cost: {end - start}')

        # modified entropy
        start = time.time()
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.q_m_entr = self._m_entr_comp(self.q_outputs, self.q_labels)
        # self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        # self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        end = time.time()
        #print(f'>> time cost: {end - start}')

        # print(self.s_tr_conf.shape, self.s_tr_entr.shape)


    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

    def _thre_setting(self, tr_values, te_values):
        """Decide the threshold for membership inference attacks using the shadow data

        Args:
            tr_values ([float]): a list of values for shadow train set
            te_values ([float]): a list of values for shadow test set

        Returns:
            float: the threshold
        """
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return mem_inf_acc, t_tr_acc, t_te_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, q_values, t_tr_values=None, t_te_values=None, return_acc=False, thre_given = None):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        if thre_given is not None:
            thres = thre_given
        else:
            thres = []
        bin_res_cal_train = np.zeros(len(s_tr_values))
        bin_res_cal_test = np.zeros(len(s_te_values))
        bin_res_query = np.zeros(len(q_values))
        for num in range(self.num_classes):
            if thre_given is None:
                thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
                thres.append(thre)
            else:
                thre = thres[num]

            bin_res_cal_train[np.multiply((self.s_tr_labels==num), (s_tr_values >= thre))] = 1
            bin_res_cal_test[np.multiply((self.s_te_labels==num), (s_te_values >= thre))] = 1
            bin_res_query[np.multiply((np.argmax(self.q_outputs, axis=1)==num), (q_values >= thre))] = 1   ## use the predictions for the query set
            if return_acc:
                t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
                t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
        bin_res = np.concatenate([bin_res_cal_train, bin_res_cal_test])
        if return_acc:
            mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
            print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
            return mem_inf_acc
        else:
            return thres, bin_res, bin_res_query

    def _run_mia(self, all_methods=True, benchmark_methods=[], args=None, return_matrix=True):
        # best_acc = 0
        mat_cal = pd.DataFrame()
        mat_cal_bin = pd.DataFrame()

        mat_query = pd.DataFrame()
        mat_query_bin = pd.DataFrame()

        membership_label = [1 for i in range(len(self.s_tr_outputs))]
        membership_label += [0 for i in range(len(self.s_te_outputs))]
        print(self.q_corr.shape)

        res = {'thresholds':{}}
        if (all_methods) or ('correctness' in benchmark_methods):
            if return_matrix:
                print(self.s_tr_corr.shape)
                print(self.s_te_corr.shape)
                mat_cal['correctness'] = np.concatenate((self.s_tr_corr, self.s_te_corr))
                mat_cal_bin['correctness'] = np.concatenate((self.s_tr_corr, self.s_te_corr))
                mat_query['correctness'] = self.q_corr
                mat_query_bin['correctness'] = self.q_corr

        if (all_methods) or ('confidence' in benchmark_methods):
            thre_conf, bin_conf_cal, bin_conf_query = self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.q_conf, return_acc=False)
            res['thresholds']['thre_conf'] = thre_conf
            if return_matrix:
                mat_cal['confidence'] = np.concatenate((self.s_tr_conf, self.s_te_conf))
                mat_cal_bin['confidence'] = bin_conf_cal
                mat_query['confidence'] = self.q_conf
                mat_query_bin['confidence'] = bin_conf_query

        if (all_methods) or ('entropy' in benchmark_methods):
            thre_ent, bin_ent_cal, bin_ent_query = self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.q_entr, return_acc=False)
            res['thresholds']['thre_ent'] = thre_ent
            if return_matrix:
                mat_cal['entropy'] = np.concatenate((-self.s_tr_entr, -self.s_te_entr))
                mat_cal_bin['entropy'] = bin_ent_cal
                mat_query['entropy'] = -self.q_entr
                mat_query_bin['entropy'] = bin_ent_query

        if (all_methods) or ('modified entropy' in benchmark_methods):
            thre_ment, bin_ment_cal, bin_ment_query = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.q_m_entr, return_acc=False)
            res['thresholds']['thre_ment'] = thre_ment
            if return_matrix:
                mat_cal['modified entropy'] = np.concatenate((-self.s_tr_m_entr, -self.s_te_m_entr))
                mat_cal_bin['modified entropy'] = bin_ment_cal
                mat_query['modified entropy'] = -self.q_m_entr
                mat_query_bin['modified entropy'] = bin_ment_query

        if return_matrix:
            mat_cal['membership'] = membership_label
            mat_cal_bin['membership'] = membership_label

            res['cal_values'] = mat_cal
            res['cal_values_bin'] = mat_cal_bin
            res['query_values'] = mat_query
            res['query_values_bin'] = mat_query_bin

        return res

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[], args=None):
        best_acc = 0
        res = []
        if (all_methods) or ('correctness' in benchmark_methods):
            acc_corr, train_acc, test_acc = self._mem_inf_via_corr()
            res.extend([train_acc, test_acc, acc_corr])
            if acc_corr > best_acc:
                best_acc = acc_corr
        if (all_methods) or ('confidence' in benchmark_methods):
            acc_conf = self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
            res.append(acc_conf)
            if acc_conf > best_acc:
                best_acc = acc_conf
        if (all_methods) or ('entropy' in benchmark_methods):
            acc_ent = self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
            res.append(acc_ent)
            if acc_ent > best_acc:
                best_acc = acc_ent
        if (all_methods) or ('modified entropy' in benchmark_methods):
            acc_ment = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
            res.append(acc_ment)
            if acc_ment > best_acc:
                best_acc = acc_ment


        print(f'Best attack acc: {best_acc}')
        res.append(best_acc)
        return res

class mia_plus(object):

    def __init__(self, shadow_train_performance, shadow_train_performance_y, shadow_test_performance, shadow_test_performance_y, query_output, query_output_y, num_classes=10,
                  x_target=None, y_target=None,x_shadow=None, y_shadow=None, target_train_performance=None, target_test_performance=None,):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        '''

        self.num_classes = num_classes
        self.x_target = x_target
        self.y_target = y_target
        self.x_shadow = x_shadow
        self.y_shadow = y_shadow

        # confidence vector
        start = time.time()
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance, shadow_train_performance_y
        self.s_te_outputs, self.s_te_labels = shadow_test_performance, shadow_test_performance_y
        self.q_outputs, self.q_labels =  query_output, query_output_y
        # self.t_tr_outputs, self.t_tr_labels = target_train_performance
        # self.t_te_outputs, self.t_te_labels = target_test_performance
        end = time.time()
        #print(f'>> time cost: {end-start}')

        # label matching (label-only)
        start = time.time()
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.q_corr = (np.argmax(self.q_outputs, axis=1)==self.q_labels).astype(int)
        # self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        # self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        end = time.time()

        start = time.time()
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.q_conf = np.array([self.q_outputs[i, self.q_labels[i]] for i in range(len(self.q_labels))])
        # self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        # self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        end = time.time()

        # entropy
        start = time.time()
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.q_entr = self._entr_comp(self.q_outputs)
        # print(self.q_entr)
        # print(self.q_outputs)
        # self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        # self.t_te_entr = self._entr_comp(self.t_te_outputs)
        end = time.time()
        #print(f'>> time cost: {end - start}')

        # modified entropy
        start = time.time()
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.q_m_entr = self._m_entr_comp(self.q_outputs, self.q_labels)
        # self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        # self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        end = time.time()
        #print(f'>> time cost: {end - start}')

        # print(self.s_tr_conf.shape, self.s_tr_entr.shape)


    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)

    def _thre_setting(self, tr_values, te_values):
        """Decide the threshold for membership inference attacks using the shadow data

        Args:
            tr_values ([float]): a list of values for shadow train set
            te_values ([float]): a list of values for shadow test set

        Returns:
            float: the threshold
        """
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return mem_inf_acc, t_tr_acc, t_te_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, q_values, t_tr_values=None, t_te_values=None, return_acc=False, thre_given = None):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        if thre_given is not None:
            thres = thre_given
        else:
            thres = []
        bin_res_cal_train = np.zeros(len(s_tr_values))
        bin_res_cal_test = np.zeros(len(s_te_values))
        bin_res_query = np.zeros(len(q_values))
        for num in range(self.num_classes):
            if thre_given is None:
                thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
                thres.append(thre)
            else:
                thre = thres[num]

            bin_res_cal_train[np.multiply((self.s_tr_labels==num), (s_tr_values >= thre))] = 1
            bin_res_cal_test[np.multiply((self.s_te_labels==num), (s_te_values >= thre))] = 1
            bin_res_query[np.multiply((np.argmax(self.q_outputs, axis=1)==num), (q_values >= thre))] = 1   ## use the predictions for the query set
            if return_acc:
                t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
                t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
        bin_res = np.concatenate([bin_res_cal_train, bin_res_cal_test])
        if return_acc:
            mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
            print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
            return mem_inf_acc
        else:
            return thres, bin_res, bin_res_query

    def _mem_inf_thre_single_thread(self, data):
        v_name, s_tr_values, s_te_values, q_values = data[0], data[1], data[2], data[3]
        thre, bin_cal, bin_query = self._mem_inf_thre(v_name, s_tr_values, s_te_values, q_values, return_acc=False)
        return thre, bin_cal, bin_query

    def _run_mia(self, all_methods=True, benchmark_methods=[], args=None, return_matrix=True):
        # best_acc = 0
        mat_cal = pd.DataFrame()
        mat_cal_bin = pd.DataFrame()

        mat_query = pd.DataFrame()
        mat_query_bin = pd.DataFrame()

        membership_label = [1 for i in range(len(self.s_tr_outputs))]
        membership_label += [0 for i in range(len(self.s_te_outputs))]

        res = {'thresholds':{}}
        if (all_methods) or ('correctness' in benchmark_methods):
            if return_matrix:
                mat_cal['correctness'] = np.concatenate((self.s_tr_corr, self.s_te_corr))
                mat_cal_bin['correctness'] = np.concatenate((self.s_tr_corr, self.s_te_corr))
                mat_query['correctness'] = self.q_corr
                mat_query_bin['correctness'] = self.q_corr

        data1 = ['confidence', self.s_tr_conf, self.s_te_conf, self.q_conf]
        data2 = ['entropy', -self.s_tr_entr, -self.s_te_entr, -self.q_entr]
        datas = [data1, data2]
        with multiprocessing.Pool(5) as pool:
            processed_results = pool.map(self._mem_inf_thre_single_thread, datas)

        [thre_conf, bin_conf_cal, bin_conf_query] = processed_results[0]
        [thre_ent, bin_ent_cal, bin_ent_query] = processed_results[1]

        res['thresholds']['thre_conf'] = thre_conf
        if return_matrix:
                mat_cal['confidence'] = np.concatenate((self.s_tr_conf, self.s_te_conf))
                mat_cal_bin['confidence'] = bin_conf_cal
                mat_query['confidence'] = self.q_conf
                mat_query_bin['confidence'] = bin_conf_query

        res['thresholds']['thre_ent'] = thre_ent
        if return_matrix:
                mat_cal['entropy'] = np.concatenate((-self.s_tr_entr, -self.s_te_entr))
                mat_cal_bin['entropy'] = bin_ent_cal
                mat_query['entropy'] = -self.q_entr
                mat_query_bin['entropy'] = bin_ent_query

        if return_matrix:
            mat_cal['membership'] = membership_label
            mat_cal_bin['membership'] = membership_label

            res['cal_values'] = mat_cal
            res['cal_values_bin'] = mat_cal_bin
            res['query_values'] = mat_query
            res['query_values_bin'] = mat_query_bin

        return res

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[], args=None):
        best_acc = 0
        res = []
        if (all_methods) or ('correctness' in benchmark_methods):
            acc_corr, train_acc, test_acc = self._mem_inf_via_corr()
            res.extend([train_acc, test_acc, acc_corr])
            if acc_corr > best_acc:
                best_acc = acc_corr
        if (all_methods) or ('confidence' in benchmark_methods):
            acc_conf = self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
            res.append(acc_conf)
            if acc_conf > best_acc:
                best_acc = acc_conf
        if (all_methods) or ('entropy' in benchmark_methods):
            acc_ent = self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
            res.append(acc_ent)
            if acc_ent > best_acc:
                best_acc = acc_ent
        if (all_methods) or ('modified entropy' in benchmark_methods):
            acc_ment = self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
            res.append(acc_ment)
            if acc_ment > best_acc:
                best_acc = acc_ment
        ## TODO: add API for NN attacks
        # if (all_methods) or ('NN' in benchmark_methods):
        #     print('To be implemented')
        #     acc_NN = self._mem_inf_NN(args)

        print(f'Best attack acc: {best_acc}')
        res.append(best_acc)
        return res

metrics = ['correctness', 'confidence', 'entropy']

def get_member_ratio(datadf, thre_cnt=1, skip=[], mode='ks'):
    sample2mem = []
    for idx, row in datadf.iterrows():
        sample2mem.append(np.sum([row[m] for m in (set(metrics) - set(skip))]))

    members_bool = np.asarray(sample2mem) >= thre_cnt

    if mode == 'ks':
        _, pvalue = ks_2samp(members_bool, [1 for i in range(10000)], mode='asymp')
    elif mode == 't':
        if members_bool.mean() == 1:
            return 1, 1
        else:
            _, pvalue = ttest_ind(members_bool, [1 for i in range(len(members_bool))], equal_var=True,
                                  nan_policy='raise')

    return np.sum(members_bool) / len(datadf), pvalue

if __name__ == '__main__':

    n = 50000
    calTrainPred = np.random.rand(n,10)
    calTrainY = np.random.randint(0,10,size=(n))
    calTestPred = np.random.rand(n, 10)
    calTestY = np.random.randint(0,10,size=(n))
    queryPred = np.random.rand(n, 10)
    queryY = np.random.randint(0,10,size=(n))
    nclass = 10

    start_time = time.time()

    MIA = mia_plus(calTrainPred,
              calTrainY,
              calTestPred,
              calTestY,
              queryPred,
              queryY,
              num_classes=nclass)

    MIA_results = MIA._run_mia()

    end_time = time.time()

    t, pv = get_member_ratio(MIA_results['query_values_bin'], skip=['modified entropy'], mode='t')
    EMA_res = np.around(pv, decimals=2)

    print(t, pv, EMA_res, end_time-start_time)

