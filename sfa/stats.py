
import numpy as np
import scipy as sp
import pandas as pd




__all__ = ["calc_accuracy",]
           # "calc_auroc",
           # "calc_auprc",
           # "calc_roc_curve",
           # "calc_pr_curve"]


def calc_accuracy(df1, df2, get_cons=False):
    """
    Count the same sign of each element between df1 and df2

    df1: pandas.DataFrame or numpy.ndarray to be compared
    df2: pandas.DataFrame or numpy.ndarray to be compared
    getcons: decide whether to return consensus array in DataFrame or not
    """

    np.sign(df1) + np.sign(df2)
    if df1.ndim == 1:
        num_total = df1.shape[0]
    elif df1.ndim == 2:
        num_total = df1.shape[0] * df1.shape[1]
    diff_abs = np.abs(np.sign(df1) - np.sign(df2))
    consensus = (diff_abs == 0)

    if isinstance(consensus, pd.DataFrame):
        num_cons = consensus.values.sum()
    else:
        #num_cons = consensus.sum(axis=1).sum()  # Number of consensus
        num_cons = consensus.sum()
        
    acc = (num_cons) / np.float(num_total)  # Accuracy
    if get_cons:
        return acc, consensus
    else:
        return acc


# end of def

#
# from sklearn.metrics import roc_curve, auc
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
#
# def _pre_process(class_type, arr_exp, arr_sim):
#     if not isinstance(class_type, str):
#         raise TypeError("class_type should be str.")
#
#     if class_type == 'UP':
#         arr_exp[arr_exp > 0] = 1
#         arr_exp[arr_exp <= 0] = 0
#     elif class_type == 'DN':
#         arr_exp *= -1  # Flip the sign
#         arr_sim *= -1
#         arr_exp[arr_exp > 0] = 1
#         arr_exp[arr_exp <= 0] = 0
#     elif class_type == '-':
#         arr_exp[arr_exp == 0] = 1
#         arr_exp[arr_exp != 0] = 0
#
#
# # end of def
#
# def calc_auroc(df_exp, df_sim, class_type='UP'):
#     auroc = dict()
#
#     arr_exp_ravel = np.array(df_exp.unstack())
#     arr_sim_ravel = np.array(df_sim.unstack())
#     _pre_process(class_type, arr_exp_ravel, arr_sim_ravel)
#     auroc['mean'] = roc_auc_score(arr_exp_ravel, arr_sim_ravel)
#     return auroc
#
#     for idx_roc, (name_roc, col) in enumerate(df_exp.iteritems()):
#
#         arr_exp = np.array(col)
#         arr_sim = np.array(df_sim[name_roc])
#
#         _pre_process(class_type, arr_exp, arr_sim)
#
#         try:
#             auc = roc_auc_score(arr_exp, arr_sim)
#             auroc[name_roc] = auc
#         except ValueError as ve:
#             print ("Skip calculating AUROC of %s "
#                    "due to the following." % name_roc)
#             print (ve)
#             continue
#
#     # end of for
#     auroc["mean"] = sum(auroc.values()) / len(auroc)
#     return auroc
# # end of def
#
# def calc_auprc(df_exp, df_sim, class_type='UP'):
#     auprc = dict()
#
#     arr_exp_ravel = np.array(df_exp.unstack())
#     arr_sim_ravel = np.array(df_sim.unstack())
#     _pre_process(class_type, arr_exp_ravel, arr_sim_ravel)
#     auprc['mean'] = average_precision_score(arr_exp_ravel, arr_sim_ravel)
#     return auprc
#
#     for idx_roc, (name_roc, col) in enumerate(df_exp.iteritems()):
#
#         arr_exp = np.array(col)
#         arr_sim = np.array(df_sim[name_roc])
#
#         _pre_process(class_type, arr_exp, arr_sim)
#
#         try:
#             auc = average_precision_score(arr_exp, arr_sim)
#             auprc[name_roc] = auc
#         except ValueError as ve:
#             print ("Skip calculating AUPRC of %s "
#                    "due to the following." % name_roc)
#             print (ve)
#             continue
#         except FloatingPointError as fpe:
#             print("Skip calculating AUPRC of %s "
#                   "due to the following." % name_roc)
#             print(fpe)
#             continue
#
#     # end of for
#     auprc["mean"] = sum(auprc.values()) / len(auprc)
#     return auprc
#
#
# # end of def
#
# def calc_roc_curve(df_exp, df_sim, class_type='UP'):
#     dict_fpr = dict()
#     dict_tpr = dict()
#     dict_thr = dict()
#     dict_auroc = dict()
#
#     # name: node name or ID
#     for idx_roc, (name, col) in enumerate(df_exp.iteritems()):
#         arr_exp = np.array(col)
#         arr_sim = np.array(df_sim[name])
#
#         _pre_process(class_type, arr_exp, arr_sim)
#
#         try:
#             fpr, tpr, thr = roc_curve(arr_exp, arr_sim)
#             dict_auroc[name] = auc(fpr, tpr)
#         except FloatingPointError as fpe:
#             print("Skip calculating AUROC of %s "
#                   "due to the following." % name)
#             print(fpe)
#             continue
#
#         dict_fpr[name] = fpr
#         dict_tpr[name] = tpr
#         dict_thr[name] = thr
#     # end of for
#
#     """
#     Use different thresholds for each entity (e.g., protein) to test with ROC
#     # First aggregate all false positive rates
#     fprs = []
#     names_valid = []  # Names which has no nan values in false positive rates.
#     for name in dict_fpr:
#         if np.any(np.isnan(dict_fpr[name])) \
#            or np.any(np.isnan(dict_tpr[name])):
#             continue
#         fprs.append(dict_fpr[name])
#         names_valid.append(name)
#     # end of for
#     all_fpr = np.unique(np.concatenate(fprs))
#     mean_tpr = np.zeros_like(all_fpr)
#     for name in names_valid:
#         mean_tpr += sp.interp(all_fpr, dict_fpr[name], dict_tpr[name])
#
#     mean_tpr /= len(names_valid)
#
#     dict_fpr["mean"] = all_fpr
#     dict_tpr["mean"] = mean_tpr
#     dict_auroc["mean"] = auc(dict_fpr["mean"], dict_tpr["mean"])
#     """
#     arr_exp_ravel = np.array(df_exp.unstack())
#     arr_sim_ravel = np.array(df_sim.unstack())
#     _pre_process(class_type, arr_exp_ravel, arr_sim_ravel)
#     fpr, tpr, thr = roc_curve(arr_exp_ravel, arr_sim_ravel)
#
#     dict_fpr["mean"] = fpr
#     dict_tpr["mean"] = tpr
#     dict_thr["mean"] = thr
#     dict_auroc["mean"] = roc_auc_score(arr_exp_ravel, arr_sim_ravel) #auc(fpr, tpr)
#
#     return dict_fpr, dict_tpr, dict_thr, dict_auroc
# # end of def
#
#
# def calc_pr_curve(df_exp, df_sim, class_type='UP'):
#     dict_precision = dict()
#     dict_recall = dict()
#     dict_thr = dict()
#     dict_auprc = dict()
#
#     # name: node name or ID
#     for idx_roc, (name, col) in enumerate(df_exp.iteritems()):
#         arr_exp = np.array(col)
#         arr_sim = np.array(df_sim[name])
#
#         _pre_process(class_type, arr_exp, arr_sim)
#
#         try:
#             precision, recall, thr = precision_recall_curve(arr_exp, arr_sim)
#             dict_auprc[name] = average_precision_score(arr_exp, arr_sim)
#         except FloatingPointError as fpe:
#             print("Skip calculating AUPRC of %s "
#                   "due to the following." % name)
#             print(fpe)
#             continue
#
#         dict_precision[name] = precision
#         dict_recall[name] = recall
#         dict_thr[name] = thr
#     # end of for
#
#     # First aggregate all false positive rates
#     # recalls = []
#     # names_valid = []  # Names which has no nan values in false positive rates.
#     # for name in dict_recall:
#     #     if np.any(np.isnan(dict_recall[name])) \
#     #        or np.any(np.isnan(dict_precision[name])):
#     #         continue
#     #     recalls.append(dict_recall[name])
#     #     names_valid.append(name)
#     # # end of for
#     # all_recall = np.unique(np.concatenate(recalls))
#     # #all_recall = np.array(sorted(all_recall, reverse=True))
#     # mean_precision = np.zeros_like(all_recall)
#     # for name in names_valid:
#     #     mean_precision += sp.interp(all_recall,
#     #                                 dict_recall[name],
#     #                                 dict_precision[name])
#     #
#     # mean_precision /= len(names_valid)
#
#     arr_exp_ravel = np.array(df_exp.unstack())
#     arr_sim_ravel = np.array(df_sim.unstack())
#     _pre_process(class_type, arr_exp_ravel, arr_sim_ravel)
#     precision, recall, thr = precision_recall_curve(arr_exp_ravel,
#                                                     arr_sim_ravel)
#
#     dict_recall["mean"] = recall #all_recall
#     dict_precision["mean"] = precision #mean_precision
#     dict_thr["mean"] = thr
#     #dict_auprc["mean"] = auc(dict_recall["mean"], dict_precision["mean"])
#     dict_auprc["mean"] = average_precision_score(arr_exp_ravel, arr_sim_ravel)
#     return dict_recall, dict_precision, dict_thr, dict_auprc
# # end of def