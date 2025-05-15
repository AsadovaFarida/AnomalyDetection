import sys
sys.path.append('../')
import json
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from Trainer.TrainerOpenStack import model_fn, input_fn, predict_fn

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=23, metavar='N',
                        help='to determine the time series data is an anomaly or not.')
    parser.add_argument('--num-candidates', type=int, default=1,
                        help='number of top candidates to consider for prediction')
    args = parser.parse_args()
    num_candidates = args.num_candidates

    ##############
    # Load Model #
    ##############
    model_dir = './model'
    model_info = model_fn(model_dir)

    ###########
    # predict #
    ###########
    test_abnormal_list = []
    with open('test_abnormal', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                tokens = [token for token in line.split() if token.strip() != '']
                line = [int(n) - 1 for n in tokens]
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue
            request = json.dumps({'line': line, 'num_candidates': num_candidates})
            input_data = input_fn(request, 'application/json')
            response = predict_fn(input_data, model_info)
            test_abnormal_list.append(response)

    test_normal_list = []
    with open('test_normal', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                tokens = [token for token in line.split() if token.strip() != '']
                line = [int(n) - 1 for n in tokens]
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue
            request = json.dumps({'line': line, 'num_candidates': num_candidates})
            input_data = input_fn(request, 'application/json')
            response = predict_fn(input_data, model_info)
            test_normal_list.append(response)

    ##############
    # Evaluation #
    ##############
    thres = args.threshold
    abnormal_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_abnormal_list]
    abnormal_cnt_anomaly = [t['anomaly_cnt'] for t in test_abnormal_list]
    abnormal_predict = []
    for test_abnormal in test_abnormal_list:
        abnormal_predict += test_abnormal['predict_list']

    normal_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_normal_list]
    normal_cnt_anomaly = [t['anomaly_cnt'] for t in test_normal_list]
    normal_predict = []
    for test_normal in test_normal_list:
        normal_predict += test_normal['predict_list']

    ground_truth = [1]*len(abnormal_has_anomaly) + [0]*len(normal_has_anomaly)
    predict = abnormal_has_anomaly + normal_has_anomaly
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    accu = 0
    for p, t in zip(predict, ground_truth):
        if p == t:
            accu += 1

        if p == 1 and t == 1:
            TP += 1
        elif p == 1 and t == 0:
            FP += 1
        elif p == 0 and t == 1:
            FN += 1
        else:
            TN += 1

    logger.info(f'thres: {thres}')
    logger.info(f'TP: {TP}')
    logger.info(f'FP: {FP}')
    logger.info(f'TN: {TN}')
    logger.info(f'FN: {FN}')

    accuracy = accu / len(predict)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    logger.info(f'accuracy: {accuracy}')
    logger.info(f'Precision: {precision}')
    logger.info(f'Recall: {recall}')
    logger.info(f'F1: {F1}')

    # -------- Visualization --------
    # Confusion Matrix
    cm = confusion_matrix(ground_truth, predict)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Bar Plot of Metrics
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': F1}
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 1)
    plt.title('Evaluation Metrics')
    plt.show()

    # Histogram of Anomaly Counts
    plt.figure()
    plt.hist(abnormal_cnt_anomaly, bins=20, alpha=0.7, label='Abnormal')
    plt.hist(normal_cnt_anomaly, bins=20, alpha=0.7, label='Normal')
    plt.xlabel('Anomaly Count')
    plt.ylabel('Number of Sequences')
    plt.legend()
    plt.title('Distribution of Anomaly Counts')
    plt.show()