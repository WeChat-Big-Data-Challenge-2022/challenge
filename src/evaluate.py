from category_id_map import category_id_to_lv2id
from util import evaluate


def evaluate_submission(result_file, ground_truth_file):
    ground_truth = {}
    with open(ground_truth_file, 'r') as f:
        for line in f:
            vid, category_id = line.strip().split(',')
            ground_truth[vid] = category_id_to_lv2id(category_id)

    predictions, labels = [], []
    with open(result_file, 'r') as f:
        for line in f:
            vid, category_id = line.strip().split(',')
            if vid not in ground_truth:
                raise Exception(f'ERROR id {vid} in result.csv')
            predictions.append(category_id_to_lv2id(category_id))
            labels.append(ground_truth[vid])

    if len(predictions) != len(ground_truth):
        raise Exception(f'ERROR: Wrong line numbers')

    return evaluate(predictions, labels)


if __name__ == '__main__':
    result_file = 'data/result.csv'
    ground_truth_file = 'data/private/ground_truth_test_a.csv'

    result = evaluate_submission(result_file, ground_truth_file)
    print(f'mean F1 score is {result["mean_f1"]}')
