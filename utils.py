import csv


def to_csv(preds, csv_path):
    with open(csv_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Id', 'Bound'])
        count = 0
        for dataset_id in [0, 1, 2]:
            for y in preds[dataset_id]:
                writer.writerow([count, int(y)])
                count += 1
