import csv
import sys

def main(file):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    with open(file, "r") as t, open("test-key.txt", "r") as k:
        kr = list(csv.reader(k, delimiter=" "))
        tr = list(csv.reader(t, delimiter=" "))
        for i in range(58):
            if int(kr[i][1]) == 1:
                if int(tr[i][1]) == 1:
                    TP += 1
                else:
                    FN += 1
            elif int(kr[i][1]) == 0:
                if int(tr[i][1]) == 0:
                    TN += 1
                else:
                    FP += 1
    PREC = TP / (TP + FP)
    REC = TP / (TP + FN)
    F1 = 2 * (PREC * REC) / (PREC + REC)

    print("TP: {}".format(TP))
    print("FP: {}".format(FP))
    print("TN: {}".format(TN))
    print("FN: {}".format(FN))
    print("Precision: {}".format(PREC))
    print("Recall: {}".format(REC))
    print("F1: {}".format(F1))

if __name__ == "__main__":
    main(sys.argv[1])
