import argparse
import json
from metrics import f1_metric
from parlai.core.metrics import RougeMetric, BleuMetric

NO_PASSAGE_USED = "no_passages_used"
KNOWLEDGE_SEP = "__knowledge__"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file")
    parser.add_argument("--test_data")
    parser.add_argument("--eval_metric", default="kf1", choices=["f1", "kf1", "rouge_l", "bleu"])
    args = parser.parse_args()

    hyp_list = []  
    ref_list = []  
    knowledge_list = []   

    with open(args.pred_file, mode="r", encoding="utf-8") as rf:
        read_lines = rf.readlines()
        for line in read_lines:
            _line = line.split("|||")
            assert len(_line) == 2
            hyp = _line[0].strip()
            ref = _line[1].strip()
            hyp_list.append(hyp)
            ref_list.append(ref)

    with open(args.test_data, mode="r", encoding="utf-8") as rf_test:
        for line in rf_test:
            line = json.loads(line)
            knowledge = line['knowledge'][0].split(KNOWLEDGE_SEP)[1].strip()
            
            knowledge_list.append(knowledge)

    if args.eval_metric == "kf1":
        hyp_rm_no_pass_used = []
        kno_rm_no_pass_used = []
        assert len(hyp_list) == len(knowledge_list)
        for i, (hyp, know) in enumerate(zip(hyp_list, knowledge_list)):
            if know != NO_PASSAGE_USED:
                hyp_rm_no_pass_used.append(hyp)
                kno_rm_no_pass_used.append(know)

        assert len(hyp_rm_no_pass_used) == len(kno_rm_no_pass_used)
        print(f"KF1: {f1_metric(hyp_rm_no_pass_used, kno_rm_no_pass_used)}")
    else:
        assert len(hyp_list) == len(ref_list)
        if args.eval_metric == "f1":
            print(f"F1: {f1_metric(hyp_list, ref_list)}")
        elif args.eval_metric == "rouge_l":
            rl = sum([RougeMetric.compute_many(hyp, [ref])[2].value() for hyp, ref in zip(hyp_list, ref_list)]) / len(hyp_list)
            print(f"rouge-l: {rl}")
        elif args.eval_metric == "bleu":
            b1 = sum([BleuMetric.compute(hyp, [ref], k=1).value() for hyp, ref in zip(hyp_list, ref_list)]) / len(hyp_list)
            b2 = sum([BleuMetric.compute(hyp, [ref], k=2).value() for hyp, ref in zip(hyp_list, ref_list)]) / len(hyp_list)
            b3 = sum([BleuMetric.compute(hyp, [ref], k=3).value() for hyp, ref in zip(hyp_list, ref_list)]) / len(hyp_list)
            b4 = sum([BleuMetric.compute(hyp, [ref], k=4).value() for hyp, ref in zip(hyp_list, ref_list)]) / len(hyp_list)

        else:
            assert False, "Wrong Choice"


if __name__ == "__main__":
    main()