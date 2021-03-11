from dataset import *
from model import *
from util import *

def test(args, dataset, model, small=False):
    model.eval()
    rets = {}
    with torch.no_grad():
        start = time.time()
        qid_list = []
        q_embs = []
        batch_len = (dataset.get_test_q_size() + args.batch_size_test - 1) // args.batch_size_test
        for batch in range(batch_len):
            d = dataset.get_test_q_batch(args.batch_size_test, batch, args.max_qlen)
            qid_list += q['id']
            q_emb, _ = model(q['ids'], q['seg'], q['mask'], q['topic'])
            q_embs.append(q_emb)
            end = time.time()
            print('Build Query Emb : Batch %d / %d [%s]' % (batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
        print()
        q_embs = torch.cat(q_embs, dim=0)

        start = time.time()
        did_list = []
        d_embs = []
        batch_len = (dataset.get_test_d_size(small) + args.batch_size_test - 1) // args.batch_size_test
        for batch in range(batch_len):
            d = dataset.get_test_d_batch(args.batch_size_test, batch, args.max_dlen, small)
            did_list += d['id']
            d_emb, _ = model(d['ids'], d['seg'], d['mask'], d['topic'])
            d_embs.append(d_emb)
            end = time.time()
            print('Build Doc Emb : Batch %d / %d [%s]' % (batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
        print()
        d_embs = torch.cat(d_embs, dim=0)

        start = time.time()
        trec_out_fp = open('data/%s/predict.trec' % args.dataset, 'w', encoding='utf-8')
        for qid in range(q_embs.size(0)):
            scores = torch.matmul(q_embs[qid:qid+1, :], d_embs.transpose(0, 1)).cpu().view(-1).tolist()
            ranks = sorted(zip(scores, did_list), reverse=True)[:1000]
            for i in range(len(ranks)):
                trec_out_fp.write('%s Q0 %s %d %f yonsei\n' % (qid_list[qid], ranks[i][1], i + 1, ranks[i][0]))
        trec_out_fp.close()
        trec_eval_res = subprocess.Popen(['data/trec_eval', '-m', 'all_trec', 'data/%s/answer.trec' % args.dataset, 'data/%s/predict.trec' % args.dataset], stdout=subprocess.PIPE, shell=False)
        out, err = trec_eval_res.communicate()
        lines = out.decode('utf-8').strip().split('\n')
        metrics = {}
        for line in lines[1:]:
            metric, _, value = line.split()
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
            metrics[metric.lower()] = value
    return metrics

def train(args, dataset):
    model_name = ''.join([t.capitalize() for t in args.model.split('_')])
    model = globals()[model_name](args.topic_dim)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_recall = 0.0
    for epoch in range(1, args.total_epoch + 1):
        batch_len = dataset.get_train_size() // args.batch_size
        loss_cum = []
        loss_hinge_cum = []
        loss_ortho_cum = []
        start = time.time()
        for batch in range(batch_len):
            q, pd, nd, margin = dataset.get_train_batch(args.batch_size, batch, args.max_qlen, args.max_dlen, args.margin_const)
            optimizer.zero_grad()
            q_emb, q_ortho = model(q['ids'], q['seg'], q['mask'], q['topic'])
            pd_emb, pd_ortho = model(pd['ids'], pd['seg'], pd['mask'], pd['topic'])
            nd_emb, nd_ortho = model(nd['ids'], nd['seg'], nd['mask'], nd['topic'])
            pos_scores = torch.sum(q_emb * pd_emb, dim=1)
            neg_scores = torch.sum(q_emb * nd_emb, dim=1)
            loss_hinge = F.relu(margin - (pos_scores - neg_scores)).mean() 
            loss_ortho = q_ortho + pd_ortho + nd_ortho
            loss = loss_hinge + args.ortho_const * loss_ortho
            loss.backward()
            optimizer.step()
            loss_cum.append(loss.item())
            loss_hinge_cum.append(loss_hinge.item())
            loss_ortho_cum.append(loss_ortho.item() * args.ortho_const)
            end = time.time()
            # log
            loss_now = sum(loss_cum) / len(loss_cum)
            loss_hinge_now = sum(loss_hinge_cum) / len(loss_hinge_cum)
            loss_ortho_now = sum(loss_ortho_cum) / len(loss_ortho_cum)
            print('[E%d] Batch %d / %d | L %.6f | LH %.6f | LO %.6f [%s]' % (epoch, batch + 1, batch_len, loss_now, loss_hinge_now, loss_ortho_now, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
            # eval
            if (batch + 1) % args.test_step == 0:
                print()
                metrics = test(args, dataset, model, True)
                metrics['epoch'] = epoch
                metrics['batch'] = batch
                print('| R@1000 |  P@20  | nDCG@20 |  MAP  |')
                print('| %6.2f | %6.2f | %7.2f | %5.2f |' % (metrics['recall_1000'] * 100, metrics['p_20'] * 100, metrics['ndcg_cut_20'] * 100, metrics['map'] * 100))
                if best_map < metrics['map']:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    for name in sorted(os.listdir('save_%s_%s' % (args.dataset, args.model))):
                        if float(name[:-3]) < metrics['map']:
                            os.remove('save_%s_%s/%s' % (args.dataset, args.model, name))
                    torch.save(state, 'save_%s_%s/%.4f.pt' % (args.dataset, args.model, metrics['map'] * 100))
                    best_map = metrics['map']
                    print('########## SAVED ##########')
            loss_cum = []
            loss_hinge_cum = []
            loss_ortho_cum = []

    # full-test the best model
    print('Start Full Test')
    best = sorted(os.listdir('save_%s_%s' % (args.dataset, args.model)), key=lambda name:float(name[:-3]), reverse=True)[0]
    state = torch.load('save_%s_%s/%s' % (args.dataset, args.model, best))
    model.load_state_dict(state['model'])
    metrics = test(args, dataset, model, False)
    print('===== Final Results =====')
    print('| R@1000 |  P@20  | nDCG@20 |  MAP  |')
    print('| %6.2f | %6.2f | %7.2f | %5.2f |' % (metrics['recall_1000'] * 100, metrics['p_20'] * 100, metrics['ndcg_cut_20'] * 100, metrics['map'] * 100))


def test_split(args, dataset, model, split, small=False):
    model.eval()
    rets = {}
    with torch.no_grad():
        start = time.time()
        qid_list = []
        q_embs = []
        batch_len = (dataset.get_test_q_size(split) + args.batch_size_test - 1) // args.batch_size_test
        for batch in range(batch_len):
            q, margin = dataset.get_test_q_batch(split, args.batch_size_test, batch, args.max_qlen, args.margin_const, args.topic_dim)
            qid_list += q['id']
            q_emb = model(q['ids'], q['seg'], q['mask'], q['topic'])
            q_embs.append(q_emb)
            end = time.time()
            print('Build Query Emb : Batch %d / %d [%s]' % (batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
        print()
        q_embs = torch.cat(q_embs, dim=0)

        start = time.time()
        pid_list = []
        p_embs = []
        batch_len = (dataset.get_test_p_size(split, small) + args.batch_size_test - 1) // args.batch_size_test
        for batch in range(batch_len):
            p, margin = dataset.get_test_p_batch(split, args.batch_size_test, batch, args.max_plen, args.margin_const, args.topic_dim, small)
            pid_list += p['id']
            p_emb = model(p['ids'], p['seg'], p['mask'], p['topic'])
            p_embs.append(p_emb)
            end = time.time()
            print('Build Passage Emb : Batch %d / %d [%s]' % (batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
        print()
        p_embs = torch.cat(p_embs, dim=0)

        start = time.time()
        trec_out_fp = open('data/%s/predict.trec' % args.dataset, 'w', encoding='utf-8')
        for qid in range(q_embs.size(0)):
            scores = torch.matmul(q_embs[qid:qid+1, :], p_embs.transpose(0, 1)).cpu().view(-1).tolist()
            ranks = sorted(zip(scores, pid_list), reverse=True)[:1000]
            for i in range(len(ranks)):
                trec_out_fp.write('%s Q0 %s %d %f yonsei\n' % (qid_list[qid], ranks[i][1], i + 1, ranks[i][0]))
        trec_out_fp.close()
        trec_eval_res = subprocess.Popen(['data/trec_eval', '-m', 'all_trec', 'data/%s/answer.trec' % args.dataset, 'data/%s/predict.trec' % args.dataset], stdout=subprocess.PIPE, shell=False)
        out, err = trec_eval_res.communicate()
        lines = out.decode('utf-8').strip().split('\n')
        metrics = {}
        for line in lines[1:]:
            metric, _, value = line.split()
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
            metrics[metric.lower()] = value
    return metrics

def train_split(args, dataset, split):
    model_name = ''.join([t.capitalize() for t in args.model.split('_')])
    model = globals()[model_name](args.topic_dim)
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    best_recall = 0.0
    for epoch in range(1, args.total_epoch + 1):
        batch_len = dataset.get_train_size(split) // args.batch_size
        loss_cum = []
        start = time.time()
        for batch in range(batch_len):
            q, rd, ud, margin = dataset.get_train_batch(split, args.batch_size, batch, args.max_qlen, args.max_plen, args.topic_dim)
            optimizer.zero_grad()
            q_emb = model(q['ids'], q['seg'], q['mask'], q['topic'])
            rd_emb = model(rd['ids'], rd['seg'], rd['mask'], rd['topic'])
            ud_emb = model(ud['ids'], ud['seg'], ud['mask'], ud['topic'])
            rel_scores = torch.sum(q_emb * rd_emb, dim=1)
            urel_scores = torch.sum(q_emb * ud_emb, dim=1)
#            loss_ortho = q_ortho + rd_ortho + ud_ortho
            loss = F.relu(margin - (rel_scores - urel_scores)).mean() # + args.ortho_const * loss_ortho
            loss.backward()
            optimizer.step()
            loss_cum.append(loss.item())
            end = time.time()
            # log
            loss_now = sum(loss_cum) / len(loss_cum)
            print('[E%d] Batch %d / %d Loss %.6f [%s]' % (epoch, batch + 1, batch_len, loss_now, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)
            # eval
            if (batch + 1) % args.test_step == 0:
                print()
                metrics = test_split(args, dataset, model, split, True)
                metrics['epoch'] = epoch
                metrics['batch'] = batch
                print('Recall:', metrics['recall_1000'] * 100)
                if best_recall < metrics['recall_1000']:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    for name in sorted(os.listdir('save_%s_%s' % (args.dataset, args.model))):
                        if int(name[:1]) == split and float(name[2:-3]) < metrics['recall_1000']:
                            os.remove('save_%s_%s/%s' % (args.dataset, args.model, name))
                    torch.save(state, 'save_%s_%s/%d_%.4f.pt' % (args.dataset, args.model, split, metrics['recall_1000']))
                    best_recall = metrics['recall_1000']
                    print('########## SAVED ##########')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=4)
    parser.add_argument('--batch_size_test', required=False, type=int, default=50)
    parser.add_argument('--cls', required=False, type=int, default=-1)
    parser.add_argument('--dataset', required=False, type=str, default='nq')
    parser.add_argument('--margin_const', required=False, type=float, default=10.0)
    parser.add_argument('--max_qlen', required=False, type=int, default=20)
    parser.add_argument('--max_dlen', required=False, type=int, default=400)
    parser.add_argument('--model', required=False, type=str, default='bert_topic_concat_ortho_loss')
    parser.add_argument('--ortho_const', required=False, type=float, default=0.01)
    parser.add_argument('--seed', required=False, type=int, default=1234)
    parser.add_argument('--sep', required=False, type=int, default=-1)
    parser.add_argument('--split', required=False, type=int, default=5)
    parser.add_argument('--test_step', required=False, type=int, default=1000)
    parser.add_argument('--topic_dim', required=False, type=int, default=768)
    parser.add_argument('--total_epoch', required=False, type=int, default=10)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if args.cls == -1:
        args.cls = tokenizer.vocab['[CLS]']
    if args.sep == -1:
        args.sep = tokenizer.vocab['[SEP]']

    # fix random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'robust04':
        dataset = Robust04Dataset(args.split)
        dataset.build(args.margin_const)
    elif args.dataset == 'marco_passage':
        dataset = MarcoPassageDataset()
    elif args.dataset == 'nq':
        dataset = NaturalQuestionsDataset()
    else:
        print('[%s] is not a known dataset!' % args.dataset)
        sys.exit()

    # create save dir
    if not os.path.exists('save_%s_%s' % (args.dataset, args.model)):
        os.makedirs('save_%s_%s' % (args.dataset, args.model))

    if args.dataset == 'robust04':
        for split in range(1, args.split):
            train_split(args, dataset, split)

        sys.exit()
        for split in range(args.split):
            best = '%d_0.0000.pt' % split
            for name in os.listdir('save_%s_%s' % (args.dataset, args.model)):
                if int(name[0]) == split and float(best[2:-3]) < float(name[2:-3]):
                    best = name
            model_name = ''.join([t.capitalize() for t in args.model.split('_')])
            model = globals()[model_name](args.topic_dim)
            model = model.to(device)
            state = torch.load('save_%s_%s/%s' % (args.dataset, args.model, best))
            model.load_state_dict(state['model'])
            metrics = test_split(args, dataset, model, split, False)
            print('===== Split %d =====' % split)
            print('| R@1000 |  P@20  | nDCG@20 |  MAP  |')
            print('| %6.2f | %6.2f | %7.2f | %5.2f |' % (metrics['recall_1000'] * 100, metrics['p_20'] * 100, metrics['ndcg_cut_20'] * 100, metrics['map'] * 100))

    else:
        train(args, dataset)
