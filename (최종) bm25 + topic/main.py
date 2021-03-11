from dataset import *
from model import *
from util import *

def test(args, dataset, ids_to_topic, small=False):
    if small:
        did_list = dataset.did_list_sm
    else:
        did_list = sorted(dataset.docset_test)
    corpus = []
    start = time.time()
    for did in did_list:
        doc_ids = dataset.docset_test[did]
        doc_topic = []
        for di in doc_ids:
            doc_topic += ids_to_topic[di]
        doc = ['w_%d' % di for di in doc_ids] + ['t_%d' % dt for dt in doc_topic]
        corpus.append(doc)
        end = time.time()
        print('Build Corpus : %d docs [%s]' % (len(corpus), format_time(start, end, len(did_list), len(corpus))), end='\r', flush=True)
    print()

    bm25 = BM25(corpus)

    start = time.time()
    cnt = 0
    trec_out_fp = open('data/%s/predict.trec' % args.dataset, 'w', encoding='utf-8')
    for data in dataset.dataset_test:
        query_ids = data['query_ids']
        query_topic = []
        for qi in query_ids:
            query_topic += ids_to_topic[qi]
        query = ['w_%d' % qi for qi in query_ids] + ['t_%d' % qt for qt in query_topic]
        scores = bm25.get_scores(query)
        ranks = sorted(zip(scores, did_list), reverse=True)[:1000]
        for i in range(len(ranks)):
            trec_out_fp.write('%s Q0 %s %d %f yonsei\n' % (data['qid'], ranks[i][1], i + 1, ranks[i][0]))
        end = time.time()
        cnt += 1
        print('Eval : %d/%d data [%s]' % (cnt, len(dataset.dataset_test), format_time(start, end, len(dataset.dataset_test), cnt)), end='\r', flush=True)
    trec_out_fp.close()
    print()

    del bm25

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', required=False, type=int, default=500)
    parser.add_argument('--bow_method', required=False, type=str, default='tf-idf')
    parser.add_argument('--lr', required=False, type=float, default=2e-3)
    parser.add_argument('--seed', required=False, type=int, default=1234)
    parser.add_argument('--dataset', required=False, type=str, default='nq')
    parser.add_argument('--dim_x', required=False, type=int, default=30000)
    parser.add_argument('--dim_z', required=False, type=int, default=768)
    parser.add_argument('--dirichlet_alpha', required=False, type=float, default=0.1)
    parser.add_argument('--noise_alpha', required=False, type=float, default=0.2)
    parser.add_argument('--kernel_alpha', required=False, type=float, default=1.0)
    parser.add_argument('--prior_alpha', required=False, type=float, default=1.0)
    parser.add_argument('--recon_alpha', required=False, type=float, default=-1.0)
    parser.add_argument('--n_thread', required=False, type=int, default=0)
    parser.add_argument('--topic_topk', required=False, type=int, default=10)
    parser.add_argument('--total_epoch', required=False, type=int, default=20)
    args = parser.parse_args()
    args.n_thread = min(args.n_thread, multiprocessing.cpu_count())

    if args.dataset == 'nq':
        dataset = NaturalQuestionsDataset()
    else:
        print('No dataset : %s' % args.dataset)
        exit()

    if not os.path.isdir('save_%s' % args.dataset):
        os.mkdir('save_%s' % args.dataset)

    if args.recon_alpha < 0.0:
        args.recon_alpha = dataset.get_recon_alpha(args.dim_x)

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    encoder = Encoder(dim_x=args.dim_x, dim_z=args.dim_z, enc_h_list=[100, 100])
    encoder = encoder.to(device)
    decoder = Decoder(dim_x=args.dim_x, dim_z=args.dim_z)
    decoder = decoder.to(device)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    bce_loss = nn.BCELoss()

    train_stat = []
    test_stat = []
    best_epoch = 0
    for epoch in range(1, 1 + args.total_epoch):
        batch_len = dataset.get_train_size() // args.batch_size
        start = time.time()
        for batch in range(batch_len):
            z_true = torch.tensor(np.random.dirichlet(np.ones(args.dim_z) * args.dirichlet_alpha, size=args.batch_size), dtype=torch.float32).to(device)
            x_q, x_p, x_n = dataset.get_train_batch(args.batch_size, batch, args.dim_x, args.bow_method)
            optimizer.zero_grad()

            # encode
            z_q = encoder(x_q)
            z_q = F.softmax(z_q, dim=1)

            z_p = encoder(x_p)
            z_p = F.softmax(z_p, dim=1)

            z_n = encoder(x_n)
            z_n = F.softmax(z_n, dim=1)

            # dist
            dist_p = bce_loss(z_q, z_p.detach()) + bce_loss(z_p, z_q.detach())
            dist_n = bce_loss(z_q, z_n.detach()) + bce_loss(z_n, z_q.detach())

            # recon
            z_noise = torch.tensor(np.random.dirichlet(np.ones(args.dim_z) * args.dirichlet_alpha, size=args.batch_size), dtype=torch.float32).to(device)
            z_q = (1.0 - args.noise_alpha) * z_q + args.noise_alpha * z_noise
            y_q = decoder(z_q)
            y_q = F.log_softmax(y_q, dim=1)
            recon_q = torch.mean(torch.sum(-x_q * y_q, dim=1))

            z_noise = torch.tensor(np.random.dirichlet(np.ones(args.dim_z) * args.dirichlet_alpha, size=args.batch_size), dtype=torch.float32).to(device)
            z_p = (1.0 - args.noise_alpha) * z_p + args.noise_alpha * z_noise
            y_p = decoder(z_p)
            y_p = F.log_softmax(y_p, dim=1)
            recon_p = torch.mean(torch.sum(-x_p * y_p, dim=1))

            z_noise = torch.tensor(np.random.dirichlet(np.ones(args.dim_z) * args.dirichlet_alpha, size=args.batch_size), dtype=torch.float32).to(device)
            z_n = (1.0 - args.noise_alpha) * z_n + args.noise_alpha * z_noise
            y_n = decoder(z_n)
            y_n = F.log_softmax(y_n, dim=1)
            recon_n = torch.mean(torch.sum(-x_n * y_n, dim=1))

            # mmd
            z_fake = encoder(x_q)
            z_fake = F.softmax(z_fake, dim=1)
            mmd_q = mmd_loss(z_true, z_fake, t=args.kernel_alpha)

            z_fake = encoder(x_p)
            z_fake = F.softmax(z_fake, dim=1)
            mmd_p = mmd_loss(z_true, z_fake, t=args.kernel_alpha)

            z_fake = encoder(x_n)
            z_fake = F.softmax(z_fake, dim=1)
            mmd_n = mmd_loss(z_true, z_fake, t=args.kernel_alpha)

            # loss
            loss = args.recon_alpha * (recon_q + recon_p + recon_n) + args.prior_alpha * (dist_p - dist_n) + (mmd_q + mmd_p + mmd_n)

            loss.backward()
            optimizer.step()

            end = time.time()
            print('[E%d] L %.6f | RQ %.6f | RP %.6f | RN %.6f | DP %.6f | DN %.6f (%d/%d) [%s]' % (epoch, loss.item(), args.recon_alpha * recon_q.item(), args.recon_alpha * recon_p.item(), args.recon_alpha * recon_n.item(), args.prior_alpha * dist_p.item(), args.prior_alpha* dist_n.item(), batch + 1, batch_len, format_time(start, end, batch_len, batch + 1)), end='\r', flush=True)

        print()

        # save log (train)
        train_stat.append({
            'epoch': epoch,
            'loss': loss.item(),
            'recon_q': recon_q.item(),
            'recon_p': recon_p.item(),
            'recon_n': recon_n.item(),
            'dist_p': dist_p.item(),
            'dist_n': dist_n.item(),
            'mmd_q': mmd_q.item(),
            'mmd_p': mmd_p.item(),
            'mmd_n': mmd_n.item()
        })
        save_json(train_stat, 'train_stat_%s.json' % args.dataset)

        # save model
        state = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'save_%s/model_%d.pt' % (args.dataset, epoch))

        # test
        with torch.no_grad():
            # 30000 각 단어가 768 토픽에 얼마나 가중치를 가지고 있는지,
            simmat = decoder.decode.weight # [dim_x, dim_z] [30000,768]
            #토픽과 유사도가 가장 높은 topk단어 찾기
            topic_words = torch.argsort(-simmat, dim=0)[:args.topic_topk, :].tolist() # [topic_topk, dim_z]
        ids_to_topic = [[] for _ in range(args.dim_x)]
        #각 단어가 어떤 토픽과 맞는지 정보 저장
        for i in range(args.topic_topk):
            for j in range(args.dim_z):
                ids_to_topic[topic_words[i][j]].append(j)
        metrics = test(args, dataset, ids_to_topic, True)
        print('| R@1000 | R@500 | R@200 | R@100 | R@30 |  P@20  | nDCG@20 |  MAP  |')
        print('| %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %6.2f | %7.2f | %5.2f |' % (metrics['recall_1000'] * 100, metrics['recall_500'] * 100, metrics['recall_200'] * 100, metrics['recall_100'] * 100, metrics['recall_30'] * 100, metrics['p_20'] * 100, metrics['ndcg_cut_20'] * 100, metrics['map'] * 100))
        # save log (test)
        metrics['epoch'] = epoch
        test_stat.append(metrics)
        save_json(test_stat, 'test_stat_%s.json' % args.dataset)

        if metrics['recall_1000'] > test_stat[best_epoch]['recall_1000']:
            best_epoch = epoch

    # full test for best model
    state_dict = torch.load('save_%s/model_%d.pt' % (args.dataset, best_epoch))
    with torch.no_grad():
        simmat = state_dict['decoder']['decode.weight'] # [dim_x, dim_z]
        topic_words = torch.argsort(-simmat, dim=0)[:args.topic_topk, :].tolist() # [topic_topk, dim_z]
    ids_to_topic = [[] for _ in range(args.dim_x)]
    for i in range(args.topic_topk):
        for j in range(args.dim_z):
            ids_to_topic[topic_words[i][j]].append(j)
    metrics = test(args, dataset, ids_to_topic, False)
    print('========== FULL TEST ==========')
    print('| R@1000 |  P@20  | nDCG@20 |  MAP  |')
    print('| %6.2f | %6.2f | %7.2f | %5.2f |' % (metrics['recall_1000'] * 100, metrics['p_20'] * 100, metrics['ndcg_cut_20'] * 100, metrics['map'] * 100))