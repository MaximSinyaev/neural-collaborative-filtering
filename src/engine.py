import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np

from .utils import save_checkpoint, use_optimizer
from .metrics import MetronAtK

def bpr_loss(positive_predictions, negative_predictions, mask=None):
    """
    Bayesian Personalised Ranking [1]_ pairwise loss function.
    Parameters
    ----------
    positive_predictions: tensor
        Tensor containing predictions for known positive items.
    negative_predictions: tensor
        Tensor containing predictions for sampled negative items.
    mask: tensor, optional
        A binary tensor used to zero the loss from some entries
        of the loss tensor.
    Returns
    -------
    loss, float
        The mean value of the loss function.
    References
    ----------
    .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from
       implicit feedback." Proceedings of the twenty-fifth conference on
       uncertainty in artificial intelligence. AUAI Press, 2009.
    """

    loss = (1.0 - torch.sigmoid(positive_predictions -
                                negative_predictions))

    if mask is not None:
        mask = mask.float()
        loss = loss * mask
        return loss.sum() / mask.sum()

    return loss.mean()

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config, use_bpr_loss=False):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.use_bpr_loss = use_bpr_loss
        if self.use_bpr_loss:
            self.crit = bpr_loss
        else:    
            self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        if self.use_bpr_loss:
            loss = self.crit(ratings_pred[ratings > 0.], ratings_pred[ratings == 0.])
        else:
            loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in tqdm(enumerate(train_loader), disable=True):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Training Epoch {}] Batch {}, Loss {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        print(f'[Trained Epoch {epoch_id}] batch num {batch_id + 1}, mean Loss {total_loss / (batch_id + 1)}')
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        hit_ratio_batch = []
        ndcg_batch = []
        pos_loss = 0
        neg_loss = 0
        eval_loss = 0
        with torch.no_grad():
            for i, evaluate_data in enumerate(evaluate_data_loader):
                test_users, test_items = evaluate_data[0], evaluate_data[1]
                negative_users, negative_items = evaluate_data[2], evaluate_data[3]
                if self.config['use_cuda'] is True:
                    test_users = test_users.cuda()
                    test_items = test_items.cuda()
                    negative_users = negative_users.cuda()
                    negative_items = negative_items.cuda()
                test_scores = self.model(test_users, test_items)
                negative_scores = self.model(negative_users, negative_items)
                if self.use_bpr_loss:
                    batch_loss = self.crit(test_scores, negatives_scores)
                else:
                    batch_pos_loss = self.crit(test_scores.view(-1), torch.tensor([1.] * len(test_scores)).cuda())
                    batch_neg_loss = self.crit(test_scores.view(-1), torch.tensor([0.] * negative_items.size(0)).view(-1).cuda())
                if self.config['use_cuda'] is True:
                    test_users = test_users.cpu()
                    test_items = test_items.cpu()
                    test_scores = test_scores.cpu()
                    negative_users = negative_users.cpu()
                    negative_items = negative_items.cpu()
                    negative_scores = negative_scores.cpu()
                self._metron.subjects = [test_users.data.view(-1).tolist(),
                                     test_items.data.view(-1).tolist(),
                                     test_scores.data.view(-1).tolist(),
                                     negative_users.data.view(-1).tolist(),
                                     negative_items.data.view(-1).tolist(),
                                     negative_scores.data.view(-1).tolist()]
                hit_ratio_batch.append(self._metron.cal_hit_ratio())
                ndcg_batch.append(self._metron.cal_ndcg())
                if self.use_bpr_loss:
                    eval_loss += batch_loss
                else:
                    pos_loss += batch_pos_loss
                    neg_loss += batch_neg_loss
                    eval_loss += batch_pos_loss + batch_neg_loss
        eval_loss = eval_loss / (i + 1)
        if not self.use_bpr_loss:
            pos_loss = pos_loss / (i + 1)
            neg_loss = neg_loss / (i + 1)
        hit_ratio = np.mean(hit_ratio_batch)
        ndcg = np.mean(ndcg_batch)
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        if not self.use_bpr_loss:
            print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}, total_loss = {:.4f}, pos_loss = {:.4f}, neg_loss = {:.4f}'\
                  .format(epoch_id, hit_ratio, ndcg, eval_loss, pos_loss, neg_loss))
        else:
            print('[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}, total_loss = {:.4f}'\
                  .format(epoch_id, hit_ratio, ndcg, eval_loss))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)