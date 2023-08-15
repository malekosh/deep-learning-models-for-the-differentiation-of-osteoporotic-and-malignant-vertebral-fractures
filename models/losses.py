import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch
class CrossEntropyLoss(_Loss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # self.nll_loss = nn.CrossEntropyLoss(weight)
        # self.cross_entropy_loss = CrossEntropyLoss2d()

    def forward(self, input, target, weight=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """

        if weight is not None:
            y_1 = F.cross_entropy(input=input, target=target, weight=weight, reduction='mean')
        else:
            y_1 = F.cross_entropy(input=input, target=target, weight=None, reduction='mean')

        return y_1
    
class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()
        # self.nll_loss = nn.CrossEntropyLoss(weight)
        # self.cross_entropy_loss = CrossEntropyLoss2d()

    def forward(self, input, target, weight=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """

        y_1 = F.mse_loss(input=input, target=target, reduction='mean')


        return y_1
    

class KappaLoss(_Loss):
    def __init__(self, num_classes, weightage, device):
        super(KappaLoss, self).__init__()
        if weightage not in ("linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")
        
        self.num_classes = num_classes
        self.weightage = weightage
        self.eps = 1e-6
        
        label_vec = torch.range(1,num_classes)
        self.row_label_vec = torch.reshape(label_vec, (1, num_classes)).float().to(device)
        self.col_label_vec = torch.reshape(label_vec, (num_classes, 1)).float().to(device)
        
        col_mat = torch.tile(self.col_label_vec, (1, num_classes))
        row_mat = torch.tile(self.row_label_vec, (num_classes, 1))
        if weightage == "linear":
            self.weight_mat = torch.abs(col_mat - row_mat)
        else:
            self.weight_mat = (col_mat - row_mat) ** 2
        self.weight_mat = self.weight_mat.to(device)
    def forward(self, y_pred, y_true):
        batch_size = y_true.size()[0]
        y_true = y_true.float()
        y_pred = y_pred.float()
#         print('Y_TRUE size ', y_true.view(1,-1).size())
#         print('col_label_vec size ', self.col_label_vec.size())
        cat_labels = torch.matmul( y_true.float().view(-1,1),self.col_label_vec.view(1,-1))
#         print('size cat_labels', cat_labels.size())
        cat_label_mat = cat_labels#torch.tile(cat_labels, (1, self.num_classes))
        row_label_mat = torch.tile(self.row_label_vec, (batch_size, 1))
        if self.weightage == "linear":
#             print('size cat_label_mat', cat_label_mat.size())
#             print('size row_label_mat', row_label_mat.size())
            weight = torch.abs(cat_label_mat - row_label_mat)
        else:
            weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, dim=0, keepdim=True)
        pred_dist = torch.sum(y_pred, dim=0, keepdim=True)
        w_pred_dist = torch.matmul(self.weight_mat, pred_dist.view(-1,1))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist.view(1,-1)))
        denominator /= batch_size
        loss = torch.div(numerator, denominator)
        return torch.log(loss + self.eps)
        
    