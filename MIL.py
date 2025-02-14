class MIL_fc(torch.nn.Module):
    def __init__(self, size_arg = "small", dropout = 0.5, n_classes = 2, top_k=1,
                 embed_dim = 64, batch_size = 1):
        super().__init__()
        assert n_classes == 2
        self.size_dict = {"small": [64, 64]}
        size = self.size_dict[size_arg]
        graph_model = HAN(input_size=23,hidden_channels=[128],heads=1,pooling_ratio=0.75)
        fc = [graph_model, torch.nn.Linear(size[0], size[1]), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        self.fc = torch.nn.Sequential(*fc)
        self.classifier=  torch.nn.Linear(size[1], n_classes)
        self.top_k=top_k
        self.batch_size = batch_size

    def relocate(self):
        self.fc = self.fc.to(device)
        self.classifiers = self.classifiers.to(device)
        
    def forward(self, h, return_features=False):
        h = [self.fc(item) for item in h]
        h = torch.stack(h, dim = 1)
        _ = h.to(device)
        logits  = self.classifier(h) # K x 2

        y_probs = F.softmax(logits, dim = 2)

        top_instance_idx = torch.topk(y_probs[:, :, 1], self.top_k, dim=1)[1].view(-1)

        top_instance = [torch.index_select(slide_logits, dim=0, index=top_instance_idx[slide]).unsqueeze(0) for slide, slide_logits in enumerate(logits)]
        if self.batch_size >= 2:
            top_instance = torch.stack(top_instance, dim=0)

        top_instance = torch.cat(tuple(top_instance), dim = 0)
        Y_hat = torch.topk(top_instance, self.top_k, dim = 2)[1]
        Y_prob = F.softmax(top_instance, dim = 2)
        return top_instance, Y_prob, Y_hat, y_probs


class CLAM_SB(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = 0., k_sample=2, n_classes=2, inst_loss_type = 'ce', subtyping=False, embed_dim=64, batch_size = 1):
        super().__init__()
        self.size_dict = {"small": [64, 64, 64], "big": [64, 64, 64]}
        size = self.size_dict[size_arg]
        graph_model = HAN(input_size=23,hidden_channels=[128],heads=1,pooling_ratio=0.75)
        fc = [graph_model, nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers =  nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2)  for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.batch_size = batch_size
        if inst_loss_type =='ce':
            instance_loss_fn=nn.CrossEntropyLoss()
        if inst_loss_type == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda(device)
        self.instance_loss_fn = instance_loss_fn
    
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()
    
    #instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier): 
        k_sample_t = self.k_sample
        if len(A.shape) == 1:
            A = A.view(1, -1)
        if A.shape[0] ==1:
            k_sample_t = 1
        top_p_ids = torch.topk(A, k_sample_t, dim=0)[1].squeeze()
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample_t, dim=0)[1].squeeze()
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample_t, device)
        n_targets = self.create_negative_targets(k_sample_t, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0).to(device)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances).to(device)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def relocate(self):
        self.attention_net = self.attention_net.to(device)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A = [self.attention_net(item)[0] for item in h]
        h = [self.attention_net(item)[1] for item in h]
        A = torch.stack(A, dim = 1)
        h = torch.stack(h, dim = 1)
        _ = A.to(device)
        _ = h.to(device)

        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        cur_bs = len(A)
        total_inst_loss = torch.zeros(cur_bs, 1).to(device)
        
        if instance_eval:
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
            for s in range(0, cur_bs):
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[s,i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1: #in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A[s], h[s], classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else: #out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss[s] += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)
  
        M = [torch.mm(torch.transpose(A[slide], 1, 0), h[slide]) for slide, ele in enumerate(A)]
        if self.batch_size >= 2:
            M = torch.stack(M, dim=0)
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim = 2)[1]
        Y_prob = F.softmax(logits, dim = 2)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets), 
            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, total_inst_loss
