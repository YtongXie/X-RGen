import torch


def build_optimizer(args, model):
    if (args.contras_loss_w > 0) and (args.fix_text_encoder is not True):
        ve_params = list(map(id, model.visual_extractor.parameters()))
        te_params = list(map(id, model.text_encoder_kd.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        ed_params = filter(lambda x: id(x) not in te_params, ed_params)

        optimizer = getattr(torch.optim, args.optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
            {'params': model.text_encoder_kd.parameters(), 'lr': args.lr_ve},
            {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    else:
        ve_params = list(map(id, model.visual_extractor.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        optimizer = getattr(torch.optim, args.optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
            {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
