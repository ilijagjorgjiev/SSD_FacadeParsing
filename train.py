import argparse
from ssd_project.functions.detection import *
from ssd_project.functions.multiboxloss import *
from ssd_project.model.ssd import *
from ssd_project.utils.global_variables import *
from ssd_project.utils.helpers import *
from ssd_project.utils.transformations import *
from ssd_project.utils.utils import *

def main(args):
    global device
    device = DEVICE
    best_loss = BEST_LOSS
    assert args.epochs > 0
    assert args.batch_size == 8
    torch.manual_seed(66)
    np.random.seed(66)
    start_epoch = START_EPOCH

    if(args.pretrained_model is not None):
        print("LOADED MODEL")
        best_model = torch.load(args.pretrained_model)
        model_state_dict =  best_model["model_state_dict"]
        start_epoch = best_model["epoch"]
        best_loss = best_model["loss"]
        epochs_since_improvement = best_model["epochs_since_improvement"]
        model = build_ssd(num_classes = NUM_CLASSES)
        model.load_state_dict(model_state_dict)
        t_loss_normal, t_loss_avg = best_model["training_losses_batch_values"], best_model["training_losses_batch_avgs"]
        v_loss_normal, v_loss_avg = best_model["validation_losses_batch_values"], best_model["validation_losses_batch_avgs"]

        print("Model LOADED SUCCESSFULLY")

    else:
        v_loss_avg, v_loss_normal = [], []
        t_loss_avg, t_loss_normal = [], []
        #build SSD model
        model = build_ssd(num_classes = NUM_CLASSES)

        # initialize newly added layers' weights with xavier method
        model.vgg.load_state_dict(VGG16_WEIGHTS_PRETRAINED)
        model.extras.apply(weights_init)
        model.loc.apply(weights_init)
        model.conf.apply(weights_init)

        biases = []
        not_biases = []

    #Initialize and SGD optimizer, with 2 times bigger learning rate
    #Done in original CAFFE REPO - https://github.com/weiliu89/caffe/tree/ssd
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)

    optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * args.lr}, {'params': not_biases}],
                            lr=args.lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    #Create Custom Datasets with applied transformations for both training and validation
    train_dataset = TrainDataset(args.path_imgs, args.path_bboxes, args.path_labels, "TRAIN", args.split_ratio)
    val_dataset = TrainDataset(args.path_imgs, args.path_bboxes, args.path_labels, "TEST", args.split_ratio)

    #Create the DataLoader from for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size,
                                                shuffle = True, collate_fn = train_dataset.collate_fn,
                                                num_workers = WORKERS, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size,
                                          shuffle = True, collate_fn = val_dataset.collate_fn,
                                          num_workers = WORKERS, pin_memory = True)
    model = model.to(device)
    loss_function = MultiBoxLoss(model.priors_cxcy).to(device)

    for epoch in range(start_epoch, args.epochs):

        train_losses = train(train_loader=train_loader,
              model=model,
              loss_function=loss_function,
              optimizer=optimizer,
              epoch=epoch)

        val_losses = validate(val_loader = val_loader,
                          model = model,
                          loss_function = loss_function)

        v_loss_avg.append(val_losses.avg)
        t_loss_avg.append(train_losses.avg)

        v_loss_normal.append(val_losses.val)
        t_loss_normal.append(train_losses.val)

        is_best = val_losses.avg < best_loss
        best_loss = min(val_losses.avg, best_loss)

        if not is_best:
            epochs_since_improvement +=1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0
            save_best_trained(epoch, epochs_since_improvement, model, optimizer, best_loss,
                              (t_loss_normal, t_loss_avg), (v_loss_normal, v_loss_avg))

    return model, optimizer, best_loss, epochs_since_improvement, args.epochs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an SSD model')

    parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=EPOCHS)
    parser.add_argument('--split-seed', action="store", dest="split_ratio", type=int, default=SPLIT_RATIO)
    parser.add_argument('--batch-train', action="store", dest="batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', action='store', dest='lr', type=float, default=1e-3 )
    parser.add_argument('--model', action='store', dest='pretrained_model', default=None)
    parser.add_argument('--path_imgs', action='store', dest='path_imgs', type=str,         default='/data/ssd_ilija_data/original_images/')
    parser.add_argument('--path_bboxes', action='store', dest='path_bboxes', type=str, default='/data/ssd_ilija_data/ground_truth/bboxes_labels/')
    parser.add_argument('--path_labels', action='store', dest='path_labels', type=str, default='/data/ssd_ilija_data/ground_truth/bboxes_labels/')

    main(parser.parse_args())
