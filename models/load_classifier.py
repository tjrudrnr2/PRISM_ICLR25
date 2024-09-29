def getClassifier(args, log = None):
    assert args is not None, 'please input the running classifier configuration'
    # assert len(args.netEncType) == 1, '일단은 1개의 네트워크만 받는다.'

    net = setClassifier(args.netEncType, args, log)

    set_requires_grad(net, False)
    
    check_requires_grad(net, log, args.netEncType)
    
    return net

def set_requires_grad(net, required):
    for params in net.parameters():
        params.requires_grad = required


def check_requires_grad(net, log, desc=None):
    state = False
    for name, param in net.named_parameters():
        if param.requires_grad == True:
            log.info("\t",name)
            state = True
    if not state:
        if desc is None:
            desc = 'Network'
        
def setClassifier(netEncType, opt, log):
    net = None
    if netEncType == 'vgg19-pytorch':
        from models.vgg_pytorch import VGG19 as VGG19Pytorch
        ###################################################
        #   VGG 19 of Pytorch official Loading
        ###################################################
        net = VGG19Pytorch(
            get_perceptual_feats=True,
            num_classes=opt.numClassesInFtrExt,
            image_size=opt.imageSize,
            batch_norm = True,
            setEncToEval=opt.setEncToEval
            )    
    else:
        net = None
    return net