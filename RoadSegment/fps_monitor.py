import torch
from torch.serialization import load
from ex import BiSeNet
from tqdm import tqdm
import time
from helpers import load_checkpoint

def speed_testing():

    # cuDnn configurations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    model = BiSeNet(2)
    
    
    print("     + {} Speed testing... ...")
    model = model.to('cuda')
    random_input = torch.randn(1,3,480,600).to('cuda')

    model.eval()
    load_checkpoint(torch.load('checkpoints/best_model.pth'),model)

    time_list = []
    for i in tqdm(range(1000)):
        torch.cuda.synchronize()
        tic = time.time()
        model(random_input)
        torch.cuda.synchronize()
        # the first iteration time cost much higher, so exclude the first iteration
        #print(time.time()-tic)
        time_list.append(time.time()-tic)
    time_list = time_list[1:]
    print("     + Done 10000 iterations inference !")
    print("     + Total time cost: {}s".format(sum(time_list)))
    print("     + Average time cost: {}s".format(sum(time_list)/1000))
    print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/1000)))

speed_testing()
