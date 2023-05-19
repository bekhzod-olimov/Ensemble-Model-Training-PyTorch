from matplotlib import pyplot as plt
import torch, random, os
import numpy as np
from torchvision import transforms as tfs
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, metric_to_track = "loss", patience = 5, threshold = 0):

        assert metric_to_track in ["loss", "acc"], "Kuzatadigan metric acc yoki loss bo'lishi kerak!"
        self.metric_to_track, self.patience, self.threshold, self.counter, self.early_stop = metric_to_track, patience, threshold, 0, False
        self.best_value = torch.tensor(float("inf")) if metric_to_track == "loss" else torch.tensor(float("-inf"))
        self.di = {}; self.di[str(self.counter)] = False
        
    def __call__(self, current_value): 
        
        print(f"\n{self.metric_to_track} ni kuzatyapmiz!")
        
        if self.metric_to_track == "loss":
            if current_value > (self.best_value + self.threshold): self.counter += 1
            else: self.best_value = current_value
                
        elif self.metric_to_track == "acc":
            
            if current_value < (self.best_value + self.threshold): self.counter += 1
            else: self.best_value = current_value
            
        for counter, value in self.di.items():
            if int(counter) == self.counter and value == False and int(counter) != 0:
                print(f"{self.metric_to_track} {counter} marta o'zgarmadi!")
        
        self.di[str(self.counter)] = True; self.di[str(self.counter + 1)] = False
                
        if self.counter >= self.patience: 
            print(f"\n{self.metric_to_track} {self.patience} marta o'zgarmaganligi uchun train jarayoni yakunlanmoqda...")
            self.early_stop = True

def visualize(ds, num_ims, row, cmap = None, cls_names = None):
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(ds) - 1) for _ in range(num_ims)]
    for idx, indeks in enumerate(indekslar):
        
        im, gt = ds[indeks]
        # Start plot
        plt.subplot(row, num_ims // row, idx + 1)
        if cmap:
            plt.imshow(tensor_2_im(im), cmap='gray')
        else:
            plt.imshow(tensor_2_im(im))
        plt.axis('off')
        if cls_names is not None:
            plt.title(f"GT -> {cls_names[str(gt)]}")
        else:
            plt.title(f"GT -> {gt}")
            
def data_tekshirish(ds):
    
    data = ds[0]    
    print(f"Dataning birinchi elementining turi: {type(data[0])}")
    print(f"Dataning ikkinchi elementining turi: {type(data[1])}")
    print(f"Dataning birinchi elementining hajmi: {(data[0]).shape}")
    print(f"Dataning birinchi elementidagi piksel qiymatlari: {np.unique(np.array(data[0]))}")
    print(f"Dataning ikkinchi elementi: {data[1]}")
    

def tensor_2_im(t, t_type = "rgb"):
    
    assert t_type in ["rgb", "gray"], "Rasm RGB yoki grayscale ekanligini aniqlashtirib bering."
    
    gray_tfs = tfs.Compose([tfs.Normalize(mean = [ 0.], std = [1/0.5]), tfs.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = tfs.Compose([tfs.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), tfs.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return ((t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def parametrlar_soni(model): 
    for name, param in model.named_parameters():
        print(f"{name} parametrida {param.numel()} ta parametr bor.")
    print(f"Modelning umumiy parametrlar soni -> {sum(param.numel() for param in model.parameters() if param.requires_grad)} ta.")
    
def train_setup(): return 20, "cuda:3", torch.nn.CrossEntropyLoss()

def train(models_di, device, epochs, tr_dl, val_dl, loss_fn, print_freq = 5):
    
    trained_models_di = {}
    
    for model_name, m in models_di.items():
        
        m.to(device)
        model_to_save = m        
        print(f"{model_name} bilan train boshlandi...")

        optimizer = torch.optim.Adam(m.parameters(), lr = 0.001)

        best_acc = 0
        for epoch in range(epochs):

            epoch_loss, epoch_acc, total = 0, 0, 0
            for idx, batch in tqdm(enumerate(tr_dl)):
                ims, gts = batch
                ims, gts = ims.to(device), gts.to(device)

                total += ims.shape[0]

                preds = m(ims)
                _, pred_cls = torch.max(preds.data, dim = 1)
                # print(pred_cls)
                # print(gts)
                loss = loss_fn(preds, gts)
                # loss = loss_fn(preds, gts.unsqueeze(1))

                epoch_acc += (pred_cls == gts).sum().item()
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            m.eval()
            with torch.no_grad():
                val_epoch_loss, val_epoch_acc, val_total = 0, 0, 0
                for idx, batch in enumerate(val_dl):
                    ims, gts = batch
                    ims, gts = ims.to(device), gts.to(device)
                    val_total += ims.shape[0]

                    preds = m(ims)
                    loss = loss_fn(preds, gts)
                    _, pred_cls = torch.max(preds.data, dim = 1)
                    val_epoch_acc += (pred_cls == gts).sum().item()
                    val_epoch_loss += loss.item()

                val_acc = val_epoch_acc / val_total

                if (epoch + 1) % print_freq == 0: 
                    print(f"Epoch {epoch + 1} train jarayoni tugadi")
                    print(f"Epoch {epoch + 1} dagi train loss -> {(epoch_loss / len(tr_dl)):.3f}")
                    print(f"Epoch {epoch + 1} dagi train accuracy -> {(epoch_acc / total):.3f}")
                    print(f"Epoch {epoch + 1} validation jarayoni tugadi")
                    print(f"Epoch {epoch + 1} dagi validation loss -> {(val_epoch_loss / len(val_dl)):.3f}")
                    print(f"Epoch {epoch + 1} dagi validation accuracy -> {val_acc:.3f}")

                if val_acc > best_acc:
                    os.makedirs("saved_models", exist_ok=True)
                    best_acc = val_acc
                    checkpoint_path = f"saved_models/{model_name}_best_model.pth"
                    torch.save(m.state_dict(), checkpoint_path)
                    
        print(f"{model_name} bilan train yakunlandi!")
        model_to_save.load_state_dict(torch.load(checkpoint_path))
        trained_models_di[model_name] = model_to_save
    
    return trained_models_di        
    
def inference(model, device, test_dl, num_ims, row, cls_names = None):
    
    preds, images, lbls = [], [], []
    correct, total = 0., 0
    for idx, data in enumerate(test_dl):
        im, gt = data
        total += im.shape[0]
        im, gt = im.to(device), gt.to(device)
        _, pred = torch.max(model(im), dim = 1)
        correct += (pred == gt).sum().item()
        images.append(im)
        preds.append(pred)
        lbls.append(gt)
    print(f"\nAccuracy -> {correct / total}\n")
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(images) - 1) for _ in range(num_ims)]
    for idx, indeks in enumerate(indekslar):
        
        im = images[indeks].squeeze()
        # Start plot
        plt.subplot(row, num_ims // row, idx + 1)
        plt.imshow(tensor_2_im(im)[:, :, 0], cmap='gray')
        plt.axis('off')
        if cls_names is not None: plt.title(f"GT -> {cls_names[(lbls[indeks][0])]} ; Prediction -> {cls_names[(preds[indeks][0])]}", color=("green" if {cls_names[(lbls[indeks][0])]} == {cls_names[(preds[indeks][0])]} else "red"))
        else: plt.title(f"GT -> {lbls[indeks][0]} ; Prediction -> {preds[indeks][0]}")
        
        
def ensemble_inference(models, num_models, device, test_dl, num_ims, row, cls_names = None):
    
    preds, images, lbls = [], [], []
    correct, total = 0., 0
    with torch.no_grad():
        for idx, data in enumerate(test_dl):
            # if idx == 1: break
            im, gt = data
            total += im.shape[0]
            im, gt = im.to(device), gt.to(device)

            predictions = []
            for i in range(num_models):
                pred = models[f"model_{i + 1}"](im)
                predictions.append(pred.detach().cpu().numpy())
            ensemble_pred = np.argmax(np.sum(predictions, axis = 0), axis = 1)
            correct += np.sum(ensemble_pred == [g.item() for g in gt])
            images.append(im); preds.append(ensemble_pred); lbls.append(gt)
        print(f"\nAccuracy -> {correct / total}\n")

    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(images) - 1) for _ in range(num_ims)]
    for idx, indeks in enumerate(indekslar):

        im = images[indeks].squeeze()
        # Start plot
        plt.subplot(row, num_ims // row, idx + 1)
        plt.imshow(tensor_2_im(im)[:, :, 0], cmap='gray')
        plt.axis('off')
        if cls_names is not None: plt.title(f"GT -> {cls_names[(lbls[indeks][0])]} ; Prediction -> {cls_names[(preds[indeks][0])]}", color=("green" if {cls_names[(lbls[indeks][0])]} == {cls_names[(preds[indeks][0])]} else "red"))
        else: plt.title(f"GT -> {lbls[indeks][0]} ; Prediction -> {preds[indeks][0]}") 