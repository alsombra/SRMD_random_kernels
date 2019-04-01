import torch
import torch.nn as nn
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from model import SRMD
import numpy as np
from data_loader import Scaling
from utils import Kernels, load_kernels
from PIL import Image,ImageDraw
import os

torch.set_default_tensor_type(torch.cuda.DoubleTensor)


class Solver(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        
        # Model hyper-parameters
        self.num_blocks = config.num_blocks  # num_blocks = 11
        self.num_channels = config.num_channels  # num_channels = 18
        self.conv_dim = config.conv_dim # conv_dim = 128
        self.scale_factor = config.scale_factor  # scale_factor = 2

        # Training settings
        self.total_step = config.total_step # 20000
        self.lr = config.lr 
        self.beta1 = config.beta1 # 0.5  ????????????????? testar 0.9 ou 0.001
        self.beta2 = config.beta2 # 0.99  ???????????????
        self.trained_model = config.trained_model
        self.use_tensorboard = config.use_tensorboard
        self.start_step = -1

        #Test settings
        self.test_mode = config.test_mode
        self.test_image_path = config.test_image_path
        self.evaluation_step = config.evaluation_step
        self.evaluation_size = config.evaluation_size

        # Path and step size
        self.log_path = config.log_path
        self.result_path = config.result_path
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step # log_step = 10
        self.sample_step = config.sample_step # sample_step = 100
        self.model_save_step = config.model_save_step # model_save_step = 1000

        # Device configuration
        self.device = config.device

        # Initialize model
        self.build_model() 
        
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def build_model(self):
        # model and optimizer
        self.model = SRMD(self.num_blocks, self.num_channels, self.conv_dim, self.scale_factor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2])

        self.model.to(self.device)

    def load_trained_model(self):
        self.load(os.path.join(self.model_save_path, '{}'.format(self.trained_model)))
        print('loaded trained model (step: {})..!'.format(self.trained_model.split('.')[0])) 

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S['SR'])
        try:
            self.optimizer.load_state_dict(S['optimizer_state_dict'])
        except KeyError as error:
            print('There is no '+str(error)+' in loaded model. Loading model without optimizer_params')
        try:
            self.start_step = S['epoch'] - 1
        except KeyError as error:
            print('There is no '+str(error)+' in loaded model. Loading model without epoch info')
        
    #############################################
    def build_tensorboard(self):
        pass
    ######################################################
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.optimizer.zero_grad()
    ######################################################################
    def detach(self, x):  # NOT USED. To learn more SEE https://pytorch.org/blog/pytorch-0_4_0-migration-guide/
        return x.data
    #############################################################
   

    def get_trio_images(self, lr_image,hr_image, reconst):
        tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) for i in range(self.data_loader.batch_size)]
        image_hr_bicubic= np.stack(image_list)
        image_hr_bicubic_single = np.squeeze(image_hr_bicubic)
        print('hr_bicubic_single:', image_hr_bicubic_single.shape)
        #return this ^
        image_hr_bicubic = image_hr_bicubic.transpose(0,3,1,2)
        image_hr_bicubic = Scaling(image_hr_bicubic)
        image_hr_bicubic = torch.from_numpy(image_hr_bicubic).double().to(self.device) # NUMPY to TORCH
        hr_image_hat = reconst  
        hr_image_hat = hr_image_hat.data.cpu().numpy()
        hr_image_hat = np.squeeze(hr_image_hat).transpose((1, 2, 0))
        hr_image_hat = (hr_image_hat*255).astype(np.uint8)
        print('hr_image_hat : ', hr_image_hat.shape)
        #return this ^
        hr_image = hr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        hr_image = np.squeeze(hr_image.astype(np.uint8))
        #return this ^
        return Image.fromarray(image_hr_bicubic_single), Image.fromarray(hr_image_hat), Image.fromarray(hr_image)

    def create_grid(self, lr_image,hr_image, reconst):
        'generate grid image: LR Image | HR image Hat (from model) | HR image (original)'
        'lr_image = lr_image tensor from dataloader (can be batch)'
        'hr_image = hr_image tensor from dataloader (can be batch)'
        'reconst = output of model (HR residual)'
        tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) for i in range(self.data_loader.batch_size)]
        image_hr_bicubic= np.stack(image_list).transpose(0,3,1,2)
        image_hr_bicubic = Scaling(image_hr_bicubic)
        image_hr_bicubic = torch.from_numpy(image_hr_bicubic).double().to(self.device) # NUMPY to TORCH
        hr_image_hat = reconst               
        hr_image_hat = hr_image_hat.data.cpu().numpy()
        hr_image_hat = torch.from_numpy(hr_image_hat).double().to(self.device) # NUMPY to TORCH

        pairs = torch.cat((image_hr_bicubic.data, \
                                hr_image_hat.data,\
                                hr_image.data), dim=3)
        grid = make_grid(pairs, 1) 
        tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
        grid = (255 * tmp).astype(np.uint8)
        return grid
    
    def img_add_info(self, img_paths, img, epoch, loss):
        'receives tensor as img'
        added_text = Image.new('RGB', (500, img.shape[0]), color = 'white')
        d = ImageDraw.Draw(added_text)
        d.text((10,10), "model trained for {} epochs, loss (comparing residuals): {:.4f}".format(epoch, loss.item()) + \
               "\n" + '\n'.join([os.path.basename(path) for path in img_paths]), fill='black')
        imgs_comb = np.hstack((np.array(img), added_text))
        
        d.text((10,10), "model trained for {} epochs, loss (comparing residuals): {:.4f}".format(epoch, loss.item()), fill='black')
        
        imgs_comb = Image.fromarray(imgs_comb)
        return imgs_comb    
    
    def train(self):
        self.model.train()

        # Reconst loss
        reconst_loss = nn.MSELoss()

        # Data iter
        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)
        
        #Initialize steps
        start = self.start_step + 1 # if not loading trained start = 0     
        
        for step in range(start, self.total_step):
            self.model.train() # adicionei pq o cara no fim (p/ samples) colocou modo eval() e esqueceu de voltar
            
            # Reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0: 
                data_iter = iter(self.data_loader)

            img_paths, x, y = next(data_iter)
            lr_image = x[:,:3,:,:]
            hr_image = y
            lr_image,hr_image, x, y = lr_image.to(self.device), hr_image.to(self.device), x.to(self.device), y.to(self.device)
            
            hr_image.to(torch.float64)
            y = y.to(torch.float64)

            out = self.model(x)
            loss = reconst_loss(out, y)

            self.reset_grad()

            # For decoder
            loss.backward(retain_graph=True)

            self.optimizer.step()

            # Print out log info
            if (step+1) % self.log_step == 0:
                print("[{}/{}] loss: {:.4f}".format(step+1, self.total_step, loss.item()))

            # Sample images
            if (step+1) % self.sample_step == 0:
                self.model.eval()  #aqui ele botou o modelo em modo eval() mas esqueceu de voltar pro modo train
                reconst = self.model(x)

                def to_np(x):
                    return x.data.cpu().numpy()
                
                tmp = self.create_grid(lr_image,hr_image, reconst)
                imgs_comb = self.img_add_info(img_paths, tmp, step+1, loss)                
                #from IPython.display import display
                grid_PIL = imgs_comb
                grid_PIL.save('./samples/test_{}.jpg'.format(step + 1))
                if self.data_loader.batch_size == 1: #only saves separate images if batch == 1
                    lr_image_np = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
                    lr_image_np = Image.fromarray(np.squeeze(lr_image_np).astype(np.uint8))
                    hr_bic, hr_hat, hr = self.get_trio_images(lr_image,hr_image, reconst)
                    random_number = np.random.rand(1)[0]
                    lr_image_np.save('./samples/test_{}_lr.png'.format(step + 1))
                    hr_bic.save('./samples/test_{}_hr_bic.png'.format(step + 1))
                    hr_hat.save('./samples/test_{}_hr_hat.png'.format(step + 1))
                    hr.save('./samples/test_{}_hr.png'.format(step + 1))                
                    
            # Save check points
            if (step+1) % self.model_save_step == 0:                
                self.save(step, loss.item(), os.path.join(self.model_save_path, '{}.pth.tar'.format(str(step))))

    def save(self, step, current_loss, filename):
        model = self.model.state_dict()
        torch.save({
            'epoch': step+1,
            'SR': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': str(current_loss)
            }, filename)

    def test_and_error(self): #receives batch from dataloader
        'You run it for a random batch from test_set. You can change batch_size for len(test_set)'
        self.model.eval()
        epoch = self.start_step + 1 # if not loading trained start = 0 
             # Reconst loss
        reconst_loss = nn.MSELoss()

            # Data iter
        img_paths, x, y = next(self.data_iter)
        lr_image = x[:,:3,:,:]
        hr_image = y
        lr_image,hr_image, x, y = lr_image.to(self.device), hr_image.to(self.device), x.to(self.device), y.to(self.device)
            
        hr_image.to(torch.float64)
        y = y.to(torch.float64)
        reconst = self.model(x)
        loss = reconst_loss(reconst, y)

        # Print out log info 
        print("model trained for {} epochs, loss: {:.4f}".format(epoch, loss.item()))
        
        tmp = self.create_grid(lr_image, hr_image, reconst)
        grid_PIL = self.img_add_info(img_paths, tmp, epoch, loss)  
         #from IPython.display import display
        random_number = np.random.rand(1)[0]
        if self.data_loader.batch_size != 1:
            grid_PIL.save('./results/grids/{:.3f}_grid_{}.png'.format(random_number, self.start_step + 1))
        elif self.data_loader.batch_size == 1: #only saves separate images if batch == 1
            
            grid_PIL.save('./results/grids/'+ os.path.basename(img_paths[0])+'_grid_{}.png'.format(self.start_step + 1))

            hr_bic, hr_hat, hr = self.get_trio_images(lr_image,hr_image, reconst)

            lr_image_np = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
            lr_image_np = Image.fromarray(np.squeeze(lr_image_np).astype(np.uint8))

            lr_image_np.save('./results/LR_images_snapshot/'+ os.path.basename(img_paths[0])+'_lr_{}.png'.format(self.start_step + 1))
            hr_bic.save('./results/HR_bicub_images/'+ os.path.basename(img_paths[0])+'_hr_bic_{}.png'.format(self.start_step + 1))
            hr_hat.save('./results/HR_HAT_images/'+ os.path.basename(img_paths[0])+'_hr_hat_{}.png'.format(self.start_step + 1))
            hr.save('./results/HR_images/'+ os.path.basename(img_paths[0])+'_hr_{}.png'.format(self.start_step + 1))

    def evaluate(self):
        if self.evaluation_size == -1:
            self.evaluation_size = len(self.data_loader)
            
        if self.data_loader.batch_size != 1:
            print('WAIT! PASS --batch_size = 1 to do this first. Your batch_size is not 1')
            pass
        for step in range(self.evaluation_size):
            if (step+1) % self.evaluation_step == 0:
                [print() for i in range(10)]
                print("[{}/{}] tests".format(step+1, len(self.data_loader)))
                [print() for i in range(10)]
            self.model.eval() 
            self.test_and_error();

    def test(self): #receives single image --> can be easily modified to handle multiple images        
        'Takes single LR image as input. Returns LR image + (models approx) HR image concatenated'
        'image location must be given by flag --test_image_path'
        self.model.eval()
        step = self.start_step + 1 # if not loading trained start = 0 
        lr_image = Image.open(self.test_image_path)
        lr_image_size = lr_image.size[0]
        #CONSIDER RGB IMAGE
        
        from utils import Kernels, load_kernels
        K, P = load_kernels(file_path='kernels/', scale_factor=2)
        randkern = Kernels(K, P)

        # LR_image_scaled + LR_residual_scaled (CONCAT) ---> TO TORCH
        lr_image_scaled = Scaling(lr_image)
        lr_image_with_kernel = randkern.ConcatDegraInfo(lr_image_scaled)
        lr_image_with_kernel = torch.from_numpy(lr_image_with_kernel).float().to(self.device) # NUMPY to TORCH

        # LR_image to torch

        lr_image_scaled = torch.from_numpy(lr_image_scaled).float().to(self.device) # NUMPY to TORCH

        #Transpose - Permute since for model we need input with channels first
        lr_image_scaled = lr_image_scaled.permute(2,0,1) 
        lr_image_with_kernel = lr_image_with_kernel.permute(2,0,1)

        lr_image_with_kernel = lr_image_with_kernel.unsqueeze(0) #just add one dimension (index on batch)
        lr_image_scaled = lr_image_scaled.unsqueeze(0)

        lr_image, x = lr_image_scaled.to(torch.float64), lr_image_with_kernel.to(torch.float64) 
        lr_image, x = lr_image.to(self.device), x.to(self.device)

        x = x.to(torch.float64)

        reconst = self.model(x)

        tmp1 = lr_image.data.cpu().numpy().transpose(0,2,3,1)*255
        image_list = [np.array(Image.fromarray(tmp1.astype(np.uint8)[i]).resize((128,128), Image.BICUBIC)) \
                      for i in range(self.data_loader.batch_size)]
        image_hr_bicubic= np.stack(image_list)
        image_hr_bicubic_single = np.squeeze(image_hr_bicubic)
        #return this ^
        image_hr_bicubic = image_hr_bicubic.transpose(0,3,1,2)
        image_hr_bicubic = Scaling(image_hr_bicubic)
        image_hr_bicubic = torch.from_numpy(image_hr_bicubic).double().to(self.device) # NUMPY to TORCH
        hr_image_hat = reconst
        hr_image_hat_np = hr_image_hat.data.cpu().numpy()
        hr_image_hat_np_scaled = hr_image_hat_np #just to try different types of scaling. It already comes scaled
        hr_image_hat_np_scaled = np.squeeze(hr_image_hat_np_scaled).transpose((1, 2, 0))
        hr_image_hat_np_png = (hr_image_hat_np_scaled*255).astype(np.uint8)
        #return this ^

        #Saving Image Bicubic and HR Image Hat
        Image.fromarray(image_hr_bicubic_single).save('./results/HR_bicub_images/'+ os.path.basename(self.test_image_path)+'_hr_bic_{}.png'.format(step))
        Image.fromarray(hr_image_hat_np_png).save('./results/HR_HAT_images/'+ os.path.basename(self.test_image_path)+'_hr_hat_{}.png'.format(step))

        #Create Grid
        hr_image_hat_np_scaled = hr_image_hat_np #It's already scaled (comes out of model scaled)
        hr_image_hat_torch = torch.from_numpy(hr_image_hat_np_scaled).double().to(self.device) # NUMPY to TORCH

        pairs = torch.cat((image_hr_bicubic.data, \
                        hr_image_hat_torch.data), dim=3)
        grid = make_grid(pairs, 1) 
        tmp = np.squeeze(grid.cpu().numpy().transpose((1, 2, 0)))
        tmp = (255 * tmp).astype(np.uint8)
        random_number = np.random.rand(1)[0]
        Image.fromarray(tmp).save('./results/grids/'+ os.path.basename(self.test_image_path).split('.')[0]+'_grid_{}.png'.format(step))
        
    def many_tests(self):
        import glob
        TYPES = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        image_paths = []
        root = self.test_image_path
        for ext in TYPES:
            image_paths.extend(glob.glob(os.path.join(root, ext)))
        for img_path in image_paths:
            self.test_image_path = img_path
            self.test()

