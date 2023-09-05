import torch.nn as nn
import torch
import vn_layers as vn
import copy
from tqdm import tqdm


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def SDFLoss_multishape(sdf, prediction, x_latent, sigma):
    """Loss function introduced in the paper DeepSDF for multiple shapes."""
    l1 = torch.mean(torch.abs(prediction - sdf))
    l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))
    loss = l1 + l2
    #print(f'Loss prediction: {l1:.3f}, Loss regulariser: {l2:.3f}')
    return loss, l1, l2


def generate_latent_codes(latent_size, samples_dict):
    """Generate a random latent codes for each shape form a Gaussian distribution
    Returns:
        - latent_codes: np.array, shape (num_shapes, latent_size)
        - dict_latent_codes: key: obj_index, value: corresponding idx in the latent_codes array. 
                                  e.g.  latent_codes = ([ [1, 2, 3], [7, 8, 9] ])
                                        dict_latent_codes[345] = 0, the obj that has index 345 refers to 
                                        the 0-th latent code.
    """
    latent_codes = torch.tensor([], dtype=torch.float32).reshape(0, latent_size).to(device)
    #dict_latent_codes = dict()
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        #dict_latent_codes[obj_idx] = i
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes #, dict_latent_codes

class SDFModel(torch.nn.Module):
    def __init__(self, num_layers, skip_connections, latent_size, inner_dim=512, output_dim=1, pooling_type='max'):
        """
        SDF model for multiple shapes.
        Args:
            input_dim: 128 for latent space + 3 points = 131
        """
        super(SDFModel, self).__init__()

        # Num layers of the entire network
        self.num_layers = num_layers 

        # If skip connections, add the input to one of the inner layers
        self.skip_connections = skip_connections

        self.latent_size = latent_size

        # Dimension of the input space (3D coordinates)
        dim_coords = 3 
        input_dim = self.latent_size + dim_coords

        # Copy input size to calculate the skip tensor size
        self.skip_tensor_dim = copy.copy(input_dim)

        # Compute how many layers are not Sequential
        num_extra_layers = 2 if (self.skip_connections and self.num_layers >= 8) else 1
        
        # Add VN layers
        self.conv_pos=vn.VNLinearLeakyReLU(in_channels=input_dim,out_channels=input_dim,dim=5, negative_slope=0.0)
        self.conv1=vn.VNLinearLeakyReLU(in_channels=input_dim,out_channels=input_dim,dim=4, negative_slope=0.0)
        self.conv2=vn.VNLinearLeakyReLU(in_channels=input_dim*2,out_channels=input_dim,dim=4, negative_slope=0.0)
        self.bn1=vn.VNBatchNorm(input_dim, dim=4)
        if pooling_type == 'max':
            self.pool = vn.VNMaxPool(input_dim)
        elif pooling_type == 'mean':
            self.pool = vn.mean_pool
        self.fstn = vn.STNkd(pooling_type='max', d=input_dim)


        # Add sequential layers with VN layers
        layers = []
        for _ in range(num_layers - num_extra_layers):
            layers.append(nn.Sequential(nn.utils.weight_norm(nn.Linear(input_dim, inner_dim)), vn.VNLinearLeakyReLU_2(inner_dim,inner_dim)))
            input_dim = inner_dim
        self.net = nn.Sequential(*layers)
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Tanh())
        self.skip_layer = nn.Sequential(nn.Linear(inner_dim, inner_dim - self.skip_tensor_dim), vn.VNLinearLeakyReLU_2(inner_dim - self.skip_tensor_dim,inner_dim - self.skip_tensor_dim))


    def forward(self, x):
        """
        Forward pass
        Args:
            x: input tensor of shape (batch_size, 131). It contains a stacked tensor [latent_code, samples].
        Returns:
            sdf: output tensor of shape (batch_size, 1)
        """

        batch_size,feature_num=x.size()

        #VN layers forward pass
        input_data = x.clone().detach()
        x=x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x=self.conv_pos(x)
        x=self.pool(x)
        x=self.conv1(x)
        x_global=self.fstn(x)
        x = torch.cat((x.view(batch_size,feature_num,1), x_global), 1)
        x = self.conv2(x)
        
        x=x.view(batch_size,feature_num)
        #Sequential layers forward pass
        if self.skip_connections and self.num_layers >= 5:
            for i in range(3):
                x = self.net[i](x)
            x = self.skip_layer(x)
            x = torch.hstack((x, input_data))
            for i in range(self.num_layers - 5):
                x = self.net[3 + i](x)
            sdf = self.final_layer(x)
        else:
            if self.skip_connections:
                print('The network requires at least 5 layers to skip connections. Normal forward pass is used.')
            x = self.net(x)
            sdf = self.final_layer(x)
        return sdf


    def infer_latent_code(self, cfg, pointcloud, sdf_gt, writer, latent_code_initial):
        """Infer latent code from coordinates, their sdf, and a trained model."""

        latent_code = latent_code_initial.clone().detach().requires_grad_(True)
        
        optim = torch.optim.Adam([latent_code], lr=cfg['lr'])

        if cfg['lr_scheduler']:
            scheduler_latent = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', 
                                                    factor=cfg['lr_multiplier'], 
                                                    patience=cfg['patience'], 
                                                    threshold=0.001, threshold_mode='rel')

        best_loss = 1000000

        for epoch in tqdm(range(0, cfg['epochs'])):

            latent_code_tile = torch.tile(latent_code, (pointcloud.shape[0], 1))
            x = torch.hstack((latent_code_tile, pointcloud))

            optim.zero_grad()

            predictions = self(x)

            if cfg['clamp']:
                predictions = torch.clamp(predictions, -cfg['clamp_value'], cfg['clamp_value'])

            loss_value, l1, l2 = SDFLoss_multishape(sdf_gt, predictions, x[:, :self.latent_size], sigma=cfg['sigma_regulariser'])
            loss_value.backward()

            if writer is not None:
                writer.add_scalar('Reconstruction loss', l1.data.cpu().numpy(), epoch)
                writer.add_scalar('Latent code loss', l2.data.cpu().numpy(), epoch)

            optim.step()

            if l1.detach().cpu().item() < best_loss:
                best_loss = l1.detach().cpu().item()
                best_latent_code = latent_code.clone()

            # step scheduler and store on tensorboard (optional)
            if cfg['lr_scheduler']:
                scheduler_latent.step(loss_value.item())
                if writer is not None:
                    writer.add_scalar('Learning rate', scheduler_latent._last_lr[0], epoch)

                if scheduler_latent._last_lr[0] < 1e-6:
                    print('Learning rate too small, stopping training')
                    break

            # logging
            if writer is not None:
                writer.add_scalar('Inference loss', loss_value.detach().cpu().item(), epoch)

        return best_latent_code