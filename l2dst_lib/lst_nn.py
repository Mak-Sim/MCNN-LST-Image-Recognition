import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import gelu, max_pool2d, relu, gelu, tanh, silu

class L2DST(nn.Module):
    """Learned 2D separable transform ."""
    def __init__(self, din, dout, device, p_drop = 0.1):
        super().__init__()
        
        #TODO: add check where din/dount have two elements
        self.fc1 = nn.Linear(din[1], dout[1], device=device)
        self.fc2 = nn.Linear(din[0], dout[0], device=device)
        
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
        torch.nn.init.zeros_(self.fc1.bias)            
        torch.nn.init.zeros_(self.fc2.bias)
        
        self.drop1 = nn.Dropout(p = p_drop)
        self.drop2 = nn.Dropout(p = p_drop)

    def forward(self, X):
        out = torch.nn.functional.tanh(self.fc1(self.drop1(X)))
        out1 = torch.transpose(out, -1, -2)
        out = torch.nn.functional.tanh(self.fc2(self.drop2(out1)))
        out = torch.transpose(out, -2, -1)
        return out

    def get_embeddings(self,X):
        out = torch.nn.functional.tanh(self.fc1(X))
        e1 = out
        out = torch.transpose(e1, -1, -2)
        out = torch.nn.functional.tanh(self.fc2(out))
        e2 = out
        out = torch.transpose(e2, -2, -1)
        
        return e1,e2

class L2DST_silu(nn.Module):
    """Learned 2D separable transform ."""
    def __init__(self, din, dout, device, p_drop = 0.1):
        super().__init__()
        
        #TODO: add check where din/dount have two elements
        self.fc1 = nn.Linear(din[1], dout[1], device=device)
        self.fc2 = nn.Linear(din[0], dout[0], device=device)
        
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        
        torch.nn.init.zeros_(self.fc1.bias)            
        torch.nn.init.zeros_(self.fc2.bias)
        
        self.drop1 = nn.Dropout(p = p_drop)
        self.drop2 = nn.Dropout(p = p_drop)

    def forward(self, X):
        out = silu(self.fc1(self.drop1(X)))
        out1 = torch.transpose(out, -1, -2)
        out = silu(self.fc2(self.drop2(out1)))
        out = torch.transpose(out, -2, -1)
        return out

    def get_embeddings(self,X):
        out = silu(self.fc1(X))
        e1 = out
        out = torch.transpose(e1, -1, -2)
        out = silu(self.fc2(out))
        e2 = out
        out = torch.transpose(e2, -2, -1)
        
        return e1,e2
    
class L2DST_ge(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, input_size, ffn_num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(input_size[1], ffn_num_outputs[1])
        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)        
        self.fc2 = nn.Linear(input_size[0], ffn_num_outputs[0])
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)

    def forward(self, X):
        out = torch.nn.functional.gelu(self.drop1(self.fc1(X)))
        out1 = torch.transpose(out, -1, -2)
        out = torch.nn.functional.gelu(self.drop2(self.fc2(out1)))
        out = torch.transpose(out, -2, -1)
        return out

    def get_embeddings(self,X):
        out = torch.nn.functional.gelu(self.fc1(X))
        e1 = out
        out = torch.transpose(e1, -1, -2)
        out = torch.nn.functional.gelu(self.fc2(out))
        e2 = out
        out = torch.transpose(e2, -2, -1)
        
        return e1,e2
        
class LST_1(nn.Module):
    def __init__(self, input_size, output_size, num_classes=10, device='cpu', p_drop_lst = 0.1, p_drop_fc = 0.1):
        super(LST_1, self).__init__()
        
        self.num_classes = num_classes
        self.L2DST = L2DST(input_size, output_size, device=device, p_drop = p_drop_lst)
        
        self.dropout = nn.Dropout(p=p_drop_fc)
        self.W_o = nn.Linear(output_size[0]*output_size[1], num_classes, device=device)
        
        nn.init.xavier_normal_(self.W_o.weight)
        torch.nn.init.zeros_(self.W_o.bias)

    def forward(self, x):
        # Multiply input by weights and add biases
        
        out = self.L2DST(x)
        
        out = out.reshape(out.shape[0],-1)
        
        out = self.W_o(self.dropout(out))
        
        return out
    
    def get_embeddings(self,x):
        e1,e2 = self.L2DST.get_embeddings(x)
        
        return e1,e2
    
    def get_prob(self,x):
        out = self.L2DST(x)
        
        out = out.reshape(out.shape[0],-1)
        
        out = self.W_o(out)
        
        return out
    
def multichan_to_2D(x):
    """
    Преобразует тензор из формы (B, C, W, H) в (B, W * s, H * s),
    где s = sqrt(C) - целое число
    """
    B, C, W, H = x.shape
    
    # Проверяем, что C является полным квадратом
    s = int(np.sqrt(C))
    assert s * s == C, "Количество каналов должно быть полным квадратом"
    
    # Изменяем форму тензора
    # 1. Разделяем каналы на s x s блоков
    x = x.view(B, s, s, W, H)
    
    # 2. Переставляем оси для правильного порядка
    # Теперь форма: (B, s, W, s, H)
    x = x.permute(0, 1, 3, 2, 4)
    
    # 3. Объединяем измерения
    # Форма становится: (B, s * W, s * H)
    x = x.reshape(B, s * W, s * H)

    return x

class MultiConv4_LST(nn.Module):
    
    def __init__(self, c1_kernels = 16, lst_out=4, num_classes=10, p_drop_lst = 0.1, p_drop_fc = 0.1, device='cpu'):
        super(MultiConv4_LST, self).__init__()
        
        self.conv1 = nn.Conv2d(1, c1_kernels, 2, padding='same')
        self.conv2 = nn.Conv2d(1, c1_kernels, 3)
        self.conv3 = nn.Conv2d(1, c1_kernels, 4, padding='same') 
        self.conv4 = nn.Conv2d(1, c1_kernels, 5)
        
        self.BN1_1 = nn.BatchNorm2d(1)
        self.BN1_2 = nn.BatchNorm2d(1)
        self.BN1_3 = nn.BatchNorm2d(1)
        self.BN1_4 = nn.BatchNorm2d(1)
        
        self.BN2 = nn.BatchNorm1d(4*lst_out * lst_out)
        
        self.lst_out = lst_out

        s = int(np.sqrt(c1_kernels))
        assert s * s == c1_kernels, "Количество каналов должно быть полным квадратом"
        
        self.lst1 = L2DST([14*s, 14*s], [lst_out, lst_out], device=device, p_drop=p_drop_lst)
        self.lst2 = L2DST([13*s, 13*s], [lst_out, lst_out], device=device, p_drop=p_drop_lst)
        self.lst3 = L2DST([14*s, 14*s], [lst_out, lst_out], device=device, p_drop=p_drop_lst)
        self.lst4 = L2DST([12*s, 12*s], [lst_out, lst_out], device=device, p_drop=p_drop_lst)
        
        self.fc1 = nn.Linear(4*lst_out * lst_out, num_classes)
        self.drop1 = nn.Dropout(p = p_drop_fc)

        nn.init.xavier_normal_(self.fc1.weight)


    def forward(self, x):
        x1 = max_pool2d(relu(self.conv1(x)), (2, 2))
        x2 = max_pool2d(relu(self.conv2(x)), (2, 2))
        x3 = max_pool2d(relu(self.conv3(x)), (2, 2))
        x4 = max_pool2d(relu(self.conv4(x)), (2, 2))
        
        x1 = multichan_to_2D(x1)
        x2 = multichan_to_2D(x2)
        x3 = multichan_to_2D(x3)
        x4 = multichan_to_2D(x4)
        
        x1 = torch.unsqueeze(x1, 1)
        x1 = self.BN1_1(x1)
        x1 = torch.squeeze(x1, 1)
        
        x2 = torch.unsqueeze(x2, 1)
        x2 = self.BN1_2(x2)
        x2 = torch.squeeze(x2, 1)
        
        x3 = torch.unsqueeze(x3, 1)
        x3 = self.BN1_3(x3)
        x3 = torch.squeeze(x3, 1)
        
        x4 = torch.unsqueeze(x4, 1)
        x4 = self.BN1_4(x4)
        x4 = torch.squeeze(x4, 1)
        
        
        x1 = self.lst1(x1)
        x2 = self.lst2(x2)
        x3 = self.lst3(x3)
        x4 = self.lst4(x4)
        
        # Применяем flatten к каждому тензору (сохраняем размерность батча)
        flat1 = torch.flatten(x1, 1)  # форма: (B, H1 * W1)
        flat2 = torch.flatten(x2, 1)  # форма: (B, H2 * W2)
        flat3 = torch.flatten(x3, 1)  # форма: (B, H3 * W3)
        flat4 = torch.flatten(x4, 1)  # форма: (B, H3 * W3)
        
        # Объединяем все векторы по последнему измерению
        combined = torch.cat([flat1, flat2, flat3, flat4], dim=1) 
        
        combined = self.BN2(combined)
        
        x = self.fc1(self.drop1(combined))
        
        return x
    
    def visualize_filters(self, save_dir=None):
        """
        Визуализирует сверточные фильтры первого слоя
        """
        # Получаем фильтры из всех сверточных слоев
        conv_layers = {
            'conv1': self.conv1,
            'conv2': self.conv2, 
            'conv3': self.conv3,
            'conv4': self.conv4,
        }
        
        for name, layer in conv_layers.items():
            filters = layer.weight.data.cpu()
            
            # Нормализуем фильтры для визуализации
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            
            # Создаем grid из фильтров
            n_filters = filters.size(0)
            n_cols = 8
            n_rows = n_filters // n_cols + (1 if n_filters % n_cols > 0 else 0)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
            axes = axes.ravel() if n_rows > 1 else [axes]
            
            for i in range(n_filters):
                ax = axes[i]
                # Берем первый канал (так как входное изображение grayscale)
                filter_img = filters[i, 0, :, :]
                ax.imshow(filter_img, cmap='viridis')
                ax.axis('off')
                ax.set_title(f'F{i+1}')
            
            # Скрываем пустые subplots
            for i in range(n_filters, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(f'Filters for {name} (Kernel size: {layer.kernel_size[0]}x{layer.kernel_size[1]})')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{name}_filters.png'), dpi=150, bbox_inches='tight')
            
            plt.show()
            plt.close(fig)
    
    def visualize_activations(self, x, save_dir=None, layer_names=None):
        """
        Визуализирует активации на внутренних слоях модели
        
        Args:
            x: входной тензор
            save_dir: директория для сохранения изображений
            layer_names: список имен слоев для визуализации
        """
        if layer_names is None:
            layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
        
        # Словарь для хранения активаций
        activations = {}
        
        # Хуки для захвата активаций
        hooks = []
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu()
            return hook
        
        # Регистрируем хуки для выбранных слоев
        for name in layer_names:
            layer = getattr(self, name)
            hooks.append(layer.register_forward_hook(get_activation(name)))
        
        # Переключаем модель в режим оценки
        self.eval()
        
        # Прямой проход для захвата активаций
        with torch.no_grad():
            y = self.forward(x)
        
        # Удаляем хуки
        for hook in hooks:
            hook.remove()
        
        # Визуализируем активации для каждого слоя
        for name, activation in activations.items():
            # Берем первый пример из батча
            act = activation[0]
            
            # Для сверточных слоев берем первые 16 каналов
            if act.dim() == 3:  # [channels, height, width]
                n_channels = min(act.size(0), 16)
                act = act[:n_channels]
                
                # Создаем grid из активаций
                grid = vutils.make_grid(act.unsqueeze(1), nrow=4, normalize=True, pad_value=0.5)
                grid = grid.permute(1, 2, 0).numpy()
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(grid)
                ax.axis('off')
                ax.set_title(f'Activations for {name}')
                
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{name}_activations.png'), dpi=150, bbox_inches='tight')
                
                plt.show()
                plt.close(fig)