import torch
import torch.nn as nn
import torch.nn.functional as F


def get_act(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'swish':
        return nn.SiLU(inplace=True)
    elif name == 'mish':
        return MishInline()
    else:
        return nn.Identity()


class MishInline(nn.Module):
    """ https://arxiv.org/abs/1908.08681v1 """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh( F.softplus(x) )


def save_model(model, save_path):
    data = {
        'model': model.state_dict(),
        'model_args': model.model_args,
    }
    print(f"saving to: {save_path}")
    torch.save(data, save_path)


def load_model(path, device):
    data = torch.load(path, map_location=torch.device(device))
    model = FunnyNet(**data['model_args']).to(device)
    model.load_state_dict(data['model'])
    return model


@torch.jit.script
def autocrop(encoder_features: torch.Tensor, decoder_features: torch.Tensor):
    """ Center crop the encoder down to the size of the decoder """
    if encoder_features.shape[2:] != decoder_features.shape[2:]:
        ds = encoder_features.shape[2:]
        es = decoder_features.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        encoder_features = encoder_features[:, :,
                        ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                        ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                        ]
    return encoder_features, decoder_features


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, activation=None):
        super().__init__()
        act_func = get_act(activation)
        bias = False if activation else True
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            act_func,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            act_func
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                stride=1, padding=1, activation=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, stride, padding, activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, activation=activation)

    def forward(self, encoder_features, decoder_features):
        decoder_features = self.up(decoder_features)
        encoder_features, decoder_features = autocrop(encoder_features, decoder_features)
        x = torch.cat([encoder_features, decoder_features], dim=1)
        return self.conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, filters, activation):
        super(UNetEncoder, self).__init__()
        self.down00 = DoubleConv(in_channels, filters[0], activation=activation)
        self.down10 = Down(filters[0], filters[1], activation=activation)
        self.down20 = Down(filters[1], filters[2], activation=activation)
        self.down30 = Down(filters[2], filters[3], activation=activation)
        self.down40 = Down(filters[3], filters[4], activation=activation)

    def forward(self, x):
        x00 = self.down00(x)
        x10 = self.down10(x00)
        x20 = self.down20(x10)
        x30 = self.down30(x20)
        x40 = self.down40(x30)
        return x00, x10, x20, x30, x40


class UNetDecoder(nn.Module):
    def __init__(self, out_channels, filters, activation):
        super(UNetDecoder, self).__init__()
        self.up1 = Up(filters[4], filters[3], activation=activation)
        self.up2 = Up(filters[3], filters[2], activation=activation)
        self.up3 = Up(filters[2], filters[1], activation=activation)
        self.up4 = Up(filters[1], filters[0], activation=activation)
        self.out = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, features):
        x00, x10, x20, x30, x40 = features
        x = self.up1(x30, x40)
        x = self.up2(x20, x)
        x = self.up3(x10, x)
        x = self.up4(x00, x)
        return self.out(x)


class FunnyNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=16, 
                activation='relu', num_to_cat=None, mean=[0,0,0], std=[1,1,1]):
        super(FunnyNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.num_to_cat = num_to_cat
        self.model_args = {"in_channels": in_channels, "out_channels": out_channels,
            "filters": filters, "activation": activation, "num_to_cat": num_to_cat} 
        if type(filters) == int:
            filters = [filters, filters*2, filters*4, filters*8, filters*16]

        self.normalize = nn.BatchNorm2d(in_channels)  # Layer will be frozen without learnable parameters
        self.set_normalization(mean, std)

        self.encoder = UNetEncoder(in_channels, out_channels, filters, activation)
        self.decoder = UNetDecoder(out_channels, filters, activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if activation == 'relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif activation == 'leaky_relu':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
                else:
                    nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_normalization(self, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.normalize.reset_parameters()
        self.normalize.running_mean = torch.tensor(mean, requires_grad=False, dtype=torch.float)
        self.normalize.running_var = torch.tensor([x**2 for x in std], requires_grad=False, dtype=torch.float)
        self.normalize.weight.requires_grad = False  # gamma
        self.normalize.bias.requires_grad = False   # beta
        self.normalize.running_mean.requires_grad = False  # mean
        self.normalize.running_var.requires_grad = False  # variance
        self.normalize.eval()

    def forward(self, x):
        x = self.normalize.eval()(x)
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


def train_test(model, input_size, batch_size, device):
    model.to(device).train()
    data = torch.randn(batch_size, model.in_channels, input_size, input_size).to(device)
    true_masks = torch.empty(batch_size, input_size, input_size, dtype=torch.long).random_(model.out_channels).to(device)
    print("data.shape :", data.shape)
    print("true_masks.shape :", true_masks.shape)
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)
    for i in range(100):
        opt.zero_grad()
        logits = model(data)
        loss = nn.CrossEntropyLoss()(logits, true_masks)
        print("loss={:.06f}".format(loss))
        loss.backward()
        opt.step()

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    in_channels = 3
    out_channels = 2
    filters = 4  # 16
    activation = "relu"  # relu, leaky_relu, silu, mish

    batch_size = 2
    input_size = 64

    model = FunnyNet(in_channels, out_channels, filters, activation).to(device)

    # train_test(model, input_size, batch_size, device)

    x = torch.randn(1, in_channels, input_size, input_size).to(device)
    logits = model(x)
    # logits = model.predict(x)
    print(logits.shape, torch.min(logits).item(), torch.max(logits).item())


    save_model(model, save_path="runs/save_test.pth")
    model2 = load_model("runs/save_test.pth", device=device)
    logits = model2(x)
    print(logits.shape, torch.min(logits).item(), torch.max(logits).item())
