import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, save
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import random_split
from torch.autograd import Variable

class VAE_loss(nn.Module):
	def __init__(self):
		super(VAE_loss, self).__init__()

	def forward(self, reconstructed_inputs, inputs, mu, logvar):
		MSE = nn.MSELoss()(reconstructed_inputs, inputs)
		KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
		return MSE + KLD, MSE, KLD

class conv_block(nn.Module):
	
	def __init__(self, ch_in, ch_out, stride):
		super(conv_block, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size = 5, padding = 2, dilation = 1, stride = stride), nn.BatchNorm2d(ch_out), nn.ELU())

	def forward(self, x):
		x = self.conv(x)
		return x

class deconv_block(nn.Module):
	
	def __init__(self, ch_in, ch_out, stride, output_padding=0, padding=2):
		super(deconv_block, self).__init__()
		self.deconv = nn.Sequential((nn.ConvTranspose2d(ch_in, ch_out, kernel_size = 5, padding = padding, stride = stride, output_padding = output_padding)), nn.BatchNorm2d(ch_out), nn.ELU())
		
	def forward(self, x):
		x = self.deconv(x)
		return x

class encoder(nn.Module):
	
	def __init__(self, Train=True):
		super(encoder, self).__init__()
		self.Conv1 = conv_block(ch_in=3,ch_out=128, stride = 1)
		self.Conv2 = conv_block(ch_in=128,ch_out=256, stride = 2)
		self.Conv3 = conv_block(ch_in=256,ch_out=512, stride = 2)
		self.Conv4 = conv_block(ch_in=512,ch_out=1024, stride = 2)
		self.Conv5 = conv_block(ch_in=1024,ch_out=1024, stride = 1)
		self.mulayer = nn.Linear(16384, 128)
		self.lvlayer = nn.Linear(16384,128)
		self.Train = Train

	def sample(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps*std

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		# return torch.normal(mu, std)
		esp = torch.randn(*mu.size())
		z = mu + std * esp
		return z

	def bottleneck(self, x):
		mu, logvar = self.mulayer(x), self.lvlayer(x)
		z = self.reparameterize(mu, logvar)
		return z, mu, logvar

	def forward(self, x):
		#Encoder
		# print("\ninput image size:",x.size())
		x1 = self.Conv1(x)
		# print("X1:",x1.size())
		x2 = self.Conv2(x1)
		# print("X2:",x2.size())
		x3 = self.Conv3(x2)
		# print("X3:",x3.size())
		x4 = self.Conv4(x3)
		# print("X4:",x4.size())
		x5 = self.Conv5(x4)
		# print("X5:",x5.size())
		x6 = nn.Flatten()(x5)
		# print("\nBefore bottleneck:",x6.size()) # Outputs a tensor of batch_sizex1024
		if self.Train:
			z, mu, logvar = self.bottleneck(x6) # z is batch_sizex128
		else:
			z = self.mulayer(x6)
			mu = None
			logvar = None
		# print("\nLatent z size:",z.size())
		return z, mu, logvar

class decoder(nn.Module):

	def __init__(self):
		super(decoder, self).__init__()
		self.first = nn.Linear(128,32768)
		self.Tconv1 = deconv_block(512, 1024, stride = 1)
		self.Tconv2 = deconv_block(1024, 512, stride = 1)
		self.Tconv3 = deconv_block(512, 256, stride = 2)
		self.Tconv4 = deconv_block(256, 128, stride = 2, padding = 1, output_padding=1)
		self.Conv = nn.Conv2d(128, 3, kernel_size = 5, stride = 1, padding=2)

	def forward(self, x):
		x5 = self.first(x)
		# print("\nAfter bottleneck:",x5.size())
		x5 = x5.view(x5.size(0), 512, 8, 8)
		x4 = self.Tconv1(x5)
		# print("X4:", x4.size())
		x3 = self.Tconv2(x4)
		# print("X3:", x3.size())
		x2 = self.Tconv3(x3)
		# print("X2:", x2.size())
		x1 = self.Tconv4(x2)
		# print("X1:", x1.size())
		x0 = self.Conv(x1)
		# print("X0:", x0.size())
		output = nn.Sigmoid()(x0)
		# print("Output", output.size())
		return output

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.encoder = encoder()
		self.decoder = decoder()

	def forward(self, x):
		z, mu, logvar = self.encoder(x)
		x_hat =self.decoder(z)

		return z, mu, logvar, x_hat