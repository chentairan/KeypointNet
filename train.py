#import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import models
import resnet

#import other tools
import json
from skimage import io, transform
import numpy as np 
import trans_data
import matplotlib.pyplot as plt

#train data file dir
json_file = "./dataset/keypoints.json"
root_dir = "./dataset"

class KeyPointsDataset(Dataset):
	"""KeyPoints dataset."""
	def __init__(self, json_file, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the json file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		with open(json_file,'r') as load_f:
			self.landmarks_frame = json.load(load_f)
		
		self.root_dir = root_dir
		self.transform = transform


	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		img_name = self.root_dir + '/' + self.landmarks_frame[idx]["filename"]
		image = io.imread(img_name)
		io.imshow(image)
		points = self.landmarks_frame[idx]["annotations"]
		landmarks = [
						points[0]["x"], points[0]["y"], 
					 	points[1]["x"], points[1]["y"], 
						points[2]["x"], points[2]["y"]
					]

		landmarks = np.asarray(landmarks)

		#transform the format
		image = image.transpose((2, 0, 1))
		image = torch.from_numpy(image)
		landmarks = torch.from_numpy(landmarks)


		sample = {'image': image, 'landmarks': landmarks}

		if self.transform:
			sample = self.transform(sample)
		return sample

if __name__ == "__main__":

	Net = resnet.resnet101
	net = Net()

	criterion = nn.MSELoss()
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	transformed_dataset = KeyPointsDataset(
											json_file = json_file,
                                           	root_dir = root_dir
											)

	trainloader = DataLoader(transformed_dataset, batch_size=2, shuffle=False, num_workers=0)

	for epoch in range(30):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs
			inputs = data["image"]
			labels = data["landmarks"]
			
			inputs = inputs.type(torch.FloatTensor)
			labels = labels.type(torch.FloatTensor)
			

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if i % 2 == 1:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %
					(epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0
				weight_name = "./weight/keypoints-" + str(epoch) + ".pkl"
				torch.save(net, weight_name)

	print('Finished Training')