from glob import glob
from tqdm import tqdm
import shutil
import argparse
parser = argparse.ArgumentParser()
import json
from utils.data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler




def load_data(train_manifest_list,
							valid_manifest_list,
							test_manifest_list, batch_size=12):

	audio_conf = dict(sample_rate=16000,
					  window_size=0.02,
					  window_stride=0.01,
					  window='hamming')
	PAD_CHAR = "¶"
	SOS_CHAR = "§"
	EOS_CHAR = "¤"

	labels_path = './labels.json'
	with open(labels_path) as label_file:
		labels = str(''.join(json.load(label_file)))


	# add PAD_CHAR, SOS_CHAR, EOS_CHAR
	labels = PAD_CHAR + SOS_CHAR + EOS_CHAR + labels
	label2id, id2label = {}, {}
	count = 0
	for i in range(len(labels)):
		if labels[i] not in label2id:
			label2id[labels[i]] = count
			id2label[count] = labels[i]
			count += 1

	train_data = SpectrogramDataset(audio_conf, manifest_filepath_list=train_manifest_list, label2id=label2id, normalize=True, augment=False)
	# print('train_data ', train_data)
	train_sampler = BucketingSampler(train_data, batch_size=batch_size)
	# print('train_sampler: ', train_sampler)
	train_loader = AudioDataLoader(
		train_data, num_workers=4, batch_sampler=train_sampler)

	valid_loader_list, test_loader_list = [], []
	for i in range(len(valid_manifest_list)):
		valid_data = SpectrogramDataset(audio_conf, manifest_filepath_list=[valid_manifest_list[i]], label2id=label2id,
										normalize=True, augment=False)
		valid_loader = AudioDataLoader(valid_data, num_workers=4, batch_size=batch_size)
		valid_loader_list.append(valid_loader)

	for i in range(len(test_manifest_list)):
		test_data = SpectrogramDataset(audio_conf, manifest_filepath_list=[test_manifest_list[i]], label2id=label2id,
									normalize=True, augment=False)
		test_loader = AudioDataLoader(test_data, num_workers=4)
		test_loader_list.append(test_loader)
	print('done')
	return train_loader, valid_loader_list, test_loader_list



if __name__ == '__main__':
	parser.add_argument('--train-manifest-list', nargs='+', type=str)
	parser.add_argument('--valid-manifest-list', nargs='+', type=str)
	parser.add_argument('--test-manifest-list', nargs='+', type=str)
	args = parser.parse_args()

	train_loader, valid_loader_list, test_loader_list = load_data(train_manifest_list=args.train_manifest_list,
							valid_manifest_list=args.valid_manifest_list,
							test_manifest_list=args.test_manifest_list, batch_size=12)