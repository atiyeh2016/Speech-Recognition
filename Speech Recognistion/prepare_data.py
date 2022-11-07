from glob import glob
from tqdm import tqdm
import shutil
import argparse
parser = argparse.ArgumentParser()
import json
from data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
from helper import remove_punctuation

def prepare_valid_data():
	wave_file_names = []
	txt_file_names = []

	for files in tqdm(glob('./drive/My Drive/voxforge/enlang/extracted/*')):
		for file in glob(files+'/*'):
			end_file = file.split('/')[-1]

			if end_file=='wav':
				for sound in glob(file+'/*'):
					wave_file_names.append(sound.split('/')[4]+'_' +sound.split('/')[-1].split('.')[0])

			if end_file=='etc':
				Prompt = glob(file+'/*prompt*')
				# print(Prompt)
				if Prompt:
					base_name=Prompt[0].split('/')[4]
					file_propmt = open(Prompt[0])
					for h in file_propmt.readlines():
						full_name = base_name +'_'+h.split(' ')[0] + '.txt' 
						txt_file_names.append(full_name.split('.')[0])


	wave_file_names = set(wave_file_names)
	txt_file_names = set(txt_file_names)
	valid_filename = wave_file_names.intersection(txt_file_names) 

	print(len(valid_filename))

	valid_filename = list(valid_filename)
	validation_file_names= valid_filename[:2000]
	test_file_names= valid_filename[2000:3000]
	train_file_names= valid_filename[3000:]


	data_val_manifest = open('./data_val.csv','w')
	data_test_manifest = open('./data_test.csv','w')
	data_train_manifest = open('./data_train.csv','w')

	for files in tqdm(glob('./drive/My Drive/voxforge/enlang/extracted/*')):
		for file in glob(files+'/*'):
			end_file = file.split('/')[-1]

			if end_file=='wav':
				for sound in glob(file+'/*'):
					sound_name = sound.split('/')[4]+'_' +sound.split('/')[-1] 
					sfn = sound_name.split('.')[0]
					# print('sfn', sfn)
					if sfn in validation_file_names:
						data_val_manifest.write('./drive/My Drive/data/val/wav/'+sfn+'.wav'+', ./data/val/txt/'+sfn+'.txt\n')
						shutil.copyfile(sound, './drive/My Drive/data/val/wav/'+sound_name) 
						
					elif sfn in test_file_names:
						data_test_manifest.write('./drive/My Drive/data/test/wav/'+sfn+'.wav'+',./data/test/txt/'+sfn+'.txt\n')
						shutil.copyfile(sound, './drive/My Drive/data/test/wav/'+sound_name) 
						
					elif sfn in train_file_names:
						data_train_manifest.write('./drive/My Drive/data/train/wav/'+sfn+'.wav'+',./data/train/txt/'+sfn+'.txt\n')
						shutil.copyfile(sound, './drive/My Drive/data/train/wav/'+sound_name) 
						


			if end_file=='etc':
				Prompt = glob(file+'/*prompt*')
				# print(Prompt)
				if Prompt:
					base_name=Prompt[0].split('/')[4]
					file_propmt = open(Prompt[0])
					for h in file_propmt.readlines():
						full_name = base_name +'_'+h.split(' ')[0] + '.txt' 
						sfn = full_name.split('.')[0] 

						if sfn in validation_file_names:
							transcript = ' '.join(h.split(' ')[1:])
							file_tmp = open('./drive/My Drive/data/val/txt/'+full_name, 'w')
							transcript = remove_punctuation(transcript)
							file_tmp.write(transcript)
							file_tmp.close()

						elif sfn in test_file_names:
							transcript = ' '.join(h.split(' ')[1:])
							file_tmp = open('./drive/My Drive/data/test/txt/'+full_name, 'w')
							transcript = remove_punctuation(transcript)
							file_tmp.write(transcript)
							file_tmp.close()

						elif sfn in train_file_names:
							transcript = ' '.join(h.split(' ')[1:])
							file_tmp = open('./drive/My Drive/data/train/txt/'+full_name, 'w')
							transcript = remove_punctuation(transcript)
							file_tmp.write(transcript)
							file_tmp.close()


	data_val_manifest.close()
	data_test_manifest.close() 
	data_train_manifest.close()
	return

if __name__ == '__main__':

	prepare_valid_data()


	# python prepare_voxforge.py --train-manifest-list ./data_train.csv --valid-manifest-list ./data_val.csv  --test-manifest-list  ./data_test.csv

