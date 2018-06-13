import shutil
from PIL import Image
from PIL import ImageEnhance
import random
import os

def generateDataset(recordingsFolder,targetFolder):
	if os.path.isdir(targetFolder) == False:
		os.mkdir(targetFolder) 
	for dirpath, dirnames, filenames in os.walk(recordingsFolder):
		if dirpath == recordingsFolder:
			for i in filenames:
				srcFile = os.path.join(recordingsFolder,i)
				dstFile = os.path.join(targetFolder,os.path.splitext(i)[0]+'_aug.csv')
				#dst = os.path.join(targetFolder, tail)
				shutil.copyfile(srcFile, dstFile)
		else:
			if dirnames != []:
				head, tail = os.path.split(dirpath)
				lstDir = os.path.join(targetFolder, tail + '_aug')
				os.mkdir(lstDir)
			else:
				head, tail = os.path.split(dirpath)
				# dstDir = os.path.join(targetFolder, tail)
				dstDir = os.path.join(lstDir, tail)
				os.mkdir(dstDir)
				if filenames != []:
					for i in filenames:
						srcFile = os.path.join(dirpath, i)
						dstFile = os.path.join(dstDir,i)
						#if os.path.isdir(targetFolder) == False:
						#	os.mkdir(targetFolder)
						srcImg = Image.open(srcFile)	# open i
						rndm = random.randint(1,4)
						if rndm == 1:
							dstImg = ImageEnhance.Sharpness(srcImg).enhance(random.randint(0,100))
						if rndm == 2:
							dstImg = ImageEnhance.Color(srcImg).enhance(random.randint(0,20))
						if rndm == 3:
							dstImg = ImageEnhance.Contrast(srcImg).enhance(round(random.uniform(0.100,10.000),3))
						if rndm == 4:
							dstImg = ImageEnhance.Brightness(srcImg).enhance(round(random.uniform(0.100,10.000),3))
						# shutil.copy(srcfile, dstdir)	# copy i to target
						dstImg.save(dstFile)
