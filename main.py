import os
import cv2
import numpy as np

from image_enhance import image_enhance

import skimage.morphology
import skimage

from getTerminationBifurcation import getTerminationBifurcation;
from removeSpuriousMinutiae import removeSpuriousMinutiae
from CommonFunctions import ShowResults
from extractMinutiaeFeatures import extractMinutiaeFeatures

def get_descriptors(img, name):

	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	enhanced_img = image_enhance(img)

	enhanced_img = enhanced_img.astype('uint8')*255

	#cv2.imshow(name, img)
	cv2.imshow(name +" Enhanced", enhanced_img)
	#cv2.waitKey(0)

	en_img = np.uint8(enhanced_img>128);

	skel = skimage.morphology.skeletonize(en_img)
	skel = np.uint8(skel)*255;

	cv2.imshow(name +" Skeleton", skel)
	#cv2.waitKey(0)

	mask = en_img*255;
	(minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask);

	minutiaeTerm = skimage.measure.label(minutiaeTerm, connectivity=2);
	minutiaeTerm_all = minutiaeTerm
	RP = skimage.measure.regionprops(minutiaeTerm)
	minutiaeTerm = removeSpuriousMinutiae(RP, np.uint8(en_img), 20);

	'''
	minutiaeBif = skimage.measure.label(minutiaeTerm, connectivity=2);
	RP = skimage.measure.regionprops(minutiaeTerm)
	minutiaeBif = removeSpuriousMinutiae(RP, np.uint8(img), 10);
	'''

	BifLabel = skimage.measure.label(minutiaeBif, connectivity=2);
	TermLabel = skimage.measure.label(minutiaeTerm, connectivity=2);

	FeaturesTerm, FeaturesBif = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif)

	bif_keypoints = []

	ter_keypoints = []

	#print("Bifurcations - " + name)
	for x in FeaturesBif:
		bif_keypoints.append(cv2.KeyPoint(x.locY, x.locX, 1))
		#print(x.locX, x.locY, x.Orientation, x.Type)

	#print("Termination - " + name)
	for x in FeaturesTerm:
		#if x.Orientation != float('nan'): 
		ter_keypoints.append(cv2.KeyPoint(x.locY, x.locX, 1))
		#print(x.locX, x.locY, x.Orientation, x.Type)

	#print(keypoints)
	
	# Define descriptor
	orb = cv2.ORB_create()
	# Compute descriptors
	_, bif_des = orb.compute(img, bif_keypoints)
	_, ter_des = orb.compute(img, ter_keypoints)

	ShowResults(skel, minutiaeTerm_all, BifLabel, name + "without filtering")
	ShowResults(skel, TermLabel, BifLabel, name)

	return bif_keypoints, ter_keypoints, bif_des, ter_des

def main():

	images_path = "images/TESTE_5/"

	img_list = os.listdir(images_path)

	processed = []

	for img1_name in sorted(img_list):
		for img2_name in sorted(img_list):

			if img2_name not in processed:
				processed.append(img1_name)

				if os.path.isfile(os.path.join(images_path, img1_name)) and os.path.isfile(os.path.join(images_path, img2_name)):
					print("Processing images -", img1_name, img2_name)
					image1 = cv2.imread(images_path+ img1_name)
					image2 = cv2.imread(images_path+ img2_name)

					cv2.imshow("img1_name", image1)
					cv2.imshow("img2_name", image2)

					kp1_bif, kp1_ter, des1_bif, des1_ter = get_descriptors(image1, "finger 1")
					kp2_bif, kp2_ter, des2_bif, des2_ter = get_descriptors(image2, "finger 2")

					bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
					matches_bif = sorted(bf.match(des1_bif, des2_bif), key= lambda match:match.distance)
					matches_ter = sorted(bf.match(des1_ter, des2_ter), key= lambda match:match.distance)
					
					#print(matches)
					# Calculate score
					#print("###Calculating Score####")
					score = 0;
					for match in matches_bif:
						score += match.distance
					bif_score = score/len(matches_bif) 
					#print("Bifurcation Score: ", score/len(matches_bif))

					score = 0;
					for match in matches_ter:
						score += match.distance
					ter_score = score/len(matches_ter)
					#print("Termination Score: ", score/len(matches_ter))
					#score_threshold = 70
					if bif_score > 60:
						if ter_score > 80:
							print("Fingerprint does not match!")
						else:
							print("Fingerprint matches.")
					else:
						if ter_score > 80:
							print("Fingerprint does not match!")
						else:
							print("Fingerprint matches.")
					#if score/len(matches) < score_threshold:
						#print("Fingerprint matches.")
					#else:
						#print("Fingerprint does not match.")
					
					cv2.waitKey(0)

				else:
					print(img1_name, " is not a file!")

	cv2.destroyAllWindows()

if __name__ == "__main__":
    main()