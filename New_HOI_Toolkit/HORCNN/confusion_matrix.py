import numpy as np
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import math

class confusion_matrix():
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.mtx = np.zeros((num_classes, 4))
		self.labels = [x+1 for x in range(self.num_classes)]

	def mAP(self, y_true, y_pred, rare_idxs):
		ay_true = np.column_stack(y_true[:-1])
		ay_pred = np.column_stack(y_pred[:-1])

		y_true = np.column_stack(y_true)
		y_pred = np.column_stack(y_pred)

		#comb = np.rec.fromarrays([y_pred,y_true])
		#comb.sort()

		#y_pred = comb.f0
		#y_pred = comb.f1
		#del comb

		#precision = np.zeros(self.num_classes)
		#recall = np.zeros(self.num_classes)
		avg_precision = []
		avg_precision_a = []
		avg_precision_micro = []
		avg_precision_macro = []
		avg_precision_rare = []
		avg_precision_norare = []
		# Looping for each class:
		for i in range(self.num_classes):
			precision, recall, _ = precision_recall_curve(y_true[i], y_pred[i])
			avg_precision.append(average_precision_score(y_true[i], y_pred[i],average='weighted'))
			avg_precision_a.append(average_precision_score(ay_true[i], ay_pred[i], average='weighted'))
			avg_precision_macro.append(average_precision_score(y_true[i], y_pred[i],average='macro'))
			avg_precision_micro.append(average_precision_score(y_true[i], y_pred[i],average='micro'))

			if i in rare_idxs:
				avg_precision_rare.append(average_precision_score(ay_true[i], ay_pred[i], average='weighted'))
			else:
				avg_precision_norare.append(average_precision_score(ay_true[i], ay_pred[i], average='weighted'))

			#txt='Class #{c} : Precision = {p} : Recall = {r} : Avg_Precision = {a}'
			txt='Class #{c} : Avg_Precision = {a} : AP_a = {aa} : AP_Macro = {ama} : A)_Micro = {ami}'
			print(txt.format(c=i+1, a=avg_precision[i], aa=avg_precision_a[i], ama=avg_precision_macro[i], ami=avg_precision_micro[i]))

		avg_precision_a1 = []
		avg_precision_rare1 = []
		avg_precision_norare1 = []
		for i in avg_precision_rare:
			if math.isnan(i) == False:
				avg_precision_rare1.append(i)
		for i in avg_precision_norare:
			if math.isnan(i) == False:
				avg_precision_norare1.append(i)
		for i in avg_precision_a:
			if math.isnan(i) == False:
				avg_precision_a1.append(i)

		txt = 'Final mAP [EXTRA-CLASS] = {ap} : mAP_a = {a} : mAP_Macro = {ama} : mAP_Micro = {ami} : Rare = {r} : No Rare = {nr}'
		print(txt.format(ap=(sum(avg_precision)/self.num_classes), a=(sum(avg_precision_a1) / len(avg_precision_a1)), ama=(sum(avg_precision_macro)/self.num_classes),
		 ami=(sum(avg_precision_micro) / self.num_classes), r=(sum(avg_precision_rare1) / len(avg_precision_rare1)), nr=(sum(avg_precision_norare1) / len(avg_precision_norare1))))


	def thresholds(self, y_true, y_pred):
		thresholds = [0.05,0.1,0.2,0.4,0.5,0.6,0.7,0.8]
		best_theshold = 0
		p_mAP = 0.0
		for i in thresholds:
			t_ypred = copy.deepcopy(y_pred)
			self.mtx = np.zeros((self.num_classes,4))
			for j in range(len(t_ypred)):
				#print(t_ypred[j])
				t_ypred[j] = [0 if x < i else 1 for x in t_ypred[j]]
				#print(t_ypred[j])

			self.add_sample_batch(y_true, t_ypred)
			self.report()

			txt = 'Threshold: {t}, mAP={m}, mRecall={r}'
			print(txt.format(t=i,m=self.mAP,r=self.mRecall))
			if self.mAP > p_mAP:
				p_mAP = self.mAP
				best_theshold = i
				print('BEST')
			if i ==0.1:
				self.print_report()



	def add_labels(self, labels):
		self.labels = labels

	def add_sample_batch(self, y_true, y_pred):
		for i in range(len(y_pred)):
			for c in range(self.num_classes):
				if y_true[i,c]==1 and y_pred[i,c]==1: #True Positive
					self.mtx[c,0] += 1 
				elif y_true[i,c]==1 and y_pred[i,c]==0: # False Positive
					self.mtx[c,1] += 1 
				elif y_true[i,c]==0 and y_pred[i,c]==0: # True Negative
					self.mtx[c,2] += 1 
				elif y_true[i,c]==0 and y_pred[i,c]==1: # False Negative
					self.mtx[c,3] += 1 

	def add_sample_single(self, y_true, p_pred):
		for c in range(self.num_classes):
			if y_true[i,c]==1 and y_pred[i,c]==1: #True Positive
				self.mtx[c,0] += 1 
			elif y_true[i,c]==1 and y_pred[i,c]==0: # False Positive
				self.mtx[c,1] += 1 
			elif y_true[i,c]==0 and y_pred[i,c]==0: # True Negative
				self.mtx[c,2] += 1 
			elif y_true[i,c]==0 and y_pred[i,c]==1: # False Negative
				self.mtx[c,3]

	def report(self):
		
		self.report_summary = []
		c_AP = 0.0
		c_Recall = 0.0
		for c in range(self.num_classes):
			num_s = self.mtx[c,0] + self.mtx[c,1]

			if self.mtx[c,0] + self.mtx[c,3] == 0:
				recall = 0.0
			else:
				recall = self.mtx[c,0] / (self.mtx[c,0] + self.mtx[c,3])
			
			c_Recall += recall
			
			if self.mtx[c,0] + self.mtx[c,1] == 0:
				precision = 0.0
			else:
				precision = self.mtx[c,0] / (self.mtx[c,0] + self.mtx[c,1])
			c_AP += precision

			if recall + precision == 0:
				fscore =0.0
			else:
				fscore = (2 * recall * precision) / (recall + precision)

			self.report_summary.append([self.labels[c], precision, recall, fscore, num_s])

		self.mAP = c_AP / self.num_classes
		self.mRecall = c_Recall /self.num_classes
		


	def print_report(self):
		print('CLASS\t\t\tPRECISION\tRECALL\tF-SCORE\tNUM_SAMPLES')
		
		for i in self.report_summary:
			line = '{l}\t\t\t{p:.2f}\t{r:.2f}\t{f:.2f}\t{s}'
			print(line.format(l=i[0], r=i[2], p=i[1], s=i[4], f=i[3]))




