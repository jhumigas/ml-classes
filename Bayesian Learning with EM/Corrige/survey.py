#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

def basicModel():
	''' Renvoie un modèle simple pour le débogage
	
	Returns
	-------
	(mixtureCoefs, probs) où
	- mixtureCoefs est un tableau 1D contenant les coefficients de mélange
	- probs est un tableau 3D de taille (nClusters, nQuestions, nAnswers)
		contenant les distributions de Dirichlet des réponses
		pour chaque couple (cluster, question)
	'''
	
	# Mélange de deux clusters
	mixtureCoefs = np.array([0.4,0.6])
	
	# Distribution des 3 réponses possibles pour chacune des 4 questions pour les deux clusters
	probs = np.array([
	# Cluster 1
	[
		[0.2 , 0.5, 0.3],
		[0.1 , 0.2, 0.7 ],
		[0.7 , 0.1, 0.2 ],
		[0.3 , 0.3, 0.4 ]
	],
	# Cluster 2
	[
		[0.7, 0.1, 0.2 ],
		[0.2, 0.6, 0.2 ],
		[0.2, 0.4, 0.4 ],
		[0.2, 0.6, 0.2 ]
	]])
	
	# Le modèle est 
	return (mixtureCoefs, probs)

def generateModel(nQuestions, nAnswers, nClusters):
	''' Génère un modèle de mélange de distributions de Dirichlet.
	
	Parameters
	----------
	nQuestions : 	int
			nombre de questions
	nAnswers : 	int
			nombre de réponses
	nClusters : 	int
			nombre de clusters
	
	Returns
	-------
	paire (mixtureCoefs, probs) où
	- mixtureCoefs est un tableau 1D contenant les coefficients de mélange
	- probs est un tableau 3D de taille (nClusters, nQuestions, nAnswers)
		contenant les distributions de Dirichlet des réponses
		pour chaque couple (cluster, question)
	'''
		
	mixtureCoefs = np.random.dirichlet(np.ones(nClusters))
	probs = np.random.dirichlet(np.ones(nAnswers), (nClusters, nQuestions))
	return (mixtureCoefs, probs)
	
def generateData(model, nPeople):
	''' Génère des réponses au questionnaire à partir d'un modèle de mélange
		
	Parameters
	----------
	model : 	type de modèle produit par "generateModel"
			modèle utilisé pour générer les données
	nPeople : 	int
			nombre de personnes interrogées

	Returns
	-------
	paire (data, cluster) où
	- data est un tableau 2D de taille (nPeople, nQuestions) contenant les réponses allant de 1 à nAnswers
	- cluster est un tableau 1D de longueur nPeople. cluster[i] contient le numéro de cluster associé à la personne i qui a été utilisé pour générer ses réponses data[i,:]
	'''
	
	mixtureCoefs, probs = model
	nClusters, nQuestions, nAnswers = probs.shape
	clusters = np.random.choice(nClusters, size = nPeople, p = mixtureCoefs)
	data = np.zeros((nPeople, nQuestions))
	answers = range(1, nAnswers+1)
	for i in range(nPeople):
		for j in range(nQuestions):
			data[i,j] = np.random.choice(answers, p=probs[clusters[i],j,:])	
	return (np.asarray(data, 'uint'), clusters + 1)

def drawModel(model):
	''' Trace les distributions des réponses pour chaque cluster
	
	Parameters
	----------
	model : 	type de modèle produit par "generateModel"
			modèle dont sont extraites les distributions de cluster
	'''
	
	plt.ion()
	mixtureCoefs, probs = model
	nClusters, nQuestions, nAnswers = probs.shape	
	questions = np.arange(nQuestions)
	fig = plt.figure()
	for cluster in range(nClusters):
		plt.subplot(nClusters, 1, cluster+1)	
		colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
		total = np.zeros(nQuestions)		
		for answer in range(nAnswers):
			c = next(colors)
			d = probs[cluster][:,answer]
			plt.bar(questions, d, width=1, bottom = total, color=c)
			total += d
		plt.yticks(np.arange(0, 1.0001, 0.5))
		plt.ylim((0,1))
		plt.title('Groupe ' + str(cluster+1))

	plt.xlabel('Questions', fontsize=14)
	fig.tight_layout()		
	plt.show()

def drawCluster(H):
	''' Trace les distributions des variables cachées
	
	Parameters
	----------	
	H:	matrice ou vecteur.
	Le nombre de lignes de H correspond au nombre de personnes
	* Si H est une matrice, 
		le nombre de colonnes est égal au nombre de clusters
		et le coefficient H[i,j] donne la probabilité que 
		la personne i appartient au cluster j
	* Si H est un vecteur, 
		le coefficient H[i] donne le numéro du cluster de i
	'''
	
	plt.ion()
	if(len(H.shape) >= 2):
		nPeople, nClusters = H.shape
	else:
		nPeople, = H.shape
		nClusters = np.max(H)
		Hnew = np.zeros((nPeople, nClusters))
		for (i,j) in zip(range(nPeople), H-1):
			Hnew[i,j] = 1.
		H = Hnew
	fig = plt.figure()
	people = np.arange(nPeople)
	colors = itertools.cycle(['b', 'r', 'g', 'c', 'm', 'y', 'k'])
	total = np.zeros(nPeople)		
	for cluster in range(nClusters):
		c = next(colors)
		d = H[:,cluster]
		plt.bar(people, d, width=1, bottom = total, color=c)
		total += d
	plt.yticks(np.arange(0, 1.0001, 0.25))
	plt.ylim((0,1))
	plt.title('Distribution of latent variables')
	plt.xlabel('person', fontsize=14)
	fig.tight_layout()		
	plt.show()

def compareClusters(C1,C2):
	''' Renvoie le degré de superposition de deux clusters C1 et C2 
		à une permutation près des clusters
		
		C1 et C2 donnent les clusters associés à chaque exemple.
		La ligne i de C1 et C2 correspond à l'exemple i.
		C1 et C2 peuvent être soit un vecteur soit une matrice.
		* Si C est un vecteur, C[i] contient le cluster auquel i appartient.
		* Si C est une matrice, le nombre de colonnes de C correspond au nombre de clusters
		et C[i,j] contient la probabilité que l'exemple i appartient au cluster j
		Dans ce cas le cluster le plus probable est associé à i
		
		La fonction renvoie (bestOverlap, bestPerm) où
		* bestOverlap est le degré de superposition de C1 et C2
		* bestPerm donne la correspondance des clusters de C2 avec ceux de C1
		bestPerm[n° du cluster de C2] = n° du cluster de C1 correspondant '''

	if(C1.shape[0] != C2.shape[0]):
		raise "C1 and C2 must have the same number of lines"
			
	if(len(C1.shape) >= 2 and C1.shape[1] > 1):
		C1 = np.argmax(C1,1)+1
	if(len(C2.shape) >= 2 and C2.shape[1] > 1):
		C2 = np.argmax(C2,1)+1
			
	nClusters = max(np.max(C1), np.max(C2))
	perms = itertools.permutations(range(1,nClusters+1))
	bestOverlap = 0.
	bestPerm = None
	for perm in perms:
		p = np.array(perm)
		permC2 = p[C2-1]
		overlap = np.sum(C1 == permC2) * 1. / np.size(C1)
		if(overlap > bestOverlap):
			bestOverlap = overlap
			bestPerm = p
	return (bestOverlap, bestPerm)
	
def permuteClusters(M, H, perm):
	''' Permute clusters on a model M
	and latent cluster distributions H using
	cluster mapping perm
	Returns (permuted model, permuted H)
	'''
	invperm = np.argsort(perm-1)
	mixtureCoefs, probs = M
	mixtureCoefs = mixtureCoefs[invperm]
	probs = probs[invperm, :, :]
	H = H[:, invperm]
	return ((mixtureCoefs, probs), H)
	
###########
# Corrigé #
###########

def em(data, nClusters, nIterations):
	''' Applique l'algorithme EM 
	- pendant un nombre fixe d'itérations nIterations
	- pour évaluer les paramètres d'un modèle de nClusters clusters
	- à partir des données data décrites au format des données produites par generateData
	
	Doit renvoyer le couple (model,H) où
	- model est le modèle appris au même format que les modèles produits pare generateModel
	- H est une matrice (nombre de personnes, nombre de clusters) dont les lignes
	correspondent aux distribution de cluster de chaque personne
	'''
	nPeople, nQuestions = data.shape
	nAnswers = int(np.max(data))

	# Initialization :
	
	# Clusters are initially equiprobable
	mixtureCoefs = np.ones(nClusters) / nClusters
	# But their parameters are drawn randomly
	probs = np.random.dirichlet(np.ones(nAnswers), (nClusters, nQuestions))

	# Ligne à décommenter pour tester le problème de symétrie
	# probs = np.ones((nClusters, nQuestions, nAnswers)) / nAnswers
	
	# Create array to store distribution parameters of hidden variables
	H = np.zeros((nPeople,nClusters))
	tenth = int(nIterations / 10)
	for t in range(nIterations):
		##########
		# E step #
		##########
		
		# P(utilisateur i dans cluster c) proportionnelle au produit :
		#	P(Cluster = c) *
		#	P(Réponse à Question 1 = Reponses[i,1] | Cluster = c) *
		#		...
		#	P(Réponse à Question m = Reponses[i,m] | Cluster = c)
		
		# Code non optimisé avec boucles explicites (voir em2 pour une version plus rapide)
		for i in range(nPeople):
			answers = data[i,:]
			for c in range(nClusters):
				H[i,c] = np.log(mixtureCoefs[c])
				for j in range(nQuestions):
					H[i,c] += np.log(probs[c,j,int(answers[j]-1)])
			H[i,:] = H[i,:] -np.max(H[i,:])
			H[i,:] = np.exp(H[i,:])
			
			# Normalise la distribution q_i
			H[i,:] /= np.sum(H[i,:])
		
		##########
		# M step #
		##########

		# P(Cluster = c) proportionnelle à la somme :
		#	P(utilisateur 1 dans cluster c) +
		#	...
		#	P(utilisateur n dans cluster c)
		weights = np.sum(H,0)
		mixtureCoefs = weights / np.sum(weights)		

		#	Indic(Réponses[1,j] = k) * P(utilisateur 1 dans cluster c) +
		# P(Réponse à Question j = k | Cluster = c)
		#	...
		#	Indic(Réponses[n,j] = k) * P(utilisateur n dans cluster c)

		# for k in range(1, nAnswers+1):
		# 	kanswers = (data == k)
		# 	for cluster in range(nClusters):
		# 		hcluster = np.reshape(H[:,cluster],(nPeople,1))
		# 		probs[cluster,:,k-1] = np.sum(kanswers * hcluster 
		# 			, 0) / weights[cluster]

		# Code non optimisé avec boucles explicites (voir em2 pour une version plus rapide)
		for cluster in range(nClusters):
			for j in range(nQuestions):
				for k in range(1, nAnswers+1):
					s = 0.
					total = 0.
					for i in range(nPeople):
						if(data[i,j] == k):
							s += H[i,cluster]
						total += H[i,cluster]
					probs[cluster, j, k-1] = s / total
				
				# Normalise P(Réponse à question j| H = c)
				probs[cluster, j, :] /= sum(probs[cluster, j, :])

		if(t % tenth == tenth-1):
			print(str(int((t+1)/tenth)*10) + "% done");

	model = (mixtureCoefs, probs)
	return (model,H)

def loglikelihood(model, data, H):
	''' Fonction qui calcule la vraisemblance
	'''
	
	mixtureCoefs, probs = model
	nPeople, nQuestions = data.shape
	
	logL = 0
	for i in range(nPeople):
		answers = data[i,:]
		for j in range(nQuestions):
			logL += np.log(sum(probs[:,j,answers[j]-1] * H[i,:].T))
	return logL

def em2(data, nClusters, epsilon):
	''' Applique l'algorithme EM 
	- pendant un nombre fixe d'itérations nIterations
	- pour évaluer les paramètres d'un modèle de nClusters clusters
	- à partir des données data décrites au format des données produites par generateData
	
	Doit renvoyer le couple (model,H) où
	- model est le modèle appris au même format que les modèles produits pare generateModel
	- H est une matrice (nombre de personnes, nombre de clusters) dont les lignes
	correspondent aux distribution de cluster de chaque personne
	'''
	nPeople, nQuestions = data.shape
	nAnswers = int(np.max(data))

	# Initialization :
	
	# Clusters are initially equiprobable
	mixtureCoefs = np.ones(nClusters) / nClusters
	# But their parameters are drawn randomly
	probs = np.random.dirichlet(np.ones(nAnswers), (nClusters, nQuestions))

	# Ligne à décommenter pour tester le problème de symétrie
	# probs = np.ones((nClusters, nQuestions, nAnswers)) / nAnswers
	
	# Create array to store distribution parameters of hidden variables
	H = np.zeros((nPeople,nClusters))
	oldLogL = -sys.float_info.max
	
	while(True):
		##########
		# E step #
		##########
		
		# P(utilisateur i dans cluster c) proportionnelle au produit :
		#	P(Cluster = c) *
		#	P(Réponse à Question 1 = Reponses[i,1] | Cluster = c) *
		#		...
		#	P(Réponse à Question m = Reponses[i,m] | Cluster = c)
		
		H = np.log(mixtureCoefs) * np.ones((nPeople,1))
		questions = np.arange(nQuestions, dtype='uint') * np.ones((nPeople,1), dtype='uint')
		
#		for i in range(nPeople):
#			H[i,:] += np.sum(np.log(probs[:, questions[i,:], data[i,:] - 1]),1)
		H += np.sum(np.log(probs[:, questions, data - 1]),0)
		print(probs[:, questions, data - 1].shape)
		H -= np.reshape(np.max(H,1), (nPeople,1))
		H = np.exp(H)

		# Normalise les distributions de Dirichlet
		H /= np.reshape(np.sum(H,1),(nPeople,1))
		
		# Compute loglikelihood
		logL = loglikelihood((mixtureCoefs, probs), data, H)
		print("logL = " + str(logL))
		if(abs(logL - oldLogL) < epsilon):
			break
		else:
			oldLogL = logL
			
		##########
		# M step #
		##########

		# P(Cluster = c) proportionnelle à la somme :
		#	P(utilisateur 1 dans cluster c) +
		#	...
		#	P(utilisateur n dans cluster c)
		weights = np.sum(H,0)
		mixtureCoefs = weights / np.sum(weights)		

		# P(Réponse à Question j = k | Cluster = c)
		#	Indic(Réponses[1,j] = k) * P(utilisateur 1 dans cluster c) +
		#	...
		#	Indic(Réponses[n,j] = k) * P(utilisateur n dans cluster c)

		for k in range(1, nAnswers+1):
			kanswers = (data == k)
			for cluster in range(nClusters):
				hcluster = np.reshape(H[:,cluster],(nPeople,1))
				probs[cluster,:,k-1] = np.sum(kanswers * hcluster , 0) / weights[cluster]
				
	model = (mixtureCoefs, probs)
	return (model,H,logL)

def BIC(model, logL, nPeople):
	mixtureCoefs, probs = model
	nClusters, nQuestions, nAnswers = probs.shape

	nParams = (nClusters - 1) + nClusters * nQuestions * (nAnswers - 1)
	bic = -2. * logL + nParams * np.log(nPeople)
	return bic
	
def findBestModel(data):
	nPeople, _ = data.shape

	epsilon = 1
	bestBic = sys.float_info.max
	for nClusters in range(1,10):
		print("Testing for " + str(nClusters) + " clusters")
		(model,H,logL) = em2(data, nClusters, epsilon)
		bic = BIC(model, logL, nPeople)
		print("bic(" + str(nClusters) + " clusters) = " + str(bic))
		if(bic < bestBic):
			bestBic = bic
			bestModel = model
	return bestModel

def test(M, nPeople, nClusters):
        M = basicModel()
        D,C = generateData(M,nPeople)
        m,h = em(D, nClusters, 100)
        score, mapping = compareClusters(C,h)
        pm,ph = permuteClusters(m,h, mapping)
        drawCluster(ph)
        drawModel(pm)
        drawCluster(C)
        drawModel(M)
