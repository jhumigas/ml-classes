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

###############
# A compléter #
###############

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

	# Each latent variable is suppose to be following a probability of distribution qi
	# q[i,j] = P(personne i belongs to Cluster j)
	# We store those distribution in an array
	q = np.zeros((nPeople, nClusters))

	# Probabilities of chosing a given latent variable
	# Initialized by a uniform distribution
	# ph(i) = P(Cluster = i)
	ph = np.ones(nClusters)/nClusters

	# Probabilities the choice of visible variable
	# knowing that a given latent variable was chosen
	# phv(i,j,k) = P(Reponse = A(i,j) | Cluster = k)
	phv = np.array([np.random.random((nQuestions, nAnswers)) for i in range(nClusters)])
	phv = normalize(phv)
    
	# Step E 
	# We focus only on the distribution q to maximize the MLE
	for t in range(nIterations):
		for i in range(nPeople):
			answers = data[i,:]
			for c in range(nClusters):
				# We use log since the probabilities might have small values 
				# First we add the log of our prior 
				q[i,c] = np.log(ph[c])
                
				# Then we add the log-likelihood 
				for j in range(nQuestions):
					q[i,c] += np.log(phv[c,j,int(answers[j]-1)])
				
			q[i,:] -= np.max(q[i,:])
			q[i,:] = np.exp(q[i,:])
			# normalize
			q[i,:] /= np.sum(q[i,:])
        
		# Step M
		weights = np.sum(q,0)
		ph = weights/np.sum(weights)

		for cluster in range(nClusters):
			for j in range(nQuestions):
				for k in range(1, nAnswers + 1):
					s = 0
					#total = 0.
					for i in range(nPeople):
						if (data[i,j] == k):
							s+= q[i,cluster]
						#total += q[i,cluster]
					#phv[cluster, j, k-1] = s/total
					phv[cluster, j, k-1] = s/weights[cluster]
                
				# Normalization 
				phv[cluster, j, :] /= sum(phv[cluster, j, :])

		model = (ph, phv)
		return (model, q)


def normalize(v):
	"""
	Fait pour une matrice 3-D
	"""
	dim = v.shape 
	for i in range(0, dim[0]-1):
		v[i,:,:] = (v[i,:,:].T/np.sum(v[i,:,:],1)).T

	return v

