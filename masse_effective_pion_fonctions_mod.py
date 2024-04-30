#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import curve_fit


# # Lecture des données

# In[21]:


def lecture(fichier):
    with open(fichier, "r") as f:
        nt = []
        re = []
        im = []
        for line in f:
            data = line.split()
            nt.append(data[0])
            re.append(data[1])
            im.append(data[2])
        nt = nt[1:]
        re = re[1:]
        im = im[1:]
        for i in range(len(nt)):
            nt[i] = float(nt[i])
            re[i] = float(re[i])
            im[i] = float(im[i])
        return nt, re, im


# In[23]:


#listes par configurations
def par_configs(nt,re,im):
    ntll = [nt[i:i+64] for i in range(0, len(nt),64)]
    rell = [re[i:i+64] for i in range(0, len(re),64)]
    imll = [im[i:i+64] for i in range(0, len(im),64)]
    return ntll, rell, imll


# In[25]:


#retourne la norme par nt
def normes_par_nt(ntll, rell, imll):
    nont = []
    for i in range(64):
        nonti = []
        for j in range(len(ntll)):
            nonti.append(np.sqrt((rell[j])[i]**2 + (imll[j])[i]**2))
        nont.append(nonti)
    return nont


# In[26]:


def re_im_par_nt(rell, imll):
    rent = []
    imnt = []
    for i in range(64):
        renti = []
        imnti = []
        for j in range(len(rell)):
            renti.append(rell[j][i])
            imnti.append(imll[j][i])
        rent.append(renti)
        imnt.append(imnti)
    return rent, imnt


# # Premier fit

# In[29]:


def moyenne_norme(nont): #renvoie une liste avec les moyennes des normes pour chaque nt, et la liste des erreurs stats associées
    return [np.mean(nont[i]) for i in range(len(nont))], [np.sqrt(np.var(nont[i])) for i in range(len(nont))]


# In[30]:


def moyenne_re_im(rent, imnt):
    return [np.mean(rent[i]) for i in range(len(rent))], [np.sqrt(np.var(rent[i])) for i in range(len(rent))], [np.mean(imnt[i]) for i in range(len(imnt))], [np.sqrt(np.var(imnt[i])) for i in range(len(imnt))]


# In[35]:


def fit(nt, A0, E0): #la fonction à fit
    return A0*np.cosh((nt-32)*E0)


# In[36]:


def estim(inf,sup, nomoy, sigmaobs): #renvoie [A0, E0, chi2, chi2réduit], xdata, ydata, ypred
    xdata = np.array([i for i in range(inf,sup)])
    ydata = np.array(nomoy[inf:sup])
    sigmadata = np.array(sigmaobs[inf:sup])
    popt, pcov = curve_fit(fit, xdata, ydata, sigma = sigmadata)
    A0 = popt[0]
    E0 = popt[1]
    ddl = sup-inf+1-2
    ypred = fit(xdata, A0, E0)
    chi2 = np.sum(((ydata-ypred)/sigmadata)**2)
    chi2red = chi2/ddl
    return [A0, E0, chi2, chi2red], xdata, ydata, ypred, sigmadata


# # Effective mass plateau

# In[37]:


def plateau(nomoy, sigmaobs):
    chi2 = []
    for i in range(2,31):
        params, xdata, ydata, ypred, sigmadata = estim(31-i, 31+i, nomoy, sigmaobs)
        chi2.append(params[2])
            
    dernier_indice_satisfaisant = -1  # Initialisation avec un indice invalide au cas où aucun élément ne satisfait la condition
    #print(chi2)
    for i, valeur in enumerate(chi2):
        if valeur < 1:
            dernier_indice_satisfaisant = i
    dernier_indice_satisfaisant = dernier_indice_satisfaisant+2
    inf = 31-dernier_indice_satisfaisant
    sup = 31+dernier_indice_satisfaisant
    #print("inf = " + str(inf))
    #print("sup = " + str(sup))
    return inf, sup


# ## Jacknife

# In[43]:


nconf = 999
a =  0.0652
x = (1/a)*197.327


# In[44]:


def jack(n, nt, re, im): #retourne les listes des nt, re, im par configuration ou il manque la nième configuration
    ntll = [nt[i:i+64] for i in range(0, len(nt),64)]
    rell = [re[i:i+64] for i in range(0, len(re),64)]
    imll = [im[i:i+64] for i in range(0, len(im),64)]
    ntll.pop(n)
    rell.pop(n)
    imll.pop(n)
    return ntll, rell, imll


# In[60]:


def jacknife(nconf, fichier, a, x):
    nt, re, im = lecture(fichier)
    E0jn = [] #liste des E0 pour chaque set du jacknife
    
    ntll, rell, imll = par_configs(nt,re,im)
    nont = normes_par_nt(ntll, rell, imll)
    nomoy, sigmaobs = moyenne_norme(nont)
    inf,sup = plateau(nomoy, sigmaobs)
    params, xdata, ydata, ypred, sigmadata = estim(inf, sup, nomoy, sigmaobs)
    
    E0 = params[1] #on trouve E0 sans jacknife d'abord
    
    for n in range(nconf): #on fait le jacknife
        
        ntll, rell, imll = jack(n,nt,re,im)
        
        nont = normes_par_nt(ntll, rell, imll)
        
        nomoy, sigmaobs = moyenne_norme(nont)
        
        inf,sup = plateau(nomoy, sigmaobs)
        
        params, xdata, ydata, ypred, sigmadata = estim(inf, sup, nomoy, sigmaobs)
        
        E0jn.append(params[1])
        
        if (n%100==0): #compteur
            print(n)
            
    E0tild = np.mean(np.array(E0jn))
    biais = (len(E0jn)-1)*(E0tild-E0)
    E0unb = E0 - biais
    sigmajn2 = np.sqrt(np.var(E0jn))
    sigmajn = np.sqrt(((nconf-1)/nconf)*np.sum((np.array(E0jn) - np.ones(nconf)*E0)**2))  #Erreur sur E0 estimée par le jacknife
    
    print("E0 biased= " + str(E0) + " +/- " + str(sigmajn))
    print("M biased = " + str(E0*x) + " +/- " + str(sigmajn*x) + " MeV")
    return E0jn, E0, E0unb, sigmajn
    


# ## Bootstrap

# In[46]:


def create_set(nt, re, im, nconf):
    ntll = [nt[i:i+64] for i in range(0, len(nt),64)]
    rell = [re[i:i+64] for i in range(0, len(re),64)]
    imll = [im[i:i+64] for i in range(0, len(im),64)]
    conf = [i for i in range(nconf)]
    ind = random.choices(conf,k=nconf)
    ntll = [ntll[i] for i in ind]
    rell = [rell[i] for i in ind]
    imll = [imll[i] for i in ind]
    return ntll, rell, imll


# In[47]:


def bootstrap(nbs, nconf, fichier, a, x):
    nt, re, im = lecture(fichier)
    E0bs = []
    
    for n in range(nbs):
        
        ntll, rell, imll = create_set(nt, re, im, nconf)
        
        nont = normes_par_nt(ntll, rell, imll)
        
        nomoy, sigmaobs = moyenne_norme(nont)
        
        inf,sup = plateau(nomoy, sigmaobs)
        
        params, xdata, ydata, ypred, sigmadata = estim(inf, sup, nomoy, sigmaobs)
        
        E0bs.append(params[1])
        
        if (n%100==0): #compteur
            print(n)
    
    E0bstild = np.mean(np.array(E0bs))  #estimator for E0
    sigmabs = np.sqrt(np.var(E0bs)) #error for E0 by bootstrapping
    
    print("\n")
    print("E0 = " + str(E0bstild) + " +/- " + str(sigmabs))
    print("M = " + str(E0bstild*x) + " +/- " + str(sigmabs*x) + " MeV")
    
    return E0bs, E0bstild, sigmabs

