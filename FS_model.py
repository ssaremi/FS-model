#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:15:48 2018

FS model implementation code.

@author: sepsaremi
"""

import difflib
import json
from sklearn.neighbors.kde import KernelDensity
import numpy as np
import pandas as pd
import random


train_reduce = train.select(\
                            train.business_info.NAICS_industry_digit.alias("naics"), train.business_info.address.city.alias("city"), 
                            train.business_info.address.state.alias("state"), train.business_info.be_id.alias("be_id"),
                            train.contact_id.alias("contact_id"), train.jobInfo.emails.alias("emails"), train.jobInfo.title.alias("title"), 
                            train.personInfo.firstName.alias("firstName"), train.personInfo.lastName.alias("lastName") \
                           )


def create_samples(the_data):
  match_sample = []
  unmatch_sample = []
  data_size = len(the_data)
  for i, ei in enumerate(the_data):
    for j, ej in enumerate(the_data):
      if i != j:
        temp = []
        if the_data[i]["contact_id"] == the_data[j]["contact_id"]:
        #if the_data[i]["beId"] == the_data[j]["beId"]:
          temp.append(the_data[i])
          temp.append(the_data[j])
          match_sample.append(temp)
        else:
          temp.append(the_data[i])
          temp.append(the_data[j])
          unmatch_sample.append(temp)
  return match_sample, unmatch_sample
        
match_pairs, unmatch_pairs = create_samples(train_list)

def seq_prob(str1, str2):
  if bool(str1) and bool(str2):
    str1_low = str1.lower()
    str2_low = str2.lower()
    r = difflib.SequenceMatcher(None, str1_low, str2_low).ratio()
  else:
    r = 0
  return r


comp_features = ["city", "emails", "title", "firstName", "lastName"]

# the following is a function for calculating the similarity vector (gamma) between two records
def calc_gamma(the_pair, features):
  gamma_vector = []
  for k in features:
    first_elem = the_pair[0][k]
    second_elem = the_pair[1][k]
    if type(first_elem) == list:
      first_elem = ''.join(str(e) for e in first_elem)
      second_elem = ''.join(str(e) for e in second_elem)
    comp = seq_prob(first_elem, second_elem)
    gamma_vector.append(comp)
  return gamma_vector

# the following function returns the list of gamma vectors for a sample of paired records
def create_vectors(data_category, features):
  data_size = len(data_category)
  temp_list = []
  for i in range(data_size):
    gamma_vector = calc_gamma(data_category[i], features)
    temp_list.append(gamma_vector)
  return temp_list

match_gamma = create_vectors(match_pairs, comp_features)
unmatch_gamma = create_vectors(unmatch_pairs_reduced, comp_features)


bandwidth = 1.
kde_match = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
kde_unmatch = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
#match_gamma_array = np.array(match_gamma)
#unmatch_gamma_array = np.array(unmatch_gamma)
#match_gamma_array.reshape(1,-1)
#unmatch_gamma_array.reshape(1,-1)
kde_match.fit(match_gamma)
kde_unmatch.fit(unmatch_gamma)

print("The parameters for the matched Kernel are: {}".format(kde_match.get_params))
print("The parameters for the unmatched Kernel are: {}".format(kde_unmatch.get_params))

def kde_prob(kde_category, matching_vector):
  # score_samples() returns the log-likelihood of the samples
  matching_array = np.array(matching_vector)
  matching_array.reshape(1,-1)
  log_pdf = kde_category.score_samples(matching_vector)
  return np.exp(log_pdf)
  
def link_ratio(comp_vector):
  comp_vector = np.array(comp_vector).reshape(1,-1)
  prob_match = kde_prob(kde_match, comp_vector)
  prob_unmatch = kde_prob(kde_unmatch, comp_vector)
  epsilon = 0.00000001
  ratio = prob_match/(prob_unmatch + epsilon)
  return ratio


