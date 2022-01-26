import os
import sys

import xml.etree.ElementTree as ET

import numpy as np
import math
import glob

def parse_xml_dir(filepaths, cut_size, step_pix=1000):
        
#     print(step_pix)
    all_leads = []

    for xml_filepath in filepaths:

        # parse xml
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        
        leads_12 = []
        
        tag_pre = ''
        for child in root:
            if child.tag.startswith('{urn:hl7-org:v3}'):
                tag_pre = '{urn:hl7-org:v3}'
        
        for elem in tree.iterfind('./%scomponent/%sseries/%scomponent/%ssequenceSet/%scomponent/%ssequence'%(tag_pre,tag_pre,tag_pre,tag_pre,tag_pre,tag_pre)):
    #         print('-------')
            for child_of_elem in elem:

                if child_of_elem.tag == '%scode'%(tag_pre):

                    if child_of_elem.attrib['code'] == 'TIME_ABSOLUTE': break

                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V3R': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V4R': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V5R': break

                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V7': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V8': break
                    if child_of_elem.attrib['code'] == 'MDC_ECG_LEAD_V9': break

#                     print(child_of_elem.attrib['code'])
                    for grand_child_digits in elem.iterfind('%svalue/%sdigits'%(tag_pre,tag_pre)):
                        arr = grand_child_digits.text.split(' ')
                        num_samples = np.array(arr).shape[0]
                        leads_12.append(arr)                        
            
        leads_12 = np.array(leads_12) # (12, 31600)
        leads_12_dim = leads_12.transpose(1,0) # (31600, 12)
        
        # cut size
#         step_pix = 1000
        for idx in range(0, leads_12.shape[-1] - cut_size, step_pix):

            sample = leads_12_dim[idx:idx + cut_size, :]
            sample = sample.astype(np.float32)
            mean = np.mean(sample)
            std = np.std(sample)
            if std > 0:
                ret = (sample - mean) / std
            else:
                ret = sample * 0
            all_leads.append(ret)  
    all_leads = np.array(all_leads)
        
    return all_leads








