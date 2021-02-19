from . import source
import re
import numpy as np
from difflib import SequenceMatcher

def find_samples(input_dir, channel_tag='_C', z_position_tag='_z'):
    tags = {'c': channel_tag, 'z': z_position_tag}
    
    # list all files in the input folder
    src = source.FilePatternSource(Path(input_dir), axis_tags=tags.copy(), 
                                             extensions=[".tif"], include_filters=[], exclude_filters=[])
    df = src._files.files
    
    # return subset of files that have the same z and c position to extract unique sample names
    subset = df
    for key in tags.keys():
        subset = subset[subset[key] == df.iloc[0][key]]
    fns = np.array(subset['file'])
    fns = [fn.replace(input_dir, '') for fn in fns] # exclude input directory from the file name
    
    # find substrings that match between the first and second strings
    matches = SequenceMatcher(None, fns[0], fns[1]).get_matching_blocks()
    matching_strings = []
    for match in matches:
        if match.size > 0:
            matching_strings.append(fns[0][match.a : match.a + match.size])
      
    # select substrings that match across all file names
    matching_strings_in_all = []
    for st in matching_strings:
        is_in_all = True
        for fn in fns:
            if len(fn.split(st)) == 1:
                is_in_all = False
        if is_in_all:
            matching_strings_in_all.append(st)
            
    # remove repeating substrings from the file name subset
    # what remains is a unique sample name that exclude any channel or z information
    for st in matching_strings_in_all:
        fns = [fn.replace(st, '') for fn in fns]
    
    # iterate over sample names and return intake_io sources that include each of these samples:
    sources = []
    for fn in fns:
        sources.append(source.FilePatternSource(Path(input_dir), axis_tags=tags.copy(), 
                                                          extensions=[".tif"], include_filters=[fn], exclude_filters=[]))
    return sources
