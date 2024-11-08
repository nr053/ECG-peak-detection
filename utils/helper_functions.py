import numpy as np
import random
from tqdm import tqdm

#
### Data/Model-related Classes
#
class strip_types():
    nsr         = 'NSR'     # Normal Sinus Rhythm
    tachyN      = 'TachyN'  # Normal Tachycardia
    bradyN      = 'BradyN'  # Normal Bradycardia
    
    singletS    = 'S'       # Supraventricular Singlet
    singletS_N  = 'S_N'     # Normal in Supraventricular Singlet
    coupletS    = 'SC'      # Supraventricular Couplet
    coupletS_N  = 'SC_N'    # Normal in Supraventricular Couplet
    runsS       = 'RunsS'   # Supraventricular Runs
    bigeS       = 'BigeS'   # Supraventricular Bigeminy
    bigeS_N     = 'BigeS_N' # Normal in Supraventricular Bigeminy
    trigeS      = 'TrigeS'  # Supraventricular Trigeminy
    trigeS_N    = 'TrigeS_N'# Normal in Supraventricular Trigeminy
    tachyS      = 'TachyS'  # Supraventricular Tachycardia
    
    singletV    = 'V'       # Ventricular Singlet
    singletV_N  = 'V_N'     # Normal in Ventricular Singlet
    coupletV    = 'VC'      # Ventricular Couplet
    coupletV_N  = 'VC_N'    # Normal in Ventricular Couplet
    runsV       = 'RunsV'   # Ventricular Runs
    bigeV       = 'BigeV'   # Ventricular Bigeminy
    bigeV_N     = 'BigeV_N' # Normal in Ventricular Bigeminy
    trigeV      = 'TrigeV'  # Ventricular Trigeminy
    trigeV_N    = 'TrigeV_N'# Normal in Ventricular Trigeminy
    tachyV      = 'TachyV'  # Ventricular Tachycardia
    ivr         = 'IVR'     # Idioventricular Rhythm
    
    afib         = 'AFIB'    # AFIB (+AFLU)
    afib_N       = 'AFIB_N'  # Normal in AFIB
    afib_V       = 'AFIB_V'  # Ventricular in AFIB
    
    #av_block     = 'av_block'# AV Block
    noise        = 'Noise'   # Noise

class beat_types():
    normal          = 'N'
    supraventricular= 'S'
    ventricular     = 'V'
    branch_block    = 'B'
    artifact        = 'A'

class beat_types_numerical():
    normal          = 0
    supraventricular= 1
    ventricular     = 2
    branch_block    = 3
    artifact        = 4

class rhythm_labels():
    sinus_rhythm    = 'SR'      # Sinus Rhythms
    afib            = 'AF'      # Atrial Fibrillation/Flutter
    other           = 'Other'   # Other class
    noise           = 'Noise'   # Noise

class rhythm_labels_numerical():
    sinus_rhythm    = 0         # Sinus Rhythms
    afib            = 1         # Atrial Fibrillation/Flutter
    other           = 2         # Other class
    noise           = 3         # Noise    

class signal_quality_assessment_labels_numerical():
    unsure  = -2    # Unsure quality signal
    poor    = 0     # Bad/Poor quality signal
    medium  = 1     # Medium quality signal
    good    = 2     # Good quality signal
    pause   = 3     # Pause

class signal_quality_assessment_strip_sample_sizes():
    single_mini = 597   # single mini-strip length in samples
    double_mini = 1194  # double -//-
    triple_mini = 1791  # triple -//-

class VAF_beat_extraction_methods():
    central = 'central'     # Extract N closest to center
    couplet = 'couplet'     # Extract 2 closest to center beats that are consecutive
    run     = 'run'         # Extract 1 central (beginning of Run) and 1 in the middle of Run

class VAF_sample_length_safety_thresholds():
    """ Maximum samples to consider for compensating with the 12-sample-long euclidean division remainder
        on the signal lengths included in VAFs. The optimal sample length thresholds have been found experimentally
        from statistics on the lengths of the different VAF signal types (i.e. strip, context data).
    """
    strip_7sec = 1782           # 7sec strips (maximum theoretical: 1792 samples)
    strip_7sec_sig_qual = 1781  # 7sec strips for signal quality classifier NOTE: this can be replaced with the general strip_7sec in a future iteration
    context_30sec = 7669        # 30sec context signal (max theoretical: 7680 samples)


#
### Structure Generators
#

# Generic
def gen_empty_dict_from_list(label_list_generator_function, output_dict_type='count', n_samples=0, n_channels=0):
    assert output_dict_type in ['count', 'list', 'array_given_dimensions'], 'The given output_dict_type does not belong to the available options.'
    labels_list = label_list_generator_function
    output_dict = {}
    if output_dict_type == 'count':
        for label in labels_list:
            output_dict[label] = 0
    elif output_dict_type == 'list':
        for label in labels_list:
            output_dict[label] = []
    elif output_dict_type == 'array_given_dimensions':
        assert n_samples>0 and n_channels>0, 'Error generating dict structure. Please specify both n_samples and n_channels.'
        for label in labels_list:
            output_dict[label] = np.empty((0, n_samples, n_channels))
    return output_dict

# Beat Classifier
def gen_beatclassifier_2_class_beat_labels_list():
    """ Returns the list of annotation labels for the 2 different beat types.
    """
    return [
        beat_types.normal,
        beat_types.ventricular,
        ]

def gen_beatclassifier_4_class_beat_labels_list():
    """ Returns the list of annotation labels for the 4 different beat types.
    """
    return [
        beat_types.normal,
        beat_types.ventricular,
        beat_types.branch_block,
        beat_types.artifact
        ]

def gen_beatclassifier_5_class_beat_labels_list():
    """ Returns the list of annotation labels for the 5 different beat types.
    """
    return [
        beat_types.normal,
        beat_types.supraventricular,
        beat_types.ventricular,
        beat_types.branch_block,
        beat_types.artifact
        ]

def gen_beatclassifier_strip_types_list():
    return [
        strip_types.nsr,
        strip_types.tachyN,
        strip_types.bradyN,
        strip_types.afib,
        strip_types.singletS,
        strip_types.coupletS,
        strip_types.bigeS,
        strip_types.trigeS,
        strip_types.runsS,
        strip_types.tachyS,
        strip_types.singletV,
        strip_types.coupletV,
        strip_types.bigeV,
        strip_types.trigeV,
        strip_types.runsV,
        strip_types.tachyV,
        strip_types.ivr
        ] 

def gen_VAF_strip_beat_extraction_rules():
    return {
        ####
        # Normal Sinus Rhythm
        strip_types.nsr:      [{'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.nsr}],          # subclass #0
        
        # Normal Tachycardia
        strip_types.tachyN:   [{'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.tachyN}],       # subclass #1
        
        # Normal Bradycardia
        strip_types.bradyN:   [{'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.bradyN}],       # subclass #2
        
        ####
        # AFIB
        strip_types.afib:     [{'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.afib_N},        # subclass #11
                               
                               {'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.afib_V}],       # subclass #25
        
        ####
        # Supraventricular Singlet
        strip_types.singletS: [{'beat_type': beat_types_numerical.supraventricular, 
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 1,
                                'subclass': strip_types.singletS},      # subclass #12
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.singletS_N}],   # subclass #3
        
        # Supraventricular Couplet
        strip_types.coupletS: [{'beat_type': beat_types_numerical.supraventricular,
                                'method':    VAF_beat_extraction_methods.couplet,
                                'num_beats': 2,
                                'subclass': strip_types.coupletS},      # subclass #13
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.coupletS_N}],   # subclass #4
        
        # Supraventricular Bigeminy
        strip_types.bigeS:    [{'beat_type': beat_types_numerical.supraventricular, 
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.bigeS},         # subclass #14
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.bigeS_N}],      # subclass #5
        
        # Supraventricular Trigeminy
        strip_types.trigeS:   [{'beat_type': beat_types_numerical.supraventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.trigeS},        # subclass #15
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.trigeS_N}],     # subclass #6
        
        # Supraventricular Run
        strip_types.runsS:    [{'beat_type': beat_types_numerical.supraventricular,
                                'method':    VAF_beat_extraction_methods.run,
                                'num_beats': 2,
                                'subclass': strip_types.runsS}],        # subclass #16
        
        # Supraventricular Tachycardia
        strip_types.tachyS:   [{'beat_type': beat_types_numerical.supraventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.tachyS}],       # subclass #17
        
        ####
        # Ventricular Signlet
        strip_types.singletV: [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 1,
                                'subclass': strip_types.singletV},      # subclass #18
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 1,
                                'subclass': strip_types.singletV_N}],   # subclass #7
        
        # Ventricular Couplet
        strip_types.coupletV: [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.couplet,
                                'num_beats': 2,
                                'subclass': strip_types.coupletV},      # subclass #19
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.coupletV_N}],   # subclass #8
       
        # Ventricular Bigeminy
        strip_types.bigeV:    [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.bigeV},         # subclass #20
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.bigeV_N}],      # subclass #9
        
        # Ventricular Trigeminy
        strip_types.trigeV:   [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.trigeV},        # subclass #21
                               
                               {'beat_type': beat_types_numerical.normal,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 3,
                                'subclass': strip_types.trigeV_N}],     # subclass #10
        
        # Ventricular Run
        strip_types.runsV:    [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.run,
                                'num_beats': 2,
                                'subclass': strip_types.runsV}],        # subclass #22
        
        # Ventricular Tachycardia
        strip_types.tachyV:   [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.tachyV}],       # subclass #23
        
        # Ventricular Idioventricular Rhythm
        strip_types.ivr:      [{'beat_type': beat_types_numerical.ventricular,
                                'method':    VAF_beat_extraction_methods.central,
                                'num_beats': 2,
                                'subclass': strip_types.ivr}],          # subclass #24
        }

def gen_beatclassifier_VAF_subclass_list():
    return [
        strip_types.nsr,        #0
        strip_types.tachyN,     #1
        strip_types.bradyN,     #2
        strip_types.singletS_N, #3
        strip_types.coupletS_N, #4  
        strip_types.bigeS_N,    #5
        strip_types.trigeS_N,   #6
        strip_types.singletV_N, #7
        strip_types.coupletV_N, #8
        strip_types.bigeV_N,    #9
        strip_types.trigeV_N,   #10
        strip_types.afib_N,     #11
        strip_types.singletS,   #12
        strip_types.coupletS,   #13
        strip_types.bigeS,      #14
        strip_types.trigeS,     #15
        strip_types.runsS,      #16
        strip_types.tachyS,     #17
        strip_types.singletV,   #18
        strip_types.coupletV,   #19
        strip_types.bigeV,      #20
        strip_types.trigeV,     #21
        strip_types.runsV,      #22
        strip_types.tachyV,     #23
        strip_types.ivr,        #24
        strip_types.afib_V      #25
        ]

def gen_beatclassifier_full_subclass_list():
    return gen_beatclassifier_VAF_subclass_list() + [beat_types.branch_block, beat_types.artifact]

def gen_beatclassifier_VAF_subclass_to_class_dict():
    return dict({
        strip_types.nsr         : beat_types_numerical.normal,              #0
        strip_types.tachyN      : beat_types_numerical.normal,              #1
        strip_types.bradyN      : beat_types_numerical.normal,              #2
        strip_types.singletS_N  : beat_types_numerical.normal,              #3
        strip_types.coupletS_N  : beat_types_numerical.normal,              #4  
        strip_types.bigeS_N     : beat_types_numerical.normal,              #5
        strip_types.trigeS_N    : beat_types_numerical.normal,              #6
        strip_types.singletV_N  : beat_types_numerical.normal,              #7
        strip_types.coupletV_N  : beat_types_numerical.normal,              #8
        strip_types.bigeV_N     : beat_types_numerical.normal,              #9
        strip_types.trigeV_N    : beat_types_numerical.normal,              #10
        strip_types.afib_N      : beat_types_numerical.normal,              #11
        strip_types.singletS    : beat_types_numerical.supraventricular,    #12
        strip_types.coupletS    : beat_types_numerical.supraventricular,    #13
        strip_types.bigeS       : beat_types_numerical.supraventricular,    #14
        strip_types.trigeS      : beat_types_numerical.supraventricular,    #15
        strip_types.runsS       : beat_types_numerical.supraventricular,    #16
        strip_types.tachyS      : beat_types_numerical.supraventricular,    #17
        strip_types.singletV    : beat_types_numerical.ventricular,         #18
        strip_types.coupletV    : beat_types_numerical.ventricular,         #19
        strip_types.bigeV       : beat_types_numerical.ventricular,         #20
        strip_types.trigeV      : beat_types_numerical.ventricular,         #21
        strip_types.runsV       : beat_types_numerical.ventricular,         #22
        strip_types.tachyV      : beat_types_numerical.ventricular,         #23
        strip_types.ivr         : beat_types_numerical.ventricular,         #24
        strip_types.afib_V      : beat_types_numerical.ventricular          #25
    })

def gen_beatclassifier_all_subclass_to_class_dict():
    return dict({
        strip_types.nsr         : beat_types_numerical.normal,              #0
        strip_types.tachyN      : beat_types_numerical.normal,              #1
        strip_types.bradyN      : beat_types_numerical.normal,              #2
        strip_types.singletS_N  : beat_types_numerical.normal,              #3
        strip_types.coupletS_N  : beat_types_numerical.normal,              #4  
        strip_types.bigeS_N     : beat_types_numerical.normal,              #5
        strip_types.trigeS_N    : beat_types_numerical.normal,              #6
        strip_types.singletV_N  : beat_types_numerical.normal,              #7
        strip_types.coupletV_N  : beat_types_numerical.normal,              #8
        strip_types.bigeV_N     : beat_types_numerical.normal,              #9
        strip_types.trigeV_N    : beat_types_numerical.normal,              #10
        strip_types.afib_N      : beat_types_numerical.normal,              #11
        strip_types.singletS    : beat_types_numerical.supraventricular,    #12
        strip_types.coupletS    : beat_types_numerical.supraventricular,    #13
        strip_types.bigeS       : beat_types_numerical.supraventricular,    #14
        strip_types.trigeS      : beat_types_numerical.supraventricular,    #15
        strip_types.runsS       : beat_types_numerical.supraventricular,    #16
        strip_types.tachyS      : beat_types_numerical.supraventricular,    #17
        strip_types.singletV    : beat_types_numerical.ventricular,         #18
        strip_types.coupletV    : beat_types_numerical.ventricular,         #19
        strip_types.bigeV       : beat_types_numerical.ventricular,         #20
        strip_types.trigeV      : beat_types_numerical.ventricular,         #21
        strip_types.runsV       : beat_types_numerical.ventricular,         #22
        strip_types.tachyV      : beat_types_numerical.ventricular,         #23
        strip_types.ivr         : beat_types_numerical.ventricular,         #24
        strip_types.afib_V      : beat_types_numerical.ventricular,         #25
        beat_types.branch_block : beat_types_numerical.branch_block,        #26
        beat_types.artifact     : beat_types_numerical.artifact             #27
    })

# Rhythm Classifier
def gen_AFIB_VAE_model_strip_types_list():
    return [
        strip_types.nsr,        #0
        strip_types.tachyN,     #1
        strip_types.bradyN,     #2
        strip_types.afib,       #3
        strip_types.runsS,      #4
        strip_types.runsV,      #5
        strip_types.tachyS,     #6
        strip_types.tachyV,     #7
        strip_types.bigeV,      #8
        strip_types.bigeS,      #9
        strip_types.trigeV,     #10
        strip_types.trigeS,     #11
        #strip_types.av_block,   #12
        strip_types.ivr         #13
        ]

def gen_AFIB_VAE_rhythm_class_list():
    return [
        rhythm_labels.sinus_rhythm,
        rhythm_labels.afib,
        rhythm_labels.other
        ]

def gen_3_class_strip_grouping_dict():
    """ Returns an initialized dictionary holding the 3-class division annotation for
        VAF strip types, used in the AFIB VAE rhythm model.
    """
    return dict({
        'NSR'       : rhythm_labels_numerical.sinus_rhythm, # Normal Sinus Rhythm
        'TachyN'    : rhythm_labels_numerical.sinus_rhythm, # Normal Tachycardia
        'BradyN'    : rhythm_labels_numerical.sinus_rhythm, # Normal Bradycardia
        'AFIB'      : rhythm_labels_numerical.afib,         # AFIB (this class includes also AFLU)
        'RunsS'     : rhythm_labels_numerical.other,        # Supraventricular Runs
        'RunsV'     : rhythm_labels_numerical.other,        # Ventricular Runs
        'TachyS'    : rhythm_labels_numerical.other,        # Supraventricular Tachycardia
        'TachyV'    : rhythm_labels_numerical.other,        # Ventricular Tachycardia
        'BigeV'     : rhythm_labels_numerical.other,        # Ventricular Bigeminy
        'BigeS'     : rhythm_labels_numerical.other,        # Supraventricular Bigeminy
        'TrigeV'    : rhythm_labels_numerical.other,        # Ventricular Trigeminy
        'TrigeS'    : rhythm_labels_numerical.other,        # Supraventricular Trigeminy
        #'av_block'  : rhythm_labels_numerical.other,        # AV Block
        'IVR'       : rhythm_labels_numerical.other,        # Idioventricular Rhythm
        'Noise'     : rhythm_labels_numerical.other         # Noise
        })    

def gen_SR_subclass_list():
    return [
        strip_types.nsr,
        strip_types.tachyN,
        strip_types.bradyN
        ]

def gen_Other_subclass_list():
    return [
        strip_types.runsS,
        strip_types.runsV,
        strip_types.tachyS,
        strip_types.tachyV,
        strip_types.bigeV,
        strip_types.bigeS,
        strip_types.trigeV,
        strip_types.trigeS,
        #strip_types.av_block,
        strip_types.ivr
        ]

def gen_Other_noTachySV_subclass_list():
    return [
        strip_types.runsS,
        strip_types.runsV,
        strip_types.bigeV,
        strip_types.bigeS,
        strip_types.trigeV,
        strip_types.trigeS,
        #strip_types.av_block,
        strip_types.ivr
    ]

def gen_non_AF_subclass_list():
    return [strip_type
            for strip_type in gen_AFIB_VAE_model_strip_types_list()
            if strip_type not in [strip_types.afib]]

# Signal Quality Assessment Model
def gen_sig_quality_assess_3_class_list():
    return [
        signal_quality_assessment_labels_numerical.poor,
        signal_quality_assessment_labels_numerical.medium,
        signal_quality_assessment_labels_numerical.good
        ]

def gen_sig_quality_assess_4_class_list():
    return [
        signal_quality_assessment_labels_numerical.poor,
        signal_quality_assessment_labels_numerical.medium,
        signal_quality_assessment_labels_numerical.good,
        signal_quality_assessment_labels_numerical.pause
        ]


#
### Functions
#
def check_beat_start_couplet(beat_idx, annotations, beat_type):
    """ Function for detecting the pattern of a couplet in a sequence of beats.
        beat_idx: the index of the starting beat of the couplet candidate
    """
    try:
        return (annotations[beat_idx -1] == 0 and
                annotations[beat_idx   ] == beat_type and
                annotations[beat_idx +1] == beat_type and
                annotations[beat_idx +2] == 0)
    except:
        return False

def check_beat_start_run(beat_idx, annotations, beat_type):
    """ Function for detecting the pattern of the beginning of a run of beats.
        beat_idx: the index of the starting beat of the run candidate
    """
    try:
        return (annotations[beat_idx -2] == 0 and
                annotations[beat_idx -1] == 0 and
                annotations[beat_idx   ] == beat_type and
                annotations[beat_idx +1] == beat_type)
    except:
        return False

def check_beat_middle_run(beat_idx, annotations, beat_type):
    """ Function for detecting the pattern of the middle of a run of beats.
        beat_idx: the index of any middle beat inside the run candidate
    """
    try:
        return (annotations[beat_idx -1] == beat_type and
                annotations[beat_idx   ] == beat_type and
                annotations[beat_idx +1] == beat_type)
    except:
        return False

def calculate_beat_selection_corner_samples(beat_sample, offset_mu, offset_std, pre_beat_window_size, post_beat_window_size):
    """ Function that returns the selected beats' corner samples, after applying a Gaussian offset
        to the beat's R-peak (beat sample) and the pre/post selection window samples.
        
        INPUTS
                beat_sample:            the sample of the considered beat
                offset_mu:              the mean of the Gaussian offset to apply to the extracted
                                        beat's data starting point
                offset_std:             the standard deviation of the Gaussian offset to apply to
                                        the extracted beat's data starting point
                pre_beat_window_size:   number of samples from the beginning of extracted to window
                                        to the center of considered beat (R-peak)
                post_beat_window_size:  number of samples from center of considered beat (R-peak)
                                        to end of extracted window
        OUTPUTS
                selection_start_sample: the sample index for starting the preprocessing window from
                selection_end_sample:   the sample index for ending the preprocessing window with
    """
    np.random.seed(10)
    beat_offset_sample = int(np.random.normal(offset_mu, offset_std, 1)) # apply custom offset to match the library with parameters mu/std
    selection_start_sample = beat_sample + beat_offset_sample - pre_beat_window_size
    selection_end_sample =   beat_sample + beat_offset_sample + post_beat_window_size
    return selection_start_sample, selection_end_sample

def extract_start_sample_positions_from_VAF_strips(strip_start_ms, beat_positions, beat_annotations, strip_beat_extraction_rule_list, offset_mu, offset_std, window_size_samples, window_center_fraction, strip_sample_length):
    """ Function for returning the start sample positions for the beats that will
        be extracted from the VAF strips.
        
        INPUTS
                strip_start_ms:         the starting position of the strip in the recording in milliseconds
                beat_positions:         list of sample index positions of strip's beats in the whole recording
                beat_annotations:       list of integer annotation labels of strip's beats
                strip_extraction_rules: dictionary with tuples describing the beat extraction rules per strip_type 
                offset_mu:              the mean of the Gaussian offset to apply to the extracted beat's data starting point
                offset_std:             the standard deviation of the Gaussian offset to apply to the extracted
                                        beat's data starting point
                window_size_samples:   the length of the data extraction window in samples
                window_center_fraction: float of the percentage where the peak's middle (R-peak) sample is placed
                                        from the beginning of the extracted window (i.e. its first sample)
                strip_sample_length:    the length of the input strip in samples
        OUTPUTS
                start_samples_list:     list of start sample positions from which data extraction and subsequently
                                        preprocessing will be performed
                selected_beat_idx_list: list of index selection of the beats included in the current VAF strip
                detection_comment:      string holding a comment about the pattern that has been detected
                                        in the given input VAF strip
    """

    # Initialize the list for storing the start sample points of interest
    start_samples_list = []
    selected_beat_idx_list = []
    selected_beat_type_list = []

    # Initialize (default) detection comment
    detection_comment = ''

    # Calculate the pre/post beat sample size
    PRE_BEAT_WINDOW_SIZE = int(np.floor(window_size_samples*window_center_fraction))
    POST_BEAT_WINDOW_SIZE = int(np.floor(window_size_samples*(1 - window_center_fraction)))

    # Iterate the list of rules (dicts) for the how to handle the different beat_types, per strip
    for strip_beat_extraction_rule in strip_beat_extraction_rule_list:

        # Select the indexes of the wanted beat type in the strip
        wanted_type_beat_idx = np.where(np.array(beat_annotations) == strip_beat_extraction_rule['beat_type'])[0]

        # Return if no beats of interest are found
        if len(wanted_type_beat_idx) == 0:
            continue

        ## Sort the wanted beat indexes based on their distance from the center of the strip
        # Position differences from center
        wanted_type_beat_pos_diff_from_center = np.array([beat_pos - strip_start_ms - 3500 for beat_pos in np.array(beat_positions)[wanted_type_beat_idx] ])
        # Sorted indexes of position differences from center
        wanted_type_beat_pos_diff_from_center_sorted_idx = np.argsort(abs(wanted_type_beat_pos_diff_from_center))
        # Wanted beat type indexes, sorted by their distance from center
        wanted_type_beat_from_center_idx = wanted_type_beat_idx[wanted_type_beat_pos_diff_from_center_sorted_idx]

        # Turn the beat positions into samples
        beat_samples = [int(round((beat_pos-strip_start_ms)*256/1000)) for beat_pos in beat_positions]
        
        ## CASE 1: Single beat extraction that is closest to the center of the strip
        if strip_beat_extraction_rule['method'] == VAF_beat_extraction_methods.central:
            for extracted_beat_idx in range(strip_beat_extraction_rule['num_beats']):
                
                if extracted_beat_idx < len(wanted_type_beat_from_center_idx):
                    selection_start_sample, selection_end_sample = calculate_beat_selection_corner_samples(beat_samples[wanted_type_beat_from_center_idx[extracted_beat_idx]],
                                                                                                            offset_mu,
                                                                                                            offset_std,
                                                                                                            PRE_BEAT_WINDOW_SIZE,
                                                                                                            POST_BEAT_WINDOW_SIZE)

                    if (selection_start_sample >= 0) and (selection_end_sample < strip_sample_length):
                        start_samples_list.append(selection_start_sample)
                        selected_beat_idx_list.append(wanted_type_beat_from_center_idx[extracted_beat_idx])
                        selected_beat_type_list.append(strip_beat_extraction_rule['beat_type'])
        
        ## CASE 2: Couplet (2 beats) extraction that are closest to the center of the strip
        elif strip_beat_extraction_rule['method'] == VAF_beat_extraction_methods.couplet:
            beat_offsets = [-3, -2, -1, 0, 1, 2, 3]
            for beat_offset in beat_offsets:
                detected_coup_pattern_flag = False
                # Check if the pattern of the beginning of a couplet is detected
                if check_beat_start_couplet(wanted_type_beat_from_center_idx[0] + beat_offset,
                                            beat_annotations,
                                            strip_beat_extraction_rule['beat_type']):
                    detected_coup_pattern_flag = True
                    # Select the candidate beat samples
                    couplet_candidate_beat_idx_list = [wanted_type_beat_from_center_idx[0] + beat_offset,
                                                       wanted_type_beat_from_center_idx[0] + beat_offset +1]
                    
                    for couplet_candidate_idx in couplet_candidate_beat_idx_list:
                        selection_start_sample, selection_end_sample = calculate_beat_selection_corner_samples(beat_samples[couplet_candidate_idx],
                                                                                                            offset_mu,
                                                                                                            offset_std,
                                                                                                            PRE_BEAT_WINDOW_SIZE,
                                                                                                            POST_BEAT_WINDOW_SIZE)

                        if (selection_start_sample >= 0) and (selection_end_sample < strip_sample_length):
                            start_samples_list.append(selection_start_sample)
                            selected_beat_idx_list.append(couplet_candidate_idx)
                            selected_beat_type_list.append(strip_beat_extraction_rule['beat_type'])

                    break
            # == DEBUGGING ==
            if detected_coup_pattern_flag:
                detection_comment += 'Detected Couplet'       
            else: 
                detection_comment += 'No Couplet Detected'
            #    print('\nNo Couplet Detected:')
            #    print(beat_annotations)
            #    print('ok')

        ## CASE 3: Run beats extraction, one from the start and one from the inside of the run
        elif strip_beat_extraction_rule['method'] == VAF_beat_extraction_methods.run:
            beat_offsets = [-3, -2, -1, 0, 1]
            for beat_offset in beat_offsets:
                detected_run_start = False
                detected_run_end = False
                # Check if the pattern of the beginning of a run is detected
                if check_beat_start_run(wanted_type_beat_from_center_idx[0] + beat_offset,
                                                beat_annotations,
                                                strip_beat_extraction_rule['beat_type']):
                    detected_run_start = True
                    detected_run_start = True
                    selection_start_sample, selection_end_sample = calculate_beat_selection_corner_samples(beat_samples[wanted_type_beat_from_center_idx[0] + beat_offset],
                                                                                                        offset_mu,
                                                                                                        offset_std,
                                                                                                        PRE_BEAT_WINDOW_SIZE,
                                                                                                        POST_BEAT_WINDOW_SIZE)

                    if (selection_start_sample >= 0) and (selection_end_sample < strip_sample_length):
                        start_samples_list.append(selection_start_sample)    
                        selected_beat_idx_list.append(wanted_type_beat_from_center_idx[0] + beat_offset)
                        selected_beat_type_list.append(strip_beat_extraction_rule['beat_type']) 
                
                # Check if the pattern of a middle point of a run is detected
                # == DEBUGGING ==
                #if beat_type_rule['beat_type']==2:
                #    print('ok')
                # == == == == ==
                if check_beat_middle_run(wanted_type_beat_from_center_idx[0] + beat_offset +2,
                                        beat_annotations,
                                        strip_beat_extraction_rule['beat_type']):
                    detected_run_end = True
                    selection_start_sample, selection_end_sample = calculate_beat_selection_corner_samples(beat_samples[wanted_type_beat_from_center_idx[0] + beat_offset +2],
                                                                                                        offset_mu,
                                                                                                        offset_std,
                                                                                                        PRE_BEAT_WINDOW_SIZE,
                                                                                                        POST_BEAT_WINDOW_SIZE)

                    if (selection_start_sample >= 0) and (selection_end_sample < strip_sample_length):
                        start_samples_list.append(selection_start_sample)  
                        selected_beat_idx_list.append(wanted_type_beat_from_center_idx[0] + beat_offset +2)
                        selected_beat_type_list.append(strip_beat_extraction_rule['beat_type']) 
                
                # Break the loop if at least one pattern was detected for the current beat_offset
                if any([detected_run_start, detected_run_end]):
                    break
            # == DEBUGGING ==
            # flag with not detected pattern
            if not detected_run_start and not detected_run_end:
                detection_comment += 'None Detected'
            #    print('\nNone Detected:')
            elif detected_run_start and not detected_run_end:
                detection_comment += 'Only Beginning Detected'
            #    print('\nOnly Beginning Detected:') 
            elif detected_run_end and not detected_run_start:
                detection_comment += 'Only Middle Detected'
            #    print('\nOnly Middle Detected:')
            else:
                detection_comment += 'All Detected'
            #    print('\nAll Detected:')   
            #print(beat_annotations)
            #print('ok')     

    return start_samples_list, selected_beat_idx_list, selected_beat_type_list, detection_comment

def return_good_ecg_channel_idx_based_on_lead_off(ecg, lead_off=[], num_max_ecg_channels=1):
    """ Function for selecting an ECG channel index based on the lead-off information.
        INPUTS
                ecg:                        array of the ECG signal
                lead_off:                   list of lead-off information signal
                num_returned_ecg_channels:  integer number of ECG channels to consider
                                            and return their index
        OUTPUTS
                ecg_channel_idx:            list of the "good" ecg channels
                lead_off_indicator:         boolean of whether lead-off was found
    """
    if ecg.ndim == 1:
        good_ecg_channel_idx_list = [0]
        lead_off_indicator = []
    elif lead_off.count(0) == len(lead_off):
        good_ecg_channel_idx_list = sorted(random.sample(range(ecg.shape[0]), num_max_ecg_channels))
        lead_off_indicator = False
    elif lead_off.count(7) > 0:
        good_ecg_channel_idx_list = []
        lead_off_indicator = True
    else:
        n_lead_off_6 = lead_off.count(6) # 6 -> lead-off in ch1, ch2 -> ch3 (idx=2) is good!
        n_lead_off_5 = lead_off.count(5) # 5 -> lead-off in ch1, ch3 -> ch2 (idx=1) is good!
        n_lead_off_3 = lead_off.count(3) # 3 -> lead-off in ch2, ch3 -> ch1 (idx=0) is good!
        lead_off_indicator = True
        if n_lead_off_6 > n_lead_off_5 and n_lead_off_6 > n_lead_off_3:
            good_ecg_channel_idx_list = [2]
        elif n_lead_off_5 > n_lead_off_6 and n_lead_off_5 > n_lead_off_3:
            good_ecg_channel_idx_list = [1]
        elif n_lead_off_3 > n_lead_off_6 and n_lead_off_3 > n_lead_off_5:
            good_ecg_channel_idx_list = [0]
        else:
            good_ecg_channel_idx_list = []
    return good_ecg_channel_idx_list, lead_off_indicator

def AnnotationToBinaryForm(annotations_str, type_positive):
    """ Converts a list of annotations in str() format to binary
        int() values in a list, where (1) is the parsed annotation(s) in
        "type_positive" and (0) is the rest.

        In the case of empty "type_positive" structure, all annotations
        are parsed as 1.
    """
    annotations_int = np.zeros(len(annotations_str), dtype=int)
    for idx, annotation in enumerate(annotations_str):
        if annotation in type_positive:
            annotations_int[idx] = 1
    
    return annotations_int

