import numpy as np
import csv
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter



def read_csv(file_path):
    import csv
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]
    return np.array(data)

# Butterworth filter functions
def butter_lowpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass', return_top=False):
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass(cutoff, sample_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        assert type(cutoff) in [tuple, list, np.ndarray], 'If bandpass filter is specified, cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass(cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff, Q=0.005, fs=sample_rate)
    else:
        raise ValueError('filtertype: %s is unknown, available are: lowpass, highpass, bandpass, and notch' % filtertype)

    filtered_data = filtfilt(b, a, data)
    
    if return_top:
        return np.clip(filtered_data, a_min=0, a_max=None)
    else:
        return filtered_data

def remove_baseline_wander(data, sample_rate, cutoff=0.05):
    return filter_signal(data=data, cutoff=cutoff, sample_rate=sample_rate, filtertype='notch')

def hampel_filter(data, filtsize=6):
    output = np.copy(np.asarray(data)) 
    onesided_filt = filtsize // 2
    for i in range(onesided_filt, len(data) - onesided_filt - 1):
        dataslice = output[i - onesided_filt : i + onesided_filt]
        mad = MAD(dataslice)
        median = np.median(dataslice)
        if output[i] > median + (3 * mad):
            output[i] = median
    return output

def hampel_correcter(data, sample_rate):
    return data - hampel_filter(data, filtsize=int(sample_rate))

def quotient_filter(RR_list, RR_list_mask=[], iterations=2):
    if len(RR_list_mask) == 0:
        RR_list_mask = np.zeros((len(RR_list)))
    else:
        assert len(RR_list) == len(RR_list_mask), 'error: RR_list and RR_list_mask should be same length if RR_list_mask is specified'

    for iteration in range(iterations):
        for i in range(len(RR_list) - 1):
            if RR_list_mask[i] + RR_list_mask[i + 1] != 0:
                pass 
            elif 0.8 <= RR_list[i] / RR_list[i + 1] <= 1.2:
                pass 
            else: 
                RR_list_mask[i] = 1

    return np.asarray(RR_list_mask)

def smooth_signal(data, sample_rate, window_length=None, polyorder=3):
    if window_length is None:
        window_length = sample_rate // 10
        
    if window_length % 2 == 0 or window_length == 0: window_length += 1

    smoothed = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    return smoothed

# Functions from test1.py, test2.py, and test3.py
def make_windows(data, sample_rate, windowsize=120, overlap=0, min_size=20):
    ln = len(data)
    window = windowsize * sample_rate
    stepsize = (1 - overlap) * window
    windows = []
    for start in range(0, ln - window + 1, int(stepsize)):
        end = start + window
        if len(data[start:end]) >= min_size * sample_rate:
            windows.append(data[start:end])
    return windows

def append_dict(dict_obj, measure_key, measure_value):
    if measure_key in dict_obj:
        dict_obj[measure_key].append(measure_value)
    else:
        dict_obj[measure_key] = [measure_value]

def detect_peaks(hrdata, rol_mean, ma_perc, sample_rate, update_dict=True, working_data={}):
    rmean = np.array(rol_mean)
    mn = np.mean(rmean / 100) * ma_perc
    rol_mean = rmean + mn
    peaksx = np.where((hrdata > rol_mean))[0]
    peaksy = hrdata[peaksx]
    peakedges = np.concatenate((np.array([0]), (np.where(np.diff(peaksx) > 1)[0]), np.array([len(peaksx)])))
    peaklist = []
    for i in range(0, len(peakedges)-1):
        try:
            y_values = peaksy[peakedges[i]:peakedges[i+1]].tolist()
            peaklist.append(peaksx[peakedges[i] + y_values.index(max(y_values))])
        except:
            pass
    if update_dict:
        working_data['peaklist'] = peaklist
        working_data['ybeat'] = [hrdata[x] for x in peaklist]
        working_data['rolling_mean'] = rol_mean
        working_data = calc_rr(working_data['peaklist'], sample_rate, working_data=working_data)
        if len(working_data['RR_list']) > 0:
            working_data['rrsd'] = np.std(working_data['RR_list'])
        else:
            working_data['rrsd'] = np.inf
        return working_data
    else:
        return peaklist, working_data

def fit_peaks(hrdata, rol_mean, sample_rate, bpmmin=40, bpmmax=180, working_data={}):
    ma_perc_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300]
    rrsd = []
    valid_ma = []
    for ma_perc in ma_perc_list:
        working_data = detect_peaks(hrdata, rol_mean, ma_perc, sample_rate, update_dict=True, working_data=working_data)
        bpm = ((len(working_data['peaklist'])/(len(hrdata)/sample_rate))*60)
        rrsd.append([working_data['rrsd'], bpm, ma_perc])
    for _rrsd, _bpm, _ma_perc in rrsd:
        if (_rrsd > 0.1) and ((bpmmin <= _bpm <= bpmmax)):
            valid_ma.append([_rrsd, _ma_perc])
    if len(valid_ma) > 0:
        working_data['best'] = min(valid_ma, key=lambda t: t[0])[1]
        working_data = detect_peaks(hrdata, rol_mean, min(valid_ma, key=lambda t: t[0])[1], sample_rate, update_dict=True, working_data=working_data)
        return working_data
    else:
        raise ValueError('Could not determine best fit for given signal.')

def check_peaks(rr_arr, peaklist, ybeat, reject_segmentwise=False, working_data={}):
    rr_arr = np.array(rr_arr)
    peaklist = np.array(peaklist)
    ybeat = np.array(ybeat)
    mean_rr = np.mean(rr_arr)
    thirty_perc = 0.3 * mean_rr
    if thirty_perc <= 300:
        upper_threshold = mean_rr + 300
        lower_threshold = mean_rr - 300
    else:
        upper_threshold = mean_rr + thirty_perc
        lower_threshold = mean_rr - thirty_perc
    rem_idx = np.where((rr_arr <= lower_threshold) | (rr_arr >= upper_threshold))[0] + 1
    working_data['removed_beats'] = peaklist[rem_idx]
    working_data['removed_beats_y'] = ybeat[rem_idx]
    working_data['binary_peaklist'] = np.asarray([0 if x in working_data['removed_beats'] else 1 for x in peaklist])
    if reject_segmentwise:
        working_data = check_binary_quality(peaklist, working_data['binary_peaklist'], working_data=working_data)
    return working_data

def check_binary_quality(peaklist, binary_peaklist, maxrejects=3, working_data={}):
    idx = 0
    working_data['rejected_segments'] = []
    for i in range(int(len(binary_peaklist) / 10)):
        if np.bincount(binary_peaklist[idx:idx + 10])[0] > maxrejects:
            binary_peaklist[idx:idx + 10] = [0 for i in range(len(binary_peaklist[idx:idx+10]))]
            if idx + 10 < len(peaklist):
                working_data['rejected_segments'].append((peaklist[idx], peaklist[idx + 10]))
            else:
                working_data['rejected_segments'].append((peaklist[idx], peaklist[-1]))
        idx += 10
    return working_data

def calc_rr(peaklist, sample_rate, working_data={}):
    peaklist = np.array(peaklist)
    if len(peaklist) > 0:
        if peaklist[0] <= ((sample_rate / 1000.0) * 150):
            peaklist = np.delete(peaklist, 0)
            working_data['ybeat'] = np.delete(working_data['ybeat'], 0)
    working_data['peaklist'] = peaklist

    rr_list = (np.diff(peaklist) / sample_rate) * 1000.0
    rr_indices = [(peaklist[i], peaklist[i+1]) for i in range(len(peaklist) - 1)]
    rr_diff = np.abs(np.diff(rr_list))
    rr_sqdiff = np.power(rr_diff, 2)
    working_data['RR_list'] = rr_list
    working_data['RR_indices'] = rr_indices
    working_data['RR_diff'] = rr_diff
    working_data['RR_sqdiff'] = rr_sqdiff
    return working_data

def update_rr(working_data={}):
    rr_source = working_data['RR_list']
    b_peaklist = working_data['binary_peaklist']
    rr_list = np.array([rr_source[i] for i in range(len(rr_source)) if b_peaklist[i] + b_peaklist[i+1] == 2])
    rr_mask = np.array([0 if (b_peaklist[i] + b_peaklist[i+1] == 2) else 1 for i in range(len(rr_source))])
    rr_masked = np.ma.array(rr_source, mask=rr_mask)
    rr_diff = np.abs(np.diff(rr_masked))
    rr_diff = rr_diff[~rr_diff.mask]
    rr_sqdiff = np.power(rr_diff, 2)

    working_data['RR_masklist'] = rr_mask
    working_data['RR_list_cor'] = rr_list
    working_data['RR_diff'] = rr_diff
    working_data['RR_sqdiff'] = rr_sqdiff

    return working_data

def clean_rr_intervals(working_data, method='quotient-filter'):
    RR_cor_indices = [i for i in range(len(working_data['RR_masklist'])) if working_data['RR_masklist'][i] == 0]
    if method.lower() == 'iqr':
        rr_cleaned, replaced_indices = outliers_iqr_method(working_data['RR_list_cor'])
        rr_mask = working_data['RR_masklist']
        for i in replaced_indices:
            rr_mask[RR_cor_indices[i]] = 1
    elif method.lower() == 'z-score':
        rr_cleaned, replaced_indices = outliers_modified_z(working_data['RR_list_cor'])
        rr_mask = working_data['RR_masklist']
        for i in replaced_indices:
            rr_mask[RR_cor_indices[i]] = 1
    elif method.lower() == 'quotient-filter':
        rr_mask = quotient_filter(working_data['RR_list'], working_data['RR_masklist'])
        rr_cleaned = [x for x,y in zip(working_data['RR_list'], rr_mask) if y == 0]
    else:
        raise ValueError('Incorrect method specified, use either "iqr", "z-score" or "quotient-filtering". Nothing to do!')
    rr_masked = np.ma.array(working_data['RR_list'], mask=rr_mask)
    rr_diff = np.abs(np.diff(rr_masked))
    rr_diff = rr_diff[~rr_diff.mask]
    rr_sqdiff = np.power(rr_diff, 2)
    working_data['RR_masked'] = rr_masked
    working_data['RR_list_cor'] = np.asarray(rr_cleaned)
    working_data['RR_diff'] = rr_diff
    working_data['RR_sqdiff'] = rr_sqdiff
    try:
        removed_beats = [x for x in working_data['removed_beats']]
        removed_beats_y = [x for x in working_data['removed_beats_y']]
        peaklist = working_data['peaklist']
        ybeat = working_data['ybeat']
        for i in range(len(rr_mask)):
            if rr_mask[i] == 1 and peaklist[i] not in removed_beats:
                removed_beats.append(peaklist[i])
                removed_beats_y.append(ybeat[i])
        working_data['removed_beats'] = np.asarray(removed_beats)
        working_data['removed_beats_y'] = np.asarray(removed_beats_y)
    except:
        pass
    return working_data

