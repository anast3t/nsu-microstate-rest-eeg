import pickle
import typing
import mne
import neurokit2 as nk
import pandas as pd


class Folders:
    """
    Class for storing the paths to the folders for the data.
    ALL FOLDERS SHOULD END WITH A SLASH
    """
    def __init__(
            self,
            raw_data: str,
            preprocessed_data: str,
            save_data: str,
            statistics: str,
            mhw_objects: str,
            images: str
    ):
        self.raw_data = raw_data
        self.preprocessed_data = preprocessed_data
        self.save_data = save_data
        self.statistics = statistics
        self.mhw_objects = mhw_objects
        self.images = images


class MicrostateHelperWrapper:
    def __init__(
            self,
            folders: Folders,
            raw: mne.io.Raw,
            raw_filename: str
    ):
        self.raw = raw
        self.raw_filename = raw_filename
        self.folders = folders
        self.sampling_rate = raw.info['sfreq']
        self.ms = None
        self.splitted_ms = None
        self.dynamic_statistics = None

    def load(self):
        print("Loading MHW object")
        with open(self.folders.save_data + self.folders.mhw_objects + self.raw_filename + '.pkl', 'rb') as file:
            return pickle.load(file)

    def save(self):
        print("Saving MHW object")
        with open(self.folders.save_data + self.folders.mhw_objects + self.raw_filename + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    def split_ms_sequence(
            self,
            start_sample=0,
            end_sample=0,
            start_time=0,
            time_end=0
    ):
        """
        Splits the sequence of a microstates object from a given start_sample to a given end_sample.
        If time_end is set, it will convert the time to samples using the sampling_rate, ignoring passed samples.

        :param start_sample: starting sample of the sequence
        :param end_sample: ending sample of the sequence
        :param start_time: starting time of the sequence
        :param time_end: ending time of the sequence
        :return: Microstates object with the sequence sliced from start to end
        """

        sampling_rate = self.sampling_rate
        ms_copy = self.ms.copy()

        if time_end != 0 and end_sample != 0:
            raise ValueError("Only one of time_end or end_sample should be set")

        if time_end != 0:
            end_sample = int(time_end * sampling_rate)
        if start_time != 0:
            start_sample = int(start_time * sampling_rate)

        ms_copy['Sequence'] = ms_copy['Sequence'][start_sample:end_sample]
        ms_copy['GFP'] = ms_copy['GFP'][start_sample:end_sample]
        return ms_copy

    def apply_basic_switch_threshold(
            self,
            inplace=False,
            threshold=0.02,
    ):
        """
        Applies the basic switch threshold algorithm to the microstates sequence.
        Removes "noisy" microstates that are shorter than the threshold.

        :param threshold: threshold for the minimum duration (is seconds) of a microstate
        :param inplace: if True, will apply the changes to the current microstates object
        :return: Microstates object with the basic switch threshold applied
        """

        sampling_rate = self.sampling_rate
        ms_clone = self.ms.copy()

        sequence = ms_clone['Sequence'].copy()
        gfp = ms_clone['GFP']
        threshold_samples = threshold * sampling_rate
        print("Threshold samples", threshold_samples)
        while True:
            while True:
                intervals = []
                candidates = []
                for i in range(len(sequence) - 1):
                    if sequence[i] != sequence[i + 1]:
                        intervals.append((i, (i - intervals[-1][0] if len(intervals) > 0 else i), sequence[i - 1]))
                intervals.append(
                    (len(sequence) - 1, len(sequence) - 1 - intervals[-1][0], sequence[-1], (len(sequence) - 1) / 2048))
                # print(np.array(intervals))
                i = 1
                while i < len(intervals) - 1:
                    if (intervals[i][1] < threshold_samples) and (intervals[i - 1][2] == intervals[i + 1][2]):
                        # length = intervals[i][1] + intervals[i-1][1] + intervals[i+1][1]
                        candidates.append((intervals[i - 1], intervals[i], intervals[i + 1]))
                        i += 2
                    i += 1
                if len(candidates) == 0:
                    break
                for candidate in candidates:
                    start = candidate[0][0]
                    end = candidate[1][0] + 1
                    state = candidate[0][2]
                    # print("Filling candidate", start, end, state)
                    # nk.microstates_plot(ms_clone, epoch = (start-50, end+50))
                    sequence[start:end] = state
                    # ms_clone['Sequence'] = sequence
                    # nk.microstates_plot(ms_clone, epoch = (start-50, end+50))
                # time.sleep(500)
            # ms_clone['Sequence'] = sequence
            # nk.microstates_plot(ms_clone, epoch = (0, int(2048)))
            # print("Second stage candidates")

            # while True:
            intervals = []
            candidates = []
            local_gfp_mean = 0
            for i in range(len(sequence) - 1):
                local_gfp_mean += gfp[i]
                if sequence[i] != sequence[i + 1]:
                    # print("Adding sequence", i, "State", sequence[i], "State", sequence[i+1])
                    length = (i - intervals[-1][0] if len(intervals) > 0 else i)
                    if length == 0:
                        length = 1
                    state = sequence[i - 1]
                    local_gfp_mean /= length
                    intervals.append((i, length, state, local_gfp_mean))
                    local_gfp_mean = 0
            intervals.append((len(sequence) - 1, len(sequence) - 1 - intervals[-1][0], sequence[-1], 0))
            # print(intervals)
            i = 1
            while i < len(intervals) - 1:
                # print("Checking interval", intervals[i], threshold_samples)
                if intervals[i][1] < threshold_samples:
                    # gfp_diff_l = intervals[i][3] - intervals[i-1][3]
                    # gfp_diff_r = intervals[i][3] - intervals[i+1][3]
                    # print("GFP diff", gfp_diff_l, gfp_diff_r)
                    ln_diff_l = (intervals[i][1] - intervals[i - 1][1])
                    ln_diff_r = intervals[i][1] - intervals[i + 1][1]
                    if ln_diff_l > 0 and ln_diff_r > 0:
                        i += 1
                        continue

                    # gfp_diff_l = intervals[i][3] - intervals[i-1][3]
                    # gfp_diff_r = intervals[i][3] - intervals[i+1][3]

                    # print("LN diff", ln_diff_l, ln_diff_r, intervals[i][1])

                    # if gfp_diff_l > gfp_diff_r:
                    if ln_diff_l < ln_diff_r:
                        candidates.append((intervals[i - 1], intervals[i], intervals[i - 1][2]))
                    else:
                        candidates.append((intervals[i - 1], intervals[i], intervals[i + 1][2]))
                    i += 2
                i += 1
            if len(candidates) == 0:
                break
            # start_min = 10000000
            # end_max = 0
            for candidate in candidates:
                start = candidate[0][0]
                # if start < start_min:
                #     start_min = start
                end = candidate[1][0] + 1
                # if end > end_max:
                #     end_max = end
                state = candidate[2]
                # print("Filling candidate", start, end, state)
                # nk.microstates_plot(ms_clone, epoch = (start-40, end+40))
                sequence[start:end] = state
                # ms_clone['Sequence'] = sequence
                # nk.microstates_plot(ms_clone, epoch = (start-40, end+40))
                # print('------')
        # print(sequence[:100])
        ms_clone['Sequence'] = sequence

        print("Applied basic switch threshold")

        if inplace:
            self.ms = ms_clone
            return self
        else:
            return ms_clone

    def get_event_bounds_by_event_transitions(
            self,
            key_names,
            key_namings,
            transitions,
            time_threshold=20,
    ):
        """
        Extracts event bounds from the raw data annotations using the event transitions.
        :param key_names: annotation names of the events to look for
        :param key_namings: annotation names mapped to event meaning
        :param transitions: event transitions to look for in format [[from_event_name, [to_event_names]]]
        :param time_threshold: minimum duration of the event in seconds
        :return:
            (timestamps - [[start_sample, end_sample]],
            events - [event_meaning_name],
            event_names - {annotation_number: event_meaning_name})

        """

        raw = self.raw
        sampling_rate = self.sampling_rate

        def check_event_name(pair, key_names):
            print(pair)
            key, value = pair
            for key_name in key_names:
                if key_name in key:
                    return True
            return False

        raw_events = mne.events_from_annotations(raw)
        event_sequence = list(map(lambda x: [x[0], x[2]], raw_events[0]))
        event_namemap = dict(filter(lambda x: check_event_name(x, key_names), raw_events[1].items()))
        event_numbers = list(event_namemap.values())
        threshold_samples = time_threshold * sampling_rate

        def get_event_number(event_name, _ev_namemap):
            return list(filter(lambda val: event_name in val[0], _ev_namemap.items()))[0]

        def remap_transitions_to_numbers(trans, ev_namemap):
            remapped_transitions = []
            for transition in trans:
                remapped_transitions.append(
                    [get_event_number(transition[0], ev_namemap)[1],
                     list(map(lambda x: get_event_number(x, ev_namemap)[1], transition[1]))]
                )
            return remapped_transitions

        transition_numbers = remap_transitions_to_numbers(transitions, event_namemap)
        filtered_sequence = list(filter(lambda x: x[1] in event_numbers, event_sequence))
        timestamps = []
        events = []
        for idx in range(len(filtered_sequence)):
            event = filtered_sequence[idx]
            next_event = filtered_sequence[idx + 1] if idx + 1 < len(filtered_sequence) else None
            # print("Idx", idx, "Event", event, "Next event", next_event)
            if next_event is None:
                break
            for transition in transition_numbers:
                if (event[1] == transition[0]) and (next_event[1] in transition[1]):
                    if threshold_samples > next_event[0] - event[0]:
                        print("Event too short, skipping", next_event[0] - event[0], event, next_event)
                        break
                    timestamps.append([event[0], next_event[0]])
                    events.append(event[1])
                    break
        event_names = dict(map(lambda x: (get_event_number(x[0], event_namemap)[1], x[1]), key_namings.items()))
        return timestamps, events, event_names

    def split_ms_sequence_by_events(
            self,
            key_names,
            key_namings,
            transitions,
            time_threshold
    ):
        """
        Splits the microstates sequence by the events in the raw data annotation. Sets in the splitted_ms attribute.

        splitted_ms attribute =
            (timestamps - [[start_sample, end_sample]],
            events - [event_meaning_name],
            event_names - {annotation_number: event_meaning_name})

        :param key_names: annotation names of the events to look for
        :param key_namings: annotation names mapped to event meaning
        :param transitions: event transitions to look for in format [[from_event_name, [to_event_names]]]
        :param time_threshold: minimum duration of the event in seconds
        :return: self
        """

        timestamps, events, event_names = self.get_event_bounds_by_event_transitions(
            key_names,
            key_namings,
            transitions,
            time_threshold=time_threshold
        )
        sequences = []

        for i in range(len(timestamps)):
            ms_copy = self.split_ms_sequence(start_sample=timestamps[i][0], end_sample=timestamps[i][1])
            sequences.append(ms_copy)

        self.splitted_ms = (sequences, events, event_names, timestamps)

        print("Splitted by events")
        return self

    # TODO: params for the microstates_segment
    def calc_raw_ms(
            self,
            recalc=False
    ):
        """
        Calculates the microstates of the raw data. Sets the ms attribute. If recalc is False, will return the current
        :param recalc: if True, will recalculate the microstates
        :return: self
        """
        if self.ms is not None and not recalc:
            return self

        print('Calculating microstates...')
        microstates = nk.microstates_segment(
            self.raw,
            n_microstates=4,
            method='kmod',
            random_state=42,
            optimize=True
        )
        self.ms = microstates

        return self

    def calc_event_split(
            self,
            recalc=False,
            key_names=None,
            key_namings=None,
            transitions=None,
            time_threshold=20
    ):
        """
        Calculates the microstates split by the events in the raw data annotations. Sets the splitted_ms attribute.
        If recalc is False, will return the current

        :param recalc: If True, will recalculate the split
        :param key_names: annotation names of the events to look for
        :param key_namings: annotation names mapped to event meaning
        :param transitions: event transitions to look for in format [[from_event_name, [to_event_names]]]
        :param time_threshold: minimum duration of the event in seconds
        :return: self
        """
        if self.splitted_ms is not None and not recalc:
            return self
        print('Calculating microstates split...')
        self.splitted_ms = self.split_ms_sequence_by_events(key_names, key_namings, transitions, time_threshold)
        return self

    def dynamic_calculate_statistics_for_split(self):

        ms_sequences, events, event_names, timestamps = self.splitted_ms
        dynamic = pd.DataFrame()

        for i in range(len(ms_sequences)):
            print("Event: ", event_names[events[i]])
            duration = (timestamps[i][1] - timestamps[i][0])
            # nk.microstates_plot(ms_sequences[i], epoch = (0, duration))
            # nk.microstates_static(ms_sequences[i], sampling_rate=sampling_rate, show=True)
            ms_dynamic = nk.microstates_dynamic(ms_sequences[i], show=False)
            ms_dynamic['Event'] = event_names[events[i]]
            ms_dynamic['Order'] = i
            dynamic = pd.concat([dynamic, ms_dynamic])

        print("Calculated dynamic statistics")
        self.dynamic_statistics = dynamic
        return self

    def dynamic_save_statistics(self):
        self.dynamic_statistics.to_csv(
            self.folders.save_data + self.folders.statistics + self.raw_filename + '_dynamic_stats.csv'
        )
        print("Saved dynamic statistics")
        return self

    def dynamic_drop_self_to_self(self):
        for i in range(4):
            # print(f'Microstate_{i}_to_{i}')
            self.dynamic_statistics.drop(f'Microstate_{i}_to_{i}', axis=1, inplace=True)
        print("Dropped self-to-self transitions")
        return self
