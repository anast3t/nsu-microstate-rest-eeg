from __future__ import annotations

import copy
import os
from typing import Self

import pickle
import typing
import mne
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy


class Folders:
    """
    Class for storing the paths to the folders for the data.
    ALL FOLDERS SHOULD END WITH A SLASH
    """

    def __init__(
            self,
            end_folder: str,
            raw_data: str,
            preprocessed_data: str,
            save_data: str,
            statistics: str,
            mhw_objects: str,
            images: str
    ):
        self.end_folder = end_folder
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
        self.ms_ordered = None
        self.splitted_ms = None
        self.split_dynamic_statistics = None
        self.split_static_statistics = None
        self.normative_labels = None

    @staticmethod
    def static_load(
            folders: Folders,
            raw_filename: str
    ) -> Self:
        print("Loading MHW object", raw_filename)
        with open(
                folders.save_data +
                folders.mhw_objects +
                folders.end_folder +
                raw_filename + '.pkl',
                'rb'
        ) as file:
            return pickle.load(file)

    def load(self) -> Self:
        print("Loading MHW object", self.raw_filename)
        with open(
                self.folders.save_data +
                self.folders.mhw_objects +
                self.folders.end_folder +
                self.raw_filename + '.pkl',
                'rb'
        ) as file:
            return pickle.load(file)

    def save(self) -> Self:
        print("Saving MHW object", self.raw_filename)
        folder = self.folders.save_data + self.folders.mhw_objects + self.folders.end_folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder + self.raw_filename + '.pkl', 'wb') as file:
            pickle.dump(self, file)
        return self

    def check_saved(self) -> bool:
        try:
            with open(
                    self.folders.save_data +
                    self.folders.mhw_objects +
                    self.folders.end_folder +
                    self.raw_filename + '.pkl',
                    'rb'
            ) as file:
                return True
        except FileNotFoundError:
            return False

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
    ) -> Self:
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
            full_copy = copy.deepcopy(self)
            full_copy.ms = ms_clone
            return full_copy

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
            time_threshold=20,
            recalc=False
    ) -> Self:
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
        :param recalc: if True, will recalculate the split
        :return: self
        """

        if self.splitted_ms is not None and not recalc:
            print('Already calculated microstates split')
            return self
        print('Calculating microstates split...')

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
    ) -> Self:
        """
        Calculates the microstates of the raw data. Sets the ms attribute. If recalc is False, will return the current
        :param recalc: if True, will recalculate the microstates
        :return: self
        """
        if self.ms is not None and not recalc:
            print('Already calculated microstates')
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

    def split_dynamic_calc_statistics(
            self,
            recalc=False
    ) -> Self:

        if self.split_dynamic_statistics is not None and not recalc:
            print('Already calculated dynamic statistics')
            return self

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
        dynamic.reset_index(drop=True, inplace=True)
        self.split_dynamic_statistics = dynamic
        return self

    def split_dynamic_save_statistics(self) -> Self:
        folder = self.folders.save_data + self.folders.statistics + self.folders.end_folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.split_dynamic_statistics.to_csv(folder + self.raw_filename + '_split_dynamic_stats.csv', index=False)
        print("Saved dynamic statistics")
        return self

    def split_dynamic_drop_self_to_self(self) -> Self:
        for i in range(4):
            # print(f'Microstate_{i}_to_{i}')
            if f'Microstate_{i}_to_{i}' in self.split_dynamic_statistics.columns:
                self.split_dynamic_statistics.drop(f'Microstate_{i}_to_{i}', axis=1, inplace=True)
            else:
                print(f'Microstate_{i}_to_{i} not found or already dropped')
        print("Dropped self-to-self transitions")
        return self

    def split_static_calc_statistics(
            self,
            recalc=False
    ) -> Self:

        try:
            if self.split_static_statistics is not None and not recalc:
                print('Already calculated static statistics')
                return self
        except AttributeError:
            self.split_static_statistics = None

        ms_sequences, events, event_names, timestamps = self.splitted_ms
        static = pd.DataFrame()

        for i in range(len(ms_sequences)):
            print("Event: ", event_names[events[i]])
            duration = (timestamps[i][1] - timestamps[i][0])
            # nk.microstates_plot(ms_sequences[i], epoch = (0, duration))
            ms_static = nk.microstates_static(ms_sequences[i], sampling_rate=self.sampling_rate, show=False)
            ms_static['Event'] = event_names[events[i]]
            ms_static['Order'] = i
            static = pd.concat([static, ms_static])

        print("Calculated static statistics")
        static.reset_index(drop=True, inplace=True)
        self.split_static_statistics = static
        return self

    def split_static_save_statistics(self) -> Self:
        folder = self.folders.save_data + self.folders.statistics + self.folders.end_folder
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.split_static_statistics.to_csv(folder + self.raw_filename + '_split_static_stats.csv', index=False,
                                            index_label=False)
        print("Saved static statistics")
        return self

    def calc_normative_labels(
            self,
            normative_maps,
            recalc=False
    ) -> Self:
        """
        Calculates the normative labels for the microstates sequence.
        :param recalc:
        :param normative_maps: normative maps to use
        :return: self
        """
        try:
            if self.normative_labels is None or recalc:
                print("Normative labels not set, calculating...")
            else:
                print("Normative labels already set")
                return self
        except AttributeError:
            self.normative_labels = None

        def calc_regroup_variants(available: [int], max_len=4, current_len=0):
            if len(available) == 1:
                return available
            variants = []
            current_len += 1
            if current_len == max_len:
                return available
            for num in available:
                available_copy = available.copy()
                available_copy.remove(num)
                if num < max_len and num + max_len in available_copy:
                    available_copy.remove(num + max_len)
                elif num >= max_len and num - max_len in available_copy:
                    available_copy.remove(num - max_len)
                for variant in calc_regroup_variants(available_copy, max_len=max_len, current_len=current_len):
                    if isinstance(variant, int):
                        variants.append([num, variant])
                    else:
                        variants.append([num, *variant])
            return variants

        remap_df_inv = pd.DataFrame(columns=["Distance_A", "Distance_B", "Distance_C", "Distance_D", "label", "inv"])
        labels = {
            0: "A",
            1: "B",
            2: "C",
            3: "D"
        }
        inverted_clusters_user = self.ms["Microstates"] * -1
        clusters_user_and_inv = np.append(self.ms["Microstates"], inverted_clusters_user, axis=0)
        df_clusters_user_and_inv = pd.DataFrame.from_records(clusters_user_and_inv, columns=self.raw.ch_names)

        # print("Self ch_names", self.raw.ch_names)
        # print("Normative ch_names", normative_maps.T.columns)
        # common_cols = [col for col in set(normative_maps.T.columns).intersection(self.raw.ch_names)]
        # print("Common cols", common_cols)
        # common_idx = [self.raw.ch_names.index(col) for col in common_cols]
        # common_idx.sort()
        # print("Common idx", common_idx)
        # clusters_user_and_inv_common = clusters_user_and_inv.T[common_idx].T
        # normative_maps_common = normative_maps.T[common_cols]
        # distances = scipy.spatial.distance.cdist(clusters_user_and_inv_common, normative_maps_common, "correlation")
        normative_maps_common = normative_maps.T
        print(df_clusters_user_and_inv.shape, normative_maps_common.shape)
        print(df_clusters_user_and_inv.columns, normative_maps_common.columns)
        normative_maps_common = normative_maps_common.drop([col for col in normative_maps_common.columns if col not in df_clusters_user_and_inv.columns and col in normative_maps_common.columns], axis=1)
        print(df_clusters_user_and_inv.shape, normative_maps_common.shape, normative_maps.shape)
        distances = scipy.spatial.distance.cdist(df_clusters_user_and_inv, normative_maps_common)
        min_combination = []
        min_distance = np.inf
        min_nums = []
        for variant in (calc_regroup_variants([0, 1, 2, 3, 4, 5, 6, 7], max_len=4)):
            dist = 0
            for i in range(4):
                dist += distances.T[i, variant[i]]
            if dist < min_distance:
                min_distance = dist
                min_combination = variant
                min_nums = [distances.T[i, variant[i]] for i in range(4)]

        print("##########")
        print(min_combination)
        print(min_nums)
        print(min_distance)
        print(distances)
        print("##########")

        for enum_idx, i in enumerate(min_combination):
            # print(enum_idx, i)
            # label = (labels[i] if i < 4 else labels[i-4]) if i in min_combination else ""
            real_idx = i if i < 4 else i - 4
            remap_df_inv = pd.concat([
                remap_df_inv,
                pd.DataFrame(
                    [[
                        distances.T[0, i], distances.T[1, i], distances.T[2, i], distances.T[3, i],
                        labels[enum_idx],
                        i >= 4
                    ]],
                    columns=["Distance_A", "Distance_B", "Distance_C", "Distance_D", "label", "inv"],
                    index=[real_idx]
                )
            ])
        self.normative_labels = remap_df_inv

        return self

    def apply_normative_labels(self, recalc=False) -> Self:

        try:
            if self.ms_ordered is not None and not recalc:
                print("Already applied microstates ordering")
                return self
        except AttributeError:
            print("No normative labels set")
            self.ms_ordered = None
        print("Applying normative labels")

        def create_remapper(normative_labels):
            label_idx_remapper = {
                "A": 0,
                "B": 1,
                "C": 2,
                "D": 3
            }
            idx_idx_remapper = {}
            for i in range(4):
                idx_idx_remapper[i] = label_idx_remapper[normative_labels.T[i]["label"]]
            print("Remapper", idx_idx_remapper)
            return idx_idx_remapper

        def reorder_microstates(ms, remapper, normative_labels):
            ms_clone = copy.deepcopy(ms)
            for i in range(len(ms["Sequence"])):
                ms_clone["Sequence"][i] = remapper[ms["Sequence"][i]]
            # print(ms["Sequence"][:1000], "\n\n\n", ms_clone["Sequence"][:1000])
            ms_array_clone = copy.deepcopy(ms_clone["Microstates"])
            for i in range(4):
                invert = (-1 if normative_labels.T[i].inv else 1)
                # invert = 1
                print(i, "->", remapper[i], "Inverted", invert)
                ms_clone["Microstates"][remapper[i]] = (ms_array_clone[i] * invert)

            return ms_clone

        self.ms_ordered = reorder_microstates(self.ms, create_remapper(self.normative_labels), self.normative_labels)
        return self
