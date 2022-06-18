# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 02:27:37 2022

@author: Ahmed Ben Sassi
"""

import json
import numpy as np
from tensorflow import keras as keras
import pretty_midi
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:

    def __init__(self, model_path="model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-max_sequence_length:]
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]
            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilites, temperature):
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites))  # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index

    def save_melody(self, melody, step_duration=0.125, file_name="mel.mid"):
        # Construct a PrettyMIDI object with tempo of 60bpm.
        pm = pretty_midi.PrettyMIDI(initial_tempo=60)

        # Add na instrument named 'takht' with no drums XD
        inst = pretty_midi.Instrument(program=42, is_drum=False, name='takht')
        pm.instruments.append(inst)
        velocity = 100

        start_symbol = None
        step_counter = 1
        absolut_time = 0

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    # handle rest
                    if start_symbol == "Es":
                        absolut_time += step_counter * step_duration

                    # handle note
                    else:
                        start = absolut_time
                        end = start + (step_counter * step_duration)
                        # handle pitch bends
                        if start_symbol[-1] == ">":
                            inst.pitch_bends.append(pretty_midi.PitchBend(2048, start))
                            pitch = pretty_midi.note_name_to_number(start_symbol[:-1])
                        else:
                            inst.pitch_bends.append(pretty_midi.PitchBend(0, start))
                            pitch = pretty_midi.note_name_to_number(start_symbol)

                        inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
                        absolut_time += step_counter * step_duration

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol
            else:
                step_counter += 1
        pm.write(file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "D3 _ _ _ _ _ _ _ D#3> _ _ _ _ _ _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.4)
    print(melody)
    mg.save_melody(melody)
