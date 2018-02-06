%matplotlib inline
from generate_sample_frames import *

# set up the TrackModel
track_model = TrackModel()
track_model.mu_c = 10

# generate sample background and tracks
with SampleGenerator(track_model) as generator:
    generator.background()
    generator.tracks()
    generator.save()
